import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyDecomposition(nn.Module):
    def __init__(self, seq_len, cutoff_ratio=0.5):
        super(FrequencyDecomposition, self).__init__()
        self.seq_len = seq_len
        self.freq_len = seq_len // 2 + 1  
        self.cutoff = int(self.freq_len * cutoff_ratio)  
        
    def forward(self, x):
        # x: [batch_size, N, seq_len]
        B, N, L = x.shape
        assert L == self.seq_len, f"Expected sequence length {self.seq_len}, got {L}"
        
        
        x_fft = torch.fft.rfft(x, dim=-1)  # [B, N, freq_len]
        assert x_fft.shape == (B, N, self.freq_len), f"FFT shape error: {x_fft.shape}"
        
        
        low_freq = x_fft.clone()
        high_freq = x_fft.clone()
        low_freq[..., self.cutoff:] = 0  
        high_freq[..., :self.cutoff] = 0  
        
        return low_freq, high_freq  
    
    def inverse(self, low_freq, high_freq):
        
        B, N, F = low_freq.shape
        assert high_freq.shape == low_freq.shape, f"Frequency shapes mismatch: {low_freq.shape} vs {high_freq.shape}"
        
        
        x_fft = low_freq + high_freq  # [B, N, F]
        
        
        target_len = (F - 1) * 2  
        
        
        x = torch.fft.irfft(x_fft, n=target_len, dim=-1)  # [B, N, target_len]
        return x

class InteractionBlock(nn.Module):
    def __init__(self, channels):
        super(InteractionBlock, self).__init__()
        self.channels = channels
        self.channels_half = channels // 2
        
        def create_network():
            return nn.Sequential(
                nn.Linear(self.channels_half, self.channels),
                nn.LayerNorm(self.channels),
                nn.ReLU(),
                nn.Linear(self.channels, self.channels_half)
            )
        
        self.s1 = create_network()
        self.t1 = create_network()
        self.s2 = create_network()
        self.t2 = create_network()
        self.scale_1 = nn.Parameter(torch.tensor(0.1))
        self.scale_2 = nn.Parameter(torch.tensor(0.1))
        
        
    def forward(self, x):
        # x: [B, channels]
        B = x.shape[0]
        assert x.shape == (B, self.channels), f"Expected shape (B, {self.channels}), got {x.shape}"
        
        x1, x2 = torch.split(x, self.channels_half, dim=-1)  
        
        s1 = self.scale_1 * self.s1(x2)
        t1 = self.t1(x2)
        y1 = x1 * torch.exp(torch.clamp(s1, -5, 5)) + t1  # [B, channels//2]
        
        s2 = self.scale_2 * self.s2(y1)
        t2 = self.t2(y1)
        y2 = x2 * torch.exp(torch.clamp(s2, -5, 5)) + t2  # [B, channels//2]
        
        return torch.cat([y1, y2], dim=-1)  # [B, channels]
    

class FrequencyModel(nn.Module):
    def __init__(self, configs):
        super(FrequencyModel, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        
        assert self.d_model % 2 == 0, "d_model must be even for invertible blocks"
        

        self.freq_len = self.seq_len // 2 + 1
        self.freq_dims = self.freq_len * 2
        

        self.freq_decomp = FrequencyDecomposition(seq_len=self.seq_len)
        

        self.freq_embedding_low = nn.Linear(self.freq_dims, self.d_model // 2)
        self.freq_embedding_high = nn.Linear(self.freq_dims, self.d_model // 2)
        

        self.inn_blocks = nn.ModuleList([
            InteractionBlock(self.d_model) 
            for _ in range(configs.e_layers)
        ])

        pred_freq_len = self.pred_len // 2 + 1
        self.pred_freq_dims = pred_freq_len * 2
        self.freq_predictor_low = nn.Linear(self.d_model // 2, self.pred_freq_dims)
        self.freq_predictor_high = nn.Linear(self.d_model // 2, self.pred_freq_dims)
        
    def forward(self, x_enc):
        # x_enc: [B, L, N]
        B, L, N = x_enc.shape                                                                                                                

        x_enc = x_enc.transpose(1, 2)  # [B, N, L]
        low_freq, high_freq = self.freq_decomp(x_enc)
        

        freq_features_low = torch.cat([
            low_freq.real, low_freq.imag,
        ], dim=-1)
        freq_features_high = torch.cat([
            high_freq.real, high_freq.imag,
        ], dim=-1)
        

        x_low = self.freq_embedding_low(freq_features_low)
        x_high = self.freq_embedding_high(freq_features_high)
        x = torch.cat([x_low, x_high], dim=-1)  # [B, N, d_model]
        
        x = x.reshape(B * N, self.d_model)
        

        z = x
        for block in self.inn_blocks:
            z = block(z)
            

        z = z.view(B, N, self.d_model)

        z_low, z_high = torch.split(z, self.d_model // 2, dim=-1)

        freq_pred_low = self.freq_predictor_low(z_low)
        freq_pred_high = self.freq_predictor_high(z_high)
        

        pred_freq_len = self.pred_len // 2 + 1
        pred_freq_splits_low = torch.split(freq_pred_low, pred_freq_len, dim=-1)
        pred_freq_splits_high = torch.split(freq_pred_high, pred_freq_len, dim=-1)
        pred_low_freq = torch.complex(pred_freq_splits_low[0], pred_freq_splits_low[1])
        pred_high_freq = torch.complex(pred_freq_splits_high[0], pred_freq_splits_high[1])
        

        pred = self.freq_decomp.inverse(pred_low_freq, pred_high_freq)
        pred = pred.transpose(1, 2)  # [B, pred_len, N]
        
        return pred



class UnfoldAndReshape(nn.Module):
    def __init__(self, period):
        super().__init__()
        self.period = period
        
    def forward(self, x):
        # x shape: [batch_size, pred_len, features]
        batch_size, pred_len, features = x.shape

        x = x.transpose(1, 2)  # [batch_size, features, pred_len]
        x = x.unfold(dimension=-1, size=self.period, step=self.period)

        x = x.transpose(1, 2) 
        new_len = pred_len // self.period
        x = x.reshape(batch_size, new_len, features, self.period)
        return x


class FoldBackToSequence(nn.Module):
    def __init__(self, pred_len):
        super().__init__()
        self.pred_len = pred_len
        
    def forward(self, x):
        # x shape: [batch_size, new_len, features, period]
        batch_size, new_len, features, period = x.shape

        x = x.reshape(batch_size, self.pred_len, features)
        return x

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
    
        self.freq_model = FrequencyModel(configs)


        
        self.time_dim = len(configs.time_feature_types)
        self.time_proj = nn.Linear(configs.pred_len, configs.pred_len)
        
        self.time_enc = nn.Sequential(
                    nn.Linear(self.time_dim, configs.c_out//configs.rda),
                    nn.LayerNorm(configs.c_out//configs.rda),
                    nn.ReLU(),
                    nn.Linear(configs.c_out//configs.rda, configs.c_out//configs.rdb),
                    nn.LayerNorm(configs.c_out//configs.rdb),
                    nn.ReLU(),
                    
                    UnfoldAndReshape(period=configs.period),
                    
                    nn.Conv2d(in_channels=configs.pred_len//configs.period,  
                            out_channels=configs.pred_len//configs.period,  
                            kernel_size=configs.ksize,  
                            padding='same'),
                    
                    FoldBackToSequence(pred_len=configs.pred_len),
                    nn.Linear(configs.c_out//configs.rdb, configs.c_out),
        )
        
        self.beta = configs.beta
        
    def forecast(self, x_enc, x_mark_dec):
        B, L, N = x_enc.shape
        assert L == self.freq_model.seq_len, f"Expected sequence length {self.freq_model.seq_len}, got {L}"
        
        
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev
        
        
        freq_pred = self.freq_model(x_enc)
        
        pred  = freq_pred
        
        
        time_embed = self.time_enc(x_mark_dec)
        time_pred = self.time_proj(time_embed.transpose(1, 2)).transpose(1, 2)
        
        
        pred = self.beta * freq_pred + (1 - self.beta) * time_pred
        
       
        pred = pred * stdev + means
        
        return pred

    
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        pred = self.forecast(x_enc, x_mark_dec)
        return pred