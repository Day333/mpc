from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        self.masks = self._get_mask()

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
    
    def _get_mask(self):
        dtype = torch.float32
        L = self.args.seq_len * self.args.c_out // self.args.patch_len
        N = self.args.seq_len // self.args.patch_len
        masks = []
        for k in range(L):
            S = ((torch.arange(L) % N == k % N) & (torch.arange(L) != k)).to(dtype).to(self.device)
            T = ((torch.arange(L) >= k // N * N) & (torch.arange(L) < k // N * N + N) & (torch.arange(L) != k)).to(dtype).to(self.device)
            ST = torch.ones(L).to(dtype).to(self.device) - S - T
            ST[k] = 0.0
            masks.append(torch.stack([S, T, ST], dim=0))
        masks = torch.stack(masks, dim=0)
        return masks
    
    def _get_mask_2(self):
        dtype = torch.float32
        dtype = torch.float32
        L = self.args.seq_len * self.args.c_out // self.args.patch_len
        N = self.args.seq_len // self.args.patch_len

        mask_base = torch.eye(L, device=self.device, dtype=dtype).unsqueeze(0).unsqueeze(0)
        mask0 = torch.eye(L, device=self.device, dtype=dtype)
        mask0.view(self.args.c_out, N, self.args.c_out, N).diagonal(dim1=0, dim2=2).fill_(1)
        mask0 = mask0.unsqueeze(0).unsqueeze(0) - mask_base
        mask1 = torch.kron(torch.ones(self.args.c_out, self.args.c_out, device=self.device, dtype=dtype), 
                            torch.eye(N, device=self.device, dtype=dtype))
        mask1 = mask1.unsqueeze(0).unsqueeze(0) - mask_base
        mask2 = torch.ones((1, 1, L, L), device=self.device, dtype=dtype) - mask1 - mask0 - mask_base
        masks = torch.cat([mask0, mask1, mask2], dim=0)  # [3, 1, L, L]
        return masks

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                # encoder - decoder
                outputs, _ = self.model(batch_x, self.masks, is_training=False)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # encoder - decoder
                outputs, moe_loss = self.model(batch_x, self.masks, is_training=True)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                alpha = 0.05
                loss_moe = alpha * moe_loss
                
                ##################### add #####################
                loss_tmp = criterion(outputs, batch_y)
                B, T, D = outputs.shape
                device = outputs.device

                if self.args.add_loss == "corr":
                    # corr diff
                    X = outputs
                    Y = batch_y

                    mean_X = X.mean(dim=1, keepdim=True)     # [B, 1, D]
                    std_X  = X.std(dim=1, keepdim=True)      # [B, 1, D]

                    mean_Y = Y.mean(dim=1, keepdim=True)
                    std_Y  = Y.std(dim=1, keepdim=True)

                    X = (X - mean_X) / (std_X + 1e-6)
                    Y = (Y - mean_Y) / (std_Y + 1e-6)

                    # X: [B, T, D] -> [B, D, T]
                    X_t = X.transpose(1, 2)
                    Y_t = Y.transpose(1, 2)

                    Cx = torch.bmm(X_t, X) / T   # [B, D, D]
                    Cy = torch.bmm(Y_t, Y) / T   # [B, D, D]

                    loss_add = (Cx - Cy).abs().mean()
                elif self.args.add_loss == "scv":    # spatial cross-variable
                    # channel diff
                    if self.args.num_pairs == "max":
                        num_pairs = max_pairs
                    elif self.args.num_pairs.isdigit():
                        num_pairs = min(num_pairs, max_pairs)
                    else:
                        raise ValueError("num_pair error")

                    idx_i = torch.randint(0, D, (num_pairs,), device=outputs.device)
                    idx_j = torch.randint(0, D, (num_pairs,), device=outputs.device)

                    pred_diff = outputs[:,:,idx_i] - outputs[:,:,idx_j]
                    true_diff = batch_y[:,:,idx_i] - batch_y[:,:,idx_j]

                    loss_add = (pred_diff - true_diff).abs().mean()
                elif self.args.add_loss == "stcv":   # spatio-temporal cross-variable
                    # patch loss diff

                    patch_len = self.args.loss_patchlen
                    stride    = patch_len

                    if (T - patch_len) % stride != 0:
                        raise ValueError("(T - patch_len) % stride != 0")

                    out_p = outputs.unfold(1, patch_len, stride)   # [B, P, D, L]
                    y_p   = batch_y.unfold(1, patch_len, stride)

                    out_p = out_p.permute(0, 1, 3, 2).contiguous()  # [B, P, L, D]
                    y_p   = y_p.permute(0, 1, 3, 2).contiguous()

                    B, P, L, D = out_p.shape

                    out_nodes = out_p.permute(0,1,3,2).reshape(B, P*D, L)
                    y_nodes   = y_p.permute(0,1,3,2).reshape(B, P*D, L)

                    N = P * D

                    max_pairs = N * (N - 1) // 2
                    if self.args.num_pairs == "max":
                        num_pairs = max_pairs
                    elif self.args.num_pairs.isdigit():
                        num_pairs = min(num_pairs, max_pairs)
                    else:
                        raise ValueError("num_pair error")

                    idx_i = torch.randint(0, N, (num_pairs,), device=device)
                    idx_j = torch.randint(0, N, (num_pairs,), device=device)

                    # 禁止同变量patch间交互
                    patch_i = idx_i // D
                    patch_j = idx_j // D

                    var_i = idx_i % D
                    var_j = idx_j % D

                    mask = ~((var_i == var_j) & (patch_i != patch_j))

                    mask = mask & (idx_i != idx_j)
                    # 禁止同变量patch间交互

                    # 取消相同时间 patch 的交互
                    mask = mask & (patch_i != patch_j)
                    # 取消相同时间 patch 的交互

                    idx_i = idx_i[mask]
                    idx_j = idx_j[mask]

                    pred_diff = out_nodes[:, idx_i] - out_nodes[:, idx_j]   # [B, num_pairs, L]
                    true_diff = y_nodes[:, idx_i]   - y_nodes[:, idx_j]

                    loss_add = (pred_diff - true_diff).abs().mean()
                elif self.args.add_loss == "fcv":    # full cross-variable
                    # patch loss diff

                    # patch_len = self.args.loss_patchlen
                    patch_len = self.args.pred_len // self.args.loss_patchlen
                    stride    = patch_len

                    if (T - patch_len) % stride != 0:
                        raise ValueError("(T - patch_len) % stride != 0")

                    out_p = outputs.unfold(1, patch_len, stride)   # [B, P, D, L]
                    y_p   = batch_y.unfold(1, patch_len, stride)

                    out_p = out_p.permute(0, 1, 3, 2).contiguous()  # [B, P, L, D]
                    y_p   = y_p.permute(0, 1, 3, 2).contiguous()

                    B, P, L, D = out_p.shape

                    out_nodes = out_p.permute(0,1,3,2).reshape(B, P*D, L)
                    y_nodes   = y_p.permute(0,1,3,2).reshape(B, P*D, L)

                    N = P * D

                    max_pairs = N * (N - 1) // 2
                    if self.args.num_pairs == "max":
                        num_pairs = max_pairs
                    elif self.args.num_pairs.isdigit():
                        num_pairs = min(num_pairs, max_pairs)
                    else:
                        raise ValueError("num_pair error")

                    idx_i = torch.randint(0, N, (num_pairs,), device=device)
                    idx_j = torch.randint(0, N, (num_pairs,), device=device)
                    
                    # 禁止同变量patch间交互
                    patch_i = idx_i // D
                    patch_j = idx_j // D

                    var_i = idx_i % D
                    var_j = idx_j % D

                    mask = ~((var_i == var_j) & (patch_i != patch_j))

                    mask = mask & (idx_i != idx_j)
                    mask = mask & (idx_i < idx_j)

                    idx_i = idx_i[mask]
                    idx_j = idx_j[mask]
                    
                    # 禁止同变量patch间交互

                    # pred_diff = out_nodes[:, idx_i] - out_nodes[:, idx_j]   # [B, num_pairs, L]
                    # true_diff = y_nodes[:, idx_i]   - y_nodes[:, idx_j]

                    # loss_add = (pred_diff - true_diff).abs().mean()

                    # # === Chunking ===
                    chunk_size = 128
                    num_valid_pairs = len(idx_i)

                    if num_valid_pairs == 0:
                        loss_add = out_nodes.new_tensor(0., requires_grad=True)
                    else:
                        loss_add = 0.
                        for start in range(0, num_valid_pairs, chunk_size):
                            sl = slice(start, start + chunk_size)
                            pred_diff = out_nodes[:, idx_i[sl]] - out_nodes[:, idx_j[sl]]
                            true_diff = y_nodes[:, idx_i[sl]]   - y_nodes[:, idx_j[sl]]
                            loss_add += (pred_diff - true_diff).abs().sum()

                        loss_add /= (B * num_valid_pairs * L)

                if self.args.add_loss == "None":
                    loss = loss_tmp + loss_moe
                else:
                    loss = self.args.beta_add_loss * loss_add + self.args.alpha_add_loss * loss_tmp + loss_moe
                ##################### add #####################

                # loss = loss_tmp + loss_moe
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputs = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # encoder - decoder
                outputs, _ = self.model(batch_x, self.masks, is_training=False)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                batch_x = batch_x.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)
        
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]
                batch_x = batch_x[:, :, f_dim:]

                input_ = batch_x
                pred = outputs
                true = batch_y

                inputs.append(input_)
                preds.append(pred)
                trues.append(true)

        inputs = np.concatenate(inputs, axis=0)
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        inputs = inputs.reshape(-1, inputs.shape[-2], inputs.shape[-1])
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        
        # result save
        # folder_path = './results/' + setting + '/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)
        
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        # np.save(folder_path + 'input.npy', inputs)
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)

        self.profile_model(test_loader)
        
        best_model_path = os.path.join('./checkpoints/' + setting, 'checkpoint.pth')
        if os.path.exists(best_model_path):
            os.remove(best_model_path)
            print(f"Deleted model checkpoint at: {best_model_path}")

        return
    
    def profile_model(self, test_loader):
        self.model.eval()
        with torch.no_grad():
            batch_x, batch_y, batch_x_mark, batch_y_mark = next(iter(test_loader))
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)

            dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start_time = time.time()

            _ = self.model(batch_x, self.masks, is_training=False)

            torch.cuda.synchronize()
            end_time = time.time()

            inference_time = end_time - start_time
            gpu_mem = torch.cuda.memory_allocated(self.device) / 1024 / 1024
            peak_mem = torch.cuda.max_memory_allocated(self.device) / 1024 / 1024
            total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

            print("=" * 80)
            print("Model Profiling Summary")
            print(f"{'Total Params':<25}: {total_params:,}")
            print(f"{'Inference Time (s)':<25}: {inference_time:.6f}")
            print(f"{'GPU Mem Footprint (MB)':<25}: {gpu_mem:.2f}")
            print(f"{'Peak Mem (MB)':<25}: {peak_mem:.2f}")
            print("=" * 80)
