from torch.optim import lr_scheduler

from data_provider.data_factory import data_provider
from experiments.exp_basic import Exp_Basic
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
        if self.args.data == 'PEMS':
            criterion = nn.L1Loss()
        else:
            criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # channel_decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # fc1 - channel_decoder
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                if self.args.data == 'PEMS':
                    B, T, C = pred.shape
                    pred = pred.cpu().numpy()
                    true = true.cpu().numpy()
                    pred = vali_data.inverse_transform(pred.reshape(-1, C)).reshape(B, T, C)
                    true = vali_data.inverse_transform(true.reshape(-1, C)).reshape(B, T, C)
                    mae, mse, rmse, mape, mspe = metric(pred, true)
                    loss = mae
                else:
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

        if self.args.lradj == 'TST':
            scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                                steps_per_epoch=train_steps,
                                                pct_start=self.args.pct_start,
                                                epochs=self.args.train_epochs,
                                                max_lr=self.args.learning_rate)

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
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # channel_decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                # loss = self.time_freq_mae(batch_y, outputs)

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
                    max_pairs = D * (D - 1) // 2
                    if self.args.num_pairs == "max":
                        num_pairs = max_pairs
                    elif self.args.num_pairs.isdigit():
                        num_pairs = min(int(self.args.num_pairs), max_pairs)
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
                        num_pairs = min(int(self.args.num_pairs), max_pairs)
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
                        num_pairs = min(int(self.args.num_pairs), max_pairs)
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

                    idx_i = idx_i[mask]
                    idx_j = idx_j[mask]
                    
                    # 禁止同变量patch间交互

                    # pred_diff = out_nodes[:, idx_i] - out_nodes[:, idx_j]   # [B, num_pairs, L]
                    # true_diff = y_nodes[:, idx_i]   - y_nodes[:, idx_j]

                    # loss_add = (pred_diff - true_diff).abs().mean()

                    # === 最小改动：加入分块 (Chunking) 逻辑 ===
                    chunk_size = 4096
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
                    loss = loss_tmp
                else:
                    loss = self.args.alpha_add_loss * loss_tmp + self.args.beta_add_loss * loss_add
                ##################### add #####################

                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            # test_loss = 0
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, None, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

            # get_cka(self.args, setting, self.model, train_loader, self.device, epoch)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def time_freq_mae(self, batch_y, outputs):
        # time mae loss
        t_loss = (outputs - batch_y).abs().mean()

        # freq mae loss
        f_loss = torch.fft.rfft(outputs, dim=1) - torch.fft.rfft(batch_y, dim=1)
        f_loss = f_loss.abs().mean()

        return (1 - self.args.alpha) * t_loss + self.args.alpha * f_loss

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        # folder_path = './test_results/' + setting + '/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # channel_decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # fc1 - channel_decoder
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    outputs = test_data.inverse_transform(outputs)
                    batch_y = test_data.inverse_transform(batch_y)

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        if self.args.data == 'PEMS':
            B, T, C = preds.shape
            preds = test_data.inverse_transform(preds.reshape(-1, C)).reshape(B, T, C)
            trues = test_data.inverse_transform(trues.reshape(-1, C)).reshape(B, T, C)

        # result save
        # folder_path = './results/' + setting + '/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        print('rmse:{}, mape:{}, mspe:{}'.format(rmse, mape, mspe))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mse:{}, rmse:{}, mape:{}, mspe:{}'.format(mse, mae, rmse, mape, mspe))
        f.write('\n')
        f.write('\n')
        f.close()

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

            _ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

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
