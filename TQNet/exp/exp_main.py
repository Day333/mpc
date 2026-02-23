from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear, PatchTST, SegRNN, CycleNet, \
    iTransformer, TimeXer, TQNet, TQDLinear, TQPatchTST, TQiTransformer
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'Linear': Linear,
            'PatchTST': PatchTST,
            'SegRNN': SegRNN,
            'CycleNet': CycleNet,
            'iTransformer': iTransformer,
            'TimeXer': TimeXer,
            'TQNet': TQNet,
            'TQDLinear': TQDLinear,
            'TQPatchTST': TQPatchTST,
            'TQiTransformer': TQiTransformer
        }
        model = model_dict[self.args.model].Model(self.args).float()

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

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_cycle = batch_cycle.int().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if any(substr in self.args.model for substr in {'CycleNet', 'TQ'}):
                            outputs = self.model(batch_x, batch_cycle)
                        elif any(substr in self.args.model for substr in
                                 {'Linear', 'MLP', 'SegRNN', 'TST'}):
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if any(substr in self.args.model for substr in {'CycleNet', 'TQ'}):
                        outputs = self.model(batch_x, batch_cycle)
                    elif any(substr in self.args.model for substr in {'Linear', 'MLP', 'SegRNN', 'TST'}):
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
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
            # max_memory = 0
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_cycle = batch_cycle.int().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if any(substr in self.args.model for substr in {'CycleNet', 'TQ'}):
                            outputs = self.model(batch_x, batch_cycle)
                        elif any(substr in self.args.model for substr in
                                 {'Linear', 'MLP', 'SegRNN', 'TST'}):
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        # loss = criterion(outputs, batch_y)
                        # train_loss.append(loss.item())
                else:
                    if any(substr in self.args.model for substr in {'CycleNet', 'TQ'}):
                        outputs = self.model(batch_x, batch_cycle)
                    elif any(substr in self.args.model for substr in {'Linear', 'MLP', 'SegRNN', 'TST'}):
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
                    # print(outputs.shape,batch_y.shape)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    # loss = criterion(outputs, batch_y)
                    # train_loss.append(loss.item())
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

                    idx_i = idx_i[mask]
                    idx_j = idx_j[mask]
                    
                    # 禁止同变量patch间交互

                    pred_diff = out_nodes[:, idx_i] - out_nodes[:, idx_j]   # [B, num_pairs, L]
                    true_diff = y_nodes[:, idx_i]   - y_nodes[:, idx_j]

                    loss_add = (pred_diff - true_diff).abs().mean()

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

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                # current_memory = torch.cuda.max_memory_allocated() / 1024 ** 2
                # max_memory = max(max_memory, current_memory)

                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

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

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        # print(f"Max Memory (MB): {max_memory}")

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        # folder_path = './test_results/' + setting + '/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_cycle = batch_cycle.int().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if any(substr in self.args.model for substr in {'CycleNet', 'TQ'}):
                            outputs = self.model(batch_x, batch_cycle)
                        elif any(substr in self.args.model for substr in
                                 {'Linear', 'MLP', 'SegRNN', 'TST'}):
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if any(substr in self.args.model for substr in {'CycleNet', 'TQ'}):
                        outputs = self.model(batch_x, batch_cycle)
                    elif any(substr in self.args.model for substr in {'Linear', 'MLP', 'SegRNN', 'TST'}):
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                # print(outputs.shape,batch_y.shape)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                # inputx.append(batch_x.detach().cpu().numpy())
                # if i % 20 == 0:
                #     input = batch_x.detach().cpu().numpy()

                #     gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                #     pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)

                #     visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
                    # np.savetxt(os.path.join(folder_path, str(i) + '.txt'), pd)
                    # np.savetxt(os.path.join(folder_path, str(i) + 'true.txt'), gt)

        if self.args.test_flop:
            test_params_flop(self.model, (batch_x.shape[1], batch_x.shape[2]))
            exit()
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        # inputx = np.concatenate(inputx, axis=0)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        # inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        ### denorm ###
        # denorm_preds = np.stack([test_data.inverse_transform(pred) for pred in preds])
        # denorm_trues = np.stack([test_data.inverse_transform(true) for true in trues])

        ### denorm ###

        # result save
        # folder_path = './results/' + setting + '/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        # mae, mse, rmse, mape, mspe, rse, corr = metric(denorm_preds, denorm_trues)

        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)
        # np.save(folder_path + 'x.npy', inputx)
        self.profile_model(test_loader)
        
        best_model_path = os.path.join('./checkpoints/' + setting, 'checkpoint.pth')
        if os.path.exists(best_model_path):
            os.remove(best_model_path)
            print(f"Deleted model checkpoint at: {best_model_path}")

        return
    
    def profile_model(self, test_loader):
        self.model.eval()
        with torch.no_grad():
            batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle = next(iter(test_loader))
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)
            batch_cycle = batch_cycle.int().to(self.device)

            dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start_time = time.time()

            _ = self.model(batch_x, batch_cycle)

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

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_cycle = batch_cycle.int().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(
                    batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if any(substr in self.args.model for substr in {'CycleNet', 'TQ'}):
                            outputs = self.model(batch_x, batch_cycle)
                        elif any(substr in self.args.model for substr in
                                 {'Linear', 'MLP', 'SegRNN', 'TST'}):
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if any(substr in self.args.model for substr in {'CycleNet', 'TQ'}):
                        outputs = self.model(batch_x, batch_cycle)
                    elif any(substr in self.args.model for substr in {'Linear', 'MLP', 'SegRNN', 'TST'}):
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
