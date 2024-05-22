from data_provider.data_factory_online import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear, FEDformer, PatchTST, TimesNet
from models.multiscale_stat_prediction import Statistics_prediction
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric
from einops import rearrange
from tqdm import tqdm
from utils.tools import Buffer

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import os
import time

import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


class Exp_EvoMSN(Exp_Basic):
    def __init__(self, args):
        super(Exp_EvoMSN, self).__init__(args)
        self.station_pretrain_epoch = 5 if 'adaptive' in self.args.station_type else 0
        self.station_type = args.station_type
        self.buffer = Buffer(args.buffer_size)

    def _get_period(self):
        """
        get top k periodicity
        """
        train_data, train_loader = self._get_data(flag='train')
        amps = 0.0
        count = 0
        for data in train_loader:
            lookback_window = data[0]
            b, l, dim = lookback_window.size()
            amps += abs(torch.fft.rfft(lookback_window, dim=1)).mean(dim=0).mean(dim=1)
            count+=1
        amps = amps / count
        amps[0] = 0
        max_period = self.args.pred_len * 2
        max_freq = l // max_period + 1
        amps[0:max_freq] = 0
        top_list = amps.topk(self.args.top_k).indices
        period_list = l // top_list
        period_weight = F.softmax(amps[top_list], dim=0)
        self.args.period_list = period_list
        self.args.period_weight = period_weight

    def _build_model(self):
        self._get_period()
        self.statistics_pred = Statistics_prediction(self.args, self.device)
        model_dict = {
            'FEDformer': FEDformer,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'Linear': Linear,
            'PatchTST': PatchTST,
            'TimesNet': TimesNet
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
        self.station_optim = []
        for k in range(self.args.top_k):
            self.station_optim.append(optim.AdamW(self.statistics_pred.stat_predict[k].parameters(), lr=self.args.station_lr))
        return model_optim

    def _select_criterion(self):
        self.criterion = nn.MSELoss()

    def station_loss(self, y, statistics_pred):
        bs, len, dim = y.shape
        loss = []
        for k in range(self.args.top_k):
            period_len = self.args.period_list[k]
            if len % period_len != 0:
                length = ((len // period_len) + 1) * period_len
                padding = y[:, -(length - len):, :]
                y = torch.cat([y, padding], dim=1)
            y = y.reshape(bs, -1, period_len, dim)
            mean = torch.mean(y, dim=2)
            std = torch.std(y, dim=2)
            stat_true = torch.cat([mean, std], dim=-1)
            loss.append(self.criterion(statistics_pred[k], stat_true))
            y = y.reshape(bs, -1, dim)[:, :len, :]
        return loss

    def vali(self, vali_data, vali_loader, criterion, epoch):
        total_loss = []
        self.model.eval()
        self.statistics_pred.eval()
        f_dim = -1 if self.args.features == 'MS' else 0
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_x, statistics_pred = self.statistics_pred.normalize(batch_x)

                if epoch + 1 <= self.station_pretrain_epoch:
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = self.station_loss(batch_y, statistics_pred)
                    loss = torch.mean(torch.tensor(loss))
                else:
                    if 'adaptive' in self.args.station_type:
                        outputs = []
                        for k in range(self.args.top_k):
                            outputs.append(self.backbone_forward(batch_x[k], batch_y, batch_x_mark, batch_y_mark))
                    else:
                        outputs = self.backbone_forward(batch_x, batch_y, batch_x_mark, batch_y_mark)
                    outputs = self.statistics_pred.de_normalize(outputs, statistics_pred)
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                    pred = outputs.detach().cpu()
                    true = batch_y.detach().cpu()

                    loss = criterion(pred, true)

                total_loss.append(loss.cpu().item())
        total_loss = np.average(total_loss)
        self.model.train()
        self.statistics_pred.train()
        return total_loss

    def train(self, setting):
        f_dim = -1 if self.args.features == 'MS' else 0
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        path_station = './stat_pretrain/' + '{}_s{}_p{}'.format(self.args.data, self.args.seq_len, self.args.pred_len)
        if not os.path.exists(path_station):
            os.makedirs(path_station)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        early_stopping_station_model = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs + self.station_pretrain_epoch):
            iter_count = 0
            train_loss = []
            if epoch == self.station_pretrain_epoch and 'adaptive' in self.args.station_type:
                best_model_path = path_station + '/' + 'checkpoint.pth'
                self.statistics_pred.load_state_dict(torch.load(best_model_path))
                print('loading pretrained adaptive station model')

            self.model.train()
            self.statistics_pred.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x, statistics_pred = self.statistics_pred.normalize(batch_x)
                
                if epoch + 1 <= self.station_pretrain_epoch:
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss_list = self.station_loss(batch_y, statistics_pred)
                    loss = np.average(torch.tensor(loss_list).cpu())
                    train_loss.append(loss.item())
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)
                    
                    if 'adaptive' in self.args.station_type:
                        outputs = []
                        for k in range(self.args.top_k):
                            outputs.append(self.backbone_forward(batch_x[k], batch_y, batch_x_mark, batch_y_mark))
                    else:
                        outputs = self.backbone_forward(batch_x, batch_y, batch_x_mark, batch_y_mark)

                    outputs = self.statistics_pred.de_normalize(outputs, statistics_pred)
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = self.criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs + self.station_pretrain_epoch - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    # two-stage training schema
                    if epoch + 1 <= self.station_pretrain_epoch:
                        for k in range(self.args.top_k):
                            loss_list[k].backward()
                            self.station_optim[k].step()
                            self.station_optim[k].zero_grad()
                    else:
                        loss.backward()
                        model_optim.step()
                    model_optim.zero_grad()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, self.criterion, epoch)
            test_loss = 0

            if epoch + 1 <= self.station_pretrain_epoch:
                print(
                    "Station Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                        epoch + 1, train_steps, train_loss, vali_loss, test_loss))
                early_stopping_station_model(vali_loss, self.statistics_pred, path_station)
                for k in range(self.args.top_k):
                    adjust_learning_rate(self.station_optim[k], epoch + 1, self.args, self.args.station_lr)
            else:
                print(
                    "Backbone Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                        epoch + 1 - self.station_pretrain_epoch, train_steps, train_loss, vali_loss, test_loss))
                early_stopping(vali_loss, self.model, path)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
                adjust_learning_rate(model_optim, epoch + 1 - self.station_pretrain_epoch, self.args,
                                     self.args.learning_rate)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        f_dim = -1 if self.args.features == 'MS' else 0

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
            
        preds = []
        trues = []
        stat_preds = []

        folder_path = './EvoMSN_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        self.statistics_pred.eval()
        # with torch.no_grad():
        if self.args.online_learning != 'none':
            self.model.train()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm(test_loader)):
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)
            if i % self.args.test_freq == 0: 
                outputs, batch_y = self.process_online(batch_x, batch_y, batch_x_mark, batch_y_mark)

                pred = outputs.detach().cpu().numpy()  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y.detach().cpu().numpy()  # batch_y.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)
                trues.append(true)

            self.buffer.add((batch_x, batch_y, batch_x_mark, batch_y_mark))

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1], batch_x.shape[2]))
            exit()
        preds = np.array(preds)
        trues = np.array(trues)
        
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("online_result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)
        return mse, mae

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(
                    batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
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

    def backbone_forward(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        # decoder input
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_label = batch_x[:, -self.args.label_len:, :]
        dec_inp = torch.cat([dec_label, dec_inp], dim=1).float().to(self.device)
        # encoder - decoder
        if 'Linear' in self.args.model or 'PatchTST' in self.args.model:
            outputs = self.model(batch_x)
        else:
            if self.args.output_attention:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        f_dim = -1 if self.args.features == 'MS' else 0
        outputs = outputs[:, -self.args.pred_len:, f_dim:]
        return outputs
    
    def process_online(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        f_dim = -1 if self.args.features == 'MS' else 0
        if self.buffer.is_full():
            buffer_x, buffer_y, buffer_x_mark, buffer_y_mark = self.buffer.get()
            buffer_y = buffer_y[:, -self.args.pred_len:, f_dim:].to(self.device)
            buffer_x, buffer_statistics_pred = self.statistics_pred.normalize(buffer_x)
            if 'adaptive' in self.args.station_type:
                buffer_outputs = []
                for k in range(self.args.top_k):
                    buffer_outputs.append(self.backbone_forward(buffer_x[k], buffer_y, buffer_x_mark, buffer_y_mark))
            else:
                buffer_outputs = self.backbone_forward(buffer_x, buffer_y, buffer_x_mark, buffer_y_mark)
            buffer_outputs = self.statistics_pred.de_normalize(buffer_outputs, buffer_statistics_pred)
            buffer_y = buffer_y[:, -self.args.pred_len:, f_dim:].to(self.device)
            if self.args.online_learning != 'none':
                model_optim = self._select_optimizer()
                loss = self.criterion(buffer_outputs, buffer_y)
                loss.backward()
                model_optim.step()
                model_optim.zero_grad()
            self.buffer.clear()

        batch_x, statistics_pred = self.statistics_pred.normalize(batch_x)
        if 'adaptive' in self.args.station_type:
            outputs = []
            for k in range(self.args.top_k):
                outputs.append(self.backbone_forward(batch_x[k], batch_y, batch_x_mark, batch_y_mark))
        else:
            outputs = self.backbone_forward(batch_x, batch_y, batch_x_mark, batch_y_mark)
        outputs = self.statistics_pred.de_normalize(outputs, statistics_pred)
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
            
        if 'stat' in self.args.online_learning and 'adaptive' in self.args.station_type:
            loss = self.criterion(outputs, batch_y)
            loss.backward()
            for k in range(self.args.top_k):
                self.station_optim[k].step()
                self.station_optim[k].zero_grad()

        return outputs, batch_y
    