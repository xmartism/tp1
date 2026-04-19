import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear
from utils.metrics import metric
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop

warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        dimension_flag = 'pred' if getattr(args, 'mode', 'train') == 'predict' else 'train'
        temp_data, _ = data_provider(args, flag=dimension_flag)

        num_features = temp_data.data_x.shape[1]
        args.enc_in = num_features
        args.dec_in = num_features

        print(f'--- Dynamic Dimension Check: Found {num_features} features ---')

        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'Linear': Linear,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        return nn.MSELoss()

    def _model_dir(self):
        model_dir = Path(self.args.model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir

    def _checkpoint_path(self):
        return self._model_dir() / 'checkpoint.pth'

    def _metadata_path(self):
        return self._model_dir() / 'metadata.txt'

    def _write_metadata(self, epoch_count):
        with open(self._metadata_path(), 'w') as f:
            f.write(f'epoch: {epoch_count}')

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        f_dim = vali_data.df_data.columns.get_loc(self.args.target)
        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in vali_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.horizon:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                outputs = outputs[:, -self.args.horizon:, f_dim:f_dim + 1]
                batch_y = batch_y[:, -self.args.horizon:, f_dim:f_dim + 1].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                loss = criterion(pred, true)
                total_loss.append(loss)

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        trained_epochs = 0
        train_data, train_loader = self._get_data(flag='train')
        f_dim = train_data.df_data.columns.get_loc(self.args.target)
        if not self.args.train_only:
            vali_data, vali_loader = self._get_data(flag='val')

        model_dir = self._model_dir()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        time_now = time.time()

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
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.horizon:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        outputs = outputs[:, -self.args.horizon:, f_dim:f_dim + 1]
                        batch_y = batch_y[:, -self.args.horizon:, f_dim:f_dim + 1].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if 'Linear' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    outputs = outputs[:, -self.args.horizon:, f_dim:f_dim + 1]
                    batch_y = batch_y[:, -self.args.horizon:, f_dim:f_dim + 1].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print(f'\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}')
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print(f'\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s')
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print(f'Epoch: {epoch + 1} cost time: {time.time() - epoch_time}')
            train_loss = np.average(train_loss)
            trained_epochs = epoch + 1

            if not self.args.train_only:
                vali_loss = self.vali(vali_data, vali_loader, criterion)
                print(
                    'Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}'.format(
                        epoch + 1, train_steps, train_loss, vali_loss
                    )
                )
                early_stopping(vali_loss, self.model, str(model_dir))
            else:
                print('Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}'.format(epoch + 1, train_steps, train_loss))
                early_stopping(train_loss, self.model, str(model_dir))

            if early_stopping.early_stop:
                print('Early stopping')
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        self.model.load_state_dict(torch.load(self._checkpoint_path()))
        self._write_metadata(trained_epochs)
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(self._checkpoint_path()))

        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.horizon:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.horizon:, f_dim:]
                batch_y = batch_y[:, -self.args.horizon:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                if i % 20 == 0:
                    input_array = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input_array[0, :, -1], true[0, :, -1]), axis=0)
                    pd_line = np.concatenate((input_array[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd_line, os.path.join(folder_path, str(i) + '.pdf'))

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1], batch_x.shape[2]))
            exit()

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        inputx = np.concatenate(inputx, axis=0)

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        with open('result.txt', 'a') as f:
            f.write(setting + '  \n')
            f.write('mse:{}, mae:{}, rse:{}, corr:{}'.format(mse, mae, rse, corr))
            f.write('\n\n')

        np.save(folder_path + 'pred.npy', preds)
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')
        target_index = pred_data.df_data.columns.get_loc(self.args.target)

        if load:
            self.model.load_state_dict(torch.load(self._checkpoint_path(), map_location=self.device))

        preds = []
        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in pred_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros([batch_y.shape[0], self.args.horizon, batch_y.shape[2]]).float().to(batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if 'Linear' in self.args.model:
                    outputs = self.model(batch_x)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                outputs = outputs[:, -self.args.horizon:, target_index:target_index + 1]
                preds.append(outputs.detach().cpu().numpy())

        preds = np.concatenate(np.array(preds), axis=0)
        final_values = preds[0].reshape(-1)
        matched_dates = list(pred_data.future_dates)[-len(final_values):]

        df_output = pd.DataFrame({
            'timestamp': [pd.Timestamp(ts).isoformat() for ts in matched_dates],
            'prediction': final_values,
        })

        output_path = Path(self.args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_output.to_csv(output_path, index=False)

        print(f'\n>>>> Forecast for next {len(final_values)} steps <<<<')
        print(df_output.to_string(index=False))
        print(f'\nCSV saved to: {output_path}')
        return df_output
