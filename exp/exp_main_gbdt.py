from data_provider.data_factory import data_provider
from data_provider.data_factory import data_provider_at_test_time
from exp.exp_basic import Exp_Basic
from models import FEDformer, Autoformer, Informer, Transformer
from models.etsformer import ETSformer
from models.crossformer import Crossformer
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric

import xgboost as xgb
from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler, StandardScaler, FunctionTransformer
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import TransformedTargetRegressor
import joblib

# ETSformer
from utils.Adam import Adam

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import optim

import os
import time
import warnings


import copy
import math


warnings.filterwarnings('ignore')


class Exp_Main_GBDT(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main_GBDT, self).__init__(args)

        # # 这个可以作为超参数来设置
        # self.test_train_num = self.args.test_train_num

        # 判断哪些channels是有周期性的
        data_path = self.args.data_path
        if "ETTh1" in data_path: selected_channels = [1,3]  # [1,3, 2,4,5,6]
        # if "ETTh1" in data_path: selected_channels = [7]
        # elif "ETTh2" in data_path: selected_channels = [1,3,7]
        elif "ETTh2" in data_path: selected_channels = [7]
        elif "ETTm1" in data_path: selected_channels = [1,3, 2,4,5]
        elif "ETTm2" in data_path: selected_channels = [1,7, 3]
        elif "illness" in data_path: selected_channels = [1,2, 3,4,5]
        # elif "weather" in data_path: selected_channels = [17,18,19, 5,8,6,13,20]  # [2,3,11]
        elif "weather" in data_path: selected_channels = [17,18,19]
        # elif "weather" in data_path: selected_channels = [5,8,6,13,20]
        # elif "weather" in data_path: selected_channels = [1,4,7,9,10]
        else: selected_channels = list(range(1, self.args.c_out))
        for channel in range(len(selected_channels)):
            selected_channels[channel] -= 1  # 注意这里要读每个item变成item-1，而非item
        
        self.selected_channels = selected_channels

        # 判断各个数据集的周期是多久
        if "ETTh1" in data_path: period = 24
        elif "ETTh2" in data_path: period = 24
        elif "ETTm1" in data_path: period = 96
        elif "ETTm2" in data_path: period = 96
        elif "electricity" in data_path: period = 24
        elif "traffic" in data_path: period = 24
        elif "illness" in data_path: period = 52.142857
        elif "weather" in data_path: period = 144
        elif "Exchange" in data_path: period = 1
        elif "WTH_informer" in data_path: period = 24
        else: period = 1
        self.period = period


    def _build_model(self):
        model_dict = {
            'FEDformer': FEDformer,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'ETSformer': ETSformer,
            'Crossformer': Crossformer,
        }
        
        if self.args.model == 'Crossformer':
            model = Crossformer.Model(
                self.args.enc_in, 
                self.args.seq_len, 
                self.args.pred_len,
                self.args.seg_len,
                self.args.win_size,
                self.args.cross_factor,
                self.args.d_model, 
                self.args.d_ff,
                self.args.n_heads, 
                self.args.e_layers,
                self.args.dropout, 
                self.args.baseline,
                self.device
            ).float()
        elif 'gb' in self.args.model:
            model = XGBRegressor()
        else:
            model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader


    # 别忘了这里要加一个用data_provider_at_test_time来提供的data
    def _get_data_at_test_time(self, flag):
        data_set, data_loader = data_provider_at_test_time(self.args, flag)
        return data_set, data_loader


    def _select_optimizer(self):
        if self.args.model == 'ETSformer':
            if 'warmup' in self.args.lradj: lr = self.args.min_lr
            else: lr = self.args.learning_rate

            if self.args.smoothing_learning_rate > 0: smoothing_lr = self.args.smoothing_learning_rate
            else: smoothing_lr = 100 * self.args.learning_rate

            if self.args.damping_learning_rate > 0: damping_lr = self.args.damping_learning_rate
            else: damping_lr = 100 * self.args.learning_rate

            nn_params = []
            smoothing_params = []
            damping_params = []
            for k, v in self.model.named_parameters():
                if k[-len('_smoothing_weight'):] == '_smoothing_weight':
                    smoothing_params.append(v)
                elif k[-len('_damping_factor'):] == '_damping_factor':
                    damping_params.append(v)
                else:
                    nn_params.append(v)

            model_optim = Adam([
                {'params': nn_params, 'lr': lr, 'name': 'nn'},
                {'params': smoothing_params, 'lr': smoothing_lr, 'name': 'smoothing'},
                {'params': damping_params, 'lr': damping_lr, 'name': 'damping'},
            ])
        else:
            model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.model == 'Crossformer':
                    outputs = self.model(batch_x)
                else:
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                # ? 是否需要对outputs取出最后一段[-self.args.pred_len:]？
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)

        total_loss = np.average(total_loss)
        self.model.train()

        return total_loss
    
    
    def get_data_for_gbdt(self, data_loader):
        train_x, train_y = [], []
        train_x_mark = []
        
        # 先遍历各batch获取数据
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(data_loader):
            train_x.append(batch_x)
            train_y.append(batch_y)
            # 这里我们不需要96的维度，可以只保留4的维度
            train_x_mark.append(batch_x_mark[:, 0:1, :].permute(0,2,1))
        
        print("type(train_x):", type(train_x))
        print("type(train_x[0]):", type(train_x[0]))
        print("len(train_x):", len(train_x))
        print("train_x[0].shape:", train_x[0].shape)
        print("train_x[-1].shape:", train_x[-1].shape)
        print("train_y[0].shape:", train_y[0].shape)
        print("train_y[-1].shape:", train_y[-1].shape)
        print("train_x_mark[0].shape:", train_x_mark[0].shape)
        print("train_x_mark[-1].shape:", train_x_mark[-1].shape)
        # for item in train_x_mark:
        #     print(item.shape)
        
        # 将batch和num_batch维度合并，从而获得所有样本的个数
        train_x = np.vstack(train_x)
        print("train_x.shape", train_x.shape)
        train_y = np.vstack(train_y)
        train_y = train_y[:, -self.args.pred_len:, :]  # y中需要去掉label_len，只保留pred_len部分
        print("train_y.shape", train_y.shape)
        
        train_x_mark = np.vstack(train_x_mark)
        print("train_x_mark.shape", train_x_mark.shape)
        
        # 考虑S或M的预测任务
        if self.args.features == 'S':
            # 将时间和channel合并成一维？
            # 因为XGBoost只接受2d的输入，[samples, seq_len, channel]的3d输入是不行的
            # 因此输入会变成[samples, seq_len * channel]
            # 特别地，这里channel=1，所以输入为[samples, seq_len]
            train_x = train_x.reshape(train_x.shape[0], -1)
            print("train_x.shape:", train_x.shape)
            train_y = train_y.reshape(train_y.shape[0], -1)
            print("train_y.shape:", train_y.shape)
            
            train_x_mark = train_x_mark.reshape(train_x.shape[0], -1)
            print("train_x_mark.shape", train_x_mark.shape)
        elif self.args.features == 'M':
            if self.args.channel_strategy == "CF":
                # 如果是channel-flatten策略，
                # 那么将[samples, seq_len, channel]中后两维seq_len和channel合并成一维
                # 因此输入会变成[samples, seq_len * channel]
                train_x = train_x.reshape(train_x.shape[0], -1)
                print("train_x.shape:", train_x.shape)
                train_y = train_y.reshape(train_y.shape[0], -1)
                print("train_y.shape:", train_y.shape)
                
                train_x_mark = train_x_mark.reshape(train_x.shape[0], -1)
                print("train_x_mark.shape", train_x_mark.shape)
            elif self.args.channel_strategy == "CI_one":
                # 如果是用一个模型同时拟合各个channel的策略，
                # 那么将[samples, seq_len, channel]中samples和channel合并成一维
                # 那么需要先permute一下，变成[samples, channel, seq_len]
                # 然后再合并前两维x
                train_x = train_x.transpose((0, 2, 1))
                train_x = train_x.reshape(-1, train_x.shape[-1])
                print("train_x.shape:", train_x.shape)
                train_y = train_y.transpose((0, 2, 1))
                train_y = train_y.reshape(-1, train_y.shape[-1])
                print("train_y.shape:", train_y.shape)
                
                train_x_mark = train_x_mark.transpose((0, 2, 1))
                train_x_mark = train_x_mark.reshape(-1, train_x_mark.shape[-1])
                print("train_x_mark.shape", train_x_mark.shape)
            elif self.args.channel_strategy == "CI_indiv":
                # train_x, train_y保持不变？
                pass
        
        return train_x, train_y, train_x_mark
    
    # def add_features(self, data_x_train, data_x_vali, data_x_test):
    def add_features(self, data_x_train, data_x_vali, data_x_test, data_x_mark_train, data_x_mark_vali, data_x_mark_test):
        data_x_new_total = {}
        phase_current = 0
        for flag in ["train", "vali", "test"]:
            # * feature 1: +RevIN
            # 注意要把data_x变成tensor之后revin才能接受
            if flag == "train": revin_model = self.revin_train; data_x = data_x_train; data_x_mark = data_x_mark_train
            elif flag == "vali": revin_model = self.revin_vali; data_x = data_x_vali; data_x_mark = data_x_mark_vali
            elif flag == "test": revin_model = self.revin_test; data_x = data_x_test; data_x_mark = data_x_mark_test
            
            x_revin = revin_model(torch.from_numpy(data_x), "norm")  # norm
            print("x_revin.shape:", x_revin.shape)
            print("data_x.shape before revin: ", data_x.shape)
            print("x before revin: ", data_x)
            print("x after revin: ", x_revin.detach().numpy())
            print("revin_model.mean:", revin_model.mean.shape, revin_model.mean)
            print("revin_model.stdev:", revin_model.stdev.shape, revin_model.stdev)
            print("revin_model.affine_weight", revin_model.affine_weight.shape, revin_model.affine_weight)
            print("revin_model.affine_bias:", revin_model.affine_bias.shape, revin_model.affine_bias)
            
            # data_x
            if self.args.add_revin:
                # data_x_new = np.concatenate((data_x, x_revin.detach().numpy()), axis=1)
                # data_x_new = np.concatenate((x_revin.detach().numpy(), data_x), axis=1)
                data_x_new = x_revin.detach().numpy()
            else:
                data_x_new = data_x
            print("data_x_new.shape", data_x_new.shape)
            
            
            # * feature 2: mean & variance
            # mean 重要性占比很低
            # 而variance和std占比很高，但是考虑这两者之后结果也变差了
            # 沿着seq_len维度求和
            mean = np.mean(data_x_new, axis=1)
            var = np.var(data_x_new, axis=1)
            std = np.std(data_x_new, axis=1)
            mean = mean.reshape(mean.shape[0], 1)
            var = var.reshape(var.shape[0], 1)
            std = std.reshape(std.shape[0], 1)
            print("mean.shape:", mean.shape)
            
            # data_x_new = np.concatenate((data_x_new, mean), axis=1)
            # data_x_new = np.concatenate((data_x_new, var), axis=1)
            # data_x_new = np.concatenate((data_x_new, std), axis=1)
            
            # * feature 3: Patching: mean & variance
            seq_len = self.args.seq_len
            patch_len = self.args.patch_len
            stride = self.args.stride
            patches = []
            i = 0
            while i+patch_len <= seq_len:
                cur_data_x = data_x_new[:, i: i+patch_len]
                patches.append(cur_data_x)
                i += stride
            
            print("len(patches):", len(patches))
            print("patches[0].shape:", patches[0].shape)
            print("patches[-1].shape:", patches[-1].shape)
            
            
            # 考虑到如果先加了全局的mean&variance，会导致特征数变多，从而导致patch时可能会将mean&variance也当作原始序列
            # 所以feature 2和feature 3在后面一起加进去
            # 先加feature 2
            if self.args.add_mean_var:
                data_x_new = np.concatenate((data_x_new, mean), axis=1)
                data_x_new = np.concatenate((data_x_new, var), axis=1)
                data_x_new = np.concatenate((data_x_new, std), axis=1)
            # 再加feature 3
            if self.args.add_patch_info:
                for patch in patches:
                    tmp_mean = np.mean(patch, axis=1)
                    tmp_var = np.var(patch, axis=1)
                    tmp_std = np.std(patch, axis=1)
                    tmp_mean = tmp_mean.reshape(tmp_mean.shape[0], 1)
                    tmp_var = tmp_var.reshape(tmp_var.shape[0], 1)
                    tmp_std = tmp_std.reshape(tmp_std.shape[0], 1)
                    data_x_new = np.concatenate((data_x_new, tmp_mean), axis=1)
                    data_x_new = np.concatenate((data_x_new, tmp_var), axis=1)
                    data_x_new = np.concatenate((data_x_new, tmp_std), axis=1)
                
            
            # * feature 4: periodic phase
            # * we use data_x_mark instead
            # 这个新特征的重要性占比很高，但是结果却下降了
            # phase_info = np.zeros_like(data_x)
            # print("phase_info.shape:", phase_info.shape)
            # for i in range(phase_info.shape[0]):
            #     phase_array = [(phase_current + j) % self.period for j in range(phase_info.shape[1])]
            #     phase_info[i] = np.array(phase_array)
            #     phase_current += 1
            
            # phase_info = np.zeros((data_x.shape[0], 1))
            # for i in range(phase_info.shape[0]):
            #     phase_array = phase_current % self.period
            #     phase_info[i] = phase_array
            #     phase_current += 1
                
            if self.args.add_x_mark:
                if self.args.features == 'M' and self.args.channel_strategy == "CI_one":
                    data_x_mark_info = [data_x_mark for _ in range(self.args.enc_in)]
                    data_x_mark_info = np.vstack(data_x_mark_info)
                else:
                    data_x_mark_info = data_x_mark
                print("data_x_mark_info.shape", data_x_mark_info.shape)
                data_x_new = np.concatenate((data_x_new, data_x_mark_info), axis=1)
            
            
            # data_x_new = np.concatenate((data_x_new, phase_info), axis=1)
            
            print("data_x_new.shape:", data_x_new.shape)
            print("data_x_new:", data_x_new)
            
            data_x_new_total[flag] = data_x_new
        
        
        return data_x_new_total["train"], data_x_new_total["vali"], data_x_new_total["test"]
        # return x.detach().numpy()
        
    
    def train(self, training):
        # 从数据集获取数据的loader
        train_data, train_loader = self._get_data(flag='train_without_shuffle')
        vali_data, vali_loader = self._get_data(flag='val_without_shuffle')
        test_data, test_loader = self._get_data(flag='test')
        
        time_now = time.time()
        train_steps = len(train_loader)
        
        # 逐batch遍历，获取数据
        # train_x, train_y = self.get_data_for_gbdt(train_loader)
        # vali_x, vali_y = self.get_data_for_gbdt(vali_loader)
        # test_x, test_y = self.get_data_for_gbdt(test_loader)
        train_x, train_y, train_x_mark = self.get_data_for_gbdt(train_loader)
        vali_x, vali_y, vali_x_mark = self.get_data_for_gbdt(vali_loader)
        test_x, test_y, test_x_mark = self.get_data_for_gbdt(test_loader)
        
        # add_revin = True
        add_revin = self.args.add_revin
        if add_revin:
            from layers.Invertible import RevIN
            if self.args.features == 'S':
                # 事实上，这个时候的enc_in就等于1
                self.revin_train = RevIN(self.args.enc_in)
                self.revin_vali = RevIN(self.args.enc_in)
                self.revin_test = RevIN(self.args.enc_in)
            elif self.args.features == 'M' and self.args.channel_strategy == "CF":
                # # 如果channel-flatten，那么需要对后两位都做revin？
                # self.revin_train = RevIN(self.args.enc_in * self.args.seq_len)
                # self.revin_vali = RevIN(self.args.enc_in * self.args.seq_len)
                # self.revin_test = RevIN(self.args.enc_in * self.args.seq_len)
                
                # 从结果来看，因为channel合并了，所以传给revin的num_features似乎还是为1
                self.revin_train = RevIN(num_features=1)
                self.revin_vali = RevIN(num_features=1)
                self.revin_test = RevIN(num_features=1)
            elif self.args.features == 'M' and self.args.channel_strategy == "CI_one":
                # 由于channel维和num_sample维合并了，所以这个时候的enc_in就等于1
                self.revin_train = RevIN(num_features=1)
                self.revin_vali = RevIN(num_features=1)
                self.revin_test = RevIN(num_features=1)
            
            # 在x上加上新的特征
            # train_x_after, vali_x_after, test_x_after = self.add_features(train_x, vali_x, test_x)
            train_x_after, vali_x_after, test_x_after = self.add_features(train_x, vali_x, test_x, train_x_mark, vali_x_mark, test_x_mark)
            print(type(train_x_after), train_x_after.shape)
            # print(train_x)
        else:
            # 不做revin的话就不加特征
            train_x_after = train_x.copy()
            vali_x_after = vali_x.copy()
            test_x_after = test_x.copy()
        
        
        def custom_mse_after_revin(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
            sample_num = y_pred.shape[0]
            seq_len = y_pred.shape[1]
            print(sample_num, seq_len)
            
            # print("y_pred before revin", y_pred)
            y_pred = self.revin_train(torch.from_numpy(y_pred), "denorm")
            y_pred = y_pred.detach().numpy()
            print(type(y_pred), y_pred.shape)
            # print("y_pred after revin", y_pred)
            
            y_true = y_true.reshape(*y_pred.shape)
            
            grad = 2 / (seq_len) * (y_pred - y_true)
            hess = np.full(y_pred.shape, 2 / (seq_len))
            
            # 一定要reshape成
            grad = grad.reshape((sample_num*seq_len, 1))
            hess = hess.reshape((sample_num*seq_len, 1))
            
            return grad, hess
        
        # 自定义损失函数：
        # https://xgboost.readthedocs.io/en/latest/tutorials/custom_metric_obj.html 中的最后Scikit-Learn Interface一节
        def custom_mse(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
            
            sample_num = y_pred.shape[0]
            seq_len = y_pred.shape[1]
            print(sample_num, seq_len)
            
            # print(type(y_pred), y_pred.shape)
            # print(y_pred)
            
            # y_true = y_true.get_label()
            y_true = y_true.reshape(*y_pred.shape)
            # print(type(y_true), y_true.shape)
            
            # grad = 2 / pred_len * (y_pred - y_true)
            grad = 2 / (seq_len) * (y_pred - y_true)
            # print(grad.shape)
            # print(grad)
            hess = np.full(y_pred.shape, 2 / (seq_len))
            # print(hess.shape)
            # print(hess)
            
            grad = grad.reshape((sample_num*seq_len, 1))
            hess = hess.reshape((sample_num*seq_len, 1))
            # print(grad.shape, hess.shape)
            
            return grad, hess
        
        def custom_mse_vali(y_true, y_pred):
            y_true = y_true.reshape(*y_pred.shape)
            print(y_pred.shape, y_true.shape)
            mse = mean_squared_error(y_true, y_pred)
            return mse
        
        def custom_mse_vali_after_revin(y_true, y_pred):
            y_true = y_true.reshape(*y_pred.shape)
            print(y_pred.shape, y_true.shape)
            
            # print("y_pred before revin", y_pred)
            y_pred = self.revin_vali(torch.from_numpy(y_pred), "denorm")
            y_pred = y_pred.detach().numpy()
            print(type(y_pred), y_pred.shape)
            # print("y_pred after revin", y_pred)
            
            mse = mean_squared_error(y_true, y_pred)
            return mse
        
        # 根据输入超参数控制采用哪种multi-output策略
        multi_strategy = "multi_output_tree" if self.args.use_VT else "one_output_per_tree"
        
        cv_params = {"n_estimators": [100, 200, 300, 400, 500]}
        
        custom_obj = custom_mse_after_revin if self.args.add_revin else custom_mse
        custom_eval_metric = custom_mse_vali_after_revin if self.args.add_revin else custom_mse_vali
        
        model = xgb.XGBRegressor(
                        max_depth=3,          # 每一棵树最大深度，默认6；
                        learning_rate=0.1,      # 学习率，每棵树的预测结果都要乘以这个学习率，默认0.3；
                        n_estimators=100,        # 使用多少棵树来拟合，也可以理解为多少次迭代。默认100；
                        # objective='reg:linear',   # 此默认参数与 XGBClassifier 不同
                        # objective='reg:squarederror',
                        # objective=custom_mse,
                        objective=custom_obj,
                        # eval_metric=custom_mse_vali,
                        eval_metric=custom_eval_metric,
                        booster='gbtree',         # 有两种模型可以选择gbtree和gblinear。gbtree使用基于树的模型进行提升计算，gblinear使用线性模型进行提升计算。默认为gbtree
                        gamma=0,                 # 叶节点上进行进一步分裂所需的最小"损失减少"。默认0；
                        min_child_weight=1,      # 可以理解为叶子节点最小样本数，默认1；
                        subsample=1,              # 训练集抽样比例，每次拟合一棵树之前，都会进行该抽样步骤。默认1，取值范围(0, 1]
                        colsample_bytree=1,       # 每次拟合一棵树之前，决定使用多少个特征，参数默认1，取值范围(0, 1]。
                        reg_alpha=0,             # 默认为0，控制模型复杂程度的权重值的 L1 正则项参数，参数值越大，模型越不容易过拟合。
                        reg_lambda=1,            # 默认为1，控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
                        random_state=0,           # 随机种子
                        # XGBoost树的行为可以由multi_strategy训练参数控制。
                        # 该参数可以取值one_output_per_tree（默认值）用于为每个目标构建一个模型，(当pl=96时对应于建96棵树)
                        # 或者取值multi_output_tree用于构建多输出树。(只用一棵树建模，但是每个叶子都包含一个长为96的向量)
                        multi_strategy=multi_strategy,
                    )

        
        # def custom_mse_after_revin(y_pred, y_true):
        #     y_pred = self.revin(torch.from_numpy(y_pred), "denorm")
        #     y_pred = y_pred.detach().numpy()
        
        # def custom_mse(y_pred: np.ndarray, y_true: xgb.QuantileDMatrix) -> np.ndarray:
            
        #     y_true = y_true.get_label()
        #     print(type(y_true), y_true.shape)
            
        #     y_pred = y_pred.reshape(*y_true.shape)
        #     print(type(y_pred), y_pred.shape)
            
        #     pred_len = y_pred.shape[0]
            
        #     grad = 2 / pred_len * (y_pred - y_true)
        #     print(grad.shape)
        #     print(grad)
        #     hess = np.full(y_pred.shape, 2 / pred_len)
        #     print(hess.shape)
        #     print(hess)
            
        #     return grad, hess
        
        # if add_revin:
        # model.fit(train_x_after, train_y, eval_set=[(vali_x_after, vali_y)], early_stopping_rounds=20, eval_metric='rmse')
        # model.fit(train_x_after, train_y, eval_set=[(vali_x_after, vali_y)], early_stopping_rounds=20, eval_metric=custom_mse)
        print("train_x_after.shape:", train_x_after.shape)
        print("train_y.shape:", train_y.shape)
        
        grid_search = False
        if grid_search:
            optimized_model = GridSearchCV(estimator=model, param_grid=cv_params, verbose=2, n_jobs=-1, scoring="neg_mean_squared_error")
            # optimized_model.fit(train_x_after, train_y, eval_set=[(vali_x_after, vali_y)], early_stopping_rounds=20)
            
            # print('参数的最佳取值：{0}'.format(optimized_model.best_params_))
            # print('最佳模型得分:{0}'.format(optimized_model.best_score_))
            
            # best_score = optimized_model.best_score_
            
            # print("rmse: ", best_score)
            # print("mse: ", best_score**2)
            
            # val_mse = best_score**2
            
            # pred_y = optimized_model.predict(test_x_after)
            # print("pred_y.shape: ", pred_y.shape)
            # print(pred_y)
            # print("true_y.shape", test_y.shape)
            # print(test_y)
            # print("MSE on test: ", mean_squared_error(test_y, pred_y))
        else:
            model.fit(train_x_after, train_y, eval_set=[(vali_x_after, vali_y)], early_stopping_rounds=20)
        
        
        # a = xgb.QuantileDMatrix()
        
        
        
        # # 如果又revin的话，还需要denorm回来
        # if add_revin:
        #     # # solution 1: failed
        #     # class Denorm(BaseEstimator, TransformerMixin):       
        #     #     def fit(self, X, y=None):
        #     #         return self
        #     #     def transform(self, X):
        #     #         return self.revin(X, "denorm")
        #     # steps = [('xgb_model', model), ('denorm', Denorm)]
        #     # # 将model变成两步的pipeline
        #     # model = Pipeline(steps)
        #     # # solution 2: run but not correct
        #     # # 用该类包装后完成后处理
        #     # model = TransformedTargetRegressor(regressor=model,
        #     #                                    inverse_func=lambda x: self.revin(x, "denorm"),
        #     #                                    check_inverse=False)
        #     # solution 3:
        #     # 创建自定义转换器，封装xgboost模型  
        #     class XGBoostTransformer:  
        #         def __init__(self, model):  
        #             self.model = model    
        #         def fit(self, X, y, **fit_params):  
        #             self.model.fit(X, y, **fit_params)
        #             return self
        #         def transform(self, X):  
        #             return self.model.predict(X)
        #     transformed_model = XGBoostTransformer(model)
        #     # 自定义后处理函数
        #     def post_processing(predictions): 
        #         processed_predictions = self.revin(torch.from_numpy(predictions), "denorm")
        #         processed_predictions = processed_predictions.detach().numpy()
        #         return processed_predictions
        #     pipeline = Pipeline([("model", transformed_model), 
        #                          ("post_processing", FunctionTransformer(post_processing, validate=False))])
        #     model = pipeline
        
        # # # Solution 2:
        # # trained_model = MultiOutputRegressor(model)
        # # trained_model.fit(train_x, train_y)
        
        # # train_forecasts = trained_model.predict(train_x)
        # # vali_forecasts = trained_model.predict(vali_x)
        # # test_forecasts = trained_model.predict(test_x)
        
        # # vali_mse = mean_squared_error(vali_forecasts, vali_y)
        # # print("vali_mse", vali_mse)
        # # test_mse = mean_squared_error(test_forecasts, test_y)
        # # print("test_mse: ", test_mse)
        
        
        # # Solution 1:
        # # model.fit(train_x, train_y, eval_set=[(test_x, test_y)], early_stopping_rounds=20, eval_metric='rmse')
        # # 理论上应该得用验证集而非测试集去验证
        # print(train_x_after.shape, vali_x_after.shape)
        # print(train_y.shape, vali_y.shape)
        # if add_revin:
        #     # ref: https://stackoverflow.com/questions/40329576/sklearn-pass-fit-parameters-to-xgboost-in-pipeline
        #     model.fit(train_x_after, train_y, model__eval_set=[(vali_x_after, vali_y)], model__early_stopping_rounds=20, model__eval_metric='rmse')
        # else:
        #     model.fit(train_x_after, train_y, eval_set=[(vali_x_after, vali_y)], early_stopping_rounds=20, eval_metric='rmse')
        
        # if add_revin:
        #     best_score = model.named_steps["model"].model.best_score
        # else:
        #     best_score = model.best_score
        best_score = model.best_score
        
        # print("rmse: ", best_score)
        # print("mse: ", best_score**2)
        
        print()
        print("best score on vali: ", best_score)
        print()
        
        # val_mse = best_score**2
        val_mse = best_score
        
        
        # # 在测试集上测试
        # if add_revin:
        #     model = model.named_steps["model"].model
        
        # 自定义后处理函数
        def post_processing(predictions): 
            processed_predictions = self.revin_test(torch.from_numpy(predictions), "denorm")
            processed_predictions = processed_predictions.detach().numpy()
            return processed_predictions
        
        pred_y = model.predict(test_x_after)
        # 如果做了revin，别忘记denorm回来
        if add_revin:
            pred_y = post_processing(pred_y)
        print("pred_y.shape: ", pred_y.shape)
        print(pred_y)
        print("true_y.shape", test_y.shape)
        print(test_y)
        print("MSE on test:", mean_squared_error(test_y, pred_y))
        print("MAE on test:", mean_absolute_error(test_y, pred_y))
        
        test_mse = mean_squared_error(test_y, pred_y)
        test_mae = mean_absolute_error(test_y, pred_y)
        
        feature_importance_weight = model.get_booster().get_score(importance_type='weight')
        print("features length: ", len(feature_importance_weight.items()))
        tmp = sorted(feature_importance_weight.items(), key=lambda d: d[1], reverse=True)
        # print(tmp[:10])
        print(tmp)
        feature_importance_gain = model.get_booster().get_score(importance_type='gain')
        tmp = sorted(feature_importance_gain.items(), key=lambda d: d[1], reverse=True)
        # print(tmp[:10])
        print(tmp)
        
        # # xgb.plot_tree(model, num_trees=0, fmap='xgb.fmap')
        # xgb.plot_tree(model, num_trees=0)
        # fig = plt.gcf()
        # fig.set_size_inches(150, 100)
        # fig.savefig("plot_tree.pdf")
        # xgb.plot_importance(model)
        # plt.savefig("plot_importance.pdf")
        
        # # visualization
        # fontsize = 16
        # for i in range(0, test_y.shape[0], 100):
        #     # plot_df = pd.DataFrame({"Forecasts": test_forecasts.flatten(), "Targets": test_y.flatten()}, index=range(len(test_y.flatten())))
        #     plot_df = pd.DataFrame({"Forecasts": pred_y[i], "Targets": test_y[i]}, index=range(len(test_y[i])))

        #     fig = plt.figure(figsize=(20, 12))
        #     # plt.plot(plot_df.index, plot_df["Forecasts"].rolling(3).mean(), label="Forecasts")
        #     plt.plot(plot_df.index, plot_df["Forecasts"], label="Forecasts")
        #     # plt.plot(plot_df.index, plot_df["Targets"].rolling(3).mean(), label="Targets")
        #     plt.plot(plot_df.index, plot_df["Targets"], label="Targets")

        #     plt.xlabel('Time', fontsize=fontsize)
        #     plt.xticks(fontsize=fontsize)
        #     plt.yticks(fontsize=fontsize)
        #     plt.ylabel("value", fontsize=fontsize)
        #     plt.grid(True)
        #     plt.legend(fontsize=fontsize)
        #     plt.tight_layout()
        #     plt.savefig(f"gbdt_figures/test_forecsats_{i}.pdf")
        
        return val_mse, test_mse, test_mae




    def train_backup(self, setting):
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
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.model == 'Crossformer':
                    outputs = self.model(batch_x)
                else:
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        if "gb" in self.args.model:
                            outputs = self.GBDT(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                # ? 是否需要对outputs取出最后一段？
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                
                loss = criterion(outputs, batch_y)
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
                    if self.args.model == 'ETSformer': torch.nn.utils.clip_grad_norm(self.model.parameters(), 1.0)
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
        
        # 这里为了防止异常，需要做一些修改，要在torch.load后加上map_location='cuda:0'
        # self.model.load_state_dict(torch.load(best_model_path))
        self.model.load_state_dict(torch.load(best_model_path, map_location='cuda:0'))

        return self.model

    def test(self, setting, test=0, flag='test'):
        # test_data, test_loader = self._get_data(flag='test')
        test_data, test_loader = self._get_data(flag=flag)
        if test:
            print('loading model from checkpoint !!!')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location='cuda:0'))

        preds = []
        trues = []

        criterion = nn.MSELoss()  # 使用MSELoss
        loss_list = []

        test_time_start = time.time()

        # result save
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

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                # encoder - decoder
                if self.args.model == 'Crossformer':
                    outputs = self.model(batch_x)
                else:
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                # ? 是否需要对outputs取出最后一段？
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                # selected_channels = self.selected_channels
                if self.args.adapt_part_channels:
                    outputs = outputs[:, :, self.selected_channels]
                    batch_y = batch_y[:, :, self.selected_channels]

                # 计算MSE loss
                loss = criterion(outputs, batch_y)
                loss_list.append(loss.item())
                # print(loss)


                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)

                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
        
        # file_name = f"batchsize_32_{setting}" if flag == 'test' else f"batchsize_1_{setting}"
        # # 将adaptation前的loss、adaptation中逐样本做adapt后的loss、以及adaptation之后的loss做统计
        # with open(f"./loss_before_and_after_adapt/{file_name}.txt", "w") as f:
        #     for loss in loss_list:
        #         f.write(f"{loss}\n")


        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        # f = open("result.txt", 'a')
        # f.write(setting + "  \n")
        # f.write('mse:{}, mae:{}'.format(mse, mae))
        # f.write('\n')
        # f.write('\n')
        # f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)

        print(f"Test - cost time: {time.time() - test_time_start}s")

        # return
        return loss_list


    def GBDT(self, batch_x):
        print(batch_x.shape)
        

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
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.model == 'Crossformer':
                    outputs = self.model(batch_x)
                else:
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
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
