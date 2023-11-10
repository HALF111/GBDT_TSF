import argparse
import os
import sys
import torch
from exp.exp_main import Exp_Main
from exp.exp_main_test import Exp_Main_Test
from exp.exp_main_gbdt import Exp_Main_GBDT
from exp.exp_main_gbdt_decom import Exp_Main_GBDT_Decom
import random
import numpy as np


def main():
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

    # basic config
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--task_id', type=str, default='test', help='task id')
    parser.add_argument('--model', type=str, default='FEDformer',
                        help='model name, options: [FEDformer, Autoformer, Informer, Transformer]')

    # data loader
    parser.add_argument('--data', type=str, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, '
                             'S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, '
                             'b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    # parser.add_argument('--cross_activation', type=str, default='tanh'

    # model define
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    # n_heads = 4 for Crossformer
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    # e_layers == 3 for Crossformer
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', default=[24], help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    # dropout == 0.2 for ETSformer & Crossformer
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    # activation == 'sigmoid' for ETSformer
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
    
    
    # DLinear
    parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')
    # Reformer & Autoformer
    parser.add_argument('--bucket_size', type=int, default=4, help='for Reformer')
    parser.add_argument('--n_hashes', type=int, default=4, help='for Reformer')
    # FEDformer
    parser.add_argument('--version', type=str, default='Fourier',
                        help='for FEDformer, there are two versions to choose, options: [Fourier, Wavelets]')
    parser.add_argument('--mode_select', type=str, default='random',
                        help='for FEDformer, there are two mode selection method, options: [random, low]')
    parser.add_argument('--modes', type=int, default=64, help='modes to be selected random 64')
    parser.add_argument('--L', type=int, default=3, help='ignore level')
    parser.add_argument('--base', type=str, default='legendre', help='mwt base')
    parser.add_argument('--cross_activation', type=str, default='tanh',
                        help='mwt cross atention activation function tanh or softmax')
    # ETSformer
    parser.add_argument('--K', type=int, default=1, help='Top-K Fourier bases')
    parser.add_argument('--min_lr', type=float, default=1e-30)
    parser.add_argument('--warmup_epochs', type=int, default=3)
    parser.add_argument('--std', type=float, default=0.2)
    parser.add_argument('--smoothing_learning_rate', type=float, default=0, help='optimizer learning rate')
    parser.add_argument('--damping_learning_rate', type=float, default=0, help='optimizer learning rate')
    parser.add_argument('--optim', type=str, default='adam', help='optimizer')
    # Crossformer
    parser.add_argument('--seg_len', type=int, default=6, help='segment length (L_seg)')
    parser.add_argument('--win_size', type=int, default=2, help='window size for segment merge')
    parser.add_argument('--cross_factor', type=int, default=10, help='num of routers in Cross-Dimension Stage of TSA (c)')
    parser.add_argument('--baseline', action='store_true', help='whether to use mean of past series as baseline for prediction', default=False)

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    # lradj == 'exponential_with_warmup' for ETSformer
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1', help='device ids of multi gpus')

    # test_train_num
    parser.add_argument('--run_train', action='store_true')
    parser.add_argument('--run_test', action='store_true')
    parser.add_argument('--run_test_batchsize1', action='store_true')
    parser.add_argument('--run_adapt', action='store_true')
    parser.add_argument('--run_calc', action='store_true')
    
    # 特征工程
    parser.add_argument('--add_revin', action='store_true')
    parser.add_argument('--use_VT', action='store_true')
    parser.add_argument('--channel_strategy', default="CI_one", type=str, help='CF/CI_one/CI_indiv')
    parser.add_argument('--add_x_mark', action='store_true')
    parser.add_argument('--add_mean_std', action='store_true')
    parser.add_argument('--add_patch_info', action='store_true')
    parser.add_argument('--patch_len', type=int, default=16, help='patch_length')
    parser.add_argument('--stride', type=int, default=8, help='patch_length')
    parser.add_argument('--add_fft', action='store_true')
    parser.add_argument('--fft_top_k', type=int, default=3, help='Top-K Fourier bases')
    parser.add_argument('--use_decomp', action='store_true')
    # XGBoost参数
    parser.add_argument('--n_estimators', type=int, default=300, help='[300]')  # 有早停的话不需要设置这一项了
    parser.add_argument('--min_child_weight', type=int, default=1, help='[1,2,3]')
    parser.add_argument('--max_depth', type=int, default=3, help='[3,4,5]')
    parser.add_argument('--gamma', type=float, default=0.0, help='[0.0, 0.1, 0.2]')
    parser.add_argument('--subsample', type=float, default=1.0, help='[0.6, 0.8, 1]')
    parser.add_argument('--colsample_bytree', type=float, default=1.0, help='[0.6, 0.8, 1]')
    # parser.add_argument('--min_child_weight', type=int, default=1, help='[1,2,3]')
    # parser.add_argument('--min_child_weight', type=int, default=1, help='[1,2,3]')

    args = parser.parse_args()

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    if "gb" in args.model:
        if args.use_decomp:
            Exp = Exp_Main_GBDT_Decom
        else:
            Exp = Exp_Main_GBDT
    else:
        # Exp = Exp_Main
        Exp = Exp_Main_Test

    if args.is_training:
        for ii in range(args.itr):
            print(f"-------Start iteration {ii+1}--------------------------")

            # setting record of experiments
            # 别忘记加上test_train_num一项！！！
            setting = '{}_{}_{}_modes{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.task_id,
                args.model,
                args.mode_select,
                args.modes,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                # args.test_train_num,
                args.des,
                ii)

            exp = Exp(args)  # set experiments
            if args.run_train:
                print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                if "gb" in args.model:
                    if args.use_decomp:
                        train_mse, val_mse_season, val_mse_trend, test_mse, test_mae = exp.train(setting)
                    else:
                        train_mse, val_mse, test_mse, test_mae = exp.train(setting)
                else:
                    exp.train(setting)
                
                print("train_mse:", train_mse)
                if args.use_decomp:
                    print("val_mse_season:", val_mse_season)
                    print("val_mse_trend:", val_mse_trend)
                else:
                    print("val_mse:", val_mse)
                print("test_mse:", test_mse)
                print("test_mae:", test_mae)
                
                # result_dir = "./mse_results"
                # if not os.path.exists(result_dir):
                #     os.makedirs(result_dir)
                # dataset_name = args.data_path.replace(".csv", "")
                # file_name = f"{dataset_name}_pl{args.pred_len}_n{args.n_estimators}_mcw{args.min_child_weight}_md{args.max_depth}_ga{args.gamma:.2f}_ss{args.subsample:.2f}_cb{args.colsample_bytree:.2f}.txt"
                # file_path = os.path.join(result_dir, file_name)
                # with open(file_path, "w") as f:
                #     f.write(f"{val_mse}, {test_mse}")
            

            if args.run_test:
                print('>>>>>>>normal testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                # exp.test(setting, flag="test")
                exp.test(setting, test=1, flag="test")

            if args.run_test_batchsize1:
                print('>>>>>>>normal testing but batch_size is 1 : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                # exp.test(setting, flag="test_with_batchsize_1")
                exp.test(setting, test=1, flag="test_with_batchsize_1")


            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(args.model_id,
                                                                                                      args.model,
                                                                                                      args.data,
                                                                                                      args.features,
                                                                                                      args.seq_len,
                                                                                                      args.label_len,
                                                                                                      args.pred_len,
                                                                                                      args.d_model,
                                                                                                      args.n_heads,
                                                                                                      args.e_layers,
                                                                                                      args.d_layers,
                                                                                                      args.d_ff,
                                                                                                      args.factor,
                                                                                                      args.embed,
                                                                                                      args.distil,
                                                                                                      args.des, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
