
# ETTh1 + seq_len=96, pred_len=96 + S
# 1.1 FEDformer
# 注意是S且enc_in,dec_in等均为1
# mse: 0.0841
python -u run.py --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --task_id ETTh1 --model FEDformer --data ETTh1 --features S --seq_len 96 --label_len 48 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 1 --dec_in 1 --c_out 1 --des 'Exp' --d_model 512 --itr 1  --gpu 0 --run_train --run_test
# 1.2 Linear
# mse:0.056, mae:0. 
python -u run.py  --is_training 1  --root_path ./dataset/ETT-small/  --data_path ETTh1.csv --task_id ETTh1_336_96  --model DLinear  --data ETTh1  --features S  --seq_len 336  --pred_len 96  --enc_in 7  --des 'Exp'  --itr 1  --batch_size 32  --learning_rate 0.005  --gpu 1 --run_train --run_test
# Linear: test_mse: 0.374, test_mae: 0.394
# PatchTST: test_mse: 0.370, test_mae: 0.400
# 1.3 XGBoost
# (0) CI_one + no RevIN + sl96: val_mse: 0.6697, test_mse: 0.3758, test_mae: 0.3903 yes
# (0) CI_one + no RevIN + sl336: val_mse: 0.6795, test_mse: 0.3711, test_mae: 0.3909 yes
# (1) CI_one + only RevIN + sl96: val_mse: 0.7086, test_mse: 0.3773, test_mae: 0.3869 yes
# (1) CI_one + only RevIN + sl336: val_mse: 0.6966, test_mse: 0.3734, test_mae: 0.3850 yes
# (2) CI_one + only RevIN + sl96 + x_mark: val_mse: 0.7220, test_mse: 0.3864, test_mae: 0.3931 yes
# (2) CI_one + only RevIN + sl336 + x_mark: val_mse: 0.7090, test_mse: 0.3848, test_mae: 0.3954 yes
# (3) CI_one + only RevIN + sl96 + mean&var: val_mse: 0.7075, test_mse: 0.3737, test_mae: 0.3867 yes
# (3) CI_one + only RevIN + sl336 + mean&var: val_mse: 0.6939, test_mse: 0.3737, test_mae: 0.3858 yes
# (4) CI_one + only RevIN + sl96 + mean&var + patch16+8: val_mse: 0.7069, test_mse: 0.3751, test_mae: 0.3876 yes
# (4) CI_one + only RevIN + sl336 + mean&var + patch16+8: val_mse: 0.6883, test_mse: 0.3746, test_mae: 0.3865 yes
# (5) CI_one + only RevIN + sl96 + patch16+8: val_mse: 0.7095, test_mse: 0.3768, test_mae: 0.3874 yes
# (5) CI_one + only RevIN + sl336 + patch16+8: val_mse: 0.6964, test_mse: 0.3732, test_mae: 0.3853 yes
python -u run.py --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --task_id ETTh1 --model gbdt --data ETTh1 --features M --seq_len 96 --label_len 48 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --d_model 512 --itr 1  --gpu 0 --run_train



# 2. ETTh2 + pred_len=96
# 2.1 FEDformer
# 2.2 DLinear

# Linear: test_mse: 0.277, test_mae: 0.338
# PatchTST: test_mse: 0.274, test_mae: 0.337
# 2.3 XGBoost
# (0) CI_one + no RevIN + sl96: val_mse: 0.9543, test_mse: 0.6571 no
# (0) CI_one + no RevIN + sl96: val_mse: 0.9543, test_mse: 0.6571 no
# (1) CI_one + only RevIN + sl96: val_mse: 0.2151; test_mse: 0.2875, test_mae: 0.3349 yes
# (1) CI_one + only RevIN + sl336: val_mse: 0.2162; test_mse: 0.2813, test_mae: 0.3369 yes
# (2) CI_one + only RevIN + sl96 + x_mark: val_mse: 0.2161; test_mse: 0.2931, test_mae: 0.3380 yes
# (2) CI_one + only RevIN + sl336 + x_mark: val_mse: 0.2156; test_mse: 0.2861, test_mae: 0.3392 yes
python -u run.py --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh2.csv --task_id ETTh2 --model gbdt --data ETTh2 --features S --seq_len 96 --label_len 48 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 1 --dec_in 1 --c_out 1 --des 'Exp' --d_model 512 --itr 1  --gpu 0 --run_train --use_VT --add_revin



# 3. Traffic + pl=96
# Linear: test_mse: 0.410, test_mae: 0.279
# PatchTST: test_mse: 0.360, test_mae: 0.249
# 3.3 XGBoost
python -u run.py --is_training 1 --root_path ./dataset/traffic/ --data_path traffic.csv --task_id trafic --model gbdt --data custom --features M --seq_len 96 --label_len 48 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 862 --dec_in 862 --c_out 862 --des 'Exp' --d_model 512 --itr 1  --gpu 0 --run_train --add_revin


# 4. Exchange + pl=96
# Linear: test_mse: 0.081, test_mae: 0.203
# 4.3 XGBoost
# (0) CI_one + no RevIN + sl96: val_mse: 0.5514, test_mse: 0.1941, test_mae: 0.3128 yes
# (0) CI_one + no RevIN + sl336: val_mse: 1.022, test_mse: 0.3490, test_mae: 0.4047 yes
# (1) CI_one + only RevIN + sl96: val_mse: 0.1251, test_mse: 0.0841, test_mae: 0.2014 yes
# (1) CI_one + only RevIN + sl336: val_mse: 0.1260, test_mse: 0.0887, test_mae: 0.2093 yes
# (2) CI_one + only RevIN + sl96 + x_mark: val_mse: 0.1251, test_mse: 0.0841, test_mae: 0.2014 yes ???
# (2) CI_one + only RevIN + sl336 + x_mark: val_mse: 0.1260, test_mse: 0.0887, test_mae: 0.2093 yes ???
python -u run.py --is_training 1 --root_path ./dataset/exchange_rate/ --data_path exchange_rate.csv --task_id exchange --model gbdt --data custom --features M --seq_len 96 --label_len 48 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 8 --dec_in 8 --c_out 8 --des 'Exp' --d_model 512 --itr 1  --gpu 0 --run_train --add_revin


# 5.1 Illness + pl=24
# seq_len = 104
# Linear: test_mse: 1.683, test_mae: 0.858
# PatchTST: test_mse: 1.319, test_mae: 0.754

# XGBoost
# (0) CI_one + no RevIN + sl36: val_mse: 0.2466, test_mse: 5.4373, test_mae: 1.5163 yes
# (0) CI_one + no RevIN + sl104: val_mse: 0.2627, test_mse: 3.6280, test_mae: 1.1987 yes
# (0).1 CF + no RevIN + sl80: val_mse: 0.2758, test_mse: 4.3887, test_mae: 1.3483 yes
# (1) CI_one + only RevIN + sl36: val_mse: 0.2210, test_mse: 2.1549, test_mae: 0.8370 yes
# (1) CI_one + only RevIN + sl52: val_mse: 0.2558, test_mse: 1.4176, test_mae: 0.7191 yes
# (1) CI_one + only RevIN + sl60: val_mse: 0.2283, test_mse: 1.3313, test_mae: 0.6959 yes
# (1) CI_one + only RevIN + sl80: val_mse: 0.2616, test_mse: 1.2576, test_mae: 0.6796 yes
# (1) CI_one + only RevIN + sl104: val_mse: 0.2741, test_mse: 1.5721, test_mae: 0.7865 yes
# (1).1 CF + only RevIN + sl80: val_mse: 0.2782, test_mse: 2.0717, test_mae: 0.9404 yes
# (2) CI_one + only RevIN + sl36 + x_mark: val_mse: 0.2203, test_mse: 2.1630, test_mae: 0.8371 yes
# (2) CI_one + only RevIN + sl52 + x_mark: val_mse: 0.2544, test_mse: 1.4146, test_mae: 0.7175 yes
# (2) CI_one + only RevIN + sl60 + x_mark: val_mse: 0.2287, test_mse: 1.3316, test_mae: 0.6959 yes
# (2) CI_one + only RevIN + sl80 + x_mark : val_mse: 0.2592, test_mse: 1.2538, test_mae: 0.6786 yes
# (2) CI_one + only RevIN + sl104 + x_mark: val_mse: 0.2741, test_mse: 1.6366, test_mae: 0.8112 yes
# (2).1 CF + only RevIN + sl80 + x_mark: val_mse: 0.2777, test_mse: 2.0717, test_mae: 0.9401 yes
# (3) CI_one + only RevIN + sl80 + mean&var : val_mse: 0.2218, test_mse: 1.8091, test_mae: 0.8051 yes
# (4) CI_one + only RevIN + sl80 + mean&var + patch16+8 : val_mse: 0.2281, test_mse: 1.7426, test_mae: 0.7943 yes
# (4) CI_one + only RevIN + sl80 + mean&var + patch24+2 : val_mse: 0.2356, test_mse: 1.6862, test_mae: 0.7861 yes
# (4) CI_one + only RevIN + sl80 + mean&var + patch12+4 : val_mse: 0.2114, test_mse: 1.7300, test_mae: 0.7933 yes
# (5) CI_one + only RevIN + sl80 + patch24+2 : val_mse: 0.2652, test_mse: 1.2687, test_mae: 0.6861 yes
# (5) CI_one + only RevIN + sl80 + patch12+4 : val_mse: 0.2597, test_mse: 1.2601, test_mae: 0.6868 yes
python -u run.py --is_training 1 --root_path ./dataset/illness/ --data_path national_illness.csv --task_id ili --model gbdt --data custom --features M --seq_len 36 --label_len 18 --pred_len 24 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --d_model 512 --itr 1  --gpu 0 --run_train --add_revin

# 5.2 Illness + pl=60
# Linear: test_mse: 1.819, test_mae: 0.917
# PatchTST: test_mse: 1.470, test_mae: 0.788
# (0) CI_one + no RevIN + sl52: val_mse: 0.2208, test_mse: 4.9537, test_mae: 1.4655 yes
# (0) CI_one + no RevIN + sl104: val_mse: 0.2341, test_mse: 4.0471, test_mae: 1.2980 yes
# (1) CI_one + only RevIN + sl52: val_mse: 0.2379, test_mse: 1.5472, test_mae: 0.7917 yes
# (1) CI_one + only RevIN + sl104: val_mse: 0.2612, test_mse: 1.7984, test_mae: 0.8965 yes
# (2) CI_one + only RevIN + sl36 + x_mark: val_mse: 0.2372, test_mse: 1.5518, test_mae: 0.7925 yes
# (2) CI_one + only RevIN + sl104 + x_mark: val_mse: 0.2612, test_mse: 1.7981, test_mae: 0.8964 yes
python -u run.py --is_training 1 --root_path ./dataset/illness/ --data_path national_illness.csv --task_id ili --model gbdt --data custom --features M --seq_len 52 --label_len 18 --pred_len 60 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --d_model 512 --itr 1  --gpu 0 --run_train --add_revin


