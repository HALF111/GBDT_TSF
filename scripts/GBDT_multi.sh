
# ETTh1 + seq_len=96, pred_len=96 + S
# 1.1 FEDformer
# 注意是S且enc_in,dec_in等均为1
# mse: 0.0841
python -u run.py --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --task_id ETTh1 --model FEDformer --data ETTh1 --features S --seq_len 96 --label_len 48 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 1 --dec_in 1 --c_out 1 --des 'Exp' --d_model 512 --itr 1  --gpu 0 --run_train --run_test
# 1.2 Linear
# mse:0.056, mae:0.179
python -u run.py  --is_training 1  --root_path ./dataset/ETT-small/  --data_path ETTh1.csv --task_id ETTh1_336_96  --model DLinear  --data ETTh1  --features S  --seq_len 336  --pred_len 96  --enc_in 7  --des 'Exp'  --itr 1  --batch_size 32  --learning_rate 0.005  --gpu 1 --run_train --run_test
# 1.3 XGBoost
# (0) CI_one + no RevIN + sl96: val_mse: 0.6697, test_mse: 0.3758, test_mae: 0.3903 yes
# (0) CI_one + no RevIN + sl96: val_mse: 0.6795, test_mse: 0.0.3711, test_mae: 0.3909 yes
# (1) CI_one + only RevIN + sl96: val_mse: 0.7086, test_mse: 0.3773, test_mae: 0.3869 yes
# (1) CI_one + only RevIN + sl336: val_mse: 0.6966, test_mse: 0.3734, test_mae: 0.3850 yes
python -u run.py --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --task_id ETTh1 --model gbdt --data ETTh1 --features M --seq_len 96 --label_len 48 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --d_model 512 --itr 1  --gpu 0 --run_train



# 2. ETTh2 + pred_len=96
# 2.1 FEDformer
# 2.2 DLinear
# 2.3 XGBoost
# (0) CI_one + no RevIN + sl96: val_mse: 0.9543, test_mse: 0.6571 no
# (0) CI_one + no RevIN + sl96: val_mse: 0.9543, test_mse: 0.6571 no
# (1) CI_one + only RevIN + sl96: val_mse: 0.2151; test_mse: 0.2875, test_mae: 0.3349 yes
# (1) CI_one + only RevIN + sl336: val_mse: 0.2162; test_mse: 0.2813, test_mae: 0.3369 no
python -u run.py --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh2.csv --task_id ETTh2 --model gbdt --data ETTh2 --features S --seq_len 96 --label_len 48 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 1 --dec_in 1 --c_out 1 --des 'Exp' --d_model 512 --itr 1  --gpu 0 --run_train --use_VT --add_revin



# 3. Traffic + pl=96
# 3.3 XGBoost
# (1) inidiv: val_mse: 0.1059, test_mse: 0.2187
# (2) VT: val_mse: 0.1130, test_mse: 0.2514
# (3) indiv + RevIN: val_mse: 0.1899, test_mse: 0.1644
# (4) VT + RevIN: val_mse: 0.1982, test_mse: 0.1953
python -u run.py --is_training 1 --root_path ./dataset/traffic/ --data_path traffic.csv --task_id trafic --model gbdt --data custom --features S --seq_len 96 --label_len 48 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 1 --dec_in 1 --c_out 1 --des 'Exp' --d_model 512 --itr 1  --gpu 0 --run_train --use_VT --add_revin


# 4. Exchange + pl=96
# 4.3 XGBoost
# (1) inidiv: val_mse: 0.6899, test_mse: 1.2106
# (2) VT: val_mse: 0.6905, test_mse: 1.1612
# (3) indiv + RevIN: val_mse: 3.4469, test_mse: 0.1601
# (4) VT + RevIN: val_mse: 3.4375, test_mse: 0.143
python -u run.py --is_training 1 --root_path ./dataset/exchange_rate/ --data_path exchange_rate.csv --task_id exchange --model gbdt --data custom --features S --seq_len 96 --label_len 48 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 1 --dec_in 1 --c_out 1 --des 'Exp' --d_model 512 --itr 1  --gpu 0 --run_train --use_VT --add_revin

