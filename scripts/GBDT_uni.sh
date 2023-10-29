
# ETTh1 + seq_len=96, pred_len=96 + S
# 1.1 FEDformer
# 注意是S且enc_in,dec_in等均为1
# mse: 0.0841
python -u run.py --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --task_id ETTh1 --model FEDformer --data ETTh1 --features S --seq_len 96 --label_len 48 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 1 --dec_in 1 --c_out 1 --des 'Exp' --d_model 512 --itr 1  --gpu 0 --run_train --run_test
# 1.2 Linear
# mse:0.056, mae:0.179
python -u run.py  --is_training 1  --root_path ./dataset/ETT-small/  --data_path ETTh1.csv --task_id ETTh1_336_96  --model DLinear  --data ETTh1  --features S  --seq_len 336  --pred_len 96  --enc_in 7  --des 'Exp'  --itr 1  --batch_size 32  --learning_rate 0.005  --gpu 1 --run_train --run_test
# 1.3 XGBoost
# (1) inidiv: val_mse: 0.1020, test_mse: 0.2870
# (1)+sl192: val_mse: 0.0999, test_mse: 0.2371
# (1)+sl336: val_mse: 0.1118, test_mse: 0.2152
# (1)+sl720: val_mse: 0.1162, test_mse: 0.1697

# (2) VT: val_mse: 0.1022, test_mse: 0.2479
# (2)+sl192: VT: val_mse: 0.0994, test_mse: 0.2695
# (2)+sl336: VT: val_mse: 0.1189, test_mse: 0.2288
# (2)+sl720: VT: val_mse: 0.1236, test_mse: 0.2426

# (3) indiv + only RevIN + sl96: val_mse: 0.0997, test_mse: 0.0561
# (3) indiv + both RevIN + sl96: val_mse: 0.0958, test_mse: 0.0793
# (3) indiv + only RevIN + sl336: val_mse: 0.0963, test_mse: 0.0586
# (3) indiv + both RevIN + sl336: val_mse: 0.0990, test_mse: 0.0653

# (3).2 indiv + only RevIN + sl96: val_mse: test_mse:
# (3).2 indiv + only RevIN + sl192: val_mse: test_mse:
# (3).2 indiv + only RevIN + sl96: val_mse: test_mse:

# (4) VT + only RevIN + sl96: val_mse: 0.1010, test_mse: 0.0563
# (4) VT + only RevIN + sl336: val_mse: 0.0974, test_mse: 0.0588
python -u run.py --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --task_id ETTh1 --model gbdt --data ETTh1 --features S --seq_len 96 --label_len 48 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 1 --dec_in 1 --c_out 1 --des 'Exp' --d_model 512 --itr 1  --gpu 0 --run_train



# 2. ETTh2 + pred_len=96
# 2.1 FEDformer
# 2.2 DLinear
# 2.3 XGBoost
# (1) inidiv: val_mse: 0.1892, test_mse: 0.1942
# (2) VT: val_mse: 0.1989, test_mse: 0.1897
# (3) indiv + only RevIN + sl96: val_mse: 0.1875, test_mse: 0.1296
# (3) indiv + both RevIN + sl96: val_mse: 0.1839, test_mse: 0.1391
# (3) indiv + only RevIN + sl336: val_mse: 0.1793, test_mse: 0.1231
# (3) indiv + both RevIN + sl336: val_mse: 0.1741, test_mse: 0.1370
# (4) VT + only RevIN + sl96: val_mse: 0.1890, test_mse: 0.1322
# (4) VT + only RevIN + sl336: val_mse: 0.1756, test_mse: 0.1357
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


# 5. Illness + pl=96
# 5.3 XGBoost
# (1) inidiv: val_mse: 0.0626, test_mse: 3.6293
# (2) VT: val_mse: 0.0628, test_mse: 3.5094
# (3) indiv + RevIN: val_mse: 0.7491, test_mse: 0.5681
# (4) VT + RevIN: val_mse: 0.9041, test_mse: 0.5820
python -u run.py --is_training 1 --root_path ./dataset/illness/ --data_path national_illness.csv --task_id ili --model gbdt --data custom --features S --seq_len 36 --label_len 18 --pred_len 24 --e_layers 2 --d_layers 1 --factor 3 --enc_in 1 --dec_in 1 --c_out 1 --des 'Exp' --d_model 512 --itr 1  --gpu 0 --run_train --use_VT --add_revin
