# train.py
from tunetools import decorator, Parameter

# This is the main function, which will be executed during training. 
# Tunetools will recognize the parameters automatically, and construct 
# grid search with the given domains. 
# "num_sample=2" means each experiment will run for 2 times.
# @decorator.main(num_sample=2)
@decorator.main(num_sample=1)
def main(
        # Register the hyper-parameters manifest here. 
        
        n_estimators: Parameter(default=300, domain=[300]),
        
        min_child_weight: Parameter(default=1, domain=[1, 2, 3]),

        max_depth: Parameter(default=3, domain=[3, 4, 5]),
        gamma: Parameter(default=0.0, domain=[0.0, 0.1, 0.2]),
        
        # subsample: Parameter(default=1.0, domain=[0.6, 0.8, 1.0]),
        # colsample_bytree: Parameter(default=1.0, domain=[0.6, 0.8, 1.0]),
        subsample: Parameter(default=1.0, domain=[1.0]),
        colsample_bytree: Parameter(default=1.0, domain=[1.0]),
        
        model: Parameter(default="gbdt", domain=["gbdt"]),
        # dataset: Parameter(default="ETTh1", domain=["ETTh1", "ETTh2", "ETTm1", "ETTm2", "ECL", "Traffic", "Weather", "Illness"]),
        dataset: Parameter(default="ETTh1", domain=["ETTh1"]),

        seq_len: Parameter(default=96, domain=[96, 336]),
        pred_len: Parameter(default=96, domain=[96]),
        gpu: Parameter(default=0, domain=[0]),
):
    # Do training here, use all the parameters...
    import os
    import random
    
    mapping = {
        "ETTh1": {"root_path": "./dataset/ETT-small", "data_path": "ETTh1", "data": "ETTh1", "task_id": "ETTh1", "variant_num": 7, "lr_range": [20, 50, 100, 200, 300], "lr_range_ets": [5, 10, 20, 50, 100], "lr_range_cross": [100, 200, 300, 500], "K": 3, "lr_ets": 1e-5, "lr_cross": [3e-5]*4, "seq_len_ets": 96, "seq_len_cross": [336,720,720,720], "seg_len_cross": [12,24,24,24], "d_model": 256, "d_ff":512, "e_layers": 3, "n_heads": 4},
        "ETTh2": {"root_path": "./dataset/ETT-small", "data_path": "ETTh2", "data": "ETTh2", "task_id": "ETTh2", "variant_num": 7, "lr_range": [10, 20, 50, 100, 200], "lr_range_ets": [200, 300, 500, 1000, 1500], "lr_range_cross": [200, 300, 500, 1000], "K": 3, "lr_ets": 1e-5, "lr_cross": [3e-5]*4, "seq_len_ets": 96, "seq_len_cross": [336,720,720,720], "seg_len_cross": [12,24,24,24], "d_model": 256, "d_ff":512, "e_layers": 3, "n_heads": 4},
        "ETTm1": {"root_path": "./dataset/ETT-small", "data_path": "ETTm1", "data": "ETTm1", "task_id": "ETTm1", "variant_num": 7, "lr_range": [5, 10, 20, 50, 100], "lr_range_ets": [2, 5, 10, 20, 50], "lr_range_cross": [2, 5, 10, 20, 50], "K": 3, "lr_ets": 1e-5, "lr_cross": [1e-4]*4, "seq_len_ets": 96, "seq_len_cross": [672,672,672,672], "seg_len_cross": [12,24,24,24], "d_model": 256, "d_ff":512, "e_layers": 3, "n_heads": 4},
        "ETTm2": {"root_path": "./dataset/ETT-small", "data_path": "ETTm2", "data": "ETTm2", "task_id": "ETTm2", "variant_num": 7, "lr_range": [5, 10, 20, 50, 100], "lr_range_ets": [50, 100, 200, 500], "lr_range_cross": [10, 20, 50, 100, 200], "K": 3, "lr_ets": 1e-5, "lr_cross": [1e-4]*4, "seq_len_ets": 96, "seq_len_cross": [672,672,672,672], "seg_len_cross": [12,24,24,24], "d_model": 256, "d_ff":512, "e_layers": 3, "n_heads": 4},

        "ECL": {"root_path": "./dataset/electricity", "data_path": "electricity", "data": "custom", "task_id": "ECL", "variant_num": 321, "lr_range": [200, 500, 1000, 1500], "lr_range_ets": [500, 1000, 1500], "lr_range_cross": [10, 20, 50, 100], "K": 3, "lr_ets": 3e-4, "lr_cross": [5e-4]*4, "seq_len_ets": 336, "seq_len_cross": [336,720,720,720], "seg_len_cross": [12,24,24,24], "d_model": 64, "d_ff":128, "e_layers": 3, "n_heads": 2},
        "Exchange": {"root_path": "./dataset/exchange_rate", "data_path": "exchange_rate", "data": "custom", "task_id": "Exchange", "variant_num": 8, "lr_range": [2, 5, 10], "lr_range_ets": [2, 5, 10], "lr_range_cross": [50, 100, 200], "K": 0, "lr_ets": 1e-4, "lr_cross": [3e-5]*4, "seq_len_ets": 336, "seq_len_cross": [336,720,720,720], "seg_len_cross": [12,24,24,24], "d_model": 256, "d_ff":512, "e_layers": 3, "n_heads": 4},
        "Traffic": {"root_path": "./dataset/traffic", "data_path": "traffic", "data": "custom", "task_id": "traffic", "variant_num": 862, "lr_range": [1000, 1500, 2000, 3000], "lr_range_ets": [100, 150, 200, 300, 500], "lr_range_cross": [2, 5, 10], "K": 3, "lr_ets": 1e-3, "lr_cross": [5e-4]*4, "seq_len_ets": 336, "seq_len_cross": [336,720,720,720], "seg_len_cross": [12,24,24,24], "d_model": 64, "d_ff":128, "e_layers": 3, "n_heads": 2},
        "Weather": {"root_path": "./dataset/weather", "data_path": "weather", "data": "custom", "task_id": "weather", "variant_num": 21, "lr_range": [20, 30, 50, 100], "lr_range_ets": [5, 10, 20, 50], "lr_range_cross": [5, 10, 20], "K": 3, "lr_ets": 3e-4, "lr_cross": [3e-5]*4, "seq_len_ets": 336, "seq_len_cross": [336,336,336,720], "seg_len_cross": [12,24,24,24], "d_model": 256, "d_ff":512, "e_layers": 3, "n_heads": 4},
        "Illness": {"root_path": "./dataset/illness", "data_path": "national_illness", "data": "custom", "task_id": "ili", "variant_num": 7, "lr_range": [100, 150, 200, 300, 500, 1000], "lr_range_ets": [0.2, 0.5, 1, 2, 5], "lr_range_cross": [5, 10, 20, 50, 70, 100], "K": 1, "lr_ets": 1e-3, "lr_cross": [5e-4]*4, "seq_len_ets": 60, "seq_len_cross": [48,48,60,60], "seg_len_cross": [6,6,6,6], "d_model": 256, "d_ff":512, "e_layers": 3, "n_heads": 4},
    }
    
    root_path = mapping[dataset]["root_path"]
    data_path = mapping[dataset]["data_path"]
    data = mapping[dataset]["data"]
    task_id = mapping[dataset]["task_id"]
    variant_num = mapping[dataset]["variant_num"]
    
    label_len = pred_len // 2

    # 读取mse和mae结果
    # 先获取存储结果的txt文件
    result_dir = "./mse_results"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    # dataset_name = data_path.replace(".csv", "")
    dataset_name = data_path
    file_name = f"{dataset_name}_pl{pred_len}_n{n_estimators}_mcw{min_child_weight}_md{max_depth}_ga{gamma:.2f}_ss{subsample:.2f}_cb{colsample_bytree:.2f}.txt"
    file_path = os.path.join(result_dir, file_name)
    
    # 如果结果文件存在，就无需再跑一遍了，不存在的话则再跑一遍：
    if not os.path.exists(file_path):
        # 新建log日志目录
        log_path = f"./all_result/{model}_{dataset}_pl{pred_len}"
        if not os.path.exists(log_path): 
            os.makedirs(log_path)
        
        # Uni-variate task
        os.system(f"python -u ./run.py   --is_training 1   --root_path {root_path}   \
            --data_path {data_path}.csv   --task_id {task_id}   \
            --model {model}   --data {data}   --features S   \
            --seq_len {seq_len}   --label_len {label_len}   --pred_len {pred_len}   \
            --e_layers 2   --d_layers 1   --factor 3   \
            --enc_in 1   --dec_in 1   \
            --c_out 1   --des 'Exp'  --itr 1  --d_model 512  --gpu {gpu} \
            --n_estimators {n_estimators}  --min_child_weight {min_child_weight}  --max_depth {max_depth} \
            --gamma {gamma}  --subsample {subsample}  --colsample_bytree {colsample_bytree} \
            --run_train \
            > {log_path}/n{n_estimators}_mcw{min_child_weight}_md{max_depth}_ga{gamma:.2f}_ss{subsample:.2f}_cb{colsample_bytree:.2f}.log")
        
        # # Multi-variate task
        # os.system(f"python -u ./run.py   --is_training 1   --root_path {root_path}   \
        #     --data_path {data_path}.csv   --task_id {task_id}   \
        #     --model {model}   --data {data}   --features M   \
        #     --seq_len {seq_len}   --label_len {label_len}   --pred_len {pred_len}   \
        #     --e_layers 2   --d_layers 1   --factor 3   \
        #     --enc_in {variant_num}   --dec_in {variant_num}   \
        #     --c_out {variant_num}   --des 'Exp'   --itr 1  --d_model 512 --gpu {gpu} \
        #     --n_estimators {n_estimators}  --min_child_weight {min_child_weight}  --max_depth {max_depth} \
        #     --gamma {gamma}  --subsample {subsample}  --colsample_bytree {colsample_bytree} \
        #     > {log_path}/n{n_estimators}_mcw{min_child_weight}_md{max_depth}_ga{gamma:.2f}_ss{subsample:.2f}_cb{colsample_bytree:.2f}.log")

    # 将结果读出来
    with open(file_path) as f:
        line = f.readline()
        line = line.split(",")
        val_mse, test_mse = float(line[0]), float(line[1])

    return {
        "val_mse": val_mse,
        "test_mse": test_mse,
    }

# @decorator.filtering
# def filter_func(alpha, test_train_num, lambda_reg, model, dataset, gpu):
#     # Filter some parameter combinations you don't want to use.
#     return dataset != 'd3'