# GBDT for TSF

## 如何开始？
1. 从 [[Autoformer](https://github.com/thuml/Autoformer)] 以及 [[Informer](https://github.com/zhouhaoyi/Informer2020)] 处下载数据集。
2. 打开`./scripts/GBDT_multi.sh`并执行里面对应的命令。（可以复制粘贴到终端执行）
3. ./scipts目录下重点是GBDT_multi.sh和GBDT_uni.sh文件，其他一大堆的都可以先不用管。


## Get Started

1. Install Python>=3.8, PyTorch 1.9.0.
2. Download data. You can obtain all the six benchmarks from [[Autoformer](https://github.com/thuml/Autoformer)] or [[Informer](https://github.com/zhouhaoyi/Informer2020)].
3. Train the model. We provide the experiment scripts of all benchmarks under the folder `./scripts`. You can reproduce the multivariate and univariate experiment results by running the following shell code separately:

```bash
bash ./scripts/run_M.sh
bash ./scripts/run_S.sh
```
