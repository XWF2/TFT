# China Real Estate Demand – TFT Pipeline

本项目使用 Temporal Fusion Transformer (TFT) 做房产需求预测，训练 / 预测脚本已拆分，另附可选网格搜索脚本。

## 环境准备
- Python 3.12（已提供 .venv312，可直接用）
- 安装依赖（如需新环境）：`pip install -r requirements.txt`（若无文件，可基于 .venv312 直接用）

在 PowerShell 中使用虚拟环境的 Python：
```
.\.venv312\Scripts\python.exe your_script.py ...
```
若已激活 venv，则直接 `python your_script.py ...`。

## 数据
- 训练特征：`out/features_train.csv`
- 测试特征：`out/features_test.csv`
- 原始测试 ID：`test.csv`
- 可选离线评估：`test_solution.csv`（若存在，预测后会自动评估，按整体和 Usage 分组输出）

## 训练
使用原始管线的训练脚本：
```
python tft_train.py --epochs 30 --batch-size 128 --encoder-len 24 --holdout-last 6 ^
  --hidden 64 --attn-head 4 --dropout 0.15 --lr 1e-3 --patience 5 ^
  --ckpt-path out/tft_best.ckpt --train-ds-path out/tft_train_ds --feat-cols-path out/tft_feat_cols.json
```
输出：
- 模型权重：`out/tft_best.ckpt`
- 数据集编码：`out/tft_train_ds`
- 特征列：`out/tft_feat_cols.json`

## 预测
使用训练好的权重生成提交文件：
```
python tft_predict.py ^
  --ckpt-path out/tft_best.ckpt ^
  --train-ds-path out/tft_train_ds ^
  --feat-cols-path out/tft_feat_cols.json ^
  --submission-path out/submission_tft.csv
```
- 结果：`out/submission_tft.csv`（列：`id`, `new_house_transaction_amount`）
- 若存在 `test_solution.csv`，会自动打印整体和按 `Usage` 分组的评估指标。

## 可选：网格搜索
脚本 `tft_grid_search.py` 会遍历一组超参组合，按 holdout MAE 选出最佳，保存到：
- `out/tft_grid_best.ckpt`
- `out/tft_grid_best_train_ds`
- `out/tft_grid_best_meta.json`

运行：
```
python tft_grid_search.py
```
用最佳模型预测：
```
python tft_predict.py ^
  --ckpt-path out/tft_grid_best.ckpt ^
  --train-ds-path out/tft_grid_best_train_ds ^
  --feat-cols-path out/tft_grid_best_meta.json ^
  --submission-path out/submission_tft.csv
```
如需使用某一具体组合（如 08），将路径换成 `out/tft_grid_08_best.ckpt`, `out/tft_grid_08_train_ds`, `out/tft_grid_08_meta.json`。

## 常见问题
- 行尾警告 (CRLF/LF)：可在 `.gitattributes` 中指定 `*.py text eol=lf` 或关闭 `core.autocrlf`。
- SSH 推送失败：改用 HTTPS 远程 `https://github.com/<user>/<repo>.git`，或配置好 SSH key。
- 不想提交大文件：在 `.gitignore` 中忽略 `out/*.ckpt`, `out/tft_grid_*_train_ds/`, `lightning_logs/` 等。

## 目录说明（核心）
- `tft.py`：主模型/数据流水线、评估、预测辅助函数
- `tft_train.py`：训练入口
- `tft_predict.py`：预测入口
- `tft_grid_search.py`：可选网格搜索
- `out/`：特征文件、模型权重、数据集编码、提交文件等
