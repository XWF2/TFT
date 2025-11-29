# China Real Estate Demand – TFT Pipeline

基于 Temporal Fusion Transformer 的预测流程，含数据预处理、训练、预测和可选网格搜索。

## 目录说明（核心）
- `prep_real_estate_data.py`：原始数据预处理，生成特征文件。
- `tft.py`：主模型/数据流水线、评估、预测辅助函数。
- `tft_train.py`：训练入口。
- `tft_predict.py`：预测入口。
- `tft_grid_search.py`：可选网格搜索入口。
- `out/`：预处理输出、特征文件、模型权重、数据集编码、提交文件等。

## 环境
- Python 3.12（已提供 `.venv312` 可直接用）。
- 如需新环境：`pip install -r requirements.txt`（若无文件，可基于 `.venv312` 直接运行）。
- 在 PowerShell 中使用虚拟环境的 Python：
  ```
  .\.venv312\Scripts\python.exe your_script.py ...
  ```
  或激活 venv 后直接 `python your_script.py ...`。

## 数据预处理
使用 `prep_real_estate_data.py` 生成特征文件（假设原始数据已在项目根目录下，按脚本默认路径读取）：
```
python prep_real_estate_data.py
```
输出（默认路径）：
- `out/features_train.csv`
- `out/features_test.csv`
- 以及其他中间/检查文件（如需）。

## 训练
原始管线训练命令示例：
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
用训练好的权重生成提交文件：
```
python tft_predict.py ^
  --ckpt-path out/tft_best.ckpt ^
  --train-ds-path out/tft_train_ds ^
  --feat-cols-path out/tft_feat_cols.json ^
  --submission-path out/submission_tft.csv
```
结果：
- `out/submission_tft.csv`（列：`id`, `new_house_transaction_amount`）
- 若存在 `test_solution.csv`，会自动评估（整体 & 按 Usage 分组）。

## 可选：网格搜索
`tft_grid_search.py` 遍历若干超参组合，按 holdout MAE 选最优，输出：
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
如需指定某一组合（如 08），改用 `out/tft_grid_08_best.ckpt`, `out/tft_grid_08_train_ds`, `out/tft_grid_08_meta.json`。

## 常见问题
- 行尾警告 (CRLF/LF)：可在 `.gitattributes` 指定 `*.py text eol=lf`，或关闭 `core.autocrlf`。
- SSH 推送失败：可改用 HTTPS 远程 `https://github.com/<user>/<repo>.git`，或配置 SSH key。
- 不想提交大文件：在 `.gitignore` 中忽略 `out/*.ckpt`, `out/tft_grid_*_train_ds/`, `lightning_logs/` 等。
