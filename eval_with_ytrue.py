# eval_with_ytrue.py
# 用官方 y_true 验证预测效果：支持 MAE / MAPE / 1-MAPE / R² / Two-Stage
import argparse
import numpy as np
import pandas as pd

def two_stage_score(y_true, y_pred, eps=1e-6):
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    ape = np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), eps)
    # 第一阶段：大错率（APE>100%）超过30%直接为0
    if (ape > 1).mean() > 0.30:
        return 0.0
    # 第二阶段：用 (1 - 平均APE) 按样本数归一（等价于 1 - MAPE，因为这里所有样本都参与）
    return float(np.clip(1.0 - ape.mean(), 0.0, 1.0))

def mae(y, yhat): 
    y=np.asarray(y,float); yhat=np.asarray(yhat,float)
    return float(np.mean(np.abs(y-yhat)))

def mape(y, yhat, eps=1e-6):
    y=np.asarray(y,float); yhat=np.asarray(yhat,float)
    return float(np.mean(np.abs(yhat-y)/np.maximum(np.abs(y),eps)))

def r2(y, yhat):
    y=np.asarray(y,float); yhat=np.asarray(yhat,float)
    ss_res=np.sum((y-yhat)**2); ss_tot=np.sum((y-y.mean())**2)
    return float(1.0 - ss_res / (ss_tot + 1e-12))

def guess_col(df, candidates):
    for c in candidates:
        if c in df.columns: return c
    raise ValueError(f"在列名 {list(df.columns)[:10]}... 中找不到任何候选列：{candidates}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", required=True, help="官方 y_true CSV 路径（如 test_solution.csv）")
    ap.add_argument("--pred",  required=True, help="你的预测 CSV 路径（需含 id 和预测列）")
    ap.add_argument("--pred-col", default="", help="预测列列名（不写则自动猜测）")
    args = ap.parse_args()

    ytrue = pd.read_csv(args.truth)
    # 官方常见列名：id, new_house_transaction_amount / amount_new_house_transactions / y / target
    id_col = guess_col(ytrue, ["id"])
    y_col  = guess_col(ytrue, [
        "new_house_transaction_amount", "amount_new_house_transactions", "y", "target"
    ])

    pred = pd.read_csv(args.pred)
    pred_id_col = guess_col(pred, ["id"])

    if args.pred_col:
        p_col = args.pred_col
        if p_col not in pred.columns:
            raise ValueError(f"预测列 {p_col} 不在你的预测文件中。可用列：{list(pred.columns)}")
    else:
        p_col = guess_col(pred, ["prediction","pred","y_pred","new_house_transaction_amount","amount_new_house_transactions"])

    # 合并
    df = ytrue[[id_col, y_col]].merge(pred[[pred_id_col, p_col]], left_on=id_col, right_on=pred_id_col, how="inner")
    if df.empty:
        raise ValueError("合并后为空。检查两边 id 是否一致/是否有空格大小写差异。")

    # 转成数值
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
    df[p_col] = pd.to_numeric(df[p_col], errors="coerce")
    df = df.dropna(subset=[y_col, p_col]).copy()

    y = df[y_col].to_numpy()
    yhat = df[p_col].to_numpy()

    m_mae = mae(y, yhat)
    m_mape = mape(y, yhat)
    m_r2 = r2(y, yhat)
    m_two = two_stage_score(y, yhat)

    print(f"[info] samples used = {len(df)}")
    print("=== Evaluation on provided y_true ===")
    print(f"MAE = {m_mae:,.2f} 万元")
    print(f"MAPE = {m_mape*100:.2f}%   (1-MAPE = {(1-m_mape)*100:.2f}%)")
    print(f"R² = {m_r2:.4f}")
    print(f"Two-Stage = {m_two:.4f}")

    # 额外诊断：大错比例
    ape = np.abs(yhat - y) / np.maximum(np.abs(y), 1e-6)
    big_ratio = float((ape > 1).mean())
    print(f"big-error ratio (APE>100%) = {big_ratio*100:.2f}%")

if __name__ == "__main__":
    main()
