# make_submission_and_eval.py
# 生成 submission 并用 test_solution.csv 线下评测
# pip install numpy pandas scikit-learn statsmodels lightgbm torch

from pathlib import Path
import warnings
import numpy as np
import pandas as pd

# ---- 路径 ----
TRAIN_CSV = Path("out/features_train.csv")
TEST_CSV  = Path("out/features_test.csv")
RAW_TEST  = Path("test.csv")                 # 官方给的 test id 顺序
TEST_SOL  = Path("test_solution.csv")        # 你提供的真值文件
OUT_DIR   = Path("out"); OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- 评分 ----
def two_stage_score(y_true, y_pred, eps=1e-6):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ape = np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), eps)
    if (ape > 1).mean() > 0.30:
        return 0.0
    mask = ape <= 1
    return float(np.clip(1.0 - ape[mask].sum() / len(ape), 0.0, 1.0))

def mae(a,b): a=np.asarray(a,float); b=np.asarray(b,float); return float(np.mean(np.abs(a-b)))
def mape(y, yhat, eps=1e-6): y=np.asarray(y,float); yhat=np.asarray(yhat,float); return float(np.mean(np.abs(yhat-y)/np.maximum(np.abs(y),eps)))
def r2(y, yhat): y=np.asarray(y,float); yhat=np.asarray(yhat,float); ss_res=np.sum((y-yhat)**2); ss_tot=np.sum((y-y.mean())**2); return float(1-ss_res/(ss_tot+1e-12))

# ---- 通用 IO ----
def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "month" in df.columns:
        df["month"] = pd.to_datetime(df["month"], errors="coerce")
    return df

def ensure_test_ids(df_pred: pd.DataFrame) -> pd.DataFrame:
    """按官方 test.csv 的 id 顺序对齐"""
    ids = pd.read_csv(RAW_TEST)["id"]
    sub = pd.DataFrame({"id": ids})
    sub = sub.merge(df_pred, on="id", how="left")
    return sub

def save_submission(name: str, preds: pd.Series):
    sub = ensure_test_ids(pd.DataFrame({
        "id": pd.read_csv(RAW_TEST)["id"],
        "new_house_transaction_amount": preds
    }))
    sub["new_house_transaction_amount"] = sub["new_house_transaction_amount"].clip(lower=0)
    out = OUT_DIR / f"submission_{name}.csv"
    sub.to_csv(out, index=False)
    print(f"[save] {out}  shape={sub.shape}")
    return out

# ---- 模型 1：ARIMA（SARIMAX）----
def predict_arima(train_df: pd.DataFrame, test_df: pd.DataFrame):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    warnings.filterwarnings("ignore")
    order  = (1,1,1); sorder = (1,1,0,12)

    train_df = train_df.sort_values(["sector","month"])
    test_df  = test_df.sort_values(["sector","month"])

    preds = []
    for sec, gtest in test_df.groupby("sector"):
        hist = train_df[train_df["sector"]==sec].sort_values("month")["y"].values
        if len(hist) < 6:
            preds.extend([np.nan]*len(gtest)); continue
        try:
            model = SARIMAX(hist, order=order, seasonal_order=sorder,
                            enforce_stationarity=False, enforce_invertibility=False)
            res = model.fit(disp=False)
            for _ in range(len(gtest)):
                yhat = float(res.forecast(steps=1)[0])
                preds.append(yhat)
        except Exception:
            preds.extend([hist[-1]]*len(gtest))

    out = pd.Series(preds, index=test_df.index).sort_index()
    out = out.fillna(np.nanmedian(train_df["y"]))
    return out.reindex(test_df.index)

# ---- 模型 2：LightGBM（带列对齐）----
# ===== LightGBM 自定义评估（每一轮都在 valid 集上打分） =====
def feval_mae(y_pred, dtrain):
    y_true = dtrain.get_label()
    val = np.mean(np.abs(y_true - y_pred))
    return ('MAE', val, False)  # 越小越好

def feval_mape(y_pred, dtrain, eps=1e-6):
    y_true = dtrain.get_label()
    val = np.mean(np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), eps))
    return ('MAPE', val, False)  # 越小越好（已做 eps 平滑，兼容 y=0）

def feval_r2(y_pred, dtrain):
    y_true = dtrain.get_label()
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2) + 1e-12
    val = 1.0 - ss_res / ss_tot
    return ('R2', val, True)  # 越大越好

def feval_twostage(y_pred, dtrain, eps=1e-6):
    y_true = dtrain.get_label()
    ape = np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), eps)
    if (ape > 1).mean() > 0.30:
        val = 0.0
    else:
        val = float(np.clip(1.0 - ape.mean(), 0.0, 1.0))
    return ('TwoStage', val, True)  # 越大越好

# ===== 画验证曲线（与 TFT 验证曲线风格一致）=====
def plot_lgbm_curves(evals_result: dict, title_prefix="LightGBM", save_dir=OUT_DIR):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[plot] matplotlib 不可用，跳过曲线绘制")
        return None

    if "valid" not in evals_result:
        print("[plot] evals_result 缺少 'valid'，跳过绘制")
        return None

    hist = evals_result["valid"]
    iters = range(1, len(next(iter(hist.values()))) + 1)
    paths = []

    for key, ylabel in [
        ("MAE", "MAE (↓)"),
        ("MAPE", "MAPE (↓)"),
        ("R2", "R² (↑)"),
        ("TwoStage", "Two-Stage (↑)"),
    ]:
        if key not in hist:
            continue
        plt.figure()
        plt.plot(list(iters), hist[key])
        plt.xlabel("Iteration (trees)")
        plt.ylabel(ylabel)
        plt.title(f"{title_prefix}: validation {key}")
        plt.tight_layout()
        out_path = save_dir / f"lgbm_valid_{key.lower()}.png"
        plt.savefig(out_path, dpi=180)
        plt.close()
        paths.append(out_path)

    # 也导出 CSV 便于论文画图
    import pandas as pd
    df = pd.DataFrame({"iter": list(iters)})
    for key in hist:
        df[f"val_{key}"] = hist[key]
    csv_path = save_dir / "lgbm_valid_curves.csv"
    df.to_csv(csv_path, index=False)
    paths.append(csv_path)
    print(f"[plot] 保存 LightGBM 验证曲线到：{', '.join(map(str, paths))}")
    return paths

def predict_lgbm(train_df: pd.DataFrame, test_df: pd.DataFrame):
    import lightgbm as lgb
    from sklearn.model_selection import TimeSeriesSplit

    df = train_df.sort_values("month").copy()
    y = df["y"].values

    # 训练/测试统一列处理
    drop_all = [c for c in ["y","sector","month","id"] if c in df.columns]
    X = df.drop(columns=drop_all, errors="ignore")
    X_test = test_df.drop(columns=drop_all, errors="ignore")
    X_test = X_test.reindex(columns=X.columns, fill_value=0)  # 列对齐（缺列补0，多列丢弃）

    params = dict(
        objective="regression", metric="mae",
        learning_rate=0.05, num_leaves=64,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
        min_data_in_leaf=50, verbosity=-1, seed=42
    )

    tscv = TimeSeriesSplit(n_splits=5)
    models, val_mae = [], []

    # 我们把“最后一个 fold”作为曲线可视化的代表，其余 fold 仅参与集成
    evals_result_last = None
    last_fold_idx = None

    for fold_idx, (tr, va) in enumerate(tscv.split(X), 1):
        dtr = lgb.Dataset(X.iloc[tr], label=y[tr])
        dva = lgb.Dataset(X.iloc[va], label=y[va], reference=dtr)

        evals_result = {}
        model = lgb.train(
            params, dtr, num_boost_round=3000,
            valid_sets=[dtr, dva], valid_names=["train","valid"],
            feval=lambda y_pred, dset: [
                feval_mae(y_pred, dset),
                feval_mape(y_pred, dset),
                feval_r2(y_pred, dset),
                feval_twostage(y_pred, dset),
            ],
            callbacks=[
                lgb.early_stopping(200, verbose=False),
                lgb.record_evaluation(evals_result),
                lgb.log_evaluation(period=100),
            ],
            keep_training_booster=True,
        )
        models.append(model)
        pred_va = model.predict(X.iloc[va], num_iteration=model.best_iteration)
        val_mae.append(np.mean(np.abs(pred_va - y[va])))

        evals_result_last = evals_result  # 记录最后一折
        last_fold_idx = fold_idx

    print(f"[LGBM] CV MAE: {np.mean(val_mae):.3f} ± {np.std(val_mae):.3f}  |  best_iter(last_fold)={models[-1].best_iteration}")

    # 画“最后一折”的验证曲线（存 PNG + CSV）
    if evals_result_last is not None:
        plot_lgbm_curves(evals_result_last, title_prefix=f"LightGBM (fold {last_fold_idx})", save_dir=OUT_DIR)

    # 推理：多折预测取平均
    preds = np.mean([m.predict(X_test, num_iteration=m.best_iteration) for m in models], axis=0)
    return pd.Series(preds, index=test_df.index)


# ---- 模型 3：LSTM（单变量）----
def predict_lstm(train_df: pd.DataFrame, test_df: pd.DataFrame,
                 window=12, epochs=12, hidden=64, lr=1e-3, device="cpu"):
    import torch
    from torch import nn
    torch.set_float32_matmul_precision("high") if hasattr(torch, "set_float32_matmul_precision") else None

    class LSTMReg(nn.Module):
        def __init__(self, hidden=64):
            super().__init__()
            self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, batch_first=True)
            self.fc = nn.Linear(hidden, 1)
        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

    def make_supervised(series, win):
        X, y = [], []
        for i in range(len(series) - win):
            X.append(series[i:i+win]); y.append(series[i+win])
        return np.array(X, np.float32), np.array(y, np.float32)

    df_tr = train_df.sort_values(["sector","month"])
    df_te = test_df.sort_values(["sector","month"])

    preds_all = pd.Series(index=df_te.index, dtype=float)

    for sec, gtest in df_te.groupby("sector"):
        hist = df_tr[df_tr["sector"]==sec].sort_values("month")["y"].values.astype(np.float32)
        if len(hist) < window+1:
            preds_all.loc[gtest.index] = np.nan; continue

        X, y = make_supervised(hist, window)
        Xt = torch.from_numpy(X).unsqueeze(-1)
        yt = torch.from_numpy(y).unsqueeze(-1)

        model = LSTMReg(hidden=hidden)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.L1Loss()
        model.train()
        for _ in range(epochs):
            opt.zero_grad(); pred = model(Xt); loss = loss_fn(pred, yt)
            loss.backward(); opt.step()

        last = hist[-window:].copy()
        model.eval()
        sec_preds = []
        for _ in range(len(gtest)):
            with torch.no_grad():
                x = torch.from_numpy(last[None, :, None])
                yhat = model(x).numpy().ravel()[0]
            sec_preds.append(float(yhat))
            last = np.append(last[1:], yhat)

        preds_all.loc[gtest.index] = sec_preds

    preds_all = preds_all.fillna(np.nanmedian(train_df["y"]))
    return preds_all

# ---- 评测（对 submission vs test_solution）----
def evaluate_with_solution(sub_path: Path):
    if not TEST_SOL.exists():
        print("[eval] 未找到 test_solution.csv，跳过评测"); return
    sol = pd.read_csv(TEST_SOL)
    # 兼容列名：id + 真实值列
    y_col = "new_house_transaction_amount"
    if y_col not in sol.columns:
        # 尝试找第二列作为 y
        cand = [c for c in sol.columns if c != "id"]
        if len(cand) == 1:
            y_col = cand[0]
        else:
            raise ValueError("test_solution.csv 未找到真实值列（默认 new_house_transaction_amount）")

    sub = pd.read_csv(sub_path)
    df = sol.merge(sub, on="id", how="inner", suffixes=("_true", "_pred"))
    y_true = df[f"{y_col}_true"].values
    y_pred = df["new_house_transaction_amount_pred"].values if "new_house_transaction_amount_pred" in df.columns else df["new_house_transaction_amount"].values

    print(f"[eval] matched rows = {len(df)}")
    print(f"MAE={mae(y_true, y_pred):,.2f}  |  MAPE={mape(y_true, y_pred)*100:.2f}%  |  R²={r2(y_true, y_pred):.3f}  |  Two-Stage={two_stage_score(y_true, y_pred):.4f}")

def main():
    for p in (TRAIN_CSV, TEST_CSV, RAW_TEST):
        if not p.exists():
            raise FileNotFoundError(f"缺少文件：{p}")

    train_df = load_csv(TRAIN_CSV)
    test_df  = load_csv(TEST_CSV)

    # 1) ARIMA
    print("[run] ARIMA …")
    arima_pred = predict_arima(train_df, test_df)
    arima_path = save_submission("arima", pd.Series(arima_pred.values, index=test_df.index))
    evaluate_with_solution(arima_path)

    # 2) LightGBM
    print("[run] LightGBM …")
    lgbm_pred = predict_lgbm(train_df, test_df)
    lgbm_path = save_submission("lgbm", pd.Series(lgbm_pred.values, index=test_df.index))
    evaluate_with_solution(lgbm_path)

    # 3) LSTM
    print("[run] LSTM …")
    lstm_pred = predict_lstm(train_df, test_df, window=12, epochs=12, hidden=64, lr=1e-3, device="cpu")
    lstm_path = save_submission("lstm", pd.Series(lstm_pred.values, index=test_df.index))
    evaluate_with_solution(lstm_path)

    # 4) 简单平均
    ens = (arima_pred.values + lgbm_pred.values + lstm_pred.values) / 3.0
    ens_path = save_submission("ens_mean3", pd.Series(ens, index=test_df.index))
    evaluate_with_solution(ens_path)

if __name__ == "__main__":
    main()
