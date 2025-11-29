# train_and_eval_baselines.py
# 依赖：pip install numpy pandas scikit-learn statsmodels lightgbm torch

from pathlib import Path
import warnings
import numpy as np
import pandas as pd

# ========== 配置 ==========
TRAIN_CSV = Path("out/features_train.csv")  # 仅使用 CSV
REPORT_TXT = Path("out/baseline_report.txt")

# ARIMA/SARIMAX
ARIMA_ORDER = (1, 1, 1)
ARIMA_SEASONAL_ORDER = (1, 1, 0, 12)  # 月度季节 12
ARIMA_VAL_LAST_MONTHS = 6

# LightGBM
LGBM_N_SPLITS = 5
LGBM_PARAMS = dict(
    objective="regression",
    metric="mae",
    learning_rate=0.05,
    num_leaves=64,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    min_data_in_leaf=50,
    verbosity=-1,
    seed=42,
)

# LSTM
LSTM_WINDOW = 12
LSTM_EPOCHS = 12
LSTM_HIDDEN = 64
LSTM_LR = 1e-3
LSTM_DEVICE = "cpu"  # 如有 GPU，可改 "cuda"

# ========== 工具 ==========
def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))

def load_train_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "month" in df.columns:
        try:
            df["month"] = pd.to_datetime(df["month"])
        except Exception:
            # 尝试常见格式
            df["month"] = pd.to_datetime(df["month"], errors="coerce")
    return df

def write_report(lines):
    REPORT_TXT.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[save] {REPORT_TXT}")

# ========== 1) ARIMA / SARIMAX ==========
def eval_arima(train_df: pd.DataFrame) -> dict:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    warnings.filterwarnings("ignore")

    assert "y" in train_df.columns
    assert {"sector", "month"}.issubset(train_df.columns)

    df = train_df.sort_values(["sector", "month"]).copy()
    sectors = df["sector"].dropna().unique()
    sector_maes = []

    for sec in sectors:
        g = df[df["sector"] == sec].sort_values("month")
        y = g["y"].values
        if len(y) < max(24, ARIMA_VAL_LAST_MONTHS + 6):
            # 序列太短，跳过该 sector
            continue
        split = len(y) - ARIMA_VAL_LAST_MONTHS
        y_tr, y_va = y[:split], y[split:]
        try:
            model = SARIMAX(
                y_tr,
                order=ARIMA_ORDER,
                seasonal_order=ARIMA_SEASONAL_ORDER,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            res = model.fit(disp=False)
            pred = res.forecast(steps=ARIMA_VAL_LAST_MONTHS)
            sector_maes.append(mae(y_va, pred))
        except Exception:
            # 回退：持平预测
            pred = np.repeat(y_tr[-1], ARIMA_VAL_LAST_MONTHS)
            sector_maes.append(mae(y_va, pred))

    score = float(np.mean(sector_maes)) if sector_maes else np.nan
    return {
        "model": "ARIMA(SARIMAX)",
        "cv_mae": score,
        "n_sectors_used": int(len(sector_maes)),
        "note": f"order={ARIMA_ORDER}, seasonal_order={ARIMA_SEASONAL_ORDER}, val_last_months={ARIMA_VAL_LAST_MONTHS}",
    }

# ========== 2) LightGBM ==========
def eval_lgbm(train_df: pd.DataFrame) -> dict:
    import lightgbm as lgb
    from sklearn.model_selection import TimeSeriesSplit

    assert "y" in train_df.columns
    df = train_df.sort_values("month").copy()
    y = df["y"].values
    drop_cols = [c for c in ["y", "sector", "month"] if c in df.columns]
    X = df.drop(columns=drop_cols)

    tscv = TimeSeriesSplit(n_splits=LGBM_N_SPLITS)
    maes_ = []

    for fold, (tr, va) in enumerate(tscv.split(X), 1):
        dtr = lgb.Dataset(X.iloc[tr], label=y[tr])
        dva = lgb.Dataset(X.iloc[va], label=y[va], reference=dtr)
        model = lgb.train(
            LGBM_PARAMS,
            dtr,
            num_boost_round=2000,
            valid_sets=[dtr, dva],
            valid_names=["train", "valid"],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        pred = model.predict(X.iloc[va], num_iteration=model.best_iteration)
        fold_mae = mae(y[va], pred)
        maes_.append(fold_mae)
        print(f"[LGBM] fold {fold} MAE = {fold_mae:.4f}")

    return {
        "model": "LightGBM",
        "cv_mae": float(np.mean(maes_)),
        "cv_std": float(np.std(maes_)),
        "n_splits": LGBM_N_SPLITS,
        "note": f"drop={drop_cols}, params={{{', '.join([f'{k}:{v}' for k,v in LGBM_PARAMS.items() if k!='metric'])}}}",
    }

# ========== 3) LSTM（单变量 y，按 sector） ==========
def eval_lstm(train_df: pd.DataFrame) -> dict:
    import torch
    from torch import nn

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
            X.append(series[i:i+win])
            y.append(series[i+win])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    torch.set_num_threads(max(1, torch.get_num_threads()))
    device = torch.device(LSTM_DEVICE)

    df = train_df.sort_values(["sector", "month"]).copy()
    cv_maes = []
    sectors = df["sector"].dropna().unique()

    for sec in sectors:
        g = df[df["sector"] == sec].sort_values("month")
        y = g["y"].values.astype(np.float32)
        if len(y) < LSTM_WINDOW + 6:
            continue

        X_all, y_all = make_supervised(y, LSTM_WINDOW)
        # 最后 6 个样本做验证
        tr = len(X_all) - 6
        if tr <= 0:  # 极端短序列
            continue
        Xtr, ytr = X_all[:tr], y_all[:tr]
        Xva, yva = X_all[tr:], y_all[tr:]

        Xt = torch.from_numpy(Xtr).unsqueeze(-1).to(device)  # (N, win, 1)
        yt = torch.from_numpy(ytr).unsqueeze(-1).to(device)
        Xv = torch.from_numpy(Xva).unsqueeze(-1).to(device)
        yv = torch.from_numpy(yva).unsqueeze(-1).to(device)

        model = LSTMReg(hidden=LSTM_HIDDEN).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=LSTM_LR)
        loss_fn = nn.L1Loss()  # 直接优化 MAE

        model.train()
        for _ in range(LSTM_EPOCHS):
            opt.zero_grad()
            pred = model(Xt)
            loss = loss_fn(pred, yt)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            pred_va = model(Xv).cpu().numpy().squeeze(-1)
        cv_maes.append(mae(yva, pred_va))

    score = float(np.mean(cv_maes)) if cv_maes else np.nan
    return {
        "model": "LSTM(univariate)",
        "cv_mae": score,
        "n_sectors_used": int(len(cv_maes)),
        "note": f"window={LSTM_WINDOW}, epochs={LSTM_EPOCHS}, hidden={LSTM_HIDDEN}, lr={LSTM_LR}, device={LSTM_DEVICE}",
    }

# ========== 主流程 ==========
def main():
    if not TRAIN_CSV.exists():
        raise FileNotFoundError(f"未找到 {TRAIN_CSV}，请先生成 features_train.csv")

    df = load_train_csv(TRAIN_CSV)
    if "y" not in df.columns:
        raise ValueError("features_train.csv 中缺少 y 列")
    if not {"sector", "month"}.issubset(df.columns):
        print("[warn] 建议包含 sector / month 列用于时间与分组；已按现有列继续。")

    print(f"[info] loaded train: {df.shape}")

    # 运行三个基线
    res_arima = eval_arima(df)
    res_lgbm  = eval_lgbm(df)
    res_lstm  = eval_lstm(df)

    # 汇总与输出
    lines = [
        "=== Baseline Evaluation (MAE, lower is better) ===",
        f"ARIMA(SARIMAX): CV_MAE={res_arima['cv_mae']:.6f} | sectors_used={res_arima['n_sectors_used']} | {res_arima['note']}",
        f"LightGBM     : CV_MAE={res_lgbm['cv_mae']:.6f} (std={res_lgbm['cv_std']:.6f}, splits={res_lgbm['n_splits']}) | {res_lgbm['note']}",
        f"LSTM(univar) : CV_MAE={res_lstm['cv_mae']:.6f} | sectors_used={res_lstm['n_sectors_used']} | {res_lstm['note']}",
    ]
    print("\n" + "\n".join(lines))
    write_report(lines)

if __name__ == "__main__":
    main()
