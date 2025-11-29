# eval_lgbm_plots.py
# 依赖：numpy pandas scikit-learn lightgbm matplotlib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

TRAIN = Path("out/features_train.csv")
OUT   = Path("out"); OUT.mkdir(exist_ok=True, parents=True)

# ---------- metrics ----------
def two_stage_score(y_true, y_pred, eps=1e-6):
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    ape = np.abs(y_pred - y_true) / np.maximum(y_true, eps)
    if (ape > 1).mean() > 0.30:
        return 0.0
    mask = ape <= 1
    return float(np.clip(1.0 - ape[mask].sum() / len(ape), 0.0, 1.0))

def mae(a,b): a=np.asarray(a,float); b=np.asarray(b,float); return float(np.mean(np.abs(a-b)))
def mape(a,b,eps=1e-6): a=np.asarray(a,float); b=np.asarray(b,float); return float(np.mean(np.abs(b-a)/np.maximum(a,eps)))
def r2(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float)
    ss_res=np.sum((a-b)**2); ss_tot=np.sum((a-a.mean())**2)
    return float(1 - ss_res/(ss_tot + 1e-12))

def load_train():
    df = pd.read_csv(TRAIN)
    if "month" in df.columns:
        df["month"] = pd.to_datetime(df["month"], errors="coerce")
    return df.sort_values(["sector","month"])

# ---------- LightGBM 评估（仅此一个模型） ----------
def eval_lgbm_collect(df, n_splits=5):
    import lightgbm as lgb
    from sklearn.model_selection import TimeSeriesSplit
    df = df.sort_values("month").copy()
    y = df["y"].to_numpy()
    drop_all = [c for c in ["y","sector","month","id"] if c in df.columns]
    X = df.drop(columns=drop_all, errors="ignore")

    tscv = TimeSeriesSplit(n_splits=n_splits)
    maes=[]; mapes=[]; r2s=[]; tss=[]
    y_all=[]; yhat_all=[]; meta=[]

    params=dict(objective="regression", metric="mae", learning_rate=0.05,
                num_leaves=64, feature_fraction=0.8, bagging_fraction=0.8,
                bagging_freq=1, min_data_in_leaf=50, verbosity=-1, seed=42)

    for k, (tr, va) in enumerate(tscv.split(X), start=1):
        dtr=lgb.Dataset(X.iloc[tr], label=y[tr])
        dva=lgb.Dataset(X.iloc[va], label=y[va], reference=dtr)
        model=lgb.train(params, dtr, num_boost_round=2000,
                        valid_sets=[dtr, dva], valid_names=["train","valid"],
                        callbacks=[lgb.early_stopping(100, verbose=False)])
        y_hat = model.predict(X.iloc[va], num_iteration=model.best_iteration)
        y_va  = y[va]
        maes.append(mae(y_va, y_hat)); mapes.append(mape(y_va, y_hat))
        r2s.append(r2(y_va, y_hat));   tss.append(two_stage_score(y_va, y_hat))

        y_all.append(y_va); yhat_all.append(y_hat)
        meta.append(df.iloc[va][["sector","month"]].assign(fold=k))

    metrics = dict(MAE=float(np.mean(maes)),
                   MAPE=float(np.mean(mapes)),
                   R2=float(np.mean(r2s)),
                   TwoStage=float(np.mean(tss)),
                   Splits=n_splits,
                   R2_by_fold=r2s,
                   TS_by_fold=tss)

    y_all   = np.concatenate(y_all)
    yhat_all= np.concatenate(yhat_all)
    meta_df = pd.concat(meta, ignore_index=True)
    diag = pd.DataFrame({"y": y_all, "yhat": yhat_all})
    diag = pd.concat([meta_df.reset_index(drop=True), diag], axis=1)
    return metrics, diag

# ---------- 画图工具 ----------
def enrich(diag):
    eps=1e-6
    d = diag.copy()
    d["err"] = d["yhat"] - d["y"]
    d["ape"] = np.abs(d["err"]) / np.maximum(np.abs(d["y"]), eps)
    # 保证 month 为时间类型（防止外部 CSV 非标准化）
    if "month" in d.columns:
        d["month"] = pd.to_datetime(d["month"], errors="coerce")
    return d

def plot_residual_hist(d, save):
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(d["err"], bins=50)
    ax.set_title("LightGBM: Residual histogram")
    ax.set_xlabel("error (yhat - y)"); ax.set_ylabel("count")
    plt.tight_layout(); plt.savefig(save, dpi=200); plt.close()

def plot_true_vs_pred(d, save):
    fig, ax = plt.subplots(figsize=(5,5))
    ax.scatter(d["y"], d["yhat"], s=6, alpha=0.4)
    lim = [0, max(d["y"].max(), d["yhat"].max())]
    ax.plot(lim, lim, linestyle="--")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("y (true)"); ax.set_ylabel("yhat (pred)")
    ax.set_title("LightGBM: True vs Pred")
    plt.tight_layout(); plt.savefig(save, dpi=200); plt.close()

def plot_ape_hist(d, save):
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(np.clip(d["ape"], 0, 5.0), bins=50)
    ax.set_title("LightGBM: APE histogram (clipped at 5.0)")
    ax.set_xlabel("APE"); ax.set_ylabel("count")
    plt.tight_layout(); plt.savefig(save, dpi=200); plt.close()

# ==== 新增：Two-Stage & R² 的“随时间变化” ====
def monthly_metrics(d):
    """按 month 聚合计算 R2 / TwoStage（仅对有 month 的样本）"""
    if "month" not in d.columns or d["month"].isna().all():
        raise ValueError("诊断数据中缺少 month 列，无法绘制按时间的变化图。")
    g = d.dropna(subset=["month"]).sort_values("month").groupby("month")
    rows = []
    for m, dfm in g:
        rows.append({
            "month": m,
            "R2": r2(dfm["y"].to_numpy(), dfm["yhat"].to_numpy()),
            "TwoStage": two_stage_score(dfm["y"].to_numpy(), dfm["yhat"].to_numpy())
        })
    mdf = pd.DataFrame(rows).sort_values("month")
    return mdf

def plot_metric_ts(mdf, metric, save, title):
    fig, ax = plt.subplots(figsize=(7,3.8))
    ax.plot(mdf["month"], mdf[metric])
    ax.set_title(title)
    ax.set_xlabel("month"); ax.set_ylabel(metric)
    ax.grid(True, linestyle=":", linewidth=0.8)
    fig.autofmt_xdate()
    plt.tight_layout(); plt.savefig(save, dpi=200); plt.close()

# ==== 可选：按折数的柱状图（评审对比直观） ====
def plot_fold_bars(r2_list, ts_list, save_r2, save_ts):
    import numpy as np
    x = np.arange(1, len(r2_list)+1)
    # R2
    fig, ax = plt.subplots(figsize=(5,3.2))
    ax.bar(x, r2_list)
    ax.set_title("LightGBM: R² by fold"); ax.set_xlabel("fold"); ax.set_ylabel("R²")
    plt.tight_layout(); plt.savefig(save_r2, dpi=200); plt.close()
    # Two-Stage
    fig, ax = plt.subplots(figsize=(5,3.2))
    ax.bar(x, ts_list)
    ax.set_title("LightGBM: Two-Stage by fold"); ax.set_xlabel("fold"); ax.set_ylabel("Two-Stage")
    plt.tight_layout(); plt.savefig(save_ts, dpi=200); plt.close()

def main():
    if not TRAIN.exists():
        raise FileNotFoundError("缺少 out/features_train.csv")
    df = load_train()
    print(f"[info] train shape = {df.shape}")

    metrics, diag = eval_lgbm_collect(df, n_splits=5)
    d = enrich(diag)
    d.to_csv(OUT / "diagnostics_lgbm.csv", index=False)

    print("=== LightGBM (TimeSeriesSplit 5-fold) ===")
    print(f"MAE={metrics['MAE']:,.2f} 万元 | MAPE={metrics['MAPE']*100:.2f}% "
          f"| R²={metrics['R2']:.3f} | Two-Stage={metrics['TwoStage']:.4f}")

    # 只输出 LGBM 图（原有）
    plot_residual_hist(d, save=OUT / "lgbm_residual_hist.png")
    plot_true_vs_pred (d, save=OUT / "lgbm_true_vs_pred.png")
    plot_ape_hist     (d, save=OUT / "lgbm_ape_hist.png")

    # 新增：按月份 Two-Stage / R² 变化图
    try:
        mdf = monthly_metrics(d)
        mdf.to_csv(OUT / "lgbm_monthly_metrics.csv", index=False)
        plot_metric_ts(mdf, "TwoStage", save=OUT / "lgbm_ts_by_month.png",
                       title="LightGBM: Two-Stage over time (by month)")
        plot_metric_ts(mdf, "R2", save=OUT / "lgbm_r2_by_month.png",
                       title="LightGBM: R² over time (by month)")
    except Exception as e:
        print(f"[warn] 月度变化图未生成：{e}")

    # 可选：按折数柱状图（便于报告对比）
    plot_fold_bars(metrics["R2_by_fold"], metrics["TS_by_fold"],
                   save_r2=OUT / "lgbm_r2_by_fold.png",
                   save_ts=OUT / "lgbm_ts_by_fold.png")

    print("[save] out/diagnostics_lgbm.csv, lgbm_residual_hist.png, lgbm_true_vs_pred.png, "
          "lgbm_ape_hist.png, lgbm_monthly_metrics.csv, lgbm_ts_by_month.png, "
          "lgbm_r2_by_month.png, lgbm_r2_by_fold.png, lgbm_ts_by_fold.png")

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
