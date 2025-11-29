# tft.py —— Temporal Fusion Transformer
# - test 推理 -> 生成 submission -> 用 test_solution 评估
from pathlib import Path
import argparse, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import copy
torch.set_float32_matmul_precision("medium")

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint, Callback

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss, MAE
from torchmetrics.regression import R2Score
from pytorch_forecasting.models import TemporalFusionTransformer

# 兼容 PyTorch 2.6 的 ckpt 安全反序列化
from torch.serialization import add_safe_globals
from pandas.core.internals.managers import BlockManager
from pytorch_forecasting.data.encoders import GroupNormalizer as _PFGroupNormalizer
from pytorch_forecasting.data.encoders import NaNLabelEncoder
add_safe_globals([_PFGroupNormalizer, pd.DataFrame, pd.Series, BlockManager])
from torchmetrics import Metric

BASE = Path(".")
TRAIN_CSV = BASE / "out/features_train.csv"
TEST_CSV  = BASE / "out/features_test.csv"
RAW_TEST  = BASE / "test.csv"              # 提交用 id 顺序
SOL_CSV   = BASE / "test_solution.csv"     # 官方真值（可选）
OUT_DIR   = BASE / "out"; OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "y"

# ---------------- 指标 ----------------
def two_stage_score(y_true, y_pred, eps=1e-6):
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    ape = np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), eps)
    if (ape > 1).mean() > 0.30:
        return 0.0
    mask = ape <= 1
    return float(np.clip(1.0 - ape[mask].sum() / len(ape), 0.0, 1.0))

def mae(a,b): a=np.asarray(a,float); b=np.asarray(b,float); return float(np.mean(np.abs(a-b)))
def mape(y, yhat, eps=1e-6): y=np.asarray(y,float); yhat=np.asarray(yhat,float); return float(np.mean(np.abs(yhat-y)/np.maximum(np.abs(y),eps)))
def r2(y, yhat): y=np.asarray(y,float); yhat=np.asarray(yhat,float); ss_res=np.sum((y-yhat)**2); ss_tot=np.sum((y-y.mean())**2); return float(1-ss_res/(ss_tot+1e-12))

class LastStepMAPE(Metric):
    full_state_update = False
    def __init__(self, eps: float = 1e-6):
        super().__init__(dist_sync_on_step=False)
        self.eps = eps
        self.add_state("ape_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count",   default=torch.tensor(0),   dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        if preds.ndim == 3:
            preds = preds[:, -1, 0]
        elif preds.ndim == 2:
            preds = preds[:, -1]
        if target.ndim == 2:
            target = target[:, -1]
        preds = preds.detach()
        target = target.detach()
        ape = (preds - target).abs() / torch.maximum(target.abs(), torch.tensor(self.eps, device=target.device))
        self.ape_sum += ape.sum()
        self.count += torch.tensor(ape.numel(), device=target.device, dtype=torch.int64)

    def compute(self):
        if self.count == 0:
            return torch.tensor(0.0, device=self.ape_sum.device)
        return self.ape_sum / self.count

class TwoStageMetric(Metric):
    full_state_update = False
    def __init__(self, eps: float = 1e-6):
        super().__init__(dist_sync_on_step=False)
        self.eps = eps
        self.add_state("big_err",   default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("ape_small", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count",     default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        if preds.ndim == 3:
            preds = preds[:, -1, 0]
        elif preds.ndim == 2:
            preds = preds[:, -1]
        if target.ndim == 2:
            target = target[:, -1]
        preds = preds.detach()
        target = target.detach()
        ape = (preds - target).abs() / torch.maximum(target.abs(), torch.tensor(self.eps, device=target.device))
        self.big_err += (ape > 1).sum()
        self.ape_small += ape[ape <= 1].sum()
        self.count += torch.tensor(ape.numel(), device=target.device, dtype=torch.int64)

    def compute(self):
        if self.count == 0:
            return torch.tensor(0.0, device=self.ape_small.device)
        big_ratio = self.big_err.float() / self.count.float()
        if big_ratio > 0.30:
            return torch.tensor(0.0, device=self.ape_small.device)
        return torch.clamp(1.0 - self.ape_small / self.count.float(), min=0.0, max=1.0)

# ---------------- 小工具 ----------------
def _dedup_columns(df: pd.DataFrame, where: str = "") -> pd.DataFrame:
    dup = df.columns[df.columns.duplicated()].tolist()
    if dup:
        print(f"[warn] duplicated columns in {where}: {dup} -> keep first occurrence")
        df = df.loc[:, ~df.columns.duplicated()].copy()
    return df

def add_calendar_feats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    df["month_num"] = df["month"].dt.month.astype("int16")
    df["quarter"]   = df["month"].dt.quarter.astype("int16")
    uniq = np.sort(df["month"].dropna().unique())
    mapper = {m:i for i,m in enumerate(uniq)}
    df["time_idx"] = df["month"].map(mapper).astype("int32")
    return _dedup_columns(df, "add_calendar_feats")

def build_time_idx_map(train_months, test_months):
    """保证 test 的 time_idx 严格延续 train"""
    t = pd.to_datetime(pd.Series(train_months)).dropna().sort_values().unique()
    mapper = {m:i for i,m in enumerate(t)}
    cur = len(mapper)
    for m in pd.to_datetime(pd.Series(test_months)).dropna().sort_values().unique():
        if m not in mapper:
            mapper[m] = cur; cur += 1
    return mapper

# ---------------- 清洗 ----------------
def clean_train_df(df: pd.DataFrame):
    need = {"sector","month","y"}
    if not need.issubset(df.columns):
        miss = need - set(df.columns)
        raise ValueError(f"features_train.csv 需包含：{need}，缺少：{miss}")

    df = add_calendar_feats(df)
    df["sector"] = pd.to_numeric(df["sector"], errors="coerce").astype("int64")
    df.drop(columns=["month_idx_global"], errors="ignore", inplace=True)
    # drop rows with missing month/time_idx to avoid NaNs downstream
    df = df[df["month"].notna() & df["time_idx"].notna()].copy()
    df = df[df["y"].notna()].copy()

    drop_non_features = ["id","y","sector","month","time_idx","month_num","quarter"]
    feat_cols = [c for c in df.columns if c not in drop_non_features]

    if feat_cols:
        df[feat_cols] = df[feat_cols].replace([np.inf, -np.inf], np.nan)
    # 删全空列
    all_nan_cols = [c for c in feat_cols if df[c].isna().all()]
    if all_nan_cols:
        print(f"[clean] drop all-NaN cols: {len(all_nan_cols)} e.g. {all_nan_cols[:5]}")
        df.drop(columns=all_nan_cols, inplace=True)
        feat_cols = [c for c in feat_cols if c not in all_nan_cols]
    # 缺失指示列
    has_nan_cols = [c for c in feat_cols if df[c].isna().any()]
    for c in has_nan_cols:
        df[c+"_nan"] = df[c].isna().astype("int8")
    feat_cols += [c+"_nan" for c in has_nan_cols]
    # 数值填充：组内中位数→全局中位数
    num_cols = [c for c in feat_cols if pd.api.types.is_numeric_dtype(df[c])]
    if num_cols:
        df[num_cols] = df.groupby("sector")[num_cols].transform(lambda g: g.fillna(g.median()))
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
        df[num_cols] = df[num_cols].replace([np.inf,-np.inf], 0)

    df = _dedup_columns(df, "clean_train(before select)")
    df = df[["sector","month","time_idx","y"] + feat_cols].sort_values(["sector","time_idx"])
    if "month_num" not in df.columns or "quarter" not in df.columns:
        df = add_calendar_feats(df)
    feat_cols = [c for c in df.columns if c not in ["sector","month","time_idx","y"]]
    feat_cols = [c for c in feat_cols if c not in ("month_num","quarter")]
    df = _dedup_columns(df, "clean_train(final)")
    return df, feat_cols

def clean_test_df(df_test: pd.DataFrame, feat_cols_train, month2idx_map, train_df_for_fill: pd.DataFrame | None = None):
    need = {"sector","month"}
    if not need.issubset(df_test.columns):
        miss = need - set(df_test.columns)
        raise ValueError(f"features_test.csv 需包含：{need}，缺少：{miss}")
    df = add_calendar_feats(df_test)
    df["sector"] = pd.to_numeric(df["sector"], errors="coerce").astype("int64")
    df["time_idx"] = pd.to_datetime(df["month"]).map(month2idx_map)
    missing_rows = df[df["month"].notna() & df["time_idx"].isna()][["sector","month","time_idx"]]
    if not missing_rows.empty:
        print(f"[warn] drop test rows with missing month->time_idx mapping: {len(missing_rows)}")
        print(missing_rows.head())
    # drop rows with missing month/time_idx to avoid NaNs in encoded tensors
    df = df[df["month"].notna() & df["time_idx"].notna()].copy()
    df["time_idx"] = df["time_idx"].astype("int32")

    # 对齐训练特征列：多余丢弃、缺列补0
    extra = [c for c in df.columns if c not in (["sector","month","time_idx","month_num","quarter"] + feat_cols_train)]
    if extra:
        df = df.drop(columns=extra)
    for c in feat_cols_train:
        if c not in df.columns:
            df[c] = 0
    # 将 train+test 拼接，按 sector+time_idx 前向填充，优先消灭交界处 NaN
    df["__is_test__"] = True
    combo_parts = [df]
    if train_df_for_fill is not None:
        tmp_train = train_df_for_fill.copy()
        tmp_train["__is_test__"] = False
        combo_parts.append(tmp_train)
    combo = pd.concat(combo_parts, ignore_index=True, sort=False)
    combo = combo.sort_values(["sector","time_idx"])

    num_cols = [c for c in feat_cols_train if pd.api.types.is_numeric_dtype(combo[c])]
    if num_cols:
        combo[num_cols] = combo[num_cols].replace([np.inf, -np.inf], np.nan)
        combo[num_cols] = combo.groupby("sector")[num_cols].ffill()
        # 剩余缺失（每个 sector 的起点）再用分组中位数→全局中位数，保持与训练一致
        combo[num_cols] = combo.groupby("sector")[num_cols].transform(lambda g: g.fillna(g.median()))
        combo[num_cols] = combo[num_cols].fillna(combo[num_cols].median())
        combo[num_cols] = combo[num_cols].replace([np.inf, -np.inf], 0)

    df = combo[combo["__is_test__"]].drop(columns="__is_test__", errors="ignore")

    df = df[["sector","month","time_idx","month_num","quarter"] + feat_cols_train].sort_values(["sector","time_idx"])
    # 预测阶段 TimeSeriesDataSet 不接受目标列 NaN，这里用 0 占位（不会被用作标签）
    df[TARGET] = 0.0
    df = _dedup_columns(df, "clean_test(final)")
    return df

def print_series_diagnostics(df: pd.DataFrame):
    g = df.groupby("sector")["time_idx"].agg(["min","max","count"]).reset_index()
    g["span"] = g["max"] - g["min"] + 1
    print(f"[diag] sectors={len(g)}, min_span={g['span'].min()}, max_span={g['span'].max()}, median_span={int(g['span'].median())}, min_count={g['count'].min()}")

# ---------------- 构建数据集（稳健版） ----------------
def make_tsdataset_hardened(df: pd.DataFrame, feature_cols, encoder_len: int, holdout_last: int):
    known_reals   = [c for c in feature_cols if c in ["month_num","quarter"]]
    unknown_reals = [c for c in feature_cols if c not in known_reals]

    enc = int(encoder_len)
    hol = int(holdout_last)

    while True:
        cutoff = int(df["time_idx"].max() - hol)   # 训练≤cutoff，验证>cutoff
        df_tr  = df[df["time_idx"] <= cutoff].copy()

        # 预筛 sector：cutoff 前至少 enc 历史，且 cutoff 后仍有样本
        gg = df.groupby("sector")["time_idx"].agg(["min","max"])
        ok_sec = gg.index[(gg["min"] <= cutoff - enc) & (gg["max"] > cutoff)]
        df_tr  = df_tr[df_tr["sector"].isin(ok_sec)].copy()
        df_all = df[df["sector"].isin(ok_sec)].copy()

        print(f"[try] encoder_len={enc}, holdout_last={hol} -> ok_sectors={len(ok_sec)}")
        if df_tr.empty or df_all[df_all["time_idx"] > cutoff].empty:
            if enc > 6: enc = max(6, enc - 4); continue
            if hol > 3: hol = max(3, hol - 1); continue
            raise RuntimeError("验证集仍为空：请降低 --encoder-len 或 --holdout-last。")

        training = TimeSeriesDataSet(
            df_tr,
            time_idx="time_idx",
            target=TARGET,
            group_ids=["sector"],
            categorical_encoders={"sector": NaNLabelEncoder(add_nan=True)},
            min_encoder_length=1,
            max_encoder_length=enc,
            min_prediction_length=1,
            max_prediction_length=1,
            time_varying_known_reals=["time_idx","month_num","quarter"],
            time_varying_unknown_reals=unknown_reals + [TARGET],
            target_normalizer=GroupNormalizer(groups=["sector"], transformation="softplus"),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            allow_missing_timesteps=True,
        )
        validation = TimeSeriesDataSet.from_dataset(
            training,
            df_all,
            min_prediction_idx=cutoff + 1,
            stop_randomization=True,
        )
        if len(validation) == 0:
            if enc > 6: enc = max(6, enc - 4); continue
            if hol > 3: hol = max(3, hol - 1); continue
            raise RuntimeError("验证数据条目为 0。")

        print(f"[ok] train_ds={len(training)} samples, valid_ds={len(validation)} samples")
        return training, validation, enc, hol

# ---------------- 每个“验证轮结束”打印指标的回调 ----------------
class EpochMetricsLogger(Callback):
    def __init__(self, valid_loader=None):
        super().__init__()
        self.valid_loader = valid_loader

    def on_validation_epoch_end(self, trainer, pl_module):
        if getattr(trainer, "sanity_checking", False):
            return
        if self.valid_loader is None:
            return
        # use a CPU copy to avoid disturbing the training model/device
        with torch.no_grad():
            tmp_model = copy.deepcopy(pl_module).cpu().eval()
            m = eval_on_loader(tmp_model, self.valid_loader)
        print(f"[epoch {trainer.current_epoch:02d}] "
              f"MAE={m['MAE']:,.2f} "
              f"MAPE={m['MAPE']*100:.2f}% "
              f"R2={m['R2']:.4f} "
              f"TwoStage={m['TwoStage']:.4f}")

# ---------------- 训练 / 评估 ----------------
def train_one_tft(train_ds, valid_ds, epochs, batch_size, hidden, attn_head, dropout, lr, patience):
    train_loader = train_ds.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    valid_loader = valid_ds.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

    model = TemporalFusionTransformer.from_dataset(
        train_ds,
        learning_rate=lr,
        hidden_size=hidden,
        attention_head_size=attn_head,
        dropout=dropout,
        loss=QuantileLoss(quantiles=[0.5]),  # 只学中位数
        optimizer="adamw",
        reduce_on_plateau_patience=2,
        logging_metrics=[MAE(), LastStepMAPE(), R2Score(), TwoStageMetric()],
    )

    ckpt = ModelCheckpoint(
        dirpath="lightning_logs",
        filename="tft-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss", mode="min",
        save_top_k=1,
    )
    early = EarlyStopping(monitor="val_loss", patience=patience, mode="min")
    lrmon = LearningRateMonitor(logging_interval="epoch")
    epilog = EpochMetricsLogger(valid_loader=valid_loader)

    trainer = Trainer(
        max_epochs=epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[ckpt, early, lrmon, epilog],
        enable_model_summary=False,
        gradient_clip_val=0.1,
        log_every_n_steps=50,
    )
    trainer.fit(model, train_loader, valid_loader, ckpt_path=None)

    best_path = ckpt.best_model_path or None
    if best_path:
        try:
            model = TemporalFusionTransformer.load_from_checkpoint(best_path)
        except Exception as e:
            print(f"[warn] load best ckpt failed, use in-memory model. err={e}")
    return model, valid_loader, best_path

def eval_on_loader(model, valid_loader):
    with torch.no_grad():
        pred = model.predict(valid_loader)
        yhat = pred if isinstance(pred, torch.Tensor) else torch.as_tensor(pred)
        if yhat.ndim == 3:  # (N, pred_len, Q)
            yhat = yhat[:, -1, 0]
        elif yhat.ndim == 2:
            yhat = yhat[:, -1]
        y_pred = yhat.detach().cpu().numpy().ravel()

    ys = []
    for _, y in valid_loader:
        y_tensor = y[0] if isinstance(y, (tuple, list)) else y
        if y_tensor.ndim == 2:
            y_tensor = y_tensor[:, -1]
        ys.append(y_tensor.detach().cpu().numpy().ravel())
    y_true = np.concatenate(ys, axis=0) if ys else np.array([])

    return dict(
        MAE = mae(y_true, y_pred),
        MAPE = mape(y_true, y_pred),
        ONE_MINUS_MAPE = 1.0 - mape(y_true, y_pred),
        R2  = r2(y_true, y_pred),
        TwoStage = two_stage_score(y_true, y_pred),
    )

def holdout_eval(df, feat_cols, args):
    train_ds, valid_ds, enc_used, hol_used = make_tsdataset_hardened(df, feat_cols, args.encoder_len, holdout_last=6)
    print(f"[holdout] encoder_len={enc_used}, holdout_last={hol_used}")
    model, vloader, best_path = train_one_tft(
        train_ds, valid_ds,
        epochs=args.epochs, batch_size=args.batch_size,
        hidden=args.hidden, attn_head=args.attn_head, dropout=args.dropout,
        lr=args.lr, patience=args.patience
    )
    m = eval_on_loader(model, vloader)
    print("\n=== Holdout (last months, best ckpt) ===")
    print(f"MAE={m['MAE']:,.2f} 万元 | MAPE={m['MAPE']*100:.2f}% | 1-MAPE={(1-m['MAPE'])*100:.2f}%")
    print(f"R²={m['R2']:.3f} | Two-Stage={m['TwoStage']:.4f}")
    return model, train_ds, m, best_path

# ---------------- test 推理 + 评测 ----------------
def parse_test_id(id_str):
    parts = str(id_str).split("_sector")
    ym_str = parts[0].strip()        # 'YYYY Mon'
    sec    = int(parts[1].strip())
    ym = pd.to_datetime(ym_str, format="%Y %b")
    return pd.to_datetime(ym.strftime("%Y-%m-01")), sec

def predict_on_test(model, train_ds, train_df, test_df, batch_size=128):
    # 组合历史 + 测试月份
    df_pred_all = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    pred_ds = TimeSeriesDataSet.from_dataset(
        train_ds, df_pred_all, predict=True,
        min_prediction_idx=int(test_df["time_idx"].min()),
        stop_randomization=True
    )
    pred_loader = pred_ds.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

    # 先收集索引，再预测，确保长度对齐
    index_records = []
    for batch_x, _ in pred_loader:
        idx_df = pred_ds.x_to_index(batch_x)
        index_records.extend(idx_df.to_dict("records"))

    with torch.no_grad():
        yhat = model.predict(pred_loader)
        yhat = yhat if isinstance(yhat, torch.Tensor) else torch.as_tensor(yhat)
    if yhat.ndim == 3:
        yhat = yhat[:, -1, 0]
    elif yhat.ndim == 2:
        yhat = yhat[:, -1]
    yhat = yhat.detach().cpu().numpy().ravel()

    if len(index_records) != len(yhat):
        raise RuntimeError(f"prediction length mismatch: index {len(index_records)} vs preds {len(yhat)}")
    pred_index_df = pd.DataFrame(index_records)
    if "sector" not in pred_index_df.columns and "groups" in pred_index_df.columns:
        pred_index_df["sector"] = pred_index_df["groups"].apply(lambda g: g[0] if isinstance(g, (list, tuple)) else g)
    pred_index_df["time_idx"] = pred_index_df["time_idx"].astype("int32")
    pred_index_df["yhat"] = yhat

    test_pred = test_df.merge(pred_index_df, on=["sector","time_idx"], how="left")
    if test_pred["yhat"].isna().any():
        missing_secs = test_pred.loc[test_pred["yhat"].isna(), "sector"].unique().tolist()
        raise RuntimeError(f"未找到预测结果的 sector：{missing_secs}。请检查这些序列是否出现在训练集中。")
    return test_pred[["sector","month","time_idx","yhat"]].sort_values(["sector","time_idx"])

# safer version: ensures no NaN time_idx before building prediction dataset
# safer version: ensures no NaN time_idx before building prediction dataset
def predict_on_test_safe(model, train_ds, train_df, test_df, batch_size=128):
    import numpy as np

    train_df = train_df.copy()
    test_df = test_df.copy()
    for name, df_ref in [("train_df", train_df), ("test_df", test_df)]:
        if df_ref["time_idx"].isna().any():
            cnt = int(df_ref["time_idx"].isna().sum())
            print(f"[warn] drop rows with NaN time_idx in {name}: {cnt}")
            df_ref.dropna(subset=["time_idx"], inplace=True)
            df_ref["time_idx"] = df_ref["time_idx"].astype("int32")

    # ensure calendar features exist
    for df_ref in (train_df, test_df):
        df_ref["month_num"] = pd.to_datetime(df_ref["month"]).dt.month.astype("int16")
        df_ref["quarter"] = pd.to_datetime(df_ref["month"]).dt.quarter.astype("int16")

    base_cols = ["sector", "month", "time_idx", "month_num", "quarter", "y"]
    feat_cols = [c for c in train_df.columns if c not in base_cols]
    pred_in = (
        pd.concat(
            [
                train_df[base_cols + feat_cols].copy(),
                test_df[base_cols + feat_cols].copy().assign(y=0.0),
            ],
            ignore_index=True,
        )
        .sort_values(["sector", "time_idx"])
    )
    pred_in = _dedup_columns(pred_in, "predict_on_test_safe(pred_in)")
    work_base = pred_in.copy()
    encoders = getattr(train_ds, "get_parameters", lambda: {})()
    sec_encoder = encoders.get("categorical_encoders", {}).get("sector") if isinstance(encoders, dict) else None

    uniq_times = sorted(test_df["time_idx"].unique())
    min_pred_time = int(min(uniq_times))
    enc_len = getattr(train_ds, "max_encoder_length", 1)
    global_y_med = float(train_df["y"].median())
    pad_rows = []
    for sec, grp in work_base.groupby("sector"):
        hist = grp[(grp["time_idx"] < min_pred_time) & (grp["time_idx"] >= min_pred_time - enc_len)]
        if hist.empty:
            first_row = grp.iloc[0].copy()
            pad_row = first_row.copy()
            pad_row["time_idx"] = np.int32(min_pred_time - 1)
            base_month = pd.to_datetime(first_row["month"])
            if pd.notna(base_month):
                pad_month = (base_month - pd.offsets.MonthBegin(1)).normalize()
                pad_row["month"] = pad_month
                pad_row["month_num"] = np.int16(pad_month.month)
                pad_row["quarter"] = np.int16(pad_month.quarter)
            sec_med = train_df.loc[train_df["sector"] == sec, "y"].median()
            pad_row["y"] = float(sec_med) if not np.isnan(sec_med) else global_y_med
            pad_rows.append(pad_row)
    if pad_rows:
        work_base = (
            pd.concat([work_base, pd.DataFrame(pad_rows)], ignore_index=True)
            .sort_values(["sector", "time_idx"])
            .reset_index(drop=True)
        )

    all_chunks = []
    model.eval()
    with torch.no_grad():
        for t_val in uniq_times:
            cur_base = work_base[work_base["time_idx"] <= t_val].copy()
            pred_ds = TimeSeriesDataSet.from_dataset(
                train_ds,
                cur_base,
                predict=True,
                min_prediction_idx=int(t_val),
                stop_randomization=True,
            )
            pred_out = model.predict(pred_ds, return_index=True, batch_size=batch_size)

            index_df = None
            yhat = None
            if isinstance(pred_out, tuple):
                for item in pred_out:
                    if isinstance(item, pd.DataFrame):
                        index_df = item
                    if isinstance(item, torch.Tensor):
                        yhat = item
            else:
                yhat = pred_out
            if yhat is None or index_df is None:
                raise RuntimeError("model.predict did not return both predictions and index")

            yhat = yhat if isinstance(yhat, torch.Tensor) else torch.as_tensor(yhat)
            if yhat.ndim == 3:
                yhat = yhat[:, -1, 0]
            elif yhat.ndim == 2:
                yhat = yhat[:, -1]
            yhat = yhat.detach().cpu().numpy().ravel()

            chunk = index_df.copy()
            if "sector" not in chunk.columns and "groups" in chunk.columns:
                chunk["sector"] = chunk["groups"].apply(lambda g: g[0] if isinstance(g, (list, tuple)) else g)
            if "group_ids" in chunk.columns and "sector" not in chunk.columns:
                chunk["sector"] = chunk["group_ids"].apply(lambda g: g[0] if isinstance(g, (list, tuple)) else g)
            if sec_encoder is not None and "sector" in chunk.columns:
                try:
                    chunk["sector"] = sec_encoder.inverse_transform(chunk["sector"])
                except Exception:
                    pass
            if len(chunk) != len(yhat):
                raise RuntimeError(f"prediction length mismatch at t={t_val}: index {len(chunk)} vs preds {len(yhat)}")

            chunk["time_idx"] = chunk["time_idx"].astype("int32")
            chunk["sector"] = chunk["sector"].astype("int64")
            chunk["yhat"] = yhat
            chunk = chunk.loc[chunk["time_idx"] == int(t_val)]
            if chunk.empty:
                raise RuntimeError(f"no predictions produced at time_idx={t_val}; check dataset/index mapping")
            for _, row in chunk.iterrows():
                mask = (work_base["sector"] == row["sector"]) & (work_base["time_idx"] == int(t_val))
                work_base.loc[mask, "y"] = row["yhat"]
            all_chunks.append(chunk)

    pred_index_df = pd.concat(all_chunks, ignore_index=True)
    test_pred = test_df.merge(pred_index_df, on=["sector", "time_idx"], how="left")
    if test_pred["yhat"].isna().any():
        missing_secs = test_pred.loc[test_pred["yhat"].isna(), "sector"].unique().tolist()
        present_secs = pred_index_df["sector"].unique().tolist() if not pred_index_df.empty else []
        t_min, t_max = (pred_index_df["time_idx"].min(), pred_index_df["time_idx"].max()) if not pred_index_df.empty else (None, None)
        raise RuntimeError(
            f"预测缺失，sector={missing_secs}，请检查时间索引映射是否完整；"
            f" 已预测sector数量={len(present_secs)}, time_idx范围=({t_min},{t_max})"
        )
    return test_pred[["sector", "month", "time_idx", "yhat"]].sort_values(["sector", "time_idx"])


def make_submission(pred_df: pd.DataFrame, raw_test_csv: Path, out_path: Path):
    ids = pd.read_csv(raw_test_csv)["id"]
    month_sector = ids.map(parse_test_id)
    id_df = pd.DataFrame(list(month_sector), columns=["month","sector"])
    id_df["id"] = ids
    sub = id_df.merge(pred_df, on=["sector","month"], how="left")
    sub["new_house_transaction_amount"] = sub["yhat"].clip(lower=0)
    sub = sub[["id","new_house_transaction_amount"]]
    sub.to_csv(out_path, index=False)
    print(f"[save] {out_path} ({sub.shape})")
    return sub

def eval_with_solution(sub_df: pd.DataFrame, sol_csv: Path):
    """Offline eval with test_solution.csv; handles extra columns and Usage split."""
    if not sol_csv.exists():
        print("[eval] test_solution.csv not found; skip eval")
        return
    sol = pd.read_csv(sol_csv)
    sol_cols = [c for c in sol.columns if c.lower() != "id" and not c.lower().startswith("usage")]
    if not sol_cols:
        raise ValueError(f"test_solution.csv missing ground truth column: {sol.columns.tolist()}")
    if len(sol_cols) > 1:
        print(f"[warn] multiple candidate GT cols in test_solution.csv {sol_cols}; using first: {sol_cols[0]}")
    gt_col = sol_cols[0]

    merged = sub_df.merge(sol, on="id", how="inner", suffixes=("", "_gt"))
    if "new_house_transaction_amount" not in merged.columns:
        raise KeyError(f"predicted column missing after merge; got columns {merged.columns.tolist()}")
    gt_col_merged = gt_col + "_gt" if gt_col + "_gt" in merged.columns else gt_col
    if gt_col_merged not in merged.columns:
        raise KeyError(f"ground truth column '{gt_col}' missing after merge; got columns {merged.columns.tolist()}")

    def _metrics(df, tag):
        y_pred = pd.to_numeric(df["new_house_transaction_amount"], errors="coerce").fillna(0).values
        y_true = pd.to_numeric(df[gt_col_merged], errors="coerce").fillna(0).values
        m_mae = mae(y_true, y_pred)
        m_mape = mape(y_true, y_pred)
        m_r2 = r2(y_true, y_pred)
        m_two = two_stage_score(y_true, y_pred)
        ape = np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), 1e-6)
        print(f"\n=== TFT on TEST ({tag}) ===")
        print(f"MAE={m_mae:,.2f} | MAPE={m_mape*100:.2f}% | 1-MAPE={(1-m_mape)*100:.2f}%")
        print(f"R2={m_r2:.3f} | Two-Stage={m_two:.4f}")
        print(f"big-error ratio (APE>100%) = {(ape>1).mean()*100:.2f}%")

    _metrics(merged, "overall")
    if "Usage" in merged.columns:
        for u, df_u in merged.groupby("Usage"):
            _metrics(df_u, f"Usage={u}")

# ---------------- 主流程 ----------------
def main():
    raise SystemExit('Use tft_train.py for training/validation, tft_predict.py for test prediction.')

if __name__ == "__main__":
    main()
