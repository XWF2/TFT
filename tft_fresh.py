#!/usr/bin/env python3
# Clean TFT pipeline: data cleaning + training + prediction in one file.
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import NaNLabelEncoder
from pytorch_forecasting.metrics import QuantileLoss, MAE
from pytorch_forecasting.models import TemporalFusionTransformer
from torchmetrics.regression import R2Score

BASE = Path(".")
TRAIN_CSV = BASE / "out/features_train.csv"
TEST_CSV = BASE / "out/features_test.csv"
RAW_TEST = BASE / "test.csv"
SOL_CSV = BASE / "test_solution.csv"
OUT_DIR = BASE / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "y"

# ---------------- metrics & eval ----------------
def two_stage_score(y_true, y_pred, eps: float = 1e-6) -> float:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    ape = np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), eps)
    if (ape > 1).mean() > 0.30:
        return 0.0
    mask = ape <= 1
    return float(np.clip(1.0 - ape[mask].sum() / len(ape), 0.0, 1.0))


def mae_np(a, b) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    return float(np.mean(np.abs(a - b)))


def mape_np(y, yhat, eps: float = 1e-6) -> float:
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    return float(np.mean(np.abs(yhat - y) / np.maximum(np.abs(y), eps)))


def r2_np(y, yhat) -> float:
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-12))


def eval_with_solution(sub_df: pd.DataFrame, sol_csv: Path):
    """Offline eval with test_solution.csv; tolerant to extra columns and encoding-safe prints."""
    if not sol_csv.exists():
        print("[eval] test_solution.csv not found; skip eval")
        return
    sol = pd.read_csv(sol_csv)
    sol_cols = [c for c in sol.columns if c.lower() != "id" and not c.lower().startswith("usage")]
    if not sol_cols:
        raise ValueError(f"test_solution.csv missing ground-truth column: {sol.columns.tolist()}")
    if len(sol_cols) > 1:
        print(f"[warn] multiple candidate GT cols in test_solution.csv {sol_cols}; using first: {sol_cols[0]}")
    gt_col = sol_cols[0]

    # use submission column if present; otherwise fall back to renamed version
    if "new_house_transaction_amount" in sub_df.columns:
        pred_col = "new_house_transaction_amount"
        sub_eval = sub_df.copy()
    else:
        pred_col = "predicted_amount"
        sub_eval = sub_df.rename(columns={"new_house_transaction_amount": pred_col})

    merged = sub_eval.merge(sol, on="id", how="inner", suffixes=("", "_gt"))
    # guard against suffix collisions
    if pred_col not in merged.columns:
        alt_col = pred_col + "_gt"
        if alt_col in merged.columns:
            pred_col = alt_col
        else:
            raise KeyError(f"predicted column '{pred_col}' missing after merge; got columns {merged.columns.tolist()}")

    # ground truth may have been suffixed during merge
    gt_col_merged = gt_col + "_gt" if gt_col + "_gt" in merged.columns else gt_col
    if gt_col_merged not in merged.columns:
        raise KeyError(f"ground truth column '{gt_col}' missing after merge; got columns {merged.columns.tolist()}")

    y_pred = pd.to_numeric(merged[pred_col], errors="coerce").fillna(0).values
    y_true = pd.to_numeric(merged[gt_col_merged], errors="coerce").fillna(0).values

    m_mae = mae_np(y_true, y_pred)
    m_mape = mape_np(y_true, y_pred)
    m_r2 = r2_np(y_true, y_pred)
    m_two = two_stage_score(y_true, y_pred)

    print("\\n=== TFT on TEST (w/ test_solution) ===")
    print(f"MAE={m_mae:,.2f} | MAPE={m_mape*100:.2f}% | 1-MAPE={(1-m_mape)*100:.2f}%")
    print(f"R2={m_r2:.3f} | Two-Stage={m_two:.4f}")
    ape = np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), 1e-6)
    print(f"big-error ratio (APE>100%) = {(ape>1).mean()*100:.2f}%")


# ---------------- utils ----------------
def _dedup_columns(df: pd.DataFrame, where: str = "") -> pd.DataFrame:
    dup = df.columns[df.columns.duplicated()].tolist()
    if dup:
        print(f"[warn] duplicated columns in {where}: {dup} -> keep first")
        df = df.loc[:, ~df.columns.duplicated()].copy()
    return df


def add_calendar_feats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    df["month_num"] = df["month"].dt.month.astype("int16")
    df["quarter"] = df["month"].dt.quarter.astype("int16")
    uniq = np.sort(df["month"].dropna().unique())
    mapper = {m: i for i, m in enumerate(uniq)}
    df["time_idx"] = df["month"].map(mapper).astype("int32")
    return _dedup_columns(df, "add_calendar_feats")


def build_time_idx_map(train_months, test_months) -> Dict[pd.Timestamp, int]:
    t = pd.to_datetime(pd.Series(train_months)).dropna().sort_values().unique()
    mapper = {m: i for i, m in enumerate(t)}
    cur = len(mapper)
    for m in pd.to_datetime(pd.Series(test_months)).dropna().sort_values().unique():
        if m not in mapper:
            mapper[m] = cur
            cur += 1
    return mapper


def clean_train_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    need = {"sector", "month", "y"}
    if not need.issubset(df.columns):
        miss = need - set(df.columns)
        raise ValueError(f"features_train.csv missing columns: {miss}")
    df = add_calendar_feats(df)
    df["sector"] = pd.to_numeric(df["sector"], errors="coerce").astype("int64")
    df.drop(columns=["month_idx_global"], errors="ignore", inplace=True)
    df = df[df["month"].notna() & df["time_idx"].notna() & df["y"].notna()].copy()

    drop_non_features = ["id", "y", "sector", "month", "time_idx", "month_num", "quarter"]
    feat_cols = [c for c in df.columns if c not in drop_non_features]

    if feat_cols:
        df[feat_cols] = df[feat_cols].replace([np.inf, -np.inf], np.nan)
    all_nan_cols = [c for c in feat_cols if df[c].isna().all()]
    if all_nan_cols:
        print(f"[clean] drop all-NaN cols: {len(all_nan_cols)} e.g. {all_nan_cols[:5]}")
        df.drop(columns=all_nan_cols, inplace=True)
        feat_cols = [c for c in feat_cols if c not in all_nan_cols]
    has_nan_cols = [c for c in feat_cols if df[c].isna().any()]
    for c in has_nan_cols:
        df[c + "_nan"] = df[c].isna().astype("int8")
    feat_cols += [c + "_nan" for c in has_nan_cols]

    num_cols = [c for c in feat_cols if pd.api.types.is_numeric_dtype(df[c])]
    if num_cols:
        df[num_cols] = df.groupby("sector")[num_cols].transform(lambda g: g.fillna(g.median()))
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
        df[num_cols] = df[num_cols].replace([np.inf, -np.inf], 0)

    df = _dedup_columns(df, "clean_train(before select)")
    df = df[["sector", "month", "time_idx", "y"] + feat_cols].sort_values(["sector", "time_idx"])
    if "month_num" not in df.columns or "quarter" not in df.columns:
        df = add_calendar_feats(df)
    feat_cols = [c for c in df.columns if c not in ["sector", "month", "time_idx", "y"]]
    feat_cols = [c for c in feat_cols if c not in ("month_num", "quarter")]
    df = _dedup_columns(df, "clean_train(final)")
    return df, feat_cols


def clean_test_df(df_test: pd.DataFrame, feat_cols_train: List[str], month2idx_map, train_df_for_fill: pd.DataFrame) -> pd.DataFrame:
    need = {"sector", "month"}
    if not need.issubset(df_test.columns):
        miss = need - set(df_test.columns)
        raise ValueError(f"features_test.csv missing columns: {miss}")
    df = add_calendar_feats(df_test)
    df["sector"] = pd.to_numeric(df["sector"], errors="coerce").astype("int64")
    df["time_idx"] = pd.to_datetime(df["month"]).map(month2idx_map)
    df = df[df["month"].notna() & df["time_idx"].notna()].copy()
    df["time_idx"] = df["time_idx"].astype("int32")

    extra = [c for c in df.columns if c not in (["sector", "month", "time_idx", "month_num", "quarter"] + feat_cols_train)]
    if extra:
        df = df.drop(columns=extra)
    for c in feat_cols_train:
        if c not in df.columns:
            df[c] = 0

    df = df[["sector", "month", "time_idx", "month_num", "quarter"] + feat_cols_train].sort_values(["sector", "time_idx"])
    df[TARGET] = np.nan

    combo = pd.concat([train_df_for_fill, df], ignore_index=True, sort=False).sort_values(["sector", "time_idx"])
    num_cols = [c for c in feat_cols_train if pd.api.types.is_numeric_dtype(combo[c])]
    if num_cols:
        combo[num_cols] = combo[num_cols].replace([np.inf, -np.inf], np.nan)
        combo[num_cols] = combo.groupby("sector")[num_cols].ffill()
        combo[num_cols] = combo.groupby("sector")[num_cols].transform(lambda g: g.fillna(g.median()))
        combo[num_cols] = combo[num_cols].fillna(combo[num_cols].median())
        combo[num_cols] = combo[num_cols].replace([np.inf, -np.inf], 0)
    df_filled = combo[(combo["time_idx"].isin(df["time_idx"])) & (combo["sector"].isin(df["sector"]))].copy()
    df_filled = _dedup_columns(df_filled, "clean_test(final)")
    df_filled[TARGET] = 0.0
    return df_filled


# ------------- dataset ----------------
def make_datasets(df: pd.DataFrame, feat_cols: List[str], encoder_len: int, holdout_last: int):
    cutoff = int(df["time_idx"].max() - holdout_last)
    df_train = df[df["time_idx"] <= cutoff].copy()
    if df_train.empty:
        raise RuntimeError("训练数据为空，请调小 holdout_last")
    training = TimeSeriesDataSet(
        df_train,
        time_idx="time_idx",
        target=TARGET,
        group_ids=["sector"],
        categorical_encoders={"sector": NaNLabelEncoder(add_nan=True)},
        max_encoder_length=encoder_len,
        min_encoder_length=1,
        max_prediction_length=1,
        min_prediction_length=1,
        time_varying_known_reals=["time_idx", "month_num", "quarter"],
        time_varying_unknown_reals=feat_cols + [TARGET],
        target_normalizer=None,
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )
    validation = TimeSeriesDataSet.from_dataset(
        training,
        df,
        min_prediction_idx=cutoff + 1,
        stop_randomization=True,
    )
    return training, validation


def predict_full(model, train_ds, train_df, test_df, feat_cols: List[str], batch_size: int = 128) -> pd.DataFrame:
    """
    覆盖测试集所有 time_idx：对每个未来 time_idx 单独构造 predict 数据集并取预测。
    """
    train_df = train_df.copy()
    test_df = test_df.copy()
    # calendar features
    for df_ref in (train_df, test_df):
        df_ref["month_num"] = pd.to_datetime(df_ref["month"]).dt.month.astype("int16")
        df_ref["quarter"] = pd.to_datetime(df_ref["month"]).dt.quarter.astype("int16")
    base_cols = ["sector", "month", "time_idx", "month_num", "quarter", "y"]
    pred_in_base = (
        pd.concat(
            [
                train_df[base_cols + feat_cols].copy(),
                test_df[base_cols + feat_cols].copy().assign(y=0.0),
            ],
            ignore_index=True,
        )
        .sort_values(["sector", "time_idx"])
    )
    pred_in_base = _dedup_columns(pred_in_base, "predict_full(pred_in_base)")
    work_base = pred_in_base.copy()

    all_chunks = []
    model.eval()
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

    with torch.no_grad():
        for t_val in uniq_times:
            # IMPORTANT: trim future rows so the dataset's max time_idx equals the one we want to predict.
            # Otherwise, TimeSeriesDataSet.from_dataset will only emit the last available timestep.
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

            if "sector" not in index_df.columns and "groups" in index_df.columns:
                index_df["sector"] = index_df["groups"].apply(lambda g: g[0] if isinstance(g, (list, tuple)) else g)
            if "group_ids" in index_df.columns and "sector" not in index_df.columns:
                index_df["sector"] = index_df["group_ids"].apply(lambda g: g[0] if isinstance(g, (list, tuple)) else g)
            if len(index_df) != len(yhat):
                raise RuntimeError(f"prediction length mismatch at t={t_val}: index {len(index_df)} vs preds {len(yhat)}")

            chunk = index_df.copy()
            chunk["time_idx"] = chunk["time_idx"].astype("int32")
            chunk["sector"] = chunk["sector"].astype("int64")
            chunk["yhat"] = yhat
            # keep only the current forecast horizon to avoid duplicates across iterations
            chunk = chunk.loc[chunk["time_idx"] == int(t_val)]
            if chunk.empty:
                raise RuntimeError(f"no predictions produced at time_idx={t_val}; check dataset/index mapping")
            # feed the predicted target back for subsequent horizons so it can be used as history
            for _, row in chunk.iterrows():
                mask = (work_base["sector"] == row["sector"]) & (work_base["time_idx"] == int(t_val))
                work_base.loc[mask, "y"] = row["yhat"]
            all_chunks.append(chunk)

    pred_index_df = pd.concat(all_chunks, ignore_index=True)
    test_pred = test_df.merge(pred_index_df, on=["sector", "time_idx"], how="left")
    if test_pred["yhat"].isna().any():
        missing_secs = test_pred.loc[test_pred["yhat"].isna(), "sector"].unique().tolist()
        raise RuntimeError(f"预测缺失，sector={missing_secs}，请检查时间索引映射是否完整")
    return test_pred[["sector", "month", "time_idx", "yhat"]].sort_values(["sector", "time_idx"])


# ------------- train & predict ----------------
def train_one(args):
    seed_everything(args.seed, workers=True)
    if not TRAIN_CSV.exists():
        raise FileNotFoundError(f"missing {TRAIN_CSV}")
    raw_train = pd.read_csv(TRAIN_CSV)
    train_df, feat_cols = clean_train_df(raw_train)
    print(f"[info] train shape={train_df.shape}, feats={len(feat_cols)}")

    train_ds, val_ds = make_datasets(train_df, feat_cols, args.encoder_len, args.holdout_last)
    train_loader = train_ds.to_dataloader(train=True, batch_size=args.batch_size, num_workers=0)
    val_loader = val_ds.to_dataloader(train=False, batch_size=args.batch_size, num_workers=0)

    model = TemporalFusionTransformer.from_dataset(
        train_ds,
        learning_rate=args.lr,
        hidden_size=args.hidden,
        attention_head_size=args.attn_head,
        dropout=args.dropout,
        loss=QuantileLoss(quantiles=[0.5]),
        optimizer="adamw",
        logging_metrics=[MAE(), R2Score()],
        reduce_on_plateau_patience=2,
    )

    ckpt = ModelCheckpoint(
        dirpath=OUT_DIR,
        filename="tft_fresh-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[ckpt, EarlyStopping(monitor="val_loss", patience=args.patience, mode="min"), LearningRateMonitor()],
        enable_model_summary=False,
        gradient_clip_val=0.1,
        log_every_n_steps=50,
    )
    trainer.fit(model, train_loader, val_loader)
    best_ckpt = ckpt.best_model_path
    if best_ckpt:
        shutil.copy(best_ckpt, args.ckpt_path)
        print(f"[save] best ckpt -> {args.ckpt_path}")
    else:
        print("[warn] no best checkpoint found; skipping copy")

    train_ds.save(args.train_ds_path)
    print(f"[save] train TimeSeriesDataSet -> {args.train_ds_path}")
    meta = {"feat_cols": feat_cols, "encoder_len": args.encoder_len, "holdout_last": args.holdout_last}
    args.meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[save] meta -> {args.meta_path}")


def predict_one(args):
    seed_everything(args.seed, workers=True)
    for p in [TRAIN_CSV, TEST_CSV, RAW_TEST]:
        if not p.exists():
            raise FileNotFoundError(f"missing {p}")
    meta = json.loads(args.meta_path.read_text(encoding="utf-8"))
    feat_cols = meta["feat_cols"]

    raw_train = pd.read_csv(TRAIN_CSV)
    te_raw = pd.read_csv(TEST_CSV)
    train_df, _ = clean_train_df(raw_train)

    month2idx = build_time_idx_map(train_df["month"], te_raw["month"])
    train_df["time_idx"] = pd.to_datetime(train_df["month"]).map(month2idx).astype("int32")
    train_for_fill = train_df[["sector", "month", "time_idx", "month_num", "quarter"] + feat_cols].copy()
    test_df = clean_test_df(te_raw, feat_cols, month2idx, train_df_for_fill=train_for_fill)

    train_ds = TimeSeriesDataSet.load(args.train_ds_path)
    model = TemporalFusionTransformer.load_from_checkpoint(args.ckpt_path)

    pred_df = predict_full(model, train_ds, train_df, test_df, feat_cols, batch_size=args.batch_size)

    ids = pd.read_csv(RAW_TEST)["id"]
    month_sector = ids.map(parse_test_id)
    id_df = pd.DataFrame(list(month_sector), columns=["month", "sector"])
    id_df["id"] = ids
    sub = id_df.merge(pred_df, on=["sector", "month"], how="left")
    sub["new_house_transaction_amount"] = sub["yhat"].clip(lower=0)
    sub = sub[["id", "new_house_transaction_amount"]]
    sub.to_csv(args.submission_path, index=False)
    print(f"[save] submission -> {args.submission_path} ({sub.shape})")
    if SOL_CSV.exists():
        eval_with_solution(sub, SOL_CSV)
    else:
        print("[eval] test_solution.csv not found; skip offline eval")
    return sub


# ---------------- parse test id ----------------
def parse_test_id(id_str):
    parts = str(id_str).split("_sector")
    ym_str = parts[0].strip()  # 'YYYY Mon'
    sec = int(parts[1].strip())
    ym = pd.to_datetime(ym_str, format="%Y %b")
    return pd.to_datetime(ym.strftime("%Y-%m-01")), sec


# ---------------- main CLI ----------------
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_train = sub.add_parser("train")
    ap_train.add_argument("--seed", type=int, default=42)
    ap_train.add_argument("--encoder-len", type=int, default=24)
    ap_train.add_argument("--holdout-last", type=int, default=6)
    ap_train.add_argument("--epochs", type=int, default=20)
    ap_train.add_argument("--batch-size", type=int, default=128)
    ap_train.add_argument("--hidden", type=int, default=64)
    ap_train.add_argument("--attn-head", type=int, default=4)
    ap_train.add_argument("--dropout", type=float, default=0.15)
    ap_train.add_argument("--lr", type=float, default=1e-3)
    ap_train.add_argument("--patience", type=int, default=5)
    ap_train.add_argument("--ckpt-path", type=Path, default=OUT_DIR / "tft_fresh_best.ckpt")
    ap_train.add_argument("--train-ds-path", type=Path, default=OUT_DIR / "tft_fresh_train_ds")
    ap_train.add_argument("--meta-path", type=Path, default=OUT_DIR / "tft_fresh_meta.json")

    ap_pred = sub.add_parser("predict")
    ap_pred.add_argument("--seed", type=int, default=42)
    ap_pred.add_argument("--batch-size", type=int, default=128)
    ap_pred.add_argument("--ckpt-path", type=Path, default=OUT_DIR / "tft_fresh_best.ckpt")
    ap_pred.add_argument("--train-ds-path", type=Path, default=OUT_DIR / "tft_fresh_train_ds")
    ap_pred.add_argument("--meta-path", type=Path, default=OUT_DIR / "tft_fresh_meta.json")
    ap_pred.add_argument("--submission-path", type=Path, default=OUT_DIR / "submission_tft_fresh.csv")

    args = ap.parse_args()
    if args.cmd == "train":
        train_one(args)
    else:
        predict_one(args)


if __name__ == "__main__":
    main()
