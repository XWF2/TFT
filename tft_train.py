#!/usr/bin/env python3
# Train + validate TFT, save model and dataset
import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet

from tft import (
    OUT_DIR,
    TRAIN_CSV,
    clean_train_df,
    holdout_eval,
    print_series_diagnostics,
    seed_everything,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--encoder-len", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--attn-head", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--ckpt-path", type=Path, default=OUT_DIR / "tft_best.ckpt")
    ap.add_argument("--train-ds-path", type=Path, default=OUT_DIR / "tft_train_ds")
    ap.add_argument("--feat-cols-path", type=Path, default=OUT_DIR / "tft_feat_cols.json")
    args = ap.parse_args()

    seed_everything(args.seed, workers=True)

    if not TRAIN_CSV.exists():
        raise FileNotFoundError(f"missing {TRAIN_CSV}")

    raw_train = pd.read_csv(TRAIN_CSV)
    train_df, feat_cols = clean_train_df(raw_train)
    print(f"[info] train shape = {train_df.shape}, features = {len(feat_cols)}")
    print_series_diagnostics(train_df)

    model, train_ds, metrics, best_ckpt = holdout_eval(train_df, feat_cols, args)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    args.ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    if best_ckpt and Path(best_ckpt).exists():
        shutil.copy(Path(best_ckpt), args.ckpt_path)
        print(f"[save] best ckpt -> {args.ckpt_path}")
    else:
        print("[warn] no best checkpoint found; skip saving model ckpt")

    TimeSeriesDataSet.save(train_ds, args.train_ds_path)
    print(f"[save] train TimeSeriesDataSet -> {args.train_ds_path}")

    with args.feat_cols_path.open("w", encoding="utf-8") as f:
        json.dump(feat_cols, f, ensure_ascii=False, indent=2)
    print(f"[save] feature columns -> {args.feat_cols_path}")

    print(f"[done] holdout metrics: {metrics}")


if __name__ == "__main__":
    main()
