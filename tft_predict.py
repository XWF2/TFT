#!/usr/bin/env python3
# Load saved TFT and generate test predictions/submission
import argparse
import json
from pathlib import Path

import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.models import TemporalFusionTransformer

from tft import (
    OUT_DIR,
    RAW_TEST,
    SOL_CSV,
    TEST_CSV,
    TRAIN_CSV,
    build_time_idx_map,
    clean_test_df,
    clean_train_df,
    make_submission,
    predict_on_test_safe,
    seed_everything,
    eval_with_solution,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--ckpt-path", type=Path, default=OUT_DIR / "tft_best.ckpt")
    ap.add_argument("--train-ds-path", type=Path, default=OUT_DIR / "tft_train_ds")
    ap.add_argument("--feat-cols-path", type=Path, default=OUT_DIR / "tft_feat_cols.json")
    ap.add_argument("--submission-path", type=Path, default=OUT_DIR / "submission_tft.csv")
    args = ap.parse_args()

    seed_everything(args.seed, workers=True)

    if not (TRAIN_CSV.exists() and TEST_CSV.exists() and RAW_TEST.exists()):
        raise FileNotFoundError("missing features_train.csv / features_test.csv / test.csv")

    raw_train = pd.read_csv(TRAIN_CSV)
    te_raw = pd.read_csv(TEST_CSV)
    feat_cols = json.loads(args.feat_cols_path.read_text(encoding="utf-8")) if args.feat_cols_path.exists() else None
    # tolerate meta files that store {"feat_cols": [...], ...}
    if isinstance(feat_cols, dict):
        feat_cols = feat_cols.get("feat_cols", list(feat_cols.values())[0] if feat_cols else None)
    train_df, feat_cols_now = clean_train_df(raw_train)
    feat_cols = feat_cols or feat_cols_now

    month2idx = build_time_idx_map(train_df["month"], te_raw["month"])
    train_df = train_df.copy()
    train_df["time_idx"] = pd.to_datetime(train_df["month"]).map(month2idx).astype("int32")
    train_for_fill = train_df[["sector","month","time_idx","month_num","quarter"] + feat_cols].copy()
    test_df = clean_test_df(te_raw, feat_cols, month2idx, train_df_for_fill=train_for_fill)

    train_ds = TimeSeriesDataSet.load(args.train_ds_path)
    model = TemporalFusionTransformer.load_from_checkpoint(args.ckpt_path)

    pred_df = predict_on_test_safe(model, train_ds, train_df, test_df, batch_size=args.batch_size)

    sub_df = make_submission(pred_df, RAW_TEST, args.submission_path)
    if SOL_CSV.exists():
        eval_with_solution(sub_df, SOL_CSV)
    else:
        print(f"[done] submission saved to {args.submission_path}")


if __name__ == "__main__":
    main()
