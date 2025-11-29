#!/usr/bin/env python3
"""
Grid search for the original tft_train / tft_predict pipeline.
Uses the same data cleaning and holdout evaluation logic from tft.py (no changes to existing code).
Results:
  - per-run artifacts: out/tft_grid_XX_best.ckpt, _train_ds, _meta.json
  - best overall copies: out/tft_grid_best.ckpt, _train_ds, _meta.json (for tft_predict.py)
"""
from __future__ import annotations

import itertools
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import pandas as pd
from lightning.pytorch import seed_everything
from pytorch_forecasting import TimeSeriesDataSet

from tft import OUT_DIR, TRAIN_CSV, clean_train_df, holdout_eval


def train_one_cfg(run_id: str, cfg: Dict):
    seed_everything(cfg.get("seed", 42), workers=True)
    if not Path(TRAIN_CSV).exists():
        raise FileNotFoundError(f"missing {TRAIN_CSV}")
    raw_train = pd.read_csv(TRAIN_CSV)
    train_df, feat_cols = clean_train_df(raw_train)

    args = SimpleNamespace(
        seed=cfg["seed"],
        encoder_len=cfg["encoder_len"],
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        hidden=cfg["hidden"],
        attn_head=cfg["attn_head"],
        dropout=cfg["dropout"],
        lr=cfg["lr"],
        patience=cfg["patience"],
    )
    model, train_ds, metrics, best_ckpt = holdout_eval(train_df, feat_cols, args)

    run_ckpt = OUT_DIR / f"tft_grid_{run_id}_best.ckpt"
    if best_ckpt and Path(best_ckpt).exists():
        shutil.copy(best_ckpt, run_ckpt)
    train_ds_path = OUT_DIR / f"tft_grid_{run_id}_train_ds"
    if train_ds_path.exists():
        if train_ds_path.is_dir():
            shutil.rmtree(train_ds_path)
        else:
            train_ds_path.unlink()
    TimeSeriesDataSet.save(train_ds, train_ds_path)
    meta = {"feat_cols": feat_cols, "encoder_len": cfg["encoder_len"]}
    meta_path = OUT_DIR / f"tft_grid_{run_id}_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # choose MAE as selection metric
    score = float(metrics.get("MAE", float("inf")))
    return {"score": score, "run_id": run_id, "ckpt": run_ckpt, "train_ds": train_ds_path, "meta": meta_path, "cfg": cfg}


def main():
    seed_everything(42, workers=True)

    grid = {
        "encoder_len": [18, 24],
        "hidden": [64, 96],
        "dropout": [0.10, 0.15],
        "lr": [1e-3],
        "attn_head": [4],
        "batch_size": [128],
        "epochs": [25],
        "patience": [5],
    }
    keys = list(grid.keys())
    combos = list(itertools.product(*(grid[k] for k in keys)))

    results: List[Dict] = []
    for idx, values in enumerate(combos):
        cfg = dict(zip(keys, values))
        cfg["seed"] = 42 + idx
        run_id = f"{idx+1:02d}"
        print(f"\n[grid] run {run_id}/{len(combos)} cfg={cfg}")
        res = train_one_cfg(run_id, cfg)
        print(f"[grid] run {run_id} MAE={res['score']:.4f} -> {res['ckpt']}")
        results.append(res)

    results.sort(key=lambda r: r["score"])
    best = results[0]
    final_ckpt = OUT_DIR / "tft_grid_best.ckpt"
    final_train_ds = OUT_DIR / "tft_grid_best_train_ds"
    final_meta = OUT_DIR / "tft_grid_best_meta.json"
    shutil.copy(best["ckpt"], final_ckpt)
    if final_train_ds.exists():
        if final_train_ds.is_dir():
            shutil.rmtree(final_train_ds)
        else:
            final_train_ds.unlink()
    src_ds = best["train_ds"]
    if src_ds.is_dir():
        shutil.copytree(src_ds, final_train_ds, dirs_exist_ok=True)
    elif src_ds.is_file():
        shutil.copy(src_ds, final_train_ds)
    else:
        raise FileNotFoundError(f"train_ds not found: {src_ds}")
    shutil.copy(best["meta"], final_meta)
    print(f"\n[best] MAE={best['score']:.4f} cfg={best['cfg']}")
    print(f"[best] ckpt -> {final_ckpt}")
    print(f"[best] train_ds -> {final_train_ds}")
    print(f"[best] meta -> {final_meta}")


if __name__ == "__main__":
    main()
