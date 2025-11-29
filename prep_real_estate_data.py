#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# === 路径配置（Windows 桌面）===
from pathlib import Path
BASE = Path(__file__).resolve().parent
TRAIN_DIR = BASE / "train"
TEST_CSV  = BASE / "test.csv"
OUT_DIR   = BASE / "out"; OUT_DIR.mkdir(parents=True, exist_ok=True)

# === 依赖 ===
import pandas as pd
import numpy as np
import re
import warnings
pd.set_option("future.no_silent_downcasting", True)

# —— 定向警告处理（仅静音噪音，不影响结果）——
warnings.filterwarnings(
    "ignore",
    message="Mean of empty slice",
    category=RuntimeWarning
)
warnings.filterwarnings(  # 静音“未指定日期格式”的提示
    "ignore",
    message=r"Could not infer format, so each element will be parsed individually, falling back to `dateutil`",
    category=UserWarning
)

# === 常量 ===
TARGET_COL = "amount_new_house_transactions"  # 预测目标（万元）
TIME_TABLES = [
    ("new_house_transactions.csv",                         "new"),
    ("new_house_transactions_nearby_sectors.csv",          "new_nb"),
    ("pre_owned_house_transactions.csv",                   "old"),
    ("pre_owned_house_transactions_nearby_sectors.csv",    "old_nb"),
    ("land_transactions.csv",                              "land"),
    ("land_transactions_nearby_sectors.csv",               "land_nb"),
]
STATIC_TABLE = "sector_POI.csv"             # 静态画像
CITY_MONTH_TABLE = "city_search_index.csv"  # 城市月度搜索（全市单城）
CITY_YEAR_TABLE  = "city_indexes.csv"       # 城市年度指标（全市单城）

LAGS  = [1, 2, 3, 6, 12]
ROLLS = [3, 6, 12]

# === 工具函数 ===
def _to_month(x, fmt: str | None = None):
    """
    把任意日期列归到每月月初。
    如果你知道格式（如 "%Y-%m-%d" / "%Y-%m" / "%Y %b"），传 fmt 可更快更稳。
    """
    if fmt:
        s = pd.to_datetime(x, format=fmt, errors='coerce')
    else:
        s = pd.to_datetime(x, errors='coerce')
    return s.dt.to_period('M').dt.to_timestamp()

def _to_sector_int(s):
    """把 sector 列稳健转为整数（容忍 'sector 1' / 'Sector1' / ' 2 ' 等），返回 pandas 可空整型 Int64。"""
    return (
        s.astype(str)
         .str.lower().str.strip()
         .str.extract(r"(\d+)")[0]
         .pipe(pd.to_numeric, errors="coerce")
         .astype("Int64")
    )

def parse_test_id(id_str):
    # 'YYYY Mon_sector n' -> (Timestamp, sector:int)
    parts = str(id_str).split("_sector")
    ym_str = parts[0].strip()         # '2024 Aug'
    sec    = int(parts[1].strip())
    ym = pd.to_datetime(ym_str, format="%Y %b")
    return pd.to_datetime(ym.strftime("%Y-%m-01")), sec

def read_csv_safe(path: Path):
    return pd.read_csv(path)

def add_time_index_cols(df):
    df = df.copy()
    df["month_num"] = df["month"].dt.month
    df["quarter"]   = df["month"].dt.quarter
    uniq = np.sort(df["month"].unique())
    mapper = {m: i for i, m in enumerate(uniq)}
    df["month_idx_global"] = df["month"].map(mapper).astype(int)
    return df

def add_strict_lag_roll(base: pd.DataFrame, src: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    向量化生成严格滞后与 rolling 特征并合并至 base。
    不回填当月原值；mom/yoy 分母使用 shift() 的历史。
    """
    # 先把 (sector, month) 框架对齐到 base
    tmp = (
        base[["sector","month"]]
        .drop_duplicates()
        .merge(src, on=["sector","month"], how="left")
        .sort_values(["sector","month"])
        .reset_index(drop=True)
    )

    value_cols = [c for c in tmp.columns if c not in ["sector","month"] and pd.api.types.is_numeric_dtype(tmp[c])]
    if not value_cols:
        return base

    # 针对每个数值列，使用 groupby('sector') 的 transform/shift/rolling 完成计算（无 apply，无警告，且列不丢）
    g = tmp.groupby("sector", group_keys=False)
    for col in value_cols:
        s = g[col]  # SeriesGroupBy
        # lag
        for L in LAGS:
            tmp[f"{prefix}__{col}_lag{L}"] = s.shift(L)
        # rolling on shifted series
        s1 = s.shift(1)
        for W in ROLLS:
            tmp[f"{prefix}__{col}_roll{W}_mean"] = s1.rolling(W, min_periods=1).mean()
        # mom / yoy
        prev1  = s.shift(1)
        prev12 = s.shift(12)
        tmp[f"{prefix}__{col}_mom"] = np.where(prev1.notna() & (prev1.abs() > 1e-9), tmp[col]/prev1 - 1, np.nan)
        tmp[f"{prefix}__{col}_yoy"] = np.where(prev12.notna() & (prev12.abs() > 1e-9), tmp[col]/prev12 - 1, np.nan)

    keep_cols = ["sector","month"] + [c for c in tmp.columns if c.startswith(f"{prefix}__")]
    return base.merge(tmp[keep_cols], on=["sector","month"], how="left")

# === 主流程：构建 base（train 时间轴 + 严格滞后特征）===
def build_base_table():
    # 1) 主表（含 y）
    new_main = read_csv_safe(TRAIN_DIR / "new_house_transactions.csv")
    new_main["month"]  = _to_month(new_main["month"])  # 若你知道格式，可传 fmt 参数
    new_main["sector_raw"] = new_main["sector"]        # 便于排查
    new_main["sector"] = _to_sector_int(new_main["sector"])

    # 去重聚合（防御）
    agg_map = {}
    for c in new_main.columns:
        if c in ["month","sector"]:
            continue
        agg_map[c] = "sum" if pd.api.types.is_numeric_dtype(new_main[c]) else "first"
    new_main = new_main.groupby(["month","sector"], as_index=False).agg(agg_map)

    assert TARGET_COL in new_main.columns, f"目标列 {TARGET_COL} 不在 new_house_transactions.csv 中"
    new_main["y"] = new_main[TARGET_COL].clip(lower=0)

    # 2) 以主表的 (sector, month) 为时间轴骨架
    base = new_main[["sector","month","y"]].sort_values(["sector","month"]).reset_index(drop=True)
    base = add_time_index_cols(base)

    # 3) 对“6 张随时间变化的表”逐一构造严格滞后特征
    # 3.1 new_house 自身
    new_df = read_csv_safe(TRAIN_DIR / "new_house_transactions.csv")
    new_df["month"]  = _to_month(new_df["month"])
    new_df["sector"] = _to_sector_int(new_df["sector"])
    base = add_strict_lag_roll(base, new_df, prefix="new")

    # 3.2 其余 5 张表
    for fname, prefix in TIME_TABLES:
        if fname == "new_house_transactions.csv":
            continue
        df = read_csv_safe(TRAIN_DIR / fname)
        df["month"]  = _to_month(df["month"])
        df["sector"] = _to_sector_int(df["sector"])
        base = add_strict_lag_roll(base, df, prefix=prefix)

    # 4) 静态画像并入（不涉时间，不泄漏）
    try:
        poi = read_csv_safe(TRAIN_DIR / STATIC_TABLE)
        poi["sector"] = _to_sector_int(poi["sector"])
        poi = poi.drop_duplicates("sector")
        base = base.merge(poi, on="sector", how="left")
    except FileNotFoundError:
        pass

    # 5) 城市“月度搜索”表（单城）
    try:
        csi = read_csv_safe(TRAIN_DIR / CITY_MONTH_TABLE)
        csi["month"] = _to_month(csi["month"])
        csi_m = (
            csi.groupby("month", as_index=False)["search_volume"]
               .sum()
               .rename(columns={"search_volume": "city_search_volume"})
        )
        tmp = (
            base[["sector","month"]]
            .drop_duplicates()
            .merge(csi_m, on="month", how="left")
            .sort_values(["sector","month"])
            .reset_index(drop=True)
        )
        g = tmp.groupby("sector")["city_search_volume"]
        for L in LAGS:
            tmp[f"city__search_volume_lag{L}"] = g.shift(L)
        s1 = g.shift(1)
        for W in ROLLS:
            tmp[f"city__search_volume_roll{W}_mean"] = s1.rolling(W, min_periods=1).mean()
        keep = ["sector","month"] + [c for c in tmp.columns if c.startswith("city__")]
        base = base.merge(tmp[keep], on=["sector","month"], how="left")
    except FileNotFoundError:
        pass

    # 6) 城市“年度指标”表（单城）
    try:
        ci = read_csv_safe(TRAIN_DIR / CITY_YEAR_TABLE)
        assert "city_indicator_data_year" in ci.columns, "city_indexes.csv 缺少 city_indicator_data_year"
        ci["year"] = ci["city_indicator_data_year"].astype(int)
        numeric_cols = [c for c in ci.columns if c not in ["city_indicator_data_year","year"] and pd.api.types.is_numeric_dtype(ci[c])]
        keep = ["year"] + numeric_cols
        ci = ci[keep].drop_duplicates("year")

        base["year"] = base["month"].dt.year
        tmp2 = (
            base[["sector","month","year"]]
            .merge(ci, on="year", how="left")
            .sort_values(["sector","month"])
            .reset_index(drop=True)
        )
        for col in numeric_cols:
            tmp2[f"city__{col}_lag1y"] = tmp2.groupby("sector")[col].shift(12)
        keep2 = ["sector","month"] + [f"city__{c}_lag1y" for c in numeric_cols]
        base = base.merge(tmp2[keep2], on=["sector","month"], how="left").drop(columns=["year"])
    except FileNotFoundError:
        pass

    # 7) 缺失填充（数值：分 sector 中位→全局中位；类别：'__MISSING__'）
    if "sector" not in base.columns or "month" not in base.columns:
        base = base.reset_index()

    num_cols = [c for c in base.columns if pd.api.types.is_numeric_dtype(base[c])]
    cat_cols = [c for c in base.columns if not pd.api.types.is_numeric_dtype(base[c]) and c not in ["month"]]

    def _fill_group(g):
        g = g.copy()
        for c in num_cols:
            if c == "sector":
                continue
            med = g[c].median(skipna=True)
            g[c] = g[c].fillna(med)
        return g

    # 用 transform 方式分组填充中位数，避免 apply 的未来行为差异
    med_by_sector = base.groupby("sector")[num_cols].transform(lambda x: x.median(skipna=True))
    for c in num_cols:
        if c == "sector":
            continue
        base[c] = base[c].fillna(med_by_sector[c])
        base[c] = base[c].fillna(base[c].median())
    for c in cat_cols:
        base[c] = base[c].fillna("__MISSING__")

    return base

# === 导出 Train/Test 特征（保持 test 行序；不泄漏）===
def main():
    print(f"[info] BASE = {BASE}")
    base = build_base_table()
    print(f"[info] base shape: {base.shape}")

    # 读取 test 并解析 id
    te = pd.read_csv(TEST_CSV)
    te["month"], te["sector"] = zip(*te["id"].map(parse_test_id))
    test_keys = te[["id","sector","month"]].copy()

    # 训练集：来自 base 中有 y 的行
    train_df = base[base["y"].notna()].copy()

    # 测试集：特征对齐（确保键列在）
    feature_cols = ["sector", "month"] + [c for c in base.columns if c not in ["y","sector","month"]]
    right = base[feature_cols].copy()
    if "sector" not in right.columns or "month" not in right.columns:
        right = right.reset_index()
    test_df = test_keys.merge(right, on=["sector","month"], how="left")
    test_df = test_df.set_index("id").loc[te["id"]].reset_index()

    train_df.to_csv(OUT_DIR / "features_train.csv", index=False)
    test_df.to_csv(OUT_DIR / "features_test.csv", index=False)
    print("[warn] 没有 pyarrow/fastparquet，已自动保存为 CSV。")
    print(f"[save] out/features_train.csv -> {train_df.shape}")
    print(f"[save] out/features_test.csv  -> {test_df.shape}")

    print("[done] 数据处理完成（严格防泄漏）。")

if __name__ == "__main__":
    main()
