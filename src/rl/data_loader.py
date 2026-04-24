"""
从 dws_15min_features.csv 构建 RL episode 数据
================================================
每个 episode = 一天 96 个 15min 时段的完整市场特征 + 结算电价 + 滞后日均特征。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .battery_cfg import (
    ALL_DWS_COLS,
    DATA_END,
    LAG_ACTUAL_COLS,
    LAG_PRICE_COLS,
    MARKET_SNAPSHOT_COLS,
    NODAL_PRICE_COL,
    TEST_START,
    TRAIN_START,
    VAL_START,
)

DWS_PATH = Path(__file__).resolve().parent.parent.parent / "output" / "dws_15min_features.csv"


def load_dws(path: Optional[Path] = None,
             start: str = TRAIN_START,
             end: str = DATA_END) -> pd.DataFrame:
    path = path or DWS_PATH
    df = pd.read_csv(path, parse_dates=["ts"])
    df = df[(df["ts"] >= start) & (df["ts"] <= end + " 23:59:59")]
    for col in ALL_DWS_COLS:
        if col not in df.columns:
            df[col] = np.nan
    df["date"] = df["ts"].dt.date.astype(str)
    return df


def _daily_mean(day_df: pd.DataFrame, cols: List[str]) -> np.ndarray:
    """返回一天中各列的均值（NaN 视为 0）。"""
    return np.array([day_df[c].mean() for c in cols], dtype=np.float32)


def build_episodes(df: pd.DataFrame,
                   min_valid: int = 91) -> Tuple[List[dict], Dict[str, np.ndarray]]:
    """
    构建 episode 列表。

    Args:
        df: 已筛选时间范围的 DWS DataFrame
        min_valid: 单日 NODAL_PRICE_COL 非空最少时段数

    Returns:
        (episodes, daily_cache)
        daily_cache: {date_str: (lag_price_mean, lag_actual_mean)} 用于构建 D-1/D-7
    """
    dates = sorted(df["date"].unique())

    daily_cache: Dict[str, dict] = {}
    for d in dates:
        day = df[df["date"] == d]
        daily_cache[d] = {
            "lag_price": _daily_mean(day, LAG_PRICE_COLS),
            "lag_actual": _daily_mean(day, LAG_ACTUAL_COLS),
        }

    episodes: List[dict] = []
    for d in dates:
        day = df[df["date"] == d].sort_values("ts").reset_index(drop=True)
        if len(day) < 96:
            continue
        day = day.iloc[:96]

        nodal = day[NODAL_PRICE_COL].values.astype(np.float32)
        if np.isnan(nodal).sum() > (96 - min_valid):
            continue
        nodal = np.nan_to_num(nodal, nan=0.0)

        features = day[MARKET_SNAPSHOT_COLS].values.astype(np.float32)
        features = np.nan_to_num(features, nan=0.0)

        # 电价滞后：D-1 / D-7（出清价隔日发布，D-1 可用）
        # 实测滞后：D-2 / D-8（实测值发布更晚，与 V8 LAG2 对齐）
        d_idx = dates.index(d)
        lag1_price = daily_cache[dates[d_idx - 1]]["lag_price"] if d_idx >= 1 else np.zeros(len(LAG_PRICE_COLS), dtype=np.float32)
        lag2_actual = daily_cache[dates[d_idx - 2]]["lag_actual"] if d_idx >= 2 else np.zeros(len(LAG_ACTUAL_COLS), dtype=np.float32)
        lag7_price = daily_cache[dates[d_idx - 7]]["lag_price"] if d_idx >= 7 else np.zeros(len(LAG_PRICE_COLS), dtype=np.float32)
        lag8_actual = daily_cache[dates[d_idx - 8]]["lag_actual"] if d_idx >= 8 else np.zeros(len(LAG_ACTUAL_COLS), dtype=np.float32)

        episodes.append({
            "date": d,
            "features_96": features,       # (96, n_market)
            "nodal_price_96": nodal,        # (96,)
            "lag1_price": lag1_price,        # (n_lag_price,) D-1 电价日均
            "lag2_actual": lag2_actual,      # (n_lag_actual,) D-2 实测日均
            "lag7_price": lag7_price,        # D-7 电价日均
            "lag8_actual": lag8_actual,      # D-8 实测日均
        })

    return episodes, daily_cache


def compute_norm_stats(episodes: List[dict]) -> dict:
    """从训练集 episode 计算 P99 归一化系数。"""
    all_features = np.concatenate([ep["features_96"] for ep in episodes], axis=0)
    all_lag1_p = np.stack([ep["lag1_price"] for ep in episodes])
    all_lag2_a = np.stack([ep["lag2_actual"] for ep in episodes])

    def p99(arr, axis=0):
        v = np.nanpercentile(np.abs(arr), 99, axis=axis)
        return np.where(v < 1e-6, 1.0, v)

    return {
        "features_p99": p99(all_features).tolist(),
        "lag_price_p99": p99(all_lag1_p).tolist(),
        "lag_actual_p99": p99(all_lag2_a).tolist(),
    }


def split_episodes(episodes: List[dict]) -> Tuple[List[dict], List[dict], List[dict]]:
    """按日期划分 train / val / test。"""
    train, val, test = [], [], []
    for ep in episodes:
        d = ep["date"]
        if d < VAL_START:
            train.append(ep)
        elif d < TEST_START:
            val.append(ep)
        else:
            test.append(ep)
    return train, val, test


def load_and_split(dws_path: Optional[Path] = None, with_v8: bool = False):
    """一站式：加载 -> 构建 episode -> 划分 -> 计算归一化。

    Args:
        with_v8: 若为 True，额外加载 V8 daily_arrays 和归一化参数，返回 v8_ctx dict。
    """
    df = load_dws(dws_path)
    episodes, _ = build_episodes(df)
    train, val, test = split_episodes(episodes)
    norm = compute_norm_stats(train)

    if not with_v8:
        return train, val, test, norm

    from .v8_data_builder import load_v8_daily_arrays, load_v8_norm
    day_lag0, day_lag1, day_lag2, day_targets, _ = load_v8_daily_arrays()
    v8_norm_mean, v8_norm_std = load_v8_norm()
    v8_ctx = {
        "day_lag0": day_lag0,
        "day_lag1": day_lag1,
        "day_lag2": day_lag2,
        "day_targets": day_targets,
        "norm_mean": v8_norm_mean,
        "norm_std": v8_norm_std,
    }
    return train, val, test, norm, v8_ctx
