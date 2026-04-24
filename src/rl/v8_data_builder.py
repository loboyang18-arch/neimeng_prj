"""
V8 格式数据构建器
==================
复用 V8 模型的 _build_daily_arrays / _get_hour_slots 等函数，
为 RL 环境构建 Conv2D 所需的 (C, H_SLOTS, 7) 输入张量。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

# V8 需要的环境变量（在 import 前设置）
os.environ.setdefault("NM_V8_TARGET", "price_sudun_500kv1m_nodal")
os.environ.setdefault("NM_V8_EXTRA_LAG1",
                       "price_sudun_500kv1m_energy,price_sudun_500kv1m_cong")
os.environ.setdefault("NM_V8_HOURLY_AGG", "mean4")
os.environ.setdefault("NM_MIN_FEATURE_DATE", "2023-06-01")

from src.model_v8_multitask import (
    C_TOTAL,
    H_SLOTS,
    LOOKBACK_DAYS,
    _build_daily_arrays,
    _compute_norm,
    _get_hour_slots,
    _load_dws,
)

V8_PRETRAIN_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "output" / "experiments" / "v8.0-rl-pretrain"
)


def load_v8_daily_arrays() -> Tuple[dict, dict, dict, dict, list]:
    """加载 DWS 并构建 V8 格式日数组。"""
    dws = _load_dws()
    valid_dates, day_lag0, day_lag1, day_lag2, day_targets = _build_daily_arrays(dws)
    return day_lag0, day_lag1, day_lag2, day_targets, valid_dates


def load_v8_norm(
    pretrain_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """加载 V8 预训练的归一化参数。"""
    d = pretrain_dir or V8_PRETRAIN_DIR
    mean = np.load(d / "norm_mean.npy")
    std = np.load(d / "norm_std.npy")
    return mean, std


def compute_v8_norm_from_episodes(
    day_lag0: dict, day_lag1: dict, day_lag2: dict, train_dates: list,
) -> Tuple[np.ndarray, np.ndarray]:
    """从训练日计算 V8 归一化参数（不使用预训练文件时的备选）。"""
    return _compute_norm(day_lag0, day_lag1, day_lag2, train_dates)


def build_v8_grid_for_slot(
    day_lag0: dict, day_lag1: dict, day_lag2: dict,
    date, slot: int,
    norm_mean: np.ndarray, norm_std: np.ndarray,
) -> np.ndarray:
    """为指定日期和时段构建归一化 V8 grid 张量。

    Args:
        date: 当日日期 (datetime.date)
        slot: 15min 时段索引 [0, 95]

    Returns:
        (C_TOTAL, H_SLOTS, LOOKBACK_DAYS) float32 张量
    """
    import pandas as pd

    h = slot // 4
    dates0 = [(pd.Timestamp(date) - pd.Timedelta(days=off)).date()
              for off in range(LOOKBACK_DAYS - 1, -1, -1)]
    dates1 = [(pd.Timestamp(date) - pd.Timedelta(days=off)).date()
              for off in range(LOOKBACK_DAYS, 0, -1)]
    dates2 = [(pd.Timestamp(date) - pd.Timedelta(days=off)).date()
              for off in range(LOOKBACK_DAYS + 1, 1, -1)]

    layers = []
    for k in range(LOOKBACK_DAYS):
        d0 = dates0[k]
        d1 = dates1[k]
        d2 = dates2[k]
        s0 = _get_hour_slots(day_lag0, d0, h) if d0 in day_lag0 else np.zeros((H_SLOTS, day_lag0[date].shape[1]), dtype=np.float32)
        s1 = _get_hour_slots(day_lag1, d1, h) if d1 in day_lag1 else np.zeros((H_SLOTS, day_lag1[date].shape[1]), dtype=np.float32)
        s2 = _get_hour_slots(day_lag2, d2, h) if d2 in day_lag2 else np.zeros((H_SLOTS, day_lag2[date].shape[1]), dtype=np.float32)
        layers.append(np.concatenate([s0, s1, s2], axis=1))

    grid = np.stack(layers, axis=-1)  # (H_SLOTS, C_TOTAL, 7)
    grid = grid.transpose(1, 0, 2)    # (C_TOTAL, H_SLOTS, 7)
    grid = np.nan_to_num(grid, nan=0.0)
    grid = ((grid - norm_mean.reshape(C_TOTAL, 1, 1))
            / norm_std.reshape(C_TOTAL, 1, 1)).astype(np.float32)
    return grid


class V8GridCache:
    """为一天预计算所有 96 个时段的 V8 grid，避免重复构建。

    每天有 96 个 15min 时段，但由于 V8 用小时级上下文窗口（H_SLOTS = 12 = 3h × 4），
    同一小时内的 4 个时段共享同一个 grid。因此只需构建 24 个 grid。
    """

    def __init__(
        self,
        day_lag0: dict, day_lag1: dict, day_lag2: dict,
        norm_mean: np.ndarray, norm_std: np.ndarray,
    ):
        self._lag0 = day_lag0
        self._lag1 = day_lag1
        self._lag2 = day_lag2
        self._mean = norm_mean
        self._std = norm_std
        self._cache: Dict = {}

    def get_grid(self, date, slot: int) -> np.ndarray:
        """获取 (C_TOTAL, H_SLOTS, 7) grid，带日+小时级缓存。"""
        h = slot // 4
        key = (date, h)
        if key not in self._cache:
            self._cache[key] = build_v8_grid_for_slot(
                self._lag0, self._lag1, self._lag2,
                date, slot, self._mean, self._std,
            )
        return self._cache[key]

    def clear(self):
        self._cache.clear()

    def precompute_day(self, date):
        """预计算一天所有 24 个小时的 grid。"""
        for h in range(24):
            self.get_grid(date, h * 4)
