"""
反向日标签构建工具
==================
对每一天，计算其 24 小时实际节点电价的 z-shape 与「训练集 z-shape 均值」的相关系数：
  - corr ≥ 0.7  → 典型日
  - 0.3 ≤ corr < 0.7 → 一般日
  - 0 ≤ corr < 0.3   → 非典型日
  - corr < 0          → 反向日（is_reverse=1）

参考形状仅由训练日（< test_start）计算，避免测试集泄漏。
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("NM_V8_TARGET", "price_sudun_500kv1m_nodal")
os.environ.setdefault("NM_MIN_FEATURE_DATE", "2024-12-14")

from src.config import OUTPUT_DIR
from src.fill_sudun_dws_gaps import fill_sudun_price_columns

logger = logging.getLogger(__name__)

NODAL_COL = os.environ.get("NM_V8_TARGET", "price_sudun_500kv1m_nodal")
ANOMALY_DIR = OUTPUT_DIR / "experiments" / "anomaly-detector"


def load_dws() -> pd.DataFrame:
    path = OUTPUT_DIR / "dws_15min_features.csv"
    df = pd.read_csv(path, parse_dates=["ts"], index_col="ts").sort_index()
    df = fill_sudun_price_columns(df)
    return df


def build_daily_actual_24(dws: pd.DataFrame) -> Dict:
    """从 DWS 提取每天 24 小时实际节点电价（按 4 个 15min 槽求均值）。"""
    start_date = dws.index.min().normalize().date()
    end_date = dws.index.max().date()
    date_range = pd.date_range(start_date, end_date, freq="D")
    out = {}
    for d_ts in date_range:
        d = d_ts.date()
        grid = pd.date_range(pd.Timestamp(d), periods=96, freq="15min")
        v96 = dws.reindex(grid)[NODAL_COL].values.astype(np.float32)
        if len(v96) == 96 and np.isfinite(v96).all():
            out[d] = v96.reshape(24, 4).mean(axis=1)
    return out


def compute_mean_shape(daily_24: Dict, dates: List) -> np.ndarray:
    """从指定日期集合计算 z-shape 均值。"""
    zs = []
    for d in dates:
        if d not in daily_24:
            continue
        a = daily_24[d]
        if a.std() > 1e-6:
            zs.append((a - a.mean()) / a.std())
    if not zs:
        raise RuntimeError("没有可用的日期来计算 mean_shape")
    return np.mean(zs, axis=0).astype(np.float32)


def label_days(daily_24: Dict, mean_shape: np.ndarray) -> pd.DataFrame:
    """对每一天生成 corr / category / is_reverse 标签。"""
    rows = []
    for d, a in daily_24.items():
        if a.std() < 1e-6:
            corr = 0.0
        else:
            zs = (a - a.mean()) / a.std()
            corr = float(np.corrcoef(zs, mean_shape)[0, 1])
        if corr >= 0.7:
            cat = "typical"
        elif corr >= 0.3:
            cat = "normal"
        elif corr >= 0:
            cat = "atypical"
        else:
            cat = "reverse"
        rows.append({
            "date": d,
            "shape_corr": corr,
            "category": cat,
            "is_reverse": int(corr < 0),
        })
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def main(test_start: str = "2026-01-27"):
    ANOMALY_DIR.mkdir(parents=True, exist_ok=True)

    dws = load_dws()
    logger.info("DWS: %d rows (%s ~ %s)", len(dws), dws.index.min(), dws.index.max())

    daily_24 = build_daily_actual_24(dws)
    logger.info("有效日数: %d", len(daily_24))

    test_dt = pd.Timestamp(test_start).date()
    train_dates = [d for d in daily_24 if d < test_dt]
    logger.info("训练日 (<%s): %d 天", test_start, len(train_dates))

    mean_shape = compute_mean_shape(daily_24, train_dates)
    logger.info("Mean shape (训练集): min=%.2f, max=%.2f", mean_shape.min(), mean_shape.max())

    np.save(ANOMALY_DIR / "mean_shape_train.npy", mean_shape)

    labels = label_days(daily_24, mean_shape)
    labels.to_csv(ANOMALY_DIR / "shape_labels.csv", index=False)

    train_subset = labels[labels["date"] < test_dt]
    test_subset = labels[labels["date"] >= test_dt]
    logger.info("训练集类别分布:")
    logger.info("  %s", train_subset["category"].value_counts().to_dict())
    logger.info("  反向日比例: %.1f%%",
                train_subset["is_reverse"].mean() * 100)
    logger.info("测试集类别分布:")
    logger.info("  %s", test_subset["category"].value_counts().to_dict())

    logger.info("已保存: %s", ANOMALY_DIR / "shape_labels.csv")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
