"""
日级特征构建
============
聚合 DWS 15 分钟特征到日级别，用于反向日检测。

特征类别：
  - 当日预测：load / renewable / wind / solar / east_send / reserve / da_pre
  - 前一日实际：D-1 实际节点电价/统一电价/负荷的统计
  - 前一周实际：D-7 实际节点电价的均值/形状
  - 日历：dayofweek、month、is_weekend
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("NM_V8_TARGET", "price_sudun_500kv1m_nodal")

from src.config import OUTPUT_DIR
from src.anomaly.labels import load_dws

logger = logging.getLogger(__name__)

NODAL_COL = "price_sudun_500kv1m_nodal"
ANOMALY_DIR = OUTPUT_DIR / "experiments" / "anomaly-detector"


def _stat(arr, prefix):
    """对 96 个 15min 值返回 mean/std/min/max/q25/q75/range/早晚段差。"""
    if not np.isfinite(arr).all():
        arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {f"{prefix}_{k}": np.nan for k in
                ("mean", "std", "min", "max", "q25", "q75",
                 "range", "morning_evening_diff", "peak_hour", "trough_hour")}
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_std":  float(np.std(arr)),
        f"{prefix}_min":  float(np.min(arr)),
        f"{prefix}_max":  float(np.max(arr)),
        f"{prefix}_q25":  float(np.percentile(arr, 25)),
        f"{prefix}_q75":  float(np.percentile(arr, 75)),
        f"{prefix}_range": float(np.max(arr) - np.min(arr)),
        f"{prefix}_morning_evening_diff":
            float(np.mean(arr[:48]) - np.mean(arr[48:])) if len(arr) >= 96 else np.nan,
        f"{prefix}_peak_hour":
            int(np.argmax(arr) // 4) if len(arr) >= 96 else -1,
        f"{prefix}_trough_hour":
            int(np.argmin(arr) // 4) if len(arr) >= 96 else -1,
    }


def build_day_features(dws: pd.DataFrame, target_dates):
    """对每一天生成日级特征行。"""
    rows = []
    forecast_cols = [
        "load_forecast", "renewable_forecast", "wind_forecast",
        "solar_forecast", "east_send_forecast",
        "reserve_pos_capacity", "reserve_neg_capacity",
        "price_dayahead_preclear_energy",
    ]

    for d in target_dates:
        d_ts = pd.Timestamp(d)
        grid = pd.date_range(d_ts, periods=96, freq="15min")
        raw = dws.reindex(grid)

        row = {"date": d}

        for col in forecast_cols:
            if col not in raw:
                continue
            row.update(_stat(raw[col].values.astype(float), col))

        net_load = (raw["load_forecast"] - raw["renewable_forecast"]).values
        row.update(_stat(net_load.astype(float), "net_load"))

        for off in (1, 2, 7):
            d_prev = (d_ts - pd.Timedelta(days=off)).floor("D")
            grid_p = pd.date_range(d_prev, periods=96, freq="15min")
            raw_p = dws.reindex(grid_p)
            for src_col, prefix in (
                (NODAL_COL, f"d_minus{off}_nodal"),
                ("price_unified", f"d_minus{off}_unified"),
                ("load_actual", f"d_minus{off}_load"),
                ("renewable_actual", f"d_minus{off}_renew"),
            ):
                if src_col not in raw_p:
                    continue
                row.update(_stat(raw_p[src_col].values.astype(float), prefix))

        row["dayofweek"] = d_ts.dayofweek
        row["month"] = d_ts.month
        row["is_weekend"] = int(d_ts.dayofweek >= 5)

        rows.append(row)

    return pd.DataFrame(rows)


def main(test_start: str = "2026-01-27"):
    ANOMALY_DIR.mkdir(parents=True, exist_ok=True)

    dws = load_dws()

    labels = pd.read_csv(ANOMALY_DIR / "shape_labels.csv", parse_dates=["date"])
    labels["date"] = labels["date"].dt.date

    feats = build_day_features(dws, labels["date"].tolist())
    logger.info("特征行数: %d", len(feats))
    logger.info("特征列数: %d (含 date)", feats.shape[1])

    merged = feats.merge(labels, on="date", how="inner")
    merged.to_csv(ANOMALY_DIR / "day_features.csv", index=False)
    logger.info("已保存: %s", ANOMALY_DIR / "day_features.csv")

    feature_cols = [c for c in feats.columns if c != "date"]
    nan_rate = merged[feature_cols].isna().mean()
    high_nan = nan_rate[nan_rate > 0.1]
    if len(high_nan) > 0:
        logger.warning("高 NaN 比例特征 (>10%%):")
        for c, v in high_nan.items():
            logger.warning("  %s: %.1f%%", c, v * 100)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
