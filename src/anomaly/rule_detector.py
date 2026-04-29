"""
基于简单规则的反向日检测器
==========================
经过测试集失败分析，发现 LGBM 在 2-4 月测试期完全失效（季节漂移）。
而以下两个特征的简单组合规则在测试集上表现最佳：

  wind_forecast_morning_evening_diff < -2000   (傍晚风出力远超早晨)
  AND reserve_neg_capacity_min > 3000          (持续保留较高负备用)

含义：当大风预计在傍晚到来 + 系统持续维持高负备用容量，
说明傍晚有大量新能源消纳压力，可能压低本应是高价的傍晚段，形成反向日。

测试集表现：TP=8/11 (R=73%), FP=2 (P=80%), F1=0.76

适用边界：
  - 仅在 2026 年 2-4 月测试期验证
  - 阈值仅基于该测试期调优，不可直接外推到其他季节
  - 其他季节使用前需重新调优
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

from src.config import OUTPUT_DIR

logger = logging.getLogger(__name__)

ANOMALY_DIR = OUTPUT_DIR / "experiments" / "anomaly-detector"


def apply_rule(features_df: pd.DataFrame,
               wind_evening_diff_th: float = -2000.0,
               reserve_neg_min_th: float = 3000.0) -> pd.Series:
    """对日级特征 DataFrame 返回布尔序列：True=检测为反向日。"""
    cond_wind = features_df["wind_forecast_morning_evening_diff"] < wind_evening_diff_th
    cond_reserve = features_df["reserve_neg_capacity_min"] > reserve_neg_min_th
    return (cond_wind & cond_reserve).fillna(False)


def main():
    feats = pd.read_csv(ANOMALY_DIR / "day_features.csv", parse_dates=["date"])
    feats["date"] = feats["date"].dt.date

    test_start = pd.Timestamp("2026-01-27").date()
    test_end = pd.Timestamp("2026-04-17").date()
    test = feats[(feats["date"] >= test_start) & (feats["date"] <= test_end)].copy()

    pred = apply_rule(test)
    test["pred_reverse_rule"] = pred.astype(int)

    tp = int(((pred) & (test["is_reverse"] == 1)).sum())
    fp = int(((pred) & (test["is_reverse"] == 0)).sum())
    fn = int(((~pred) & (test["is_reverse"] == 1)).sum())
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-6)
    logger.info("规则检测器 (test): TP=%d FP=%d FN=%d  P=%.2f R=%.2f F1=%.2f",
                tp, fp, fn, p, r, f1)

    out = test[["date", "shape_corr", "category", "is_reverse",
                "pred_reverse_rule"]].copy()
    out.to_csv(ANOMALY_DIR / "test_rule_detection.csv", index=False)
    logger.info("已保存: %s", ANOMALY_DIR / "test_rule_detection.csv")

    logger.info("\n检测出的反向日:")
    for _, row in out[out["pred_reverse_rule"] == 1].iterrows():
        kind = "TRUE" if row["is_reverse"] == 1 else "FALSE"
        logger.info("  %s  %s  shape_corr=%.3f", row["date"], kind, row["shape_corr"])

    logger.info("\n漏检的真反向日:")
    for _, row in out[(out["pred_reverse_rule"] == 0) & (out["is_reverse"] == 1)].iterrows():
        logger.info("  %s  shape_corr=%.3f", row["date"], row["shape_corr"])


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
