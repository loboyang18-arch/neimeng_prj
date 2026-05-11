"""
V11 Hybrid — V10 精准点估计 + Chronos 校准不确定性 + 鲁棒 MILP

逻辑：
  - V10 P50 (MAE=107): 中位数预测准
  - Chronos P10/P90 (cov80=0.79): 真不确定性区间
  - 取 V10 P50 当中位数，把 Chronos 的"半宽"加到 V10 P50 两侧 → 得到混合分位数
  - 跑 robust MILP α 网格

关键假设：Chronos 的 (P50 - P10) 和 (P90 - P50) 反映真实不确定性强度，
          但 Chronos 的 P50 偏离系统真值；V10 的 P50 系统对齐更好。
          组合后既精又有可信区间。
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("NM_V8_TARGET", "price_sudun_500kv1m_nodal")

from src.config import OUTPUT_DIR  # noqa: E402

logger = logging.getLogger(__name__)
EVAL_DIR = OUTPUT_DIR / "experiments" / "v11.0-hybrid"

V10_PRED = OUTPUT_DIR / "experiments" / "v10.0-joint" / "test_predictions_hourly.csv"


def build_hybrid_quantiles(
    chronos_q_csv: Path,
    v10_pred_csv: Path,
    width_scale: float = 1.0,
) -> pd.DataFrame:
    """构造混合分位数预测：V10 P50 + Chronos 半宽。

    Args:
        width_scale: 对 Chronos 半宽再缩放（可调，默认 1.0）。
                     设为 0.5 / 0.7 可以减弱区间，让 robust MILP 不太保守。
    Returns:
        DataFrame: ts, actual, p10, p30, p50, p70, p90, date
    """
    chronos = pd.read_csv(chronos_q_csv, parse_dates=["ts"])
    v10 = pd.read_csv(v10_pred_csv, parse_dates=["ts"])
    v10 = v10.rename(columns={"pred": "v10_p50"})

    merged = chronos.merge(
        v10[["ts", "v10_p50"]], on="ts", how="inner",
        suffixes=("", "_v10"),
    )
    if len(merged) == 0:
        raise RuntimeError(f"merge 空：chronos {len(chronos)} vs v10 {len(v10)}")
    if "actual_v10" in merged.columns:
        merged = merged.drop(columns=["actual_v10"])

    # Chronos 的半宽
    half_low = (merged["p50"] - merged["p10"]).clip(lower=0) * width_scale
    half_low_30 = (merged["p50"] - merged["p30"]).clip(lower=0) * width_scale
    half_high_70 = (merged["p70"] - merged["p50"]).clip(lower=0) * width_scale
    half_high = (merged["p90"] - merged["p50"]).clip(lower=0) * width_scale

    out = pd.DataFrame({
        "ts": merged["ts"],
        "actual": merged["actual"],
        "p10": merged["v10_p50"] - half_low,
        "p30": merged["v10_p50"] - half_low_30,
        "p50": merged["v10_p50"],
        "p70": merged["v10_p50"] + half_high_70,
        "p90": merged["v10_p50"] + half_high,
    })
    out["date"] = out["ts"].dt.date.astype(str)
    return out.sort_values("ts").reset_index(drop=True)


def coverage_metrics(df: pd.DataFrame):
    a = df["actual"].values
    p10 = df["p10"].values
    p50 = df["p50"].values
    p90 = df["p90"].values
    mask = ~np.isnan(a)
    a, p10, p50, p90 = a[mask], p10[mask], p50[mask], p90[mask]
    mae = float(np.mean(np.abs(p50 - a)))
    rmse = float(np.sqrt(np.mean((p50 - a) ** 2)))
    cov80 = float(np.mean((a >= p10) & (a <= p90)))
    width = float(np.mean(p90 - p10))
    return mae, rmse, cov80, width


def run(
    chronos_exp: str = "v11.0-foundation",
    v10_pred_csv: Path = V10_PRED,
    width_scales: tuple = (0.3, 0.5, 0.7, 1.0),
    alphas: tuple = (0.0, 0.3, 0.5, 0.7, 1.0),
):
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    chronos_csv = OUTPUT_DIR / "experiments" / chronos_exp / "test_predictions_quantile.csv"

    from src.eval_v10_quantile_robust import evaluate_alpha
    from scripts.strategy_milp_15min import load_actual_15min
    actual_xlsx = ROOT / "source_data" / "日清算结果查询电厂侧(1)_副本.xlsx"
    actual_df = load_actual_15min(actual_xlsx)

    summaries = []

    for ws in width_scales:
        logger.info("=" * 60)
        logger.info("混合策略: V10 P50 + Chronos 半宽 × %.2f", ws)
        logger.info("=" * 60)

        hybrid_df = build_hybrid_quantiles(chronos_csv, v10_pred_csv, width_scale=ws)
        sub_dir = EVAL_DIR / f"width_{int(ws * 100):03d}"
        sub_dir.mkdir(parents=True, exist_ok=True)
        hybrid_df.to_csv(sub_dir / "predictions_quantile.csv", index=False)

        mae, rmse, cov80, width = coverage_metrics(hybrid_df)
        logger.info("  混合预测: MAE=%.2f RMSE=%.2f cov80=%.3f width=%.1f",
                    mae, rmse, cov80, width)

        for alpha in alphas:
            df = evaluate_alpha(hybrid_df, actual_df, alpha=alpha, carry_soc=True)
            df.to_csv(sub_dir / f"daily_alpha_{alpha:.2f}.csv", index=False)

            net_total = df["net"].sum()
            pf_total = df["pf_net"].sum()
            n_loss = (df["net"] < 0).sum()
            ratio = net_total / pf_total if abs(pf_total) > 1 else np.nan

            logger.info(
                "  α=%.2f: 净=%.2f万 兑现率=%.1f%% 亏损天=%d",
                alpha, net_total / 1e4,
                ratio * 100 if not np.isnan(ratio) else 0, n_loss,
            )
            summaries.append({
                "width_scale": ws, "alpha": alpha,
                "mae": round(mae, 2), "cov80": round(cov80, 3),
                "width": round(width, 1),
                "n_days": len(df), "n_loss_days": int(n_loss),
                "net_total_wan": round(net_total / 1e4, 2),
                "realization": round(ratio, 4) if not np.isnan(ratio) else None,
            })

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(EVAL_DIR / "summary.csv", index=False)

    print("\n" + "=" * 80)
    print(" V11 Hybrid (V10 P50 + Chronos uncertainty) 网格")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print("=" * 80)

    return summary_df


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    p = argparse.ArgumentParser()
    p.add_argument("--chronos-exp", default="v11.0-foundation",
                   help="chronos 预测实验目录名")
    p.add_argument("--width-scales", default="0.3,0.5,0.7,1.0")
    p.add_argument("--alphas", default="0.0,0.3,0.5,0.7,1.0")
    args = p.parse_args()
    ws = tuple(float(x) for x in args.width_scales.split(","))
    al = tuple(float(x) for x in args.alphas.split(","))
    run(chronos_exp=args.chronos_exp, width_scales=ws, alphas=al)
