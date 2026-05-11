"""
V10-Quantile 鲁棒 MILP 评估：α 网格扫描

对每个 α ∈ {0, 0.3, 0.5, 0.7, 1.0}：
  - 每天用分位数预测求解 robust MILP
  - 与 V10 baseline (α=0 实际就是 P50 单价格) 对比
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("NM_V8_TARGET", "price_sudun_500kv1m_nodal")
os.environ.setdefault("NM_V8_EXTRA_LAG1",
                       "price_sudun_500kv1m_energy,price_sudun_500kv1m_cong")
os.environ.setdefault("NM_V8_HOURLY_AGG", "mean4")
os.environ.setdefault("NM_MIN_FEATURE_DATE", "2024-12-14")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.strategy_milp_15min import (  # noqa: E402
    solve_day_milp_15min,
    solve_day_milp_15min_robust,
    solve_pf_day_15min,
    eval_day_revenue_15min,
    load_actual_15min,
)

logger = logging.getLogger(__name__)

OUTPUT_DIR = ROOT / "output"
ACTUAL_XLSX = ROOT / "source_data" / "日清算结果查询电厂侧(1)_副本.xlsx"


def _expand_24_to_96(arr_24: np.ndarray) -> np.ndarray:
    """小时级预测扩展到 15 分钟（每小时重复 4 次）。"""
    return np.repeat(arr_24, 4)


def evaluate_alpha(
    pred_df: pd.DataFrame,
    actual_df: pd.DataFrame,
    alpha: float,
    carry_soc: bool = True,
) -> pd.DataFrame:
    """对所有日期跑 robust MILP（α），收集每日收益。"""
    dates = sorted(pred_df["date"].unique())
    rows = []

    soc_carry = 0.0
    soc_carry_pf = 0.0

    for i, date in enumerate(dates):
        if date not in actual_df.index:
            continue

        day = pred_df[pred_df["date"] == date].sort_values("ts")
        if len(day) != 24:
            continue

        p10_24 = day["p10"].values.astype(float)
        p50_24 = day["p50"].values.astype(float)
        p90_24 = day["p90"].values.astype(float)
        actual_24 = day["actual"].values.astype(float)

        p10_96 = _expand_24_to_96(p10_24)
        p50_96 = _expand_24_to_96(p50_24)
        p90_96 = _expand_24_to_96(p90_24)
        actual_96 = actual_df.loc[date].values.astype(float)

        is_last = i == len(dates) - 1
        force_end = is_last or (not carry_soc)

        if carry_soc and not is_last:
            next_date = dates[i + 1]
            if next_date in pred_df["date"].values:
                next_p50 = pred_df[pred_df["date"] == next_date]["p50"].values
                next_avg = float(np.mean(next_p50)) if len(next_p50) > 0 else 0.0
            else:
                next_avg = 0.0
        else:
            next_avg = 0.0

        if alpha == 0.0:
            c, d, soc = solve_day_milp_15min(
                p50_96, soc_init=soc_carry,
                force_zero_end=force_end, next_day_avg_price=next_avg,
            )
        else:
            c, d, soc = solve_day_milp_15min_robust(
                p10_96, p50_96, p90_96, alpha=alpha,
                soc_init=soc_carry,
                force_zero_end=force_end, next_day_avg_price=next_avg,
            )

        c_pf, d_pf, soc_pf = solve_pf_day_15min(
            actual_96, soc_init=soc_carry_pf,
            force_zero_end=force_end,
            next_day_avg_price=float(np.mean(actual_96)),
        )

        rev = eval_day_revenue_15min(c, d, actual_96)
        rev_pf = eval_day_revenue_15min(c_pf, d_pf, actual_96)

        if carry_soc:
            soc_carry = float(soc[-1])
            soc_carry_pf = float(soc_pf[-1])

        # 衡量分位数与真实价格关系
        actual_mean = float(np.mean(actual_24))
        p10_mean = float(np.mean(p10_24))
        p50_mean = float(np.mean(p50_24))
        p90_mean = float(np.mean(p90_24))

        rows.append({
            "date": date,
            "alpha": alpha,
            "actual_mean": round(actual_mean, 2),
            "p10_mean": round(p10_mean, 2),
            "p50_mean": round(p50_mean, 2),
            "p90_mean": round(p90_mean, 2),
            "interval_width": round(p90_mean - p10_mean, 2),
            "charge_mwh": rev["charge_mwh"],
            "discharge_mwh": rev["discharge_mwh"],
            "net": rev["net"],
            "pf_net": rev_pf["net"],
            "ratio": (rev["net"] / rev_pf["net"]) if abs(rev_pf["net"]) > 1 else np.nan,
        })

    return pd.DataFrame(rows)


def run(
    exp_dir: str = "v10.0-quantile",
    alphas: tuple = (0.0, 0.3, 0.5, 0.7, 1.0),
    carry_soc: bool = True,
):
    pred_csv = OUTPUT_DIR / "experiments" / exp_dir / "test_predictions_quantile.csv"
    if not pred_csv.exists():
        logger.error("未找到预测文件: %s", pred_csv)
        return None

    pred_df = pd.read_csv(pred_csv, parse_dates=["ts"])
    pred_df["date"] = pred_df["ts"].dt.date.astype(str)
    actual_df = load_actual_15min(ACTUAL_XLSX)

    out_dir = OUTPUT_DIR / "experiments" / exp_dir / "robust_milp"
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    all_dfs = []
    for alpha in alphas:
        logger.info("─" * 50)
        logger.info("α = %.2f, carry_soc=%s", alpha, carry_soc)
        df = evaluate_alpha(pred_df, actual_df, alpha=alpha, carry_soc=carry_soc)
        df.to_csv(out_dir / f"daily_alpha_{alpha:.2f}.csv", index=False)
        all_dfs.append(df)

        net_total = df["net"].sum()
        pf_total = df["pf_net"].sum()
        ratio = net_total / pf_total if abs(pf_total) > 1 else np.nan
        n_days = len(df)
        n_loss = (df["net"] < 0).sum()
        avg_width = df["interval_width"].mean()

        logger.info(
            "α=%.2f: 累计净收益=%.2f万元 (PF=%.2f万) 兑现率=%.1f%% | "
            "天数=%d 亏损天=%d 平均区间宽=%.1f",
            alpha, net_total / 1e4, pf_total / 1e4,
            ratio * 100 if not np.isnan(ratio) else 0,
            n_days, n_loss, avg_width,
        )
        summaries.append({
            "alpha": alpha,
            "n_days": n_days,
            "n_loss_days": int(n_loss),
            "net_total": int(net_total),
            "net_total_wan": round(net_total / 1e4, 2),
            "pf_total_wan": round(pf_total / 1e4, 2),
            "realization_rate": round(ratio, 4) if not np.isnan(ratio) else None,
            "avg_interval_width": round(avg_width, 2),
        })

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(out_dir / "summary_alpha_grid.csv", index=False)

    print("\n" + "=" * 72)
    print(f" V10-Quantile 鲁棒 MILP α 网格扫描结果（{exp_dir}, carry_soc={carry_soc}）")
    print("=" * 72)
    print(summary_df.to_string(index=False))
    print("=" * 72)

    return summary_df, all_dfs


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    p = argparse.ArgumentParser()
    p.add_argument("--exp", default="v10.0-quantile")
    p.add_argument("--alphas", default="0.0,0.3,0.5,0.7,1.0")
    p.add_argument("--no-carry-soc", action="store_true")
    args = p.parse_args()
    alphas = tuple(float(x) for x in args.alphas.split(","))
    run(exp_dir=args.exp, alphas=alphas, carry_soc=(not args.no_carry_soc))
