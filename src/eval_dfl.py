"""
DFL 评估脚本：DFL-V8 预测 → 完整 MILP → 性能对比
===================================================
对比三组：
  A. MSE baseline V8 预测 + MILP
  B. DFL fine-tuned V8 预测 + MILP
  C. 完全预知 (PF) 基准
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.strategy_milp_15min import (
    solve_day_milp_15min,
    solve_pf_day_15min,
    eval_day_revenue_15min,
    load_actual_15min,
    AUX_MWH,
)

logger = logging.getLogger(__name__)

OUTPUT_DIR = ROOT / "output"
DFL_DIR = OUTPUT_DIR / "experiments" / "v8.0-dfl"
ACTUAL_XLSX = ROOT / "source_data" / "日清算结果查询电厂侧(1)_副本.xlsx"


def evaluate_milp(pred_csv: Path, actual_df: pd.DataFrame,
                  start: str = "2026-01-25", end: str = "2026-04-18",
                  carry_soc: bool = True,
                  label: str = "Strategy") -> pd.DataFrame:
    """用预测 CSV 跑完整 MILP，返回每日结果 DataFrame。"""

    pred = pd.read_csv(pred_csv, parse_dates=["ts"])
    pred["date"] = pred["ts"].dt.date.astype(str)
    if start:
        pred = pred[pred["date"] >= start]
    if end:
        pred = pred[pred["date"] <= end]

    dates = sorted(pred["date"].unique())
    rows = []
    soc_carry = 0.0
    soc_carry_pf = 0.0

    for i, date in enumerate(dates):
        if date not in actual_df.index:
            continue

        day_pred = pred[pred["date"] == date].sort_values("ts")
        n = len(day_pred)
        if n >= 96:
            pred_96 = day_pred["pred"].values[:96].astype(float)
        elif n >= 24:
            pred_hourly = day_pred["pred"].values[:24].astype(float)
            pred_96 = np.repeat(pred_hourly, 4)
        else:
            continue

        actual_96 = actual_df.loc[date].values.astype(float)

        is_last = (i == len(dates) - 1)
        if carry_soc:
            force_end = is_last
            if not is_last and dates[i + 1] in pred["date"].values:
                next_h = pred[pred["date"] == dates[i + 1]]["pred"].values
                next_avg = float(np.mean(next_h)) if len(next_h) > 0 else 0.0
            else:
                next_avg = 0.0
        else:
            force_end = True
            next_avg = 0.0

        c, d, soc = solve_day_milp_15min(
            pred_96, soc_init=soc_carry,
            force_zero_end=force_end,
            next_day_avg_price=next_avg,
        )
        c_pf, d_pf, soc_pf = solve_pf_day_15min(
            actual_96, soc_init=soc_carry_pf,
            force_zero_end=force_end,
            next_day_avg_price=float(np.mean(actual_96)),
        )

        soc_end = float(soc[-1]) if soc.sum() > 0 else 0.0
        soc_end_pf = float(soc_pf[-1]) if soc_pf.sum() > 0 else 0.0
        if carry_soc:
            soc_carry = soc_end
            soc_carry_pf = soc_end_pf

        rev = eval_day_revenue_15min(c, d, actual_96)
        rev_pf = eval_day_revenue_15min(c_pf, d_pf, actual_96)

        rows.append({
            "date": date,
            "charge_mwh": rev["charge_mwh"],
            "discharge_mwh": rev["discharge_mwh"],
            "gross": rev["gross"],
            "aux_cost": rev["aux_cost"],
            "net": rev["net"],
            "pf_gross": rev_pf["gross"],
            "pf_net": rev_pf["net"],
            "soc_end": soc_end,
        })

    df = pd.DataFrame(rows)
    return df


def run_comparison():
    """对比 MSE vs DFL 的 MILP 评估结果。"""

    dfl_pred = DFL_DIR / "test_predictions_hourly.csv"
    mse_pred = DFL_DIR / "test_predictions_hourly_mse.csv"

    if not dfl_pred.exists() or not mse_pred.exists():
        logger.error("预测文件不存在，请先运行 train_dfl.py")
        return

    logger.info("加载实际 15 分钟价格...")
    actual_df = load_actual_15min(ACTUAL_XLSX)
    logger.info("  %d 天", len(actual_df))

    logger.info("=" * 70)
    logger.info("评估 MSE baseline V8 + MILP...")
    mse_results = evaluate_milp(mse_pred, actual_df, label="MSE-V8")

    logger.info("评估 DFL fine-tuned V8 + MILP...")
    dfl_results = evaluate_milp(dfl_pred, actual_df, label="DFL-V8")
    logger.info("=" * 70)

    # 保存结果
    mse_results.to_csv(DFL_DIR / "milp_results_mse.csv", index=False)
    dfl_results.to_csv(DFL_DIR / "milp_results_dfl.csv", index=False)

    # 汇总对比
    def _summary(df, name):
        n = len(df)
        tot_net = df["net"].sum()
        tot_pf = df["pf_net"].sum()
        ratio = tot_net / tot_pf if abs(tot_pf) > 1 else float("nan")
        ann = tot_net / n * 365 if n > 0 else 0
        return {
            "strategy": name,
            "days": n,
            "net_revenue_yuan": round(tot_net, 0),
            "pf_net_yuan": round(tot_pf, 0),
            "realization_pct": round(ratio * 100, 1),
            "annualized_yuan": round(ann, 0),
        }

    summary = [
        _summary(mse_results, "MSE-V8 + MILP"),
        _summary(dfl_results, "DFL-V8 + MILP"),
    ]
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(DFL_DIR / "dfl_vs_mse_summary.csv", index=False)

    # 打印报告
    logger.info("\n" + "=" * 70)
    logger.info("DFL vs MSE — MILP 评估汇总")
    logger.info("=" * 70)
    for s in summary:
        logger.info(
            "  %-20s  净收益: %+.1f万  PF: %.1f万  兑现率: %.1f%%  全年: %.2f亿",
            s["strategy"],
            s["net_revenue_yuan"] / 1e4,
            s["pf_net_yuan"] / 1e4,
            s["realization_pct"],
            s["annualized_yuan"] / 1e8,
        )

    # 逐日对比
    merge = mse_results[["date", "net"]].rename(columns={"net": "mse_net"}).merge(
        dfl_results[["date", "net"]].rename(columns={"net": "dfl_net"}),
        on="date", how="outer",
    )
    merge["diff"] = merge["dfl_net"] - merge["mse_net"]
    win = (merge["diff"] > 0).sum()
    lose = (merge["diff"] < 0).sum()
    tie = (merge["diff"] == 0).sum()
    logger.info("\n  逐日胜负: DFL 胜 %d 天, MSE 胜 %d 天, 平 %d 天", win, lose, tie)
    logger.info("  DFL 累计优势: %+.1f 万元", merge["diff"].sum() / 1e4)

    merge.to_csv(DFL_DIR / "daily_comparison.csv", index=False)
    logger.info("\n结果已保存至: %s", DFL_DIR)

    return summary_df


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    run_comparison()
