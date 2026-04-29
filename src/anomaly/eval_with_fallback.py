"""
异常日降级策略评估
==================
对每种降级策略生成 hybrid 预测 CSV，调用 scripts.strategy_milp_15min.run 计算 MILP 收益。

降级策略：
  A: no-op           → 输出全天均值（flat），MILP 不操作
  B: mean_shape      → mean_shape × V10 预测的当日均价（保留温和价差）
  C: v8_fallback     → 反向日改用 V8 预测

模式：
  oracle    → 使用真实标签 (shape_corr<0)，给出收益上限
  detector  → 使用 LightGBM 检测器输出
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("NM_V8_TARGET", "price_sudun_500kv1m_nodal")

from scripts.strategy_milp_15min import run as milp_run, plot_weekly
from src.config import OUTPUT_DIR

logger = logging.getLogger(__name__)

ANOMALY_DIR = OUTPUT_DIR / "experiments" / "anomaly-detector"
EVAL_DIR = OUTPUT_DIR / "experiments" / "anomaly-fallback"
ACTUAL_XLSX = ROOT / "source_data" / "日清算结果查询电厂侧(1)_副本.xlsx"

V10_PRED = OUTPUT_DIR / "experiments" / "v10.0-joint" / "test_predictions_hourly.csv"
V8_PRED = OUTPUT_DIR / "experiments" / "v8.0-jan25-sudun500" / "test_predictions_hourly.csv"


def _load_predictions():
    v10 = pd.read_csv(V10_PRED, parse_dates=["ts"])
    v8 = pd.read_csv(V8_PRED, parse_dates=["ts"])
    v10["date"] = v10["ts"].dt.date
    v8["date"] = v8["ts"].dt.date
    return v10, v8


def _load_anomaly_flags(mode: str, threshold: float | None = None) -> pd.DataFrame:
    """返回 DataFrame[date, is_anomaly]。"""
    if mode == "oracle":
        labels = pd.read_csv(ANOMALY_DIR / "shape_labels.csv", parse_dates=["date"])
        labels["date"] = labels["date"].dt.date
        labels["is_anomaly"] = labels["is_reverse"]
        return labels[["date", "is_anomaly"]]
    elif mode == "detector":
        df = pd.read_csv(ANOMALY_DIR / "test_predictions.csv", parse_dates=["date"])
        df["date"] = df["date"].dt.date
        if threshold is None:
            df["is_anomaly"] = df["pred_reverse_f1"]
        else:
            df["is_anomaly"] = (df["prob_reverse"] >= threshold).astype(int)
        return df[["date", "is_anomaly"]]
    elif mode == "rule":
        feats = pd.read_csv(ANOMALY_DIR / "day_features.csv", parse_dates=["date"])
        feats["date"] = feats["date"].dt.date
        from src.anomaly.rule_detector import apply_rule
        feats["is_anomaly"] = apply_rule(feats).astype(int)
        return feats[["date", "is_anomaly"]]
    else:
        raise ValueError(f"未知模式: {mode}")


def _build_hybrid_pred(strategy: str, v10: pd.DataFrame, v8: pd.DataFrame,
                       anomaly_dates: set, mean_shape: np.ndarray) -> pd.DataFrame:
    """对反向日应用降级策略，正常日保留 V10 预测。返回与 v10 同结构的 DataFrame。"""
    out_rows = []
    for d, group in v10.groupby("date"):
        g = group.sort_values("ts").reset_index(drop=True)
        if len(g) != 24:
            out_rows.extend(g.to_dict("records"))
            continue

        if d not in anomaly_dates:
            out_rows.extend(g.to_dict("records"))
            continue

        actual_24 = g["actual"].values
        v10_pred_24 = g["pred"].values
        ts_24 = g["ts"].values

        if strategy == "noop":
            new_pred = np.full(24, float(np.mean(v10_pred_24)))
        elif strategy == "mean_shape":
            day_mean = float(np.mean(v10_pred_24))
            day_std = max(float(np.std(v10_pred_24)), 50.0)
            new_pred = day_mean + mean_shape * day_std
        elif strategy == "v8_fallback":
            v8_d = v8[v8["date"] == d].sort_values("ts").reset_index(drop=True)
            if len(v8_d) == 24:
                new_pred = v8_d["pred"].values
            else:
                new_pred = v10_pred_24
        else:
            raise ValueError(strategy)

        for h in range(24):
            out_rows.append({"ts": pd.Timestamp(ts_24[h]),
                             "actual": float(actual_24[h]),
                             "pred": float(new_pred[h]),
                             "date": d})

    out = pd.DataFrame(out_rows).drop(columns=["date"])
    return out


def _summarize(strategy: str, milp_csv: Path, anomaly_dates: set,
               labels: pd.DataFrame) -> dict:
    """读取 MILP 结果并按日类别汇总。"""
    df = pd.read_csv(milp_csv)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.merge(labels[["date", "category", "is_reverse"]], on="date", how="left")

    test_start = pd.Timestamp("2026-01-27").date()
    test_end = pd.Timestamp("2026-04-17").date()
    df = df[(df["date"] >= test_start) & (df["date"] <= test_end)]

    summary = {"strategy": strategy, "total_days": len(df)}
    summary["net_total"] = float(df["net"].sum())
    summary["pf_total"] = float(df["pf_net"].sum())
    summary["realization"] = (
        df["net"].sum() / df["pf_net"].sum() if abs(df["pf_net"].sum()) > 1 else 0
    )

    rev = df[df["is_reverse"] == 1]
    summary["reverse_days"] = len(rev)
    summary["reverse_net"] = float(rev["net"].sum())
    summary["reverse_pf"] = float(rev["pf_net"].sum())

    nor = df[df["is_reverse"] == 0]
    summary["normal_net"] = float(nor["net"].sum())
    summary["normal_pf"] = float(nor["pf_net"].sum())

    for cat in ["typical", "normal", "atypical", "reverse"]:
        sub = df[df["category"] == cat]
        summary[f"{cat}_n"] = len(sub)
        summary[f"{cat}_net"] = float(sub["net"].sum()) if len(sub) > 0 else 0
        summary[f"{cat}_pf"] = float(sub["pf_net"].sum()) if len(sub) > 0 else 0
    return summary


def evaluate(mode: str = "oracle", threshold: float | None = None,
             draw_plots: bool = False):
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    v10, v8 = _load_predictions()
    flags = _load_anomaly_flags(mode, threshold)
    labels = pd.read_csv(ANOMALY_DIR / "shape_labels.csv", parse_dates=["date"])
    labels["date"] = labels["date"].dt.date

    mean_shape_24 = np.load(ANOMALY_DIR / "mean_shape_train.npy")  # (24,)

    test_start = pd.Timestamp("2026-01-27").date()
    test_end = pd.Timestamp("2026-04-17").date()
    flags_test = flags[(flags["date"] >= test_start) & (flags["date"] <= test_end)]
    anomaly_dates = set(flags_test[flags_test["is_anomaly"] == 1]["date"].tolist())
    logger.info("[%s] 测试期检出/标记的反向日: %d / %d",
                mode, len(anomaly_dates), len(flags_test))
    if mode == "oracle":
        logger.info("  反向日: %s", sorted([str(d) for d in anomaly_dates]))

    sub_dir = EVAL_DIR / mode
    sub_dir.mkdir(parents=True, exist_ok=True)

    summaries = []

    label_v10 = "V10 baseline (no fallback)"
    csv_v10 = sub_dir / "v10_baseline.csv"
    milp_run(pred_csv=V10_PRED, actual_xlsx=ACTUAL_XLSX,
             out_csv=csv_v10, label=label_v10, carry_soc=True)
    summaries.append(_summarize("v10_baseline", csv_v10, anomaly_dates, labels))

    for strategy in ["noop", "mean_shape", "v8_fallback"]:
        logger.info("=" * 60)
        logger.info("策略: %s", strategy)
        logger.info("=" * 60)
        hybrid = _build_hybrid_pred(strategy, v10, v8, anomaly_dates, mean_shape_24)
        hybrid_csv = sub_dir / f"hybrid_{strategy}.csv"
        hybrid.to_csv(hybrid_csv, index=False)
        out_csv = sub_dir / f"milp_{strategy}.csv"
        milp_run(pred_csv=hybrid_csv, actual_xlsx=ACTUAL_XLSX,
                 out_csv=out_csv, label=f"V10 + fallback={strategy}",
                 carry_soc=True)
        s = _summarize(strategy, out_csv, anomaly_dates, labels)
        summaries.append(s)

        if draw_plots:
            plot_weekly(pd.read_csv(out_csv), sub_dir / f"plots_{strategy}")

    csv_v8 = sub_dir / "v8_baseline.csv"
    milp_run(pred_csv=V8_PRED, actual_xlsx=ACTUAL_XLSX,
             out_csv=csv_v8, label="V8 baseline",
             start=str(test_start), end=str(test_end), carry_soc=True)
    summaries.append(_summarize("v8_baseline", csv_v8, anomaly_dates, labels))

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(sub_dir / "summary.csv", index=False)

    logger.info("=" * 80)
    logger.info("[%s] 策略对比汇总", mode)
    logger.info("=" * 80)
    fmt = "{:<15} {:>10} {:>10} {:>9} {:>12} {:>12}"
    logger.info(fmt.format("策略", "总净收益", "PF", "兑现率",
                           "反向日净收益", "正常日净收益"))
    for s in summaries:
        logger.info(fmt.format(
            s["strategy"],
            f"{s['net_total']/1e4:.1f}万",
            f"{s['pf_total']/1e4:.1f}万",
            f"{s['realization']*100:.1f}%",
            f"{s['reverse_net']/1e4:+.1f}万",
            f"{s['normal_net']/1e4:+.1f}万",
        ))
    logger.info("")
    logger.info("按反向日明细 (vs V10 baseline):")
    base = next(s for s in summaries if s["strategy"] == "v10_baseline")
    for s in summaries:
        if s["strategy"] in ["v10_baseline", "v8_baseline"]:
            continue
        delta_total = (s["net_total"] - base["net_total"]) / 1e4
        delta_rev = (s["reverse_net"] - base["reverse_net"]) / 1e4
        delta_norm = (s["normal_net"] - base["normal_net"]) / 1e4
        logger.info(
            "  %-15s 总收益 %+.1f万 (反向日 %+.1f万 + 其他 %+.1f万)",
            s["strategy"], delta_total, delta_rev, delta_norm,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["oracle", "detector", "rule"], default="oracle")
    parser.add_argument("--threshold", type=float, default=None,
                        help="检测器阈值（仅 detector 模式生效）")
    parser.add_argument("--plots", action="store_true", help="生成周图")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    evaluate(mode=args.mode, threshold=args.threshold, draw_plots=args.plots)
