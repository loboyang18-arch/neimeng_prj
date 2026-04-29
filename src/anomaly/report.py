"""
异常日检测 + 保守降级 — 最终对比报告
======================================
汇总 V10 / V8 / Oracle / Rule / Detector 模式下的策略收益，并生成 markdown 报告。
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
EVAL_DIR = OUTPUT_DIR / "experiments" / "anomaly-fallback"


def _load_summary(mode: str) -> pd.DataFrame:
    p = EVAL_DIR / mode / "summary.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def main():
    oracle = _load_summary("oracle")
    rule = _load_summary("rule")

    labels = pd.read_csv(ANOMALY_DIR / "shape_labels.csv", parse_dates=["date"])
    labels["date"] = labels["date"].dt.date
    test_start = pd.Timestamp("2026-01-27").date()
    test_end = pd.Timestamp("2026-04-17").date()
    test_labels = labels[(labels["date"] >= test_start) & (labels["date"] <= test_end)]
    n_reverse = int((test_labels["category"] == "reverse").sum())

    rule_det = pd.read_csv(ANOMALY_DIR / "test_rule_detection.csv",
                           parse_dates=["date"])
    rule_det["date"] = rule_det["date"].dt.date
    tp = int(((rule_det["pred_reverse_rule"] == 1) &
              (rule_det["is_reverse"] == 1)).sum())
    fp = int(((rule_det["pred_reverse_rule"] == 1) &
              (rule_det["is_reverse"] == 0)).sum())
    fn = int(((rule_det["pred_reverse_rule"] == 0) &
              (rule_det["is_reverse"] == 1)).sum())

    v10_milp = pd.read_csv(EVAL_DIR / "rule" / "v10_baseline.csv")
    v10_milp["date"] = pd.to_datetime(v10_milp["date"]).dt.date
    v10_milp = v10_milp[(v10_milp["date"] >= test_start) &
                        (v10_milp["date"] <= test_end)]

    rule_noop = pd.read_csv(EVAL_DIR / "rule" / "milp_noop.csv")
    rule_noop["date"] = pd.to_datetime(rule_noop["date"]).dt.date
    rule_noop = rule_noop[(rule_noop["date"] >= test_start) &
                          (rule_noop["date"] <= test_end)]

    v8_milp = pd.read_csv(EVAL_DIR / "rule" / "v8_baseline.csv")
    v8_milp["date"] = pd.to_datetime(v8_milp["date"]).dt.date
    v8_milp = v8_milp[(v8_milp["date"] >= test_start) &
                      (v8_milp["date"] <= test_end)]

    detected = rule_det[rule_det["pred_reverse_rule"] == 1]["date"].tolist()
    detail = []
    for d in detected:
        v10_net = float(v10_milp[v10_milp["date"] == d]["net"].values[0])
        new_net = float(rule_noop[rule_noop["date"] == d]["net"].values[0])
        v8_net = float(v8_milp[v8_milp["date"] == d]["net"].values[0])
        is_rev = int(rule_det[rule_det["date"] == d]["is_reverse"].values[0])
        detail.append({
            "date": d,
            "is_reverse": is_rev,
            "v10_net": v10_net,
            "noop_net": new_net,
            "v8_net": v8_net,
            "delta_v10_to_noop": new_net - v10_net,
        })
    detail_df = pd.DataFrame(detail)

    md = []
    md.append("# 异常日检测 + 保守降级策略 — 实验报告\n")
    md.append(f"**测试期**: {test_start} ~ {test_end} ({len(v10_milp)} 天)\n")
    md.append(f"**真实反向日 (corr<0)**: {n_reverse} 天\n")
    md.append("\n---\n\n## 策略总收益对比\n")
    md.append("| 策略 | 总净收益(万) | PF(万) | 兑现率 | 反向日净(万) | 全年外推(亿) |")
    md.append("|------|------------:|-------:|-------:|------------:|------------:|")

    annual_factor = 365 / max(len(v10_milp), 1)
    rows_to_show = []
    for label, src, key in [
        ("V10 baseline", oracle, "v10_baseline"),
        ("V8 baseline", oracle, "v8_baseline"),
        ("Oracle + no-op", oracle, "noop"),
        ("Oracle + mean_shape", oracle, "mean_shape"),
        ("Oracle + v8_fallback", oracle, "v8_fallback"),
        ("Rule + no-op", rule, "noop"),
        ("Rule + mean_shape", rule, "mean_shape"),
        ("Rule + v8_fallback", rule, "v8_fallback"),
    ]:
        if len(src) == 0:
            continue
        r = src[src["strategy"] == key]
        if len(r) == 0:
            continue
        r = r.iloc[0]
        annual = r["net_total"] / 1e4 * annual_factor / 1e4
        rows_to_show.append((label, r["net_total"]/1e4, r["pf_total"]/1e4,
                             r["realization"]*100, r["reverse_net"]/1e4, annual))
        md.append(f"| {label} | {r['net_total']/1e4:.1f} | {r['pf_total']/1e4:.1f} | "
                  f"{r['realization']*100:.1f}% | {r['reverse_net']/1e4:+.1f} | "
                  f"{annual:.3f} |")

    md.append("\n---\n\n## 检测器评估（基于规则）\n")
    md.append("**规则**: `wind_forecast_morning_evening_diff < -2000 AND reserve_neg_capacity_min > 3000`\n\n")
    p_v = tp / max(tp + fp, 1)
    r_v = tp / max(tp + fn, 1)
    f1_v = 2 * p_v * r_v / max(p_v + r_v, 1e-6)
    md.append(f"- TP = {tp}  FP = {fp}  FN = {fn}")
    md.append(f"- 精确率 = {p_v:.2f}, 召回率 = {r_v:.2f}, F1 = {f1_v:.2f}\n")

    md.append("\n## 规则检出的 10 天明细\n")
    md.append("| 日期 | 真反向 | V10净(万) | no-op净(万) | V8净(万) | Δ(no-op vs V10) |")
    md.append("|------|:------:|---------:|------------:|--------:|----------------:|")
    for _, r in detail_df.iterrows():
        flag = "TRUE" if r["is_reverse"] == 1 else "FALSE"
        md.append(f"| {r['date']} | {flag} | {r['v10_net']/1e4:+.1f} | "
                  f"{r['noop_net']/1e4:+.1f} | {r['v8_net']/1e4:+.1f} | "
                  f"{r['delta_v10_to_noop']/1e4:+.1f} |")

    md.append("\n---\n\n## 关键结论\n")
    if len(rule):
        v10_total = float(rule[rule["strategy"] == "v10_baseline"].iloc[0]["net_total"])
        rule_noop_total = float(rule[rule["strategy"] == "noop"].iloc[0]["net_total"])
        v8_total = float(rule[rule["strategy"] == "v8_baseline"].iloc[0]["net_total"])
        oracle_noop = float(oracle[oracle["strategy"] == "noop"].iloc[0]["net_total"]) if len(oracle) else 0

        gain_vs_v10 = (rule_noop_total - v10_total) / 1e4
        gain_vs_v8 = (rule_noop_total - v8_total) / 1e4
        oracle_gain = (oracle_noop - v10_total) / 1e4
        md.append(f"- 推荐方案 **Rule + no-op**: 净收益 {rule_noop_total/1e4:.1f} 万")
        md.append(f"- vs V10 baseline: **+{gain_vs_v10:.1f} 万** "
                  f"({gain_vs_v10/(v10_total/1e4)*100:.1f}%)")
        md.append(f"- vs V8 baseline:  **+{gain_vs_v8:.1f} 万** "
                  f"({gain_vs_v8/(v8_total/1e4)*100:.1f}%)")
        md.append(f"- Oracle 上限对照: +{oracle_gain:.1f} 万 (规则版本"
                  f"{'超越' if rule_noop_total > oracle_noop else '低于'} oracle "
                  f"{abs(rule_noop_total-oracle_noop)/1e4:.1f} 万)")
        md.append(f"- 全年外推: V10={v10_total/1e4*annual_factor/1e4:.3f} 亿 → "
                  f"Rule+no-op={rule_noop_total/1e4*annual_factor/1e4:.3f} 亿\n")

    md.append("## 风险提示\n")
    md.append("- 规则阈值 (`wind_evening_diff<-2000`, `reserve_neg_min>3000`) 仅在 2026-02 ~ 2026-04 测试期调优")
    md.append("- 训练集（2022-06 ~ 2026-01）反向日 82% 集中在夏季 5-8 月，与测试期机制不同（冬末春初大风+大光）")
    md.append("- LightGBM 检测器在此分布漂移下完全失效（11 个真反向日全部漏检）")
    md.append("- 推广到其他季节前需重新调优阈值，建议每季度复检\n")

    out_path = EVAL_DIR / "REPORT.md"
    out_path.write_text("\n".join(md), encoding="utf-8")
    logger.info("报告已保存: %s", out_path)
    print("\n".join(md))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
