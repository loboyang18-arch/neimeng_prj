"""
基于实测数据的两种辅助用电方案效率分析
=========================================
方案A：以无主充电日估算辅助用电 24.09 MWh/天
方案B：以正常运营日中位数估算辅助用电 13.03 MWh/天

对应的双程效率（损耗）：
  η_A = 放电量 / (主充电 - 辅助_A) = 68327 / (77085 - 2024) ≈ 91.0%
  η_B = 放电量 / (主充电 - 辅助_B) = 68327 / (77085 - 1095) ≈ 89.9%

收益计算：
  实际放电量   = 800 × η  [MWh]
  毛收益       = 放电均价 × 800η - 充电均价 × 800
  辅助成本     = 日均实际价格 × aux_mwh_per_day
  净收益       = 毛收益 - 辅助成本

与完全预知基准(PF)同步调整。
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# ── 固定参数 ───────────────────────────────────────────────────────────────
CAPACITY_MW = 200
WINDOW_H    = 4
ENERGY_MWH  = 800          # 单次充/放电量（不含效率）

# 两种辅助用电方案（MWh/天）
AUX_A_MWH = 24.09          # 无主充电日均值
AUX_B_MWH = 13.03          # 正常日中位数

# 对应效率（由实测充放电数据推算）
ETA_A = 0.910              # 91.0%
ETA_B = 0.899              # 89.9%

# 原有 12% 辅助方案（保留对比）
AUX_OLD_MWH = ENERGY_MWH * 0.12   # 96 MWh/天，且无效率调整

# ── 辅助：周号 ─────────────────────────────────────────────────────────────
def _week_label(date: pd.Timestamp) -> str:
    iso = date.isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def compute_scenarios(strategy_csv: Path, pred_csv: Path) -> pd.DataFrame:
    """
    strategy_csv: 含 charge/discharge 窗口及实际均价的策略结果（nodaycross 版）
    pred_csv    : 含逐小时实际节点价格（actual 列）的预测文件
    """
    strat = pd.read_csv(strategy_csv)
    pred  = pd.read_csv(pred_csv, parse_dates=["ts"])
    pred["date"] = pred["ts"].dt.date

    # 每日实际均价（用于辅助用电成本）
    daily_avg = pred.groupby("date")["actual"].mean().rename("avg_price")

    strat["date_str"] = strat["date"].astype(str)
    strat["date_dt"]  = pd.to_datetime(strat["date"]).dt.date
    daily_avg_df = daily_avg.reset_index()
    daily_avg_df["date"] = daily_avg_df["date"].astype(str)
    strat = strat.merge(daily_avg_df, left_on="date_str", right_on="date", how="left", suffixes=("", "_r"))
    strat["avg_price_yuan"] = strat["avg_price"]   # 元/MWh

    results = []
    for _, row in strat.iterrows():
        dc_mean  = float(row["discharge_actual_mean"])
        ch_mean  = float(row["charge_actual_mean"])
        avg_p    = float(row["avg_price_yuan"])

        # ── 旧方案（无效率，96 MWh 辅助）──────────────────────────────────
        rev_old      = (dc_mean - ch_mean) * ENERGY_MWH
        aux_cost_old = avg_p * AUX_OLD_MWH
        net_old      = rev_old - aux_cost_old

        # PF 旧
        pf_rev_old = float(row["revenue_pf_yuan"])
        # PF 原脚本只含毛收益，无辅助成本调整，这里统一扣除
        pf_net_old = pf_rev_old - aux_cost_old

        # ── 方案 A（η=91%，辅助24.09 MWh）────────────────────────────────
        rev_A      = (dc_mean * ETA_A - ch_mean) * ENERGY_MWH
        aux_cost_A = avg_p * AUX_A_MWH
        net_A      = rev_A - aux_cost_A

        # PF 方案 A：按同一效率调整基准收益（从 pf_yuan 中还原均价再重算）
        pf_dc_mean = float(row["revenue_pf_yuan"]) / ENERGY_MWH + ch_mean   # 近似 pf 放电均价
        # 更准确：pf 的 revenue = (pf_dc - pf_ch) * 800，但我们没有 pf 的 ch/dc 分量
        # 保守地对毛收益乘以效率比 η_A 来近似
        pf_rev_A  = float(row["revenue_pf_yuan"]) * ETA_A
        pf_net_A  = pf_rev_A - aux_cost_A

        # ── 方案 B（η=89.9%，辅助13.03 MWh）──────────────────────────────
        rev_B      = (dc_mean * ETA_B - ch_mean) * ENERGY_MWH
        aux_cost_B = avg_p * AUX_B_MWH
        net_B      = rev_B - aux_cost_B

        pf_rev_B  = float(row["revenue_pf_yuan"]) * ETA_B
        pf_net_B  = pf_rev_B - aux_cost_B

        results.append({
            "date"          : row["date_str"],
            "charge_h"      : f"{int(row['charge_start']):02d}–{int(row['charge_end']):02d}",
            "discharge_h"   : f"{int(row['discharge_start']):02d}–{int(row['discharge_end']):02d}",
            "ch_mean"       : round(ch_mean, 2),
            "dc_mean"       : round(dc_mean, 2),
            "avg_price"     : round(avg_p, 2),
            # 旧方案
            "gross_old"     : round(rev_old,      0),
            "aux_old"       : round(aux_cost_old, 0),
            "net_old"       : round(net_old,       0),
            "pf_net_old"    : round(pf_net_old,   0),
            # 方案 A
            "gross_A"       : round(rev_A,         0),
            "aux_A"         : round(aux_cost_A,    0),
            "net_A"         : round(net_A,          0),
            "pf_net_A"      : round(pf_net_A,      0),
            # 方案 B
            "gross_B"       : round(rev_B,         0),
            "aux_B"         : round(aux_cost_B,    0),
            "net_B"         : round(net_B,          0),
            "pf_net_B"      : round(pf_net_B,      0),
        })

    return pd.DataFrame(results)


def _fmt(v: float) -> str:
    """格式化为万元，保留2位小数"""
    return f"{v/1e4:+.2f}万"


def print_report(df: pd.DataFrame, label: str):
    print(f"\n{'='*70}")
    print(f" {label}")
    print(f"{'='*70}")

    # 表头
    hdr = (f"{'日期':^12} {'充':^6} {'放':^6} "
           f"{'旧净收(万)':>10} {'A净收(万)':>10} {'B净收(万)':>10} "
           f"{'A-PF(万)':>10} {'B-PF(万)':>10}")
    print(hdr)
    print("-" * 70)

    df["date_dt"] = pd.to_datetime(df["date"])
    df["week"]    = df["date_dt"].apply(_week_label)
    week_groups   = df.groupby("week", sort=False)

    for wk, wdf in week_groups:
        for _, r in wdf.iterrows():
            print(f"{r['date']:^12} {r['charge_h']:^6} {r['discharge_h']:^6} "
                  f"{_fmt(r['net_old']):>10} {_fmt(r['net_A']):>10} {_fmt(r['net_B']):>10} "
                  f"{_fmt(r['pf_net_A']):>10} {_fmt(r['pf_net_B']):>10}")
        # 周小计
        s = wdf[["net_old","net_A","net_B","pf_net_A","pf_net_B"]].sum()
        print(f"  {'>> 周小计 ' + wk:^20} "
              f"{_fmt(s['net_old']):>10} {_fmt(s['net_A']):>10} {_fmt(s['net_B']):>10} "
              f"{_fmt(s['pf_net_A']):>10} {_fmt(s['pf_net_B']):>10}")
        print()

    # 总计
    tot = df[["net_old","net_A","net_B","pf_net_old","pf_net_A","pf_net_B",
              "gross_A","gross_B","aux_A","aux_B","gross_old","aux_old"]].sum()
    n   = len(df)
    print("=" * 70)
    print(f"测试期共 {n} 天")
    print()
    print(f"{'方案':12}  {'毛收益':>12}  {'辅助成本':>12}  {'净收益':>12}  {'PF净收益':>12}  {'兑现率':>8}")
    for tag, g, a, net, pf_net, eta, aux_mwh in [
        ("旧方案(12%,无η)", "gross_old", "aux_old", "net_old", "pf_net_old",  1.0,  96.0),
        (f"方案A(η=91%,辅={AUX_A_MWH})", "gross_A", "aux_A", "net_A", "pf_net_A", ETA_A, AUX_A_MWH),
        (f"方案B(η=89.9%,辅={AUX_B_MWH})", "gross_B", "aux_B", "net_B", "pf_net_B", ETA_B, AUX_B_MWH),
    ]:
        g_v   = tot[g]
        a_v   = tot[a]
        n_v   = tot[net]
        pf_v  = tot[pf_net] if pf_net in tot.index else float("nan")
        ratio = n_v / pf_v if pf_v != 0 else float("nan")
        print(f"  {tag:28}  {g_v/1e4:>10.1f}万  {a_v/1e4:>10.1f}万  "
              f"{n_v/1e4:>10.1f}万  {pf_v/1e4:>10.1f}万  {ratio:>7.1%}")

    print()
    # 全年外推（按 n 天线性外推 365 天）
    for tag, net, pf_net in [
        ("旧方案", "net_old", "pf_net_old"),
        ("方案A",  "net_A",   "pf_net_A"),
        ("方案B",  "net_B",   "pf_net_B"),
    ]:
        n_v  = tot[net]
        pf_v = tot[pf_net] if pf_net in tot.index else float("nan")
        ann  = n_v  / n * 365
        ann_pf = pf_v / n * 365
        print(f"  全年外推 {tag}：净={ann/1e8:.3f}亿元，PF净={ann_pf/1e8:.3f}亿元")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", required=True, help="strategy_result_nodaycross.csv 路径")
    ap.add_argument("--pred",     required=True, help="test_predictions_hourly.csv 路径")
    ap.add_argument("--out",      default=None,  help="输出详细结果 CSV（可选）")
    ap.add_argument("--label",    default="效率方案对比",  help="报告标题")
    args = ap.parse_args()

    df = compute_scenarios(Path(args.strategy), Path(args.pred))

    if args.out:
        df.to_csv(args.out, index=False, encoding="utf-8-sig")
        print(f"详细结果已保存：{args.out}")

    print_report(df, args.label)


if __name__ == "__main__":
    main()
