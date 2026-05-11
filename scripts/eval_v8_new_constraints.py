"""
V8 baseline 在新约束下的 81 天充放电收益重算
==============================================

约束变化（与历史 V8 评估对比）：
  - 容量：800 MWh → 400 MWh
  - 日循环上限：1.5 次循环（1200 MWh） → 取消（每天不限循环数）
  - 容量补偿：无 → 每放电 1 MWh 补偿 350 元（= 0.35 元/kWh）

保持不变：
  - 功率上限   P_MAX_MW = 195 MW
  - 双程效率   ETA_RT = 0.91
  - 辅助用电   AUX_MWH = 13.03 MWh/天
  - 跨日 SOC（与历史 V8 评估口径一致 carry_soc=True）

输出：
  output/experiments/v8.0-new-constraints/daily.csv      逐日明细（旧/新两版）
  output/experiments/v8.0-new-constraints/summary.txt    81 天合计
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.strategy_milp_15min import (  # noqa: E402
    solve_day_milp_15min,
    solve_pf_day_15min,
    eval_day_revenue_15min,
    load_actual_15min,
    AUX_MWH,
)

logger = logging.getLogger(__name__)

# ── 新/旧约束 ─────────────────────────────────────────────────────────────
OLD_CAP_MWH        = 800.0
OLD_MAX_CHARGE_MWH = 1200.0   # 1.5 × 800

NEW_CAP_MWH        = 400.0
NEW_MAX_CHARGE_MWH = 2400.0   # 远超物理极限，等效不限循环
CAP_COMP_PER_MWH   = 350.0    # 直接进 MILP 目标函数（CBC 可稳定求解）
TIME_LIMIT_SEC     = 30.0     # CBC 单天时间上限（秒），gap=5% 下通常 <30s 收敛


def _expand_24_to_96(arr_24: np.ndarray) -> np.ndarray:
    return np.repeat(arr_24, 4)


def _count_cycles(c_96: np.ndarray, cap_mwh: float) -> float:
    """以 充电量 / 容量 估算等效循环次数。"""
    DT = 0.25
    return float((c_96 * DT).sum() / cap_mwh) if cap_mwh > 0 else 0.0


def _solve_one_day(pred_96, actual_96, cap_mwh, max_charge_mwh,
                   soc_init_pred, soc_init_pf, force_end, next_avg,
                   cap_comp_per_mwh: float = 0.0,
                   time_limit: float = 120.0,
                   date_label: str = ""):
    """对一天求解：策略侧用预测、PF 侧用真实。

    cap_comp_per_mwh 直接进入 MILP 目标函数，CBC 求解器自动决定最优循环次数。

    返回 (c, d, soc, c_pf, d_pf, soc_pf)
    """
    common = dict(
        cap_mwh=cap_mwh, max_charge_mwh=max_charge_mwh,
        force_zero_end=force_end,
        cap_comp_per_mwh=cap_comp_per_mwh,
        time_limit=time_limit,
    )
    c, d, soc = solve_day_milp_15min(
        pred_96,
        soc_init=soc_init_pred, next_day_avg_price=next_avg,
        **common,
    )
    c_pf, d_pf, soc_pf = solve_pf_day_15min(
        actual_96,
        soc_init=soc_init_pf,
        next_day_avg_price=float(np.mean(actual_96)),
        **common,
    )
    return c, d, soc, c_pf, d_pf, soc_pf


def _eval_with_compensation(c, d, actual_96, cap_comp_per_mwh):
    """套利净 + 容量补偿。复用 eval_day_revenue_15min 算套利部分。"""
    rev = eval_day_revenue_15min(c, d, actual_96)
    cap_comp = rev["discharge_mwh"] * cap_comp_per_mwh
    rev["cap_comp"] = round(cap_comp, 0)
    rev["net_total"] = round(rev["net"] + cap_comp, 0)
    return rev


def run(
    pred_csv: Path,
    actual_xlsx: Path,
    out_dir: Path,
    start: str = "2026-01-27",
    end: str = "2026-04-17",
    smoke_days: int | None = None,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    pred = pd.read_csv(pred_csv, parse_dates=["ts"])
    pred["date"] = pred["ts"].dt.date.astype(str)
    pred = pred[(pred["date"] >= start) & (pred["date"] <= end)]

    actual_df = load_actual_15min(actual_xlsx)

    dates = sorted(pred["date"].unique())
    if smoke_days:
        dates = dates[:smoke_days]
    logger.info("=" * 60)
    logger.info("V8 新约束评估 | %d 天 (%s ~ %s)", len(dates), dates[0], dates[-1])
    logger.info("旧约束: 800 MWh 容量, ≤1.5 循环, 无容量补偿")
    logger.info("新约束: 400 MWh 容量, 不限循环, 容量补偿 350 元/MWh")
    logger.info("=" * 60)

    # 跨日 SOC（old / new 各自独立）
    soc_old      = 0.0
    soc_old_pf   = 0.0
    soc_new      = 0.0
    soc_new_pf   = 0.0

    rows = []
    for i, date in enumerate(dates):
        if date not in actual_df.index:
            logger.warning("  [%s] 缺真实价格，跳过", date)
            continue
        day_pred = pred[pred["date"] == date].sort_values("ts")
        if len(day_pred) < 24:
            logger.warning("  [%s] 预测点不足 24 (got %d)，跳过", date, len(day_pred))
            continue
        pred_24 = day_pred["pred"].values[:24].astype(float)
        pred_96 = _expand_24_to_96(pred_24)
        actual_96 = actual_df.loc[date].values.astype(float)

        is_last = (i == len(dates) - 1)
        force_end = is_last
        if not is_last and dates[i + 1] in pred["date"].values:
            next_p = pred[pred["date"] == dates[i + 1]]["pred"].values
            next_avg = float(np.mean(next_p)) if len(next_p) > 0 else 0.0
        else:
            next_avg = 0.0

        # 旧约束（标准 MILP，无补偿）
        c_o, d_o, soc_o, c_o_pf, d_o_pf, soc_o_pf = _solve_one_day(
            pred_96, actual_96, OLD_CAP_MWH, OLD_MAX_CHARGE_MWH,
            soc_old, soc_old_pf, force_end, next_avg,
            cap_comp_per_mwh=0.0, time_limit=TIME_LIMIT_SEC,
            date_label=date,
        )
        # 新约束：cap_comp=350 直接进 MILP 目标，CBC 自动决定最优循环次数
        c_n, d_n, soc_n, c_n_pf, d_n_pf, soc_n_pf = _solve_one_day(
            pred_96, actual_96, NEW_CAP_MWH, NEW_MAX_CHARGE_MWH,
            soc_new, soc_new_pf, force_end, next_avg,
            cap_comp_per_mwh=CAP_COMP_PER_MWH,
            time_limit=TIME_LIMIT_SEC,
            date_label=date,
        )

        # 跨日 SOC 更新
        soc_old    = float(soc_o[-1])
        soc_old_pf = float(soc_o_pf[-1])
        soc_new    = float(soc_n[-1])
        soc_new_pf = float(soc_n_pf[-1])

        # 评估
        rev_old    = eval_day_revenue_15min(c_o, d_o, actual_96)        # 旧：无补偿
        rev_old_pf = eval_day_revenue_15min(c_o_pf, d_o_pf, actual_96)
        rev_new    = _eval_with_compensation(c_n, d_n, actual_96, CAP_COMP_PER_MWH)
        rev_new_pf = _eval_with_compensation(c_n_pf, d_n_pf, actual_96, CAP_COMP_PER_MWH)

        cyc_new = _count_cycles(c_n, NEW_CAP_MWH)
        cyc_old = _count_cycles(c_o, OLD_CAP_MWH)

        rows.append({
            "date":              date,
            # ── 旧约束 ──
            "old_charge_mwh":    rev_old["charge_mwh"],
            "old_discharge_mwh": rev_old["discharge_mwh"],
            "old_cycles":        round(cyc_old, 2),
            "old_net":           rev_old["net"],
            "old_pf_net":        rev_old_pf["net"],
            # ── 新约束 ──
            "new_charge_mwh":    rev_new["charge_mwh"],
            "new_discharge_mwh": rev_new["discharge_mwh"],
            "new_cycles":        round(cyc_new, 2),
            "new_arb_net":       rev_new["net"],
            "new_cap_comp":      rev_new["cap_comp"],
            "new_net_total":     rev_new["net_total"],
            "new_pf_net_total":  rev_new_pf["net_total"],
        })

        if (i + 1) % 10 == 0 or i + 1 == len(dates):
            logger.info(
                "  [%s] (%d/%d) 旧:净=%.1fk  新:套利=%.1fk 补偿=%.1fk 合计=%.1fk 循环=%.1f×",
                date, i + 1, len(dates),
                rev_old["net"] / 1e3,
                rev_new["net"] / 1e3, rev_new["cap_comp"] / 1e3,
                rev_new["net_total"] / 1e3, cyc_new,
            )

    daily_df = pd.DataFrame(rows)
    plot_cols = [c for c in daily_df.columns if c.startswith("_")]
    daily_df.drop(columns=plot_cols).to_csv(out_dir / "daily.csv", index=False)
    daily_df.to_pickle(out_dir / "daily_full.pkl")

    # ── 汇总 ──
    n = len(daily_df)
    summary = {
        "n_days":                 n,
        "old_charge_mwh_total":   float(daily_df["old_charge_mwh"].sum()),
        "old_discharge_mwh_total":float(daily_df["old_discharge_mwh"].sum()),
        "old_cycles_avg":         float(daily_df["old_cycles"].mean()),
        "old_net_total":          float(daily_df["old_net"].sum()),
        "old_pf_net_total":       float(daily_df["old_pf_net"].sum()),

        "new_charge_mwh_total":   float(daily_df["new_charge_mwh"].sum()),
        "new_discharge_mwh_total":float(daily_df["new_discharge_mwh"].sum()),
        "new_cycles_avg":         float(daily_df["new_cycles"].mean()),
        "new_arb_net_total":      float(daily_df["new_arb_net"].sum()),
        "new_cap_comp_total":     float(daily_df["new_cap_comp"].sum()),
        "new_net_total":          float(daily_df["new_net_total"].sum()),
        "new_pf_net_total":       float(daily_df["new_pf_net_total"].sum()),

        "old_realization":        (daily_df["old_net"].sum() /
                                   daily_df["old_pf_net"].sum())
                                  if daily_df["old_pf_net"].sum() != 0 else None,
        "new_realization":        (daily_df["new_net_total"].sum() /
                                   daily_df["new_pf_net_total"].sum())
                                  if daily_df["new_pf_net_total"].sum() != 0 else None,
    }

    lines = [
        "=" * 70,
        f"V8 baseline 81 天 — 新/旧约束对比",
        "=" * 70,
        f"日期范围      : {daily_df['date'].iloc[0]} ~ {daily_df['date'].iloc[-1]} ({n} 天)",
        f"功率上限      : 195 MW（不变）",
        f"双程效率      : 0.91（不变）",
        f"辅助用电      : {AUX_MWH} MWh/天（不变）",
        "",
        "── 旧约束（800 MWh / ≤1.5 循环 / 无补偿）──",
        f"  累计充电量    : {summary['old_charge_mwh_total']:>12,.1f} MWh",
        f"  累计放电量    : {summary['old_discharge_mwh_total']:>12,.1f} MWh",
        f"  日均循环数    : {summary['old_cycles_avg']:>12.2f} 次",
        f"  累计净收益    : {summary['old_net_total']:>12,.0f} 元 = {summary['old_net_total']/1e4:>8.1f} 万元",
        f"  PF 净收益     : {summary['old_pf_net_total']:>12,.0f} 元 = {summary['old_pf_net_total']/1e4:>8.1f} 万元",
        f"  兑现率        : {(summary['old_realization'] or 0)*100:>12.1f} %",
        "",
        "── 新约束（400 MWh / 不限循环 / 350元/MWh放电补偿）──",
        f"  累计充电量    : {summary['new_charge_mwh_total']:>12,.1f} MWh",
        f"  累计放电量    : {summary['new_discharge_mwh_total']:>12,.1f} MWh",
        f"  日均循环数    : {summary['new_cycles_avg']:>12.2f} 次",
        f"  套利净收益    : {summary['new_arb_net_total']:>12,.0f} 元 = {summary['new_arb_net_total']/1e4:>8.1f} 万元",
        f"  容量补偿合计  : {summary['new_cap_comp_total']:>12,.0f} 元 = {summary['new_cap_comp_total']/1e4:>8.1f} 万元",
        f"  总净收益      : {summary['new_net_total']:>12,.0f} 元 = {summary['new_net_total']/1e4:>8.1f} 万元",
        f"  PF 总净收益   : {summary['new_pf_net_total']:>12,.0f} 元 = {summary['new_pf_net_total']/1e4:>8.1f} 万元",
        f"  兑现率        : {(summary['new_realization'] or 0)*100:>12.1f} %",
        "",
        "── 收益变化 ──",
        f"  Δ总净收益     : {(summary['new_net_total']-summary['old_net_total'])/1e4:>+8.1f} 万元 "
        f"({(summary['new_net_total']/summary['old_net_total']-1)*100 if summary['old_net_total'] else 0:+6.1f} %)",
        f"  其中：套利变化 : {(summary['new_arb_net_total']-summary['old_net_total'])/1e4:>+8.1f} 万元",
        f"        补偿增加 : {summary['new_cap_comp_total']/1e4:>+8.1f} 万元",
        "=" * 70,
    ]
    summary_text = "\n".join(lines)
    print("\n" + summary_text)
    with open(out_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)

    pd.Series(summary).to_csv(out_dir / "summary.csv")
    logger.info("结果保存: %s", out_dir)
    return daily_df, summary


# ── 绘图 ────────────────────────────────────────────────────────────────────
def _week_label(date_str: str) -> str:
    ts = pd.Timestamp(date_str)
    iso = ts.isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def plot_weekly(df: pd.DataFrame, out_dir: Path):
    """为新约束充放电画图，按周保存 PNG。

    每天画一张子图（上下两面板）：
      上面板：新约束充放电功率 + SOC + 电价
      下面板不再需要——只画新约束即可（旧约束数据也保留以便未来对比）

    Args:
        df: run() 返回的 DataFrame（含 _new_c/_new_d/_new_soc/_actual/_pred 列）
        out_dir: 输出目录，按周存放 PNG
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.font_manager as fm

    font_path = "/usr/share/fonts/wqy-microhei/wqy-microhei.ttc"
    if Path(font_path).exists():
        fm.fontManager.addfont(font_path)
        prop = fm.FontProperties(fname=font_path)
        plt.rcParams["font.family"] = prop.get_name()
    plt.rcParams["axes.unicode_minus"] = False
    FS = 8.5

    out_dir.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df["week"] = df["date"].apply(_week_label)

    t_axis = np.arange(96) * 15 / 60  # 0~24h

    for wk, wdf in df.groupby("week", sort=False):
        days = wdf["date"].tolist()
        n = len(days)
        fig, axes = plt.subplots(n, 1, figsize=(16, 3.6 * n), constrained_layout=True)
        if n == 1:
            axes = [axes]
        fig.suptitle(f"新约束（400MWh / 不限循环 / 350元补偿）充放电决策  {wk}",
                     fontsize=11, fontweight="bold")

        for ax, (_, r) in zip(axes, wdf.iterrows()):
            actual_96 = np.array(r["_actual"])
            pred_96   = np.array(r["_pred"])
            c_new     = np.array(r["_new_c"])
            d_new     = np.array(r["_new_d"])
            soc_new   = np.array(r["_new_soc"])

            # 充放电背景色块
            for t in range(96):
                x0, x1 = t * 15 / 60, (t + 1) * 15 / 60
                if c_new[t] > 0.5:
                    ax.axvspan(x0, x1, color="#BBDEFB", alpha=0.7, zorder=0)
                if d_new[t] > 0.5:
                    ax.axvspan(x0, x1, color="#FFCDD2", alpha=0.7, zorder=0)

            mid = t_axis + 15 / 120
            ax.plot(mid, actual_96, color="#1565C0", lw=1.5,
                    label="实际价(15min)", alpha=0.9, zorder=3)
            ax.plot(mid, pred_96, color="#E53935", lw=1.2, ls="--",
                    label="预测价", alpha=0.8, zorder=3)

            # 功率 + SOC 次轴
            ax2 = ax.twinx()
            bar_w = 14 / 60
            ax2.bar(mid, c_new,  width=bar_w, color="#1565C0", alpha=0.3,
                    label="充电(MW)", zorder=2)
            ax2.bar(mid, -d_new, width=bar_w, color="#C62828", alpha=0.3,
                    label="放电(MW)", zorder=2)
            if soc_new.max() > 1:
                ax2.plot(mid, soc_new, color="#2E7D32", lw=1.2, ls=":",
                         alpha=0.85, label="SOC(MWh)", zorder=4)
            ax2.set_ylim(-280, max(280, NEW_CAP_MWH * 1.1))
            ax2.set_ylabel("功率(MW) / SOC(MWh)", fontsize=FS - 1, color="#888")
            ax2.tick_params(labelsize=FS - 1, colors="#888")
            ax2.axhline(0, color="#aaa", lw=0.5, ls=":")

            # 标题
            cyc = r["new_cycles"]
            net_w = r["new_net_total"] / 1e4
            comp_w = r["new_cap_comp"] / 1e4
            arb_w = r["new_arb_net"] / 1e4
            soc_end = float(soc_new[-1])
            ax.set_title(
                f"{r['date']}  充{r['new_charge_mwh']:.0f}MWh 放{r['new_discharge_mwh']:.0f}MWh "
                f"循环{cyc:.1f}×  套利{arb_w:+.1f}万 补偿{comp_w:.1f}万 "
                f"合计{net_w:+.1f}万  SOC末{soc_end:.0f}",
                fontsize=FS, loc="left", pad=3)

            ax.set_xlim(0, 24)
            ax.set_xticks(range(0, 25, 2))
            ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 2)], fontsize=FS - 1)
            ax.set_ylabel("电价 (元/MWh)", fontsize=FS)
            ax.tick_params(labelsize=FS - 1)
            ax.grid(axis="y", ls=":", alpha=0.4)
            ax.set_facecolor("#FAFAFA")

            if r["date"] == days[0]:
                h1, l1 = ax.get_legend_handles_labels()
                patch_c = mpatches.Patch(color="#BBDEFB", label="充电时段")
                patch_d = mpatches.Patch(color="#FFCDD2", label="放电时段")
                ax.legend(handles=h1 + [patch_c, patch_d],
                          labels=l1 + ["充电时段", "放电时段"],
                          loc="upper right", fontsize=FS - 1, framealpha=0.85, ncol=4)

        out_path = out_dir / f"{wk}.png"
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        logger.info("  绘图: %s", out_path.name)

    logger.info("绘图完成，保存至 %s", out_dir)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    p = argparse.ArgumentParser()
    p.add_argument("--pred-csv",
                   default=str(ROOT / "output" / "experiments" / "v8.0-12m-sudun500" /
                               "test_predictions_hourly.csv"))
    p.add_argument("--actual-xlsx",
                   default=str(ROOT / "source_data" / "日清算结果查询电厂侧(1)_副本.xlsx"))
    p.add_argument("--out-dir",
                   default=str(ROOT / "output" / "experiments" / "v8.0-new-constraints"))
    p.add_argument("--start", default="2026-01-27")
    p.add_argument("--end",   default="2026-04-17")
    p.add_argument("--smoke", type=int, default=None,
                   help="只跑前 N 天烟测")
    p.add_argument("--plot-only", action="store_true",
                   help="仅绘图（需已有 daily_full.pkl）")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    if args.plot_only:
        pkl_path = out_dir / "daily_full.pkl"
        if not pkl_path.exists():
            print(f"错误：{pkl_path} 不存在，请先运行完整评估")
            sys.exit(1)
        daily_df = pd.read_pickle(pkl_path)
        plot_weekly(daily_df, out_dir / "plots_new")
    else:
        daily_df, summary = run(
            pred_csv=Path(args.pred_csv),
            actual_xlsx=Path(args.actual_xlsx),
            out_dir=out_dir,
            start=args.start, end=args.end,
            smoke_days=args.smoke,
        )
        plot_weekly(daily_df, out_dir / "plots_new")
