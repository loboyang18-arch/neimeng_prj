#!/usr/bin/env python3
"""
实际运营（日清算 XLSX）逐周绘图：15min 实际电价 + 实际充放功率，
并叠加与 MILP 周图一致的「小时预测价」阶梯曲线。

默认输出目录：output/experiments/v8.0-jan25-sudun500/plots_actual_with_pred
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.compute_performance_metrics import (  # noqa: E402
    load_actual,
    _week_label,
)

DT_H = 0.25  # 15min → h
SLOT_EPS = 0.01  # MWh/15min，判定有充/放


def _window_str(slots: list[int]) -> str:
    if not slots:
        return "–"
    t0, t1 = slots[0], slots[-1]
    return (
        f"{t0 * 15 // 60:02d}:{t0 * 15 % 60:02d}–"
        f"{(t1 + 1) * 15 // 60:02d}:{(t1 + 1) * 15 % 60:02d}"
    )


def build_daily_frame(
    pred_csv: Path,
    actual_xlsx: Path,
    rev_csv: Path,
    start: str,
    end: str,
) -> pd.DataFrame:
    pred = pd.read_csv(pred_csv, parse_dates=["ts"])
    pred["date"] = pred["ts"].dt.date.astype(str)
    pred = pred[(pred["date"] >= start) & (pred["date"] <= end)]

    actual = load_actual(actual_xlsx, rev_csv, start=start, end=end)
    rows = []
    for date in sorted(actual.keys()):
        day_pred = pred[pred["date"] == date].sort_values("ts")
        if len(day_pred) < 24:
            continue
        pred_h = day_pred["pred"].values.astype(float)
        pred_96 = np.repeat(pred_h, 4)

        d = actual[date]
        actual_96 = d["prices"].astype(float)
        chg = d["chg"].astype(float)
        dis = d["dis"].astype(float)
        c_mw = chg / DT_H
        d_mw = dis / DT_H

        chg_slots = [t for t in range(96) if chg[t] > SLOT_EPS]
        dis_slots = [t for t in range(96) if dis[t] > SLOT_EPS]

        rows.append({
            "date": date,
            "charge_window": _window_str(chg_slots),
            "discharge_window": _window_str(dis_slots),
            "charge_mwh": float(np.nansum(chg)),
            "discharge_mwh": float(np.nansum(dis)),
            "net": d["net"],
            "_c": c_mw.tolist(),
            "_d": d_mw.tolist(),
            "_actual": actual_96.tolist(),
            "_pred": pred_96.tolist(),
        })

    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
    df["week"] = df["date"].apply(_week_label)
    return df


def plot_weekly(df: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    import matplotlib.patches as mpatches

    font_path = "/usr/share/fonts/wqy-microhei/wqy-microhei.ttc"
    fm.fontManager.addfont(font_path)
    prop = fm.FontProperties(fname=font_path)
    plt.rcParams["font.family"] = prop.get_name()
    plt.rcParams["axes.unicode_minus"] = False
    FS = 8.5

    out_dir.mkdir(parents=True, exist_ok=True)
    df = df.copy()

    t_axis = np.arange(96) * 15 / 60.0

    for wk, wdf in df.groupby("week", sort=False):
        days = wdf["date"].tolist()
        n = len(days)
        fig, axes = plt.subplots(n, 1, figsize=(15, 3.4 * n), constrained_layout=True)
        if n == 1:
            axes = [axes]
        fig.suptitle(f"实际运营 充放电与电价对比（含预测价）  {wk}", fontsize=11, fontweight="bold")

        for ax, (_, r) in zip(axes, wdf.iterrows()):
            actual_96 = np.array(r["_actual"])
            pred_96 = np.array(r["_pred"])
            c = np.array(r["_c"])
            d = np.array(r["_d"])

            for t in range(96):
                x0 = t * 15 / 60.0
                x1 = (t + 1) * 15 / 60.0
                if c[t] > 1e-3:
                    ax.axvspan(x0, x1, color="#BBDEFB", alpha=0.75, zorder=0)
                if d[t] > 1e-3:
                    ax.axvspan(x0, x1, color="#FFCDD2", alpha=0.75, zorder=0)

            ax.plot(
                t_axis + 15 / 120,
                actual_96,
                color="#1565C0",
                lw=1.5,
                label="实际价(15min)",
                alpha=0.9,
                zorder=3,
            )
            ax.step(
                np.arange(24) + 0.5,
                np.repeat(pred_96[::4], 1),
                color="#E53935",
                lw=1.4,
                ls="--",
                label="预测价(小时)",
                where="mid",
                zorder=3,
                alpha=0.85,
            )

            ax2 = ax.twinx()
            bar_w = 14 / 60.0
            ax2.bar(t_axis + 15 / 120, c, width=bar_w, color="#1565C0", alpha=0.3, label="充电(MW)", zorder=2)
            ax2.bar(t_axis + 15 / 120, -d, width=bar_w, color="#C62828", alpha=0.3, label="放电(MW)", zorder=2)

            pmax = float(np.nanmax(np.abs(np.concatenate([c, d])))) if len(c) else 50.0
            lim = max(280.0, pmax * 1.15)
            ax2.set_ylim(-lim, lim)
            ax2.set_ylabel("功率(MW)", fontsize=FS - 1, color="#888888")
            ax2.tick_params(labelsize=FS - 1, colors="#888888")
            ax2.axhline(0, color="#aaaaaa", lw=0.5, ls=":")

            ax.set_title(
                f"{r['date']}   充: {r['charge_window']} {r['charge_mwh']:.0f}MWh  "
                f"放: {r['discharge_window']} {r['discharge_mwh']:.0f}MWh  "
                f"净收益: {r['net']/1e4:+.2f}万",
                fontsize=FS,
                loc="left",
                pad=3,
            )
            ax.set_xlim(0, 24)
            ax.set_xticks(range(0, 25, 2))
            ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 2)], fontsize=FS - 1)
            ax.set_ylabel("节点电价 (元/MWh)", fontsize=FS)
            ax.tick_params(labelsize=FS - 1)
            ax.grid(axis="y", ls=":", alpha=0.4)
            ax.set_facecolor("#FAFAFA")

            if r["date"] == days[0]:
                h1, l1 = ax.get_legend_handles_labels()
                patch_c = mpatches.Patch(color="#BBDEFB", label="充电时段")
                patch_d = mpatches.Patch(color="#FFCDD2", label="放电时段")
                ax.legend(
                    handles=h1 + [patch_c, patch_d],
                    labels=l1 + ["充电时段", "放电时段"],
                    loc="upper right",
                    fontsize=FS - 1,
                    framealpha=0.85,
                    ncol=4,
                )

        out_path = out_dir / f"{wk}.png"
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  {out_path.name}")


def main() -> None:
    ap = argparse.ArgumentParser(description="实际运营周图 + 预测电价")
    ap.add_argument(
        "--pred",
        type=Path,
        default=ROOT / "output/experiments/v8.0-jan25-sudun500/test_predictions_hourly.csv",
        help="小时级预测 CSV（含 ts, pred）",
    )
    ap.add_argument(
        "--actual_xlsx",
        type=Path,
        default=ROOT / "source_data/日清算结果查询电厂侧(1)_副本.xlsx",
    )
    ap.add_argument(
        "--rev_csv",
        type=Path,
        default=ROOT / "output/actual_spot_revenue_jan27_apr17.csv",
    )
    ap.add_argument("--start", default="2026-01-27")
    ap.add_argument("--end", default="2026-04-17")
    ap.add_argument(
        "--plots",
        type=Path,
        default=ROOT / "output/experiments/v8.0-jan25-sudun500/plots_actual_with_pred",
        help="PNG 输出目录",
    )
    args = ap.parse_args()

    print("构建逐日数据 …")
    df = build_daily_frame(
        pred_csv=args.pred,
        actual_xlsx=args.actual_xlsx,
        rev_csv=args.rev_csv,
        start=args.start,
        end=args.end,
    )
    if df.empty:
        print("无可用日（检查预测是否覆盖、日清算是否 96 点、收益 CSV 是否含该日）。")
        sys.exit(1)
    print(f"共 {len(df)} 天，开始按周绘图 …")
    plot_weekly(df, args.plots)
    print(f"完成：{args.plots}")


if __name__ == "__main__":
    main()
