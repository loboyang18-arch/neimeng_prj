"""
导出看板所需数据文件
=====================
从 MILP 策略（跨日SOC模式）中提取并整理以下文件到 output/dashboard/ ：

  15min_timeseries.csv  — 每日 96 个15分钟时段：充放电功率、SOC、预测/实际电价、时段收益
  daily_summary.csv     — 逐日汇总：充放电量、窗口、收益、PF对照
  weekly_summary.csv    — 逐周对比：策略收益 vs 完全预知 vs 实际现货
  model_predictions.csv — 小时级预测电价（长格式，附实际值）

运行方式：
  conda run -n power python scripts/export_dashboard_data.py
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# ── 路径配置 ─────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PRED_CSV      = ROOT / "output/experiments/v8.0-jan25-sudun500/test_predictions_hourly.csv"
ACTUAL_XLSX   = ROOT / "source_data/日清算结果查询电厂侧(1)_副本.xlsx"
ACTUAL_REV    = ROOT / "output/actual_spot_revenue_jan27_apr17.csv"
OUT_DIR       = ROOT / "output/dashboard"

# 有效回测区间（排除 Jan 25-26 调试日和 Apr 18 不完整放电日）
PERIOD_START  = "2026-01-27"
PERIOD_END    = "2026-04-17"

# 统计区间（排除1月25-26调试日及4月18日不完整放电日）
PERIOD_START  = "2026-01-27"
PERIOD_END    = "2026-04-17"

# ── 导入 MILP 策略模块 ────────────────────────────────────────────────────────
from scripts.strategy_milp_15min import run as milp_run, _week_label


def expand_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """
    将 run() 返回的 DataFrame（含 _c/_d/_soc/_actual/_pred 列表列）
    展开为每行一个15分钟时段的长格式表。
    """
    DT = 0.25  # h

    records = []
    for _, row in df.iterrows():
        date   = row["date"]
        c_arr  = np.array(row["_c"],      dtype=float)
        d_arr  = np.array(row["_d"],      dtype=float)
        soc_arr= np.array(row["_soc"],    dtype=float)
        act_arr= np.array(row["_actual"], dtype=float)
        pred_arr=np.array(row["_pred"],   dtype=float)

        for t in range(96):
            hh  = t * 15 // 60
            mm  = t * 15 % 60
            dt  = pd.Timestamp(f"{date} {hh:02d}:{mm:02d}")

            c   = float(c_arr[t])
            d   = float(d_arr[t])
            soc = float(soc_arr[t])
            p_a = float(act_arr[t])
            p_p = float(pred_arr[t])

            e_c = c * DT   # MWh
            e_d = d * DT   # MWh
            slot_rev = (e_d - e_c) * p_a  # 元（正=收益，负=成本）

            if c > 0.5:
                state = "充电"
            elif d > 0.5:
                state = "放电"
            else:
                state = "待机"

            records.append({
                "datetime":          dt,
                "date":              date,
                "slot":              t,
                "hour":              hh,
                "minute":            mm,
                "state":             state,
                "charge_mw":         round(c, 3),
                "discharge_mw":      round(d, 3),
                "net_power_mw":      round(d - c, 3),   # 正=放电，负=充电
                "soc_mwh":           round(soc, 2),
                "charge_energy_mwh": round(e_c, 4),
                "discharge_energy_mwh": round(e_d, 4),
                "pred_price":        round(p_p, 2),
                "actual_price":      round(p_a, 2),
                "slot_revenue":      round(slot_rev, 2),
            })

    ts = pd.DataFrame(records)
    ts.sort_values(["date", "slot"], inplace=True)
    ts.reset_index(drop=True, inplace=True)
    return ts


def build_daily_summary(df: pd.DataFrame) -> pd.DataFrame:
    """整理逐日汇总表，附加加权平均充放电价格。"""
    result = []
    for _, row in df.iterrows():
        c_arr   = np.array(row["_c"],      dtype=float)
        d_arr   = np.array(row["_d"],      dtype=float)
        act_arr = np.array(row["_actual"], dtype=float)
        DT = 0.25

        e_c = c_arr * DT
        e_d = d_arr * DT

        avg_chg_price = (float(np.dot(act_arr, e_c)) / e_c.sum()
                         if e_c.sum() > 0 else float("nan"))
        avg_dis_price = (float(np.dot(act_arr, e_d)) / e_d.sum()
                         if e_d.sum() > 0 else float("nan"))

        realization = (row["net"] / row["pf_net"]
                       if abs(row["pf_net"]) > 1 else float("nan"))

        result.append({
            "date":               row["date"],
            "week":               _week_label(row["date"]),
            "charge_window":      row["charge_window"],
            "discharge_window":   row["discharge_window"],
            "charge_mwh":         row["charge_mwh"],
            "discharge_mwh":      row["discharge_mwh"],
            "soc_end_mwh":        row.get("soc_end", 0),
            "avg_charge_price":   round(avg_chg_price, 2),
            "avg_discharge_price":round(avg_dis_price, 2),
            "gross_yuan":         row["gross"],
            "aux_cost_yuan":      row["aux_cost"],
            "net_yuan":           row["net"],
            "pf_net_yuan":        row["pf_net"],
            "realization_pct":    round(realization * 100, 1) if not np.isnan(realization) else None,
        })

    return pd.DataFrame(result)


def build_weekly_summary(daily: pd.DataFrame,
                         actual_rev: pd.DataFrame) -> pd.DataFrame:
    """逐周汇总，合并实际现货收益。"""
    # 按周聚合策略数据
    agg = (daily.groupby("week")
           .agg(
               days         = ("date", "count"),
               charge_mwh   = ("charge_mwh", "sum"),
               discharge_mwh= ("discharge_mwh","sum"),
               net_yuan     = ("net_yuan",   "sum"),
               pf_net_yuan  = ("pf_net_yuan","sum"),
           )
           .reset_index())

    # 按周聚合实际现货
    if actual_rev is not None:
        actual_wk = (actual_rev.groupby("week")["net_revenue"]
                     .sum().reset_index()
                     .rename(columns={"net_revenue": "actual_spot_yuan"}))
        agg = agg.merge(actual_wk, on="week", how="left")
    else:
        agg["actual_spot_yuan"] = None

    agg["realization_pct"] = (agg["net_yuan"] / agg["pf_net_yuan"] * 100).round(1)
    agg["vs_actual_yuan"]  = (agg["net_yuan"] - agg["actual_spot_yuan"]).round(0)
    agg["net_wan"]         = (agg["net_yuan"] / 1e4).round(2)
    agg["pf_net_wan"]      = (agg["pf_net_yuan"] / 1e4).round(2)
    agg["actual_spot_wan"] = (agg["actual_spot_yuan"] / 1e4).round(2)
    agg["vs_actual_wan"]   = (agg["vs_actual_yuan"] / 1e4).round(2)

    return agg[[
        "week", "days", "charge_mwh", "discharge_mwh",
        "net_wan", "pf_net_wan", "actual_spot_wan",
        "realization_pct", "vs_actual_wan",
        "net_yuan", "pf_net_yuan", "actual_spot_yuan",
    ]]


def build_predictions(pred_csv: Path) -> pd.DataFrame:
    """整理模型小时级预测结果（长格式）。"""
    df = pd.read_csv(pred_csv, parse_dates=["ts"])
    df["date"]     = df["ts"].dt.date.astype(str)
    df["hour"]     = df["ts"].dt.hour
    df["week"]     = df["date"].apply(_week_label)
    df["error"]    = (df["pred"] - df["actual"]).round(2)
    df["abs_error"]= df["error"].abs().round(2)
    df = df.rename(columns={
        "ts":     "datetime",
        "actual": "actual_price",
        "pred":   "pred_price",
    })
    return df[["datetime","date","week","hour","actual_price","pred_price","error","abs_error"]]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"输出目录：{OUT_DIR}")

    # ── 运行 MILP（跨日SOC模式）────────────────────────────────────────────
    print("\n[1/4] 运行 MILP 策略（跨日SOC模式）…")
    df_result = milp_run(
        pred_csv    = PRED_CSV,
        actual_xlsx = ACTUAL_XLSX,
        out_csv     = None,     # 不保存日摘要（下面单独处理）
        label       = "dashboard-export",
        carry_soc   = True,
    )

    # ── 过滤有效回测区间 ─────────────────────────────────────────────────────
    df_result = df_result[
        (df_result["date"] >= PERIOD_START) & (df_result["date"] <= PERIOD_END)
    ].reset_index(drop=True)
    print(f"\n  过滤至 {PERIOD_START} ～ {PERIOD_END}，共 {len(df_result)} 天")

    # ── 展开15分钟时序 ──────────────────────────────────────────────────────
    print("\n[2/4] 展开15分钟时序数据…")
    ts = expand_timeseries(df_result)
    out_ts = OUT_DIR / "15min_timeseries.csv"
    ts.to_csv(out_ts, index=False, encoding="utf-8-sig")
    print(f"  已保存：{out_ts}  ({len(ts)} 行 × {len(ts.columns)} 列)")

    # ── 逐日汇总 ────────────────────────────────────────────────────────────
    print("\n[3/4] 整理逐日汇总…")
    daily = build_daily_summary(df_result)
    out_daily = OUT_DIR / "daily_summary.csv"
    daily.to_csv(out_daily, index=False, encoding="utf-8-sig")
    print(f"  已保存：{out_daily}  ({len(daily)} 行)")

    # ── 逐周汇总 ────────────────────────────────────────────────────────────
    print("\n[4/4] 整理逐周汇总（合并实际现货）…")
    actual_rev = None
    if ACTUAL_REV.exists():
        actual_rev = pd.read_csv(ACTUAL_REV)
    weekly = build_weekly_summary(daily, actual_rev)
    out_weekly = OUT_DIR / "weekly_summary.csv"
    weekly.to_csv(out_weekly, index=False, encoding="utf-8-sig")
    print(f"  已保存：{out_weekly}  ({len(weekly)} 行)")

    # ── 预测电价（长格式）──────────────────────────────────────────────────
    out_pred = OUT_DIR / "model_predictions.csv"
    pred_df  = build_predictions(PRED_CSV)
    pred_df.to_csv(out_pred, index=False, encoding="utf-8-sig")
    print(f"\n  已保存：{out_pred}  ({len(pred_df)} 行)")

    # ── 输出预览 ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("看板数据文件一览")
    print("=" * 60)
    for f in sorted(OUT_DIR.glob("*.csv")):
        size_kb = f.stat().st_size / 1024
        rows    = sum(1 for _ in open(f, encoding="utf-8-sig")) - 1
        print(f"  {f.name:<35} {rows:>6} 行  {size_kb:>8.1f} KB")

    print("\n总览（逐周）:")
    print(weekly[["week","net_wan","pf_net_wan","actual_spot_wan",
                  "realization_pct","vs_actual_wan"]].to_string(index=False))

    total = weekly[["net_yuan","pf_net_yuan","actual_spot_yuan"]].sum()
    print(f"\n合计：策略净收益 {total['net_yuan']/1e4:.1f}万"
          f"  PF {total['pf_net_yuan']/1e4:.1f}万"
          f"  实际现货 {total['actual_spot_yuan']/1e4:.1f}万")


if __name__ == "__main__":
    main()
