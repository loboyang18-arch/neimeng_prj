"""
导出三种策略的看板数据（中文表头 xlsx）
=========================================
为以下三种策略生成与现有 dashboard/{metrics_strategy_daily, strategy_15min_timeseries}.xlsx
完全同结构（中文表头 + 单位）的 6 个 xlsx 文件：

  1. 15min MILP（日清零）        — 调用 scripts.strategy_milp_15min.run(carry_soc=False)
  2. 小时级 MILP                  — 调用 scripts.strategy_milp.run()
  3. 启发式 4h 窗口（方案B）      — 从 strategy_result_nodaycross.csv 重建 96 时段轨迹

输出目录：output/dashboard/strategies/
  metrics_strategy_daily_{tag}.xlsx
  strategy_15min_timeseries_{tag}.xlsx

运行：
  conda run -n power python scripts/export_dashboard_strategies.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── 路径配置 ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PRED_CSV = ROOT / "output/experiments/v8.0-jan25-sudun500/test_predictions_hourly.csv"
ACTUAL_XLSX = ROOT / "source_data/日清算结果查询电厂侧(1)_副本.xlsx"
HEURISTIC_CSV = ROOT / "output/experiments/v8.0-jan25-sudun500/strategy_result_nodaycross.csv"

OUT_DIR = ROOT / "output/dashboard/strategies"
DASHBOARD_DIR = ROOT / "output/dashboard"

PERIOD_START = "2026-01-27"
PERIOD_END = "2026-04-18"

# ── 物理常数 ────────────────────────────────────────────────────────────────
DT = 0.25                   # 时段长度（h）
CAP_MWH = 800.0
P_MAX_MW = 195.0
AUX_MWH_PER_DAY = 13.03     # 方案 B 辅助用电

# 方案 B 启发式（与 scripts/strategy_efficiency_analysis.py 保持一致）
ETA_B = 0.899               # 双程效率（用于放电侧缩减）

# ── 复用底层模块 ─────────────────────────────────────────────────────────────
from scripts.strategy_milp_15min import (
    run as milp_15min_run,
    load_actual_15min,
)
from scripts.strategy_milp import run as milp_hourly_run
from scripts.export_dashboard_data import expand_timeseries


# ── 中文映射 ────────────────────────────────────────────────────────────────
TS_COLUMN_MAP = {
    "datetime": "时间",
    "date": "日期",
    "slot": "当日时段序号（0～95）",
    "hour": "hour",
    "minute": "minute",
    "state": "运行状态",
    "charge_mw": "本时段充电功率（电网侧输入）_MW",
    "discharge_mw": "本时段放电功率（电网侧输出）_MW",
    "net_power_mw": "净功率_MW",
    "soc_mwh": "本时段末电池荷电状态_MWh",
    "charge_energy_mwh": "本时段充电电量_MWh",
    "discharge_energy_mwh": "本时段放电电量_MWh",
    "pred_price": "预测节点电价（小时级）_元/MWh",
    "actual_price": "实际15分钟节点电价_元/MWh",
    "slot_revenue": "本时段净收益_元",
}

DAILY_COLUMNS_CN = [
    "日期",
    "当日充电量（电网侧）_MWh",
    "当日放电量（电网侧）_MWh",
    "充电成本_元",
    "放电收益_元",
    "毛收益_元",
    "辅助用电成本_元",
    "净收益_元",
    "日均循环次数_次",
    "平均循环收益_元/次",
    "充电加权均价_元/MWh",
    "放电加权均价_元/MWh",
    "度电收益_元/MWh",
    "平均充放电价差_元/MWh",
    "完全预知（PF）净收益_元",
]


# ── 通用：从 timeseries（96 行/天）汇总日级中文表 ───────────────────────────
def build_metrics_daily(ts: pd.DataFrame, pf_net_by_date: dict[str, float]) -> pd.DataFrame:
    """
    输入：
      ts            — expand_timeseries 输出的长格式表（96 行/天）
      pf_net_by_date— {date_str: pf_net_yuan}，由各策略的 PF 求解结果提供

    输出：
      DataFrame，列名为 DAILY_COLUMNS_CN
    """
    rows = []
    for date, g in ts.groupby("date", sort=True):
        c = g["charge_mw"].values        # MW
        d = g["discharge_mw"].values     # MW
        actual = g["actual_price"].values  # 元/MWh

        e_c = c * DT                     # MWh per slot
        e_d = d * DT

        charge_mwh = float(e_c.sum())
        discharge_mwh = float(e_d.sum())

        charge_cost = float((e_c * actual).sum())          # 元
        discharge_revenue = float((e_d * actual).sum())    # 元
        gross = discharge_revenue - charge_cost

        avg_actual_day = float(np.mean(actual))
        aux_cost = avg_actual_day * AUX_MWH_PER_DAY
        net = gross - aux_cost

        cycles = discharge_mwh / CAP_MWH if CAP_MWH > 0 else 0.0
        avg_cycle_rev = (gross / cycles) if cycles > 1e-9 else 0.0

        avg_charge_p = (charge_cost / charge_mwh) if charge_mwh > 1e-9 else float("nan")
        avg_dis_p = (discharge_revenue / discharge_mwh) if discharge_mwh > 1e-9 else float("nan")

        throughput = charge_mwh + discharge_mwh
        unit_rev = (gross / throughput) if throughput > 1e-9 else float("nan")
        spread = (avg_dis_p - avg_charge_p) if (charge_mwh > 0 and discharge_mwh > 0) else float("nan")

        pf_net = pf_net_by_date.get(date, float("nan"))

        rows.append({
            "日期": date,
            "当日充电量（电网侧）_MWh": round(charge_mwh, 2),
            "当日放电量（电网侧）_MWh": round(discharge_mwh, 2),
            "充电成本_元": round(charge_cost, 0),
            "放电收益_元": round(discharge_revenue, 0),
            "毛收益_元": round(gross, 0),
            "辅助用电成本_元": round(aux_cost, 0),
            "净收益_元": round(net, 0),
            "日均循环次数_次": round(cycles, 4),
            "平均循环收益_元/次": avg_cycle_rev,
            "充电加权均价_元/MWh": round(avg_charge_p, 2) if not np.isnan(avg_charge_p) else None,
            "放电加权均价_元/MWh": round(avg_dis_p, 2) if not np.isnan(avg_dis_p) else None,
            "度电收益_元/MWh": round(unit_rev, 2) if not np.isnan(unit_rev) else None,
            "平均充放电价差_元/MWh": round(spread, 2) if not np.isnan(spread) else None,
            "完全预知（PF）净收益_元": round(pf_net, 0) if not np.isnan(pf_net) else None,
        })
    df = pd.DataFrame(rows, columns=DAILY_COLUMNS_CN)
    df["日期"] = pd.to_datetime(df["日期"])
    return df


# ── 中文 timeseries：保持列顺序 + 时间字段为 datetime ────────────────────────
# 与基准 strategy_15min_timeseries.xlsx 对齐的列类型（防止 openpyxl 把整数值列推断为 int）
TS_FLOAT_COLS = [
    "本时段充电功率（电网侧输入）_MW",
    "本时段放电功率（电网侧输出）_MW",
    "净功率_MW",
    "本时段末电池荷电状态_MWh",
    "本时段充电电量_MWh",
    "本时段放电电量_MWh",
    "预测节点电价（小时级）_元/MWh",
    "实际15分钟节点电价_元/MWh",
    "本时段净收益_元",
]

DAILY_FLOAT_COLS = [
    "当日充电量（电网侧）_MWh",
    "当日放电量（电网侧）_MWh",
    "日均循环次数_次",
    "平均循环收益_元/次",
    "充电加权均价_元/MWh",
    "放电加权均价_元/MWh",
    "度电收益_元/MWh",
    "平均充放电价差_元/MWh",
]


def to_chinese_timeseries(ts: pd.DataFrame) -> pd.DataFrame:
    out = ts[list(TS_COLUMN_MAP.keys())].rename(columns=TS_COLUMN_MAP).copy()
    out["时间"] = pd.to_datetime(out["时间"])
    out["日期"] = pd.to_datetime(out["日期"])
    for c in TS_FLOAT_COLS:
        out[c] = out[c].astype(float)
    return out


# ── 区间裁剪 ────────────────────────────────────────────────────────────────
def _clip_period(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    return df[(df[date_col] >= PERIOD_START) & (df[date_col] <= PERIOD_END)].reset_index(drop=True)


# ── 策略 0：15min MILP（跨日 SOC）— 与 dashboard 主表一致 ───────────────────
def build_15min_carry_soc():
    print("\n[策略 0] 15min MILP（跨日 SOC）— 主 dashboard 策略")
    df = milp_15min_run(
        pred_csv=PRED_CSV,
        actual_xlsx=ACTUAL_XLSX,
        out_csv=None,
        label="dashboard-15min-跨日SOC",
        carry_soc=True,
    )
    df = _clip_period(df)
    print(f"  有效天数：{len(df)}")
    ts = expand_timeseries(df)
    pf_net_map = dict(zip(df["date"], df["pf_net"]))
    daily = build_metrics_daily(ts, pf_net_map)
    return ts, daily


# ── 策略 1：15min MILP（日清零）─────────────────────────────────────────────
def build_15min_daily_zero():
    print("\n[策略 1/3] 15min MILP（日清零）")
    df = milp_15min_run(
        pred_csv=PRED_CSV,
        actual_xlsx=ACTUAL_XLSX,
        out_csv=None,
        label="dashboard-15min-日清零",
        carry_soc=False,
    )
    df = _clip_period(df)
    print(f"  有效天数：{len(df)}")
    ts = expand_timeseries(df)
    pf_net_map = dict(zip(df["date"], df["pf_net"]))
    daily = build_metrics_daily(ts, pf_net_map)
    return ts, daily


# ── 策略 2：小时级 MILP ──────────────────────────────────────────────────────
def build_hourly_milp():
    print("\n[策略 2/3] 小时级 MILP")
    df_h = milp_hourly_run(pred_csv=PRED_CSV, out_csv=None, label="dashboard-小时MILP")
    df_h = _clip_period(df_h)
    print(f"  有效天数：{len(df_h)}")

    actual_15m = load_actual_15min(ACTUAL_XLSX)

    # 24 维 → 96 维：c/d/soc/pred 复制 4 次；actual 用真实 15 分钟价
    rows = []
    for _, row in df_h.iterrows():
        date = row["date"]
        if date not in actual_15m.index:
            print(f"  [{date}] 缺失 15min 真实价，跳过")
            continue
        c24 = np.asarray(row["_c"], float)
        d24 = np.asarray(row["_d"], float)
        soc24 = np.asarray(row["_soc"], float)
        pred24 = np.asarray(row["_pred"], float)
        actual96 = actual_15m.loc[date].values.astype(float)

        rows.append({
            "date": date,
            "_c": np.repeat(c24, 4).tolist(),
            "_d": np.repeat(d24, 4).tolist(),
            "_soc": np.repeat(soc24, 4).tolist(),
            "_pred": np.repeat(pred24, 4).tolist(),
            "_actual": actual96.tolist(),
            "pf_net": float(row["pf_net"]),
        })
    df_expanded = pd.DataFrame(rows)
    ts = expand_timeseries(df_expanded)
    pf_net_map = dict(zip(df_expanded["date"], df_expanded["pf_net"]))
    daily = build_metrics_daily(ts, pf_net_map)
    return ts, daily


# ── 策略 3：启发式 4h 窗口（方案B）──────────────────────────────────────────
def _hours_to_slots(start_h: int, end_h: int) -> list[int]:
    """charge_start..charge_end 闭区间（含 end_h），转成 15min slot。"""
    return list(range(start_h * 4, (end_h + 1) * 4))


def build_heuristic_4h():
    print("\n[策略 3/3] 启发式 4h 窗口（方案B）")
    if not HEURISTIC_CSV.exists():
        raise FileNotFoundError(f"找不到启发式策略源数据：{HEURISTIC_CSV}")

    raw = pd.read_csv(HEURISTIC_CSV)
    raw["date"] = raw["date"].astype(str)
    raw = raw[(raw["date"] >= PERIOD_START) & (raw["date"] <= PERIOD_END)].reset_index(drop=True)
    print(f"  有效天数：{len(raw)}")

    actual_15m = load_actual_15min(ACTUAL_XLSX)

    # 加载 V8 小时级预测（同一区间）— 仅用于填充 timeseries 的 "预测电价" 列
    pred_df = pd.read_csv(PRED_CSV, parse_dates=["ts"])
    pred_df["date"] = pred_df["ts"].dt.date.astype(str)

    rows = []
    for _, r in raw.iterrows():
        date = r["date"]
        if date not in actual_15m.index:
            print(f"  [{date}] 缺失 15min 真实价，跳过")
            continue
        cs = int(r["charge_start"])
        ce = int(r["charge_end"])
        ds = int(r["discharge_start"])
        de = int(r["discharge_end"])

        # 方案 B 口径：充电 800 MWh（电网侧），放电 800·η_B = 719.2 MWh（电网侧）
        chg_slots = _hours_to_slots(cs, ce)
        dis_slots = _hours_to_slots(ds, de)
        n_chg = len(chg_slots) if chg_slots else 1
        n_dis = len(dis_slots) if dis_slots else 1

        c96 = np.zeros(96)
        d96 = np.zeros(96)
        # 单时段功率 = 总电量 / (n_slots × DT)
        c_pwr = 800.0 / (n_chg * DT)              # 应等于 200 MW
        d_pwr = 800.0 * ETA_B / (n_dis * DT)       # 应等于 ~179.8 MW

        for s in chg_slots:
            c96[s] = c_pwr
        for s in dis_slots:
            d96[s] = d_pwr

        # SOC 模型简化（与 efficiency_analysis 口径一致：所有效率损耗折到放电侧）
        soc = np.zeros(96)
        cur = 0.0
        for t in range(96):
            cur = cur + c96[t] * DT - d96[t] * DT
            cur = max(0.0, cur)
            soc[t] = cur

        # 预测电价（24 维 → 96 维，repeat 4）
        day_pred = pred_df[pred_df["date"] == date].sort_values("ts")
        if len(day_pred) >= 24:
            pred96 = np.repeat(day_pred["pred"].values[:24].astype(float), 4)
        else:
            pred96 = np.zeros(96)

        actual96 = actual_15m.loc[date].values.astype(float)

        # PF 净收益：复用 strategy_efficiency_analysis 的方案 B 口径
        # pf_rev_B = revenue_pf_yuan × ETA_B；pf_aux = avg_actual × AUX_B
        pf_rev_b = float(r["revenue_pf_yuan"]) * ETA_B
        pf_aux = float(np.mean(actual96)) * AUX_MWH_PER_DAY
        pf_net = pf_rev_b - pf_aux

        rows.append({
            "date": date,
            "_c": c96.tolist(),
            "_d": d96.tolist(),
            "_soc": soc.tolist(),
            "_pred": pred96.tolist(),
            "_actual": actual96.tolist(),
            "pf_net": pf_net,
        })
    df_expanded = pd.DataFrame(rows)
    ts = expand_timeseries(df_expanded)
    pf_net_map = dict(zip(df_expanded["date"], df_expanded["pf_net"]))
    daily = build_metrics_daily(ts, pf_net_map)
    return ts, daily


# ── 输出 ────────────────────────────────────────────────────────────────────
def _ensure_float_dtype(df: pd.DataFrame, float_cols: list[str]) -> pd.DataFrame:
    """
    pandas.read_excel 在整列值都是整数时会推断为 int64（即使写入时是 float）。
    为保证读回时 dtype 与基准 xlsx 一致（float64），在数值整列恰好都是整数的情况下，
    给最后一行加一个极小扰动（1e-12，肉眼/Excel 显示完全看不出）。
    """
    out = df.copy()
    for c in float_cols:
        if c not in out.columns:
            continue
        out[c] = out[c].astype("float64")
        col = out[c].dropna()
        if len(col) > 0 and (col % 1 == 0).all():
            last = out.index[-1]
            out.at[last, c] = float(out.at[last, c]) + 1e-12
    return out


def export_pair(tag: str, ts: pd.DataFrame, daily: pd.DataFrame, out_dir: Path | None = None,
                ts_filename: str | None = None, daily_filename: str | None = None):
    """
    输出一对中文 xlsx（timeseries + daily）。
    out_dir / *_filename 缺省时：写入 OUT_DIR，文件名带 {tag} 后缀。
    """
    target_dir = out_dir if out_dir is not None else OUT_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    ts_cn = _ensure_float_dtype(to_chinese_timeseries(ts), TS_FLOAT_COLS)
    ts_path = target_dir / (ts_filename or f"strategy_15min_timeseries_{tag}.xlsx")
    with pd.ExcelWriter(ts_path, engine="openpyxl") as w:
        ts_cn.to_excel(w, sheet_name="15min_timeseries", index=False)
    print(f"  保存 {ts_path}  ({len(ts_cn)} 行 × {len(ts_cn.columns)} 列)")

    daily_out = _ensure_float_dtype(daily, DAILY_FLOAT_COLS)
    daily_path = target_dir / (daily_filename or f"metrics_strategy_daily_{tag}.xlsx")
    with pd.ExcelWriter(daily_path, engine="openpyxl") as w:
        daily_out.to_excel(w, sheet_name="metrics_strategy_daily", index=False)
    print(f"  保存 {daily_path}  ({len(daily_out)} 行 × {len(daily_out.columns)} 列)")


def _summary(tag: str, daily: pd.DataFrame):
    total_charge = daily["当日充电量（电网侧）_MWh"].sum()
    total_dis = daily["当日放电量（电网侧）_MWh"].sum()
    total_gross = daily["毛收益_元"].sum()
    total_aux = daily["辅助用电成本_元"].sum()
    total_net = daily["净收益_元"].sum()
    total_pf = daily["完全预知（PF）净收益_元"].sum()
    print(f"  [{tag}] 充 {total_charge:.1f} MWh  放 {total_dis:.1f} MWh  "
          f"毛 {total_gross/1e4:.2f}万  辅 {total_aux/1e4:.2f}万  "
          f"净 {total_net/1e4:.2f}万  PF净 {total_pf/1e4:.2f}万")


def main():
    print(f"统计区间：{PERIOD_START} ~ {PERIOD_END}")
    print(f"主 dashboard 输出：{DASHBOARD_DIR}")
    print(f"分策略输出：{OUT_DIR}")

    # 主策略：15min MILP 跨日 SOC → 直接覆盖 dashboard 主目录的两个 xlsx
    ts0, d0 = build_15min_carry_soc()
    export_pair(
        "carry_soc_15min", ts0, d0,
        out_dir=DASHBOARD_DIR,
        ts_filename="strategy_15min_timeseries.xlsx",
        daily_filename="metrics_strategy_daily.xlsx",
    )
    _summary("15min 跨日SOC（主表）", d0)

    ts1, d1 = build_15min_daily_zero()
    export_pair("daily_zero_15min", ts1, d1)
    _summary("15min 日清零", d1)

    ts2, d2 = build_hourly_milp()
    export_pair("hourly_milp", ts2, d2)
    _summary("小时级 MILP", d2)

    ts3, d3 = build_heuristic_4h()
    export_pair("heuristic_4h", ts3, d3)
    _summary("启发式 4h 方案B", d3)

    print("\n" + "=" * 60)
    print("全部 6 个 xlsx 已输出到", OUT_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
