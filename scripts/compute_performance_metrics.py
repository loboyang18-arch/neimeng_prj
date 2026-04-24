"""
储能现货套利评价指标计算
========================
分别计算策略回测与实际运营的以下指标：

7.1 总体收益指标
  1.1 单位功率收益   (元/MW/天)
  1.2 度电收益       (元/MWh)   — 毛收益 / 吞吐电量
  1.3 度电净收益     (元/MWh)   — 净收益 / 吞吐电量
  1.4 Top-N 收益集中度 %       — 汇总期指标
  1.5 单次循环净利润分布 P50/P90 — 汇总期指标

7.2 现货套利效率指标
  2.1 放电价差利用率  %
  2.2 放电价格百分位  %  （放电均价在当日96时段中的百分位，越高越好）
  2.3 充电价格百分位  %  （充电均价在当日96时段中的百分位，越低越好）
  2.4 平均充放电价差  (元/MWh)
  2.5 往返效率损耗盈亏门槛  (元/MWh) — 最低价差才能盈利
  2.6 Top-N 放电高价抓取率  %
  2.7 Bottom-N 充电低价抓取率 %

输出：
  output/dashboard/metrics_strategy_daily.csv  — 策略逐日指标
  output/dashboard/metrics_actual_daily.csv    — 实际运营逐日指标
  output/dashboard/metrics_summary.csv         — 两者汇总对比
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── 常量 ──────────────────────────────────────────────────────────────────────
P_MAX_MW         = 195.0      # 额定功率（MW）
CAP_MWH          = 800.0      # 额定容量（MWh）
ETA_RT           = 0.910      # 双程效率
DT               = 0.25       # 每时段小时数（15min）
WINDOW_4H        = 16         # 4h 窗口槽数（连续窗口用于最优基准）
TOPN_SLOTS       = 16         # Top-N 放电：最高 4h 共 16 个时段（非连续）
BOTN_SLOTS       = 16         # Bottom-N 充电：最低 4h 共 16 个时段（非连续）
MAIN_CHG_THRESH  = 10.0       # MWh/h：区分主充电与辅助用电的阈值
AUX_MWH_DAY      = 13.03      # 日辅助用电量（MWh）

# ── 文件路径 ──────────────────────────────────────────────────────────────────
TS_CSV        = ROOT / "output/dashboard/15min_timeseries.csv"
ACTUAL_XLSX   = ROOT / "source_data/日清算结果查询电厂侧(1)_副本.xlsx"
ACTUAL_REV    = ROOT / "output/actual_spot_revenue_jan27_apr17.csv"
OUT_DIR       = ROOT / "output/dashboard"


# ══════════════════════════════════════════════════════════════════════════════
# 辅助计算函数
# ══════════════════════════════════════════════════════════════════════════════

def best_contiguous_window(prices: np.ndarray, window: int, mode: str) -> float:
    """
    返回价格数组中最优连续窗口的均价。
    mode='high' → 最高均价窗口（放电最优基准）
    mode='low'  → 最低均价窗口（充电最优基准）
    """
    T = len(prices)
    if T < window:
        return float("nan")
    rolling = [np.nanmean(prices[i: i + window]) for i in range(T - window + 1)]
    return max(rolling) if mode == "high" else min(rolling)


def optimal_noncontiguous_avg(prices: np.ndarray, n_slots: int, mode: str) -> float:
    """
    返回非连续最优 n_slots 个时段的均价（作为捕获率分母）。
    mode='high' → 最高 n 个时段均价（放电基准）
    mode='low'  → 最低 n 个时段均价（充电基准）
    """
    if n_slots < 1:
        return float("nan")
    n_slots = min(n_slots, len(prices))
    idx = np.argsort(prices)
    chosen = idx[-n_slots:] if mode == "high" else idx[:n_slots]
    return float(np.nanmean(prices[chosen]))


def weighted_avg(values: np.ndarray, weights: np.ndarray) -> float:
    """能量加权平均价格；权重全为零时返回 NaN。"""
    total_w = np.nansum(weights)
    if total_w < 1e-6:
        return float("nan")
    return float(np.nansum(values * weights) / total_w)


def top_n_capture(prices: np.ndarray, energy: np.ndarray, n: int, mode: str) -> float:
    """
    Top-N 价格槽位的能量捕获率。
    mode='high' → 最高 n 个价格槽（放电高价抓取）
    mode='low'  → 最低 n 个价格槽（充电低价抓取）
    """
    total_e = np.nansum(energy)
    if total_e < 1e-6:
        return float("nan")
    idx = np.argsort(prices)
    target_idx = idx[-n:] if mode == "high" else idx[:n]
    captured = np.nansum(energy[target_idx])
    return float(captured / total_e * 100)


def day_equivalent_cycles(dis_mwh: float, cap_mwh: float = CAP_MWH) -> float:
    """一天的等效满充满放循环次数 = 当日放电量 / 额定容量。"""
    return dis_mwh / cap_mwh if cap_mwh > 0 else 0.0


def _week_label(date_str: str) -> str:
    ts = pd.Timestamp(date_str)
    iso = ts.isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


# ══════════════════════════════════════════════════════════════════════════════
# 数据加载
# ══════════════════════════════════════════════════════════════════════════════

def load_strategy(ts_csv: Path,
                  start: str = "2026-01-27",
                  end:   str = "2026-04-17") -> dict[str, dict]:
    """
    从 15min_timeseries.csv 加载策略每日96时段数据。
    返回 {date: {'prices':array, 'chg':array, 'dis':array,
                 'gross':float, 'net':float, 'aux_cost':float}}
    """
    df = pd.read_csv(ts_csv)
    df = df[(df["date"] >= start) & (df["date"] <= end)]

    result = {}
    for date, grp in df.groupby("date"):
        grp = grp.sort_values("slot").reset_index(drop=True)
        if len(grp) < 96:
            continue
        prices = grp["actual_price"].values.astype(float)
        chg    = grp["charge_energy_mwh"].values.astype(float)
        dis    = grp["discharge_energy_mwh"].values.astype(float)

        gross    = float(np.nansum(prices * dis) - np.nansum(prices * chg))
        avg_p    = float(np.nanmean(prices))
        aux_cost = avg_p * AUX_MWH_DAY
        net      = gross - aux_cost

        result[date] = {
            "prices": prices, "chg": chg, "dis": dis,
            "gross": gross, "net": net, "aux_cost": aux_cost,
        }
    return result


def load_actual(xlsx_path: Path, rev_csv: Path,
                start: str = "2026-01-27",
                end:   str = "2026-04-17") -> dict[str, dict]:
    """
    从日清算 XLSX 加载实际运营每日数据，扩展为每日96时段。
    充电：按 xx:15 槽的小时值，均摊到该小时的4个时段。
    放电：直接使用 一期_计量电量（15min粒度，MWh/slot）。
    """
    df = pd.read_excel(xlsx_path, header=0)
    df["date"] = pd.to_datetime(df["查询日期"]).dt.date.astype(str)
    df["price"]    = pd.to_numeric(df["一期_省内实时节点电价"], errors="coerce")
    df["dis_mwh"]  = pd.to_numeric(df["一期_计量电量"],        errors="coerce").fillna(0.0)
    df["chg_hourly"]= pd.to_numeric(df["充电_实际用电量"],     errors="coerce")

    # 实际现货收益（日总量）
    rev = pd.read_csv(rev_csv)
    rev["date"] = rev["date"].astype(str)
    rev = rev.set_index("date")

    result = {}
    for date, grp in df.groupby("date"):
        if date < start or date > end:
            continue
        grp = grp.reset_index(drop=True)
        if len(grp) < 96:
            continue

        prices   = grp["price"].values.astype(float)
        dis_slot = grp["dis_mwh"].values.astype(float)   # MWh/15min slot

        # 将小时充电值（在 xx:15 位置，每4行一个）均摊到4个时段
        chg_slot = np.zeros(96)
        hourly_vals = grp["chg_hourly"].values  # NaN 除 xx:15 外
        for i in range(len(hourly_vals)):
            v = hourly_vals[i]
            if pd.notna(v) and v > MAIN_CHG_THRESH:    # 主充电
                # 均摊到本时段及前3个时段（该小时块）
                blk_start = max(0, i - 3)
                blk_end   = i + 1
                n_slots   = blk_end - blk_start
                chg_slot[blk_start:blk_end] = v / 4.0  # MWh/h ÷ 4 = MWh/15min

        # 日收益（来自实际结算 CSV）
        if date not in rev.index:
            continue
        r = rev.loc[date]
        gross    = float(r["disc_fee"] - r["main_chg_fee"])
        aux_cost = float(r["aux_fee"])
        net      = float(r["net_revenue"])

        result[date] = {
            "prices": prices, "chg": chg_slot, "dis": dis_slot,
            "gross": gross, "net": net, "aux_cost": aux_cost,
        }
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 逐日指标计算
# ══════════════════════════════════════════════════════════════════════════════

def compute_daily_metrics(data: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for date in sorted(data):
        d      = data[date]
        prices = d["prices"]
        chg    = d["chg"]
        dis    = d["dis"]
        gross  = d["gross"]
        net    = d["net"]
        aux    = d["aux_cost"]

        chg_mwh = float(np.nansum(chg))
        dis_mwh = float(np.nansum(dis))
        throughput = chg_mwh + dis_mwh

        avg_chg_p = weighted_avg(prices, chg)
        avg_dis_p = weighted_avg(prices, dis)

        # 实际充放时段数（用于自适应基准）
        chg_slots = int(np.sum(chg > 0.01))
        dis_slots = int(np.sum(dis > 0.01))
        # 保底 4h 窗口
        n_chg_ref = max(chg_slots, WINDOW_4H)
        n_dis_ref = max(dis_slots, WINDOW_4H)

        # 最优连续4h窗口均价（用于 2.1 价差利用率基准）
        opt_dis_4h_cont = best_contiguous_window(prices, WINDOW_4H, "high")
        opt_chg_4h_cont = best_contiguous_window(prices, WINDOW_4H, "low")
        opt_spread = (opt_dis_4h_cont - opt_chg_4h_cont
                      if not (np.isnan(opt_dis_4h_cont) or np.isnan(opt_chg_4h_cont))
                      else float("nan"))

        # 非连续最优均价（用于 2.2/2.3 捕获率基准，与实际时段数对齐）
        opt_dis_nc = optimal_noncontiguous_avg(prices, n_dis_ref, "high")
        opt_chg_nc = optimal_noncontiguous_avg(prices, n_chg_ref, "low")

        actual_spread = (avg_dis_p - avg_chg_p
                         if not (np.isnan(avg_dis_p) or np.isnan(avg_chg_p))
                         else float("nan"))

        # 1.1 单位功率收益
        m11 = gross / P_MAX_MW if throughput > 0 else float("nan")

        # 1.2 度电收益
        m12 = gross / throughput if throughput > 0 else float("nan")

        # 1.3 度电净收益
        m13 = net / throughput if throughput > 0 else float("nan")

        # 2.1 放电价差利用率（基于连续4h最优窗口）
        m21 = (actual_spread / opt_spread * 100
               if (not np.isnan(actual_spread) and opt_spread and opt_spread > 1)
               else float("nan"))

        # 2.2 放电价格百分位（放电均价在当日96个时段中的百分位排名）
        # 越高越好：100% = 放在当日最高价，50% = 放在日中位数
        if not np.isnan(avg_dis_p):
            m22 = float(np.mean(prices < avg_dis_p) * 100)
        else:
            m22 = float("nan")

        # 2.3 充电价格百分位（充电均价在当日96个时段中的百分位排名）
        # 越低越好：0% = 充在当日最低价，50% = 充在日中位数，100% = 充在最高价
        daily_avg_p = float(np.nanmean(prices))
        if not np.isnan(avg_chg_p):
            m23 = float(np.mean(prices < avg_chg_p) * 100)
        else:
            m23 = float("nan")

        # 2.4 平均充放电价差
        m24 = actual_spread

        # 2.5 往返效率损耗盈亏门槛（元/MWh，最小价差才能盈利）
        m25 = ((1 - ETA_RT) * avg_dis_p
               if not np.isnan(avg_dis_p) else float("nan"))

        # 2.6 Top-N 放电高价抓取率
        m26 = top_n_capture(prices, dis, TOPN_SLOTS, "high") if dis_mwh > 0 else float("nan")

        # 2.7 Bottom-N 充电低价抓取率
        m27 = top_n_capture(prices, chg, BOTN_SLOTS, "low") if chg_mwh > 0 else float("nan")

        rows.append({
            "date":               date,
            "week":               _week_label(date),
            "charge_mwh":         round(chg_mwh, 2),
            "discharge_mwh":      round(dis_mwh, 2),
            "gross_yuan":         round(gross, 0),
            "aux_cost_yuan":      round(aux, 0),
            "net_yuan":           round(net, 0),
            "avg_charge_price":   round(avg_chg_p, 2),
            "avg_discharge_price":round(avg_dis_p, 2),
            "opt_charge_4h_cont_price":    round(opt_chg_4h_cont, 2),
            "opt_discharge_4h_cont_price": round(opt_dis_4h_cont, 2),
            "opt_charge_nc_price":    round(opt_chg_nc, 2),
            "opt_discharge_nc_price": round(opt_dis_nc, 2),
            "chg_slots": chg_slots,
            "dis_slots": dis_slots,
            "m11_unit_power_rev_yuan_mw": round(m11, 2),
            "m12_energy_rev_yuan_mwh":    round(m12, 2),
            "m13_net_energy_rev_yuan_mwh":round(m13, 2),
            "m21_spread_utilization_pct": round(m21, 1) if not np.isnan(m21) else None,
            "m22_discharge_capture_pct":  round(m22, 1) if not np.isnan(m22) else None,
            "m23_charge_capture_pct":     round(m23, 1) if not np.isnan(m23) else None,
            "m24_avg_spread_yuan_mwh":    round(m24, 2) if not np.isnan(m24) else None,
            "m25_rte_breakeven_yuan_mwh": round(m25, 2) if not np.isnan(m25) else None,
            "m26_top_dis_capture_pct":    round(m26, 1) if not np.isnan(m26) else None,
            "m27_bot_chg_capture_pct":    round(m27, 1) if not np.isnan(m27) else None,
        })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# 汇总指标计算（1.4 / 1.5）
# ══════════════════════════════════════════════════════════════════════════════

def compute_top_n_concentration(daily: pd.DataFrame, topn_pct: float = 0.10) -> float:
    """1.4 Top-N 收益集中度：前 N% 高收益日的净收益 / 总净收益。"""
    nets = daily["net_yuan"].dropna().sort_values(ascending=False).values
    total = nets.sum()
    if total <= 0 or len(nets) == 0:
        return float("nan")
    n = max(1, int(np.ceil(len(nets) * topn_pct)))
    return float(nets[:n].sum() / total * 100)


def compute_cycle_stats(data: dict[str, dict]) -> dict:
    """
    1.5 等效循环统计（按等效循环数加权）。
    返回 dict 包含：
      - total_cycles: 全期等效满充满放循环总次数
      - weighted_mean: 加权均值（= 总净收益 / 总循环，保证乘法一致性）
      - weighted_p50 / weighted_p90: 以等效循环数为权重的加权分位数
    """
    eq_cycles_list = []   # 每天等效循环数
    pnl_per_cyc_list = [] # 每天单循环净收益（元）
    for d in data.values():
        dis_mwh = float(np.nansum(d["dis"]))
        eq_cyc  = day_equivalent_cycles(dis_mwh)
        if eq_cyc > 0.01:
            eq_cycles_list.append(eq_cyc)
            pnl_per_cyc_list.append(d["net"] / eq_cyc)

    total_cycles = sum(eq_cycles_list)
    weights = np.array(eq_cycles_list)
    values  = np.array(pnl_per_cyc_list)

    if len(values) == 0:
        return {"total_cycles": 0.0, "weighted_mean": None,
                "weighted_p50": None, "weighted_p90": None}

    weighted_mean = float(np.sum(weights * values) / np.sum(weights))

    def weighted_quantile(vals, wts, q):
        idx = np.argsort(vals)
        sorted_v = vals[idx]
        sorted_w = wts[idx]
        cum_w = np.cumsum(sorted_w)
        cum_w_norm = (cum_w - 0.5 * sorted_w[idx]) / cum_w[-1]
        return float(np.interp(q, cum_w_norm, sorted_v))

    return {
        "total_cycles":  total_cycles,
        "weighted_mean":  weighted_mean,
        "weighted_p50":   weighted_quantile(values, weights, 0.50),
        "weighted_p90":   weighted_quantile(values, weights, 0.90),
    }


def compute_summary(strat_daily: pd.DataFrame,
                    actual_daily: pd.DataFrame,
                    strat_data: dict,
                    actual_data: dict) -> pd.DataFrame:
    """汇总对比表：策略 vs 实际。"""
    def agg(df: pd.DataFrame, data: dict, label: str) -> dict:
        n  = len(df)
        s  = df.select_dtypes(include="number").sum()
        m  = df.select_dtypes(include="number").mean()

        cyc = compute_cycle_stats(data)
        top10  = compute_top_n_concentration(df, 0.10)
        top20  = compute_top_n_concentration(df, 0.20)

        return {
            "指标": label,
            "天数": n,
            "总充电量MWh":         round(s["charge_mwh"], 1),
            "总放电量MWh":         round(s["discharge_mwh"], 1),
            "总毛收益万":          round(s["gross_yuan"] / 1e4, 2),
            "总净收益万":          round(s["net_yuan"] / 1e4, 2),
            "日均净收益万":        round(m["net_yuan"] / 1e4, 3),
            "全年外推亿":          round(m["net_yuan"] / 1e4 * 365 / 1e4, 3),
            # 1.1 ~ 1.3（均值）
            "1.1 单位功率收益均值(元/MW)": round(m["m11_unit_power_rev_yuan_mw"], 1),
            "1.2 度电收益均值(元/MWh)":    round(m["m12_energy_rev_yuan_mwh"], 2),
            "1.3 度电净收益均值(元/MWh)":  round(m["m13_net_energy_rev_yuan_mwh"], 2),
            # 1.4 集中度
            "1.4 Top10%收益集中度%": round(top10, 1) if not np.isnan(top10) else None,
            "1.4 Top20%收益集中度%": round(top20, 1) if not np.isnan(top20) else None,
            # 1.5 等效循环盈亏（加权统计，权重=当日等效循环数，保证均值×总循环=总净收益）
            "1.5 单循环P50(万元)": round(cyc["weighted_p50"] / 1e4, 3) if cyc["weighted_p50"] is not None else None,
            "1.5 单循环P90(万元)": round(cyc["weighted_p90"] / 1e4, 3) if cyc["weighted_p90"] is not None else None,
            "1.5 单循环均值(万元)": round(cyc["weighted_mean"] / 1e4, 3) if cyc["weighted_mean"] is not None else None,
            "1.5 等效循环总次数":   round(cyc["total_cycles"], 1),
            # 2.1 ~ 2.5（均值）
            "2.1 价差利用率均值%":     round(df["m21_spread_utilization_pct"].mean(), 1),
            "2.2 放电价格百分位均值%": round(df["m22_discharge_capture_pct"].mean(), 1),
            "2.3 充电价格百分位均值%": round(df["m23_charge_capture_pct"].mean(), 1),
            "2.4 平均价差均值(元/MWh)":round(df["m24_avg_spread_yuan_mwh"].mean(), 2),
            "2.5 盈亏门槛均值(元/MWh)":round(df["m25_rte_breakeven_yuan_mwh"].mean(), 2),
            "2.6 放电高价抓取率均值%": round(df["m26_top_dis_capture_pct"].mean(), 1),
            "2.7 充电低价抓取率均值%": round(df["m27_bot_chg_capture_pct"].mean(), 1),
        }

    row_strat  = agg(strat_daily,  strat_data,  "策略（15min MILP 跨日SOC）")
    row_actual = agg(actual_daily, actual_data, "实际现货运营")

    # 转置为竖排指标对比表
    summary_rows = []
    all_keys = [k for k in row_strat if k != "指标"]
    for k in all_keys:
        summary_rows.append({
            "指标":   k,
            "策略值": row_strat[k],
            "实际值": row_actual[k],
        })
    return pd.DataFrame(summary_rows)


# ══════════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════════

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PERIOD_START, PERIOD_END = "2026-01-27", "2026-04-17"

    print(f"统计区间：{PERIOD_START} ～ {PERIOD_END}")

    # ── 加载数据 ──────────────────────────────────────────────────────────────
    print("\n[1/4] 加载策略时序数据…")
    strat_data = load_strategy(TS_CSV, PERIOD_START, PERIOD_END)
    print(f"  策略可用天数：{len(strat_data)}")

    print("[2/4] 加载实际运营数据…")
    actual_data = load_actual(ACTUAL_XLSX, ACTUAL_REV, PERIOD_START, PERIOD_END)
    print(f"  实际可用天数：{len(actual_data)}")

    # ── 逐日指标 ──────────────────────────────────────────────────────────────
    print("[3/4] 计算逐日指标…")
    strat_daily  = compute_daily_metrics(strat_data)
    actual_daily = compute_daily_metrics(actual_data)

    out_s = OUT_DIR / "metrics_strategy_daily.csv"
    out_a = OUT_DIR / "metrics_actual_daily.csv"
    strat_daily.to_csv(out_s,  index=False, encoding="utf-8-sig")
    actual_daily.to_csv(out_a, index=False, encoding="utf-8-sig")
    print(f"  策略逐日 → {out_s.name}  ({len(strat_daily)}行)")
    print(f"  实际逐日 → {out_a.name}  ({len(actual_daily)}行)")

    # ── 汇总对比 ──────────────────────────────────────────────────────────────
    print("[4/4] 计算汇总对比…")
    summary = compute_summary(strat_daily, actual_daily, strat_data, actual_data)
    out_c = OUT_DIR / "metrics_summary.csv"
    summary.to_csv(out_c, index=False, encoding="utf-8-sig")
    print(f"  汇总对比 → {out_c.name}")

    # ── 控制台预览 ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print(f"{'指标':<34} {'策略值':>15} {'实际值':>15}")
    print("-" * 68)
    for _, row in summary.iterrows():
        s = str(row["策略值"]) if row["策略值"] is not None else "–"
        a = str(row["实际值"]) if row["实际值"] is not None else "–"
        print(f"  {row['指标']:<32} {s:>15} {a:>15}")


if __name__ == "__main__":
    main()
