"""
新约束多场景评估（含绘图）
==========================
场景列表：
  A) 400 MWh + 补贴(入目标) + V8 预测
  B) 800 MWh + 补贴(入目标) + V8 预测
  C) 400 MWh + 补贴(入目标) + 前1天实际电价
  D) 400 MWh + 补贴(入目标) + 前3天平均电价
  E) 400 MWh + 补贴(入目标) + 前1周同星期电价
  F) 400 MWh + 补贴仅事后计入 + V8 预测
     说明：MILP 求解时 cap_comp=0（策略只看节点电价做套利决策），
           事后再按实际放电量乘 350 元/MWh 加补偿。
           PF 也按同口径求解，兑现率计算保持一致。
  G) 400 MWh + 纯补贴循环（忽略电价信号）
     说明：MILP 输入价格为常数（无价差信号），cap_comp=350 入目标。
           策略唯一驱动力为"最大化放电量"以赚补贴。
           事后用真实电价计算套利损益。

所有场景均计算 PF（完全预知基准）、兑现率，保存 daily.csv / summary.txt / pkl / 周图。
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

CAP_COMP_PER_MWH = 350.0
TIME_LIMIT_SEC   = 30.0
DT = 0.25

# ── 工具函数 ─────────────────────────────────────────────────────────────────

def _expand_to_96(arr) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if len(arr) >= 96:
        return arr[:96]
    if len(arr) >= 24:
        return np.repeat(arr[:24], 4)
    raise ValueError(f"价格序列长度不足: {len(arr)}")


def _count_cycles(c_96: np.ndarray, cap_mwh: float) -> float:
    return float((c_96 * DT).sum() / cap_mwh) if cap_mwh > 0 else 0.0


def simulate_max_cycling_continuous(
    n_days: int,
    cap_mwh: float = 400.0,
    p_max: float = 195.0,
    dp_ramp: float = 65.0,
    l_min: int = 4,
    eta_c: float = float(np.sqrt(0.91)),
    eta_d: float = float(np.sqrt(0.91)),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """跨日连续贪心模拟：一次性生成 n_days×96 槽的最大循环充放电序列。

    状态机在整个时间轴上不间断运行，日边界不产生任何中断。
    返回 (c, d, soc)，长度均为 n_days*96。
    """
    T = n_days * 96
    c = np.zeros(T)
    d = np.zeros(T)
    soc = np.zeros(T)

    cur_soc = 0.0
    state = 'charge'
    prev_power = 0.0
    slots_in_state = 0

    for t in range(T):
        if state == 'charge':
            target = min(p_max, prev_power + dp_ramp)
            room = (cap_mwh - cur_soc) / (eta_c * DT) if eta_c * DT > 0 else 0
            power = max(0.0, min(target, room))
            if cur_soc >= cap_mwh - 0.1 or power < 0.5:
                if slots_in_state >= l_min:
                    state = 'gap_cd'
                    slots_in_state = 0
                    prev_power = 0.0
                else:
                    c[t] = max(0.0, min(dp_ramp, room))
                    cur_soc += eta_c * c[t] * DT
                    cur_soc = min(cur_soc, cap_mwh)
                    prev_power = c[t]
                    slots_in_state += 1
                    soc[t] = cur_soc
                    continue
            else:
                c[t] = power
                cur_soc += eta_c * power * DT
                cur_soc = min(cur_soc, cap_mwh)
                prev_power = power
                slots_in_state += 1

        elif state == 'discharge':
            target = min(p_max, prev_power + dp_ramp)
            available = cur_soc * eta_d / DT if DT > 0 else 0
            power = max(0.0, min(target, available))
            if cur_soc < 0.1 or power < 0.5:
                if slots_in_state >= l_min:
                    state = 'gap_dc'
                    slots_in_state = 0
                    prev_power = 0.0
                else:
                    d[t] = max(0.0, min(dp_ramp, available))
                    cur_soc -= d[t] * DT / eta_d
                    cur_soc = max(cur_soc, 0.0)
                    prev_power = d[t]
                    slots_in_state += 1
                    soc[t] = cur_soc
                    continue
            else:
                d[t] = power
                cur_soc -= power * DT / eta_d
                cur_soc = max(cur_soc, 0.0)
                prev_power = power
                slots_in_state += 1

        elif state == 'gap_cd':
            prev_power = 0.0
            slots_in_state += 1
            if slots_in_state >= 1:
                state = 'discharge'
                slots_in_state = 0

        elif state == 'gap_dc':
            prev_power = 0.0
            slots_in_state += 1
            if slots_in_state >= 1:
                state = 'charge'
                slots_in_state = 0

        soc[t] = max(cur_soc, 0.0)

    return c, d, soc


def run_scenario_greedy_cycling(
    label: str,
    cap_mwh: float,
    cap_comp: float,
    actual_df: pd.DataFrame,
    dates: list[str],
    out_dir: Path,
):
    """场景 G：跨日连续贪心循环 + 真实电价事后评估。

    一次性模拟全部 N×96 槽的连续充放电，再按日切片用真实电价计算收益。
    PF 直接复用场景 A（同物理约束 400MWh + 补偿入目标）的已有结果。
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # 筛选有效日
    valid = [d for d in dates if d in actual_df.index]
    n_days = len(valid)
    logger.info("=" * 70)
    logger.info("[%s] %d 天 | cap=%g MWh | 补偿=%g 元/MWh (跨日连续贪心)",
                label, n_days, cap_mwh, cap_comp)
    logger.info("=" * 70)

    if n_days == 0:
        logger.warning("[%s] 无有效天数", label)
        return pd.DataFrame()

    # ── 复用场景 A 的 PF 数据（同物理约束，无需重新求解） ──
    pf_lookup = {}
    a_dir = out_dir.parent / "A_400mwh_v8pred"
    a_csv = a_dir / "daily.csv"
    if a_csv.exists():
        a_df = pd.read_csv(a_csv)
        pf_lookup = dict(zip(a_df["date"].astype(str), a_df["pf_net_total"]))
        logger.info("  复用场景 A 的 PF 数据: %d 天", len(pf_lookup))
    else:
        logger.warning("  未找到场景 A 结果 (%s)，PF 列将填 0", a_csv)

    # ── 一次性连续模拟全部 N×96 槽 ──
    c_all, d_all, soc_all = simulate_max_cycling_continuous(
        n_days=n_days, cap_mwh=cap_mwh)

    total_cyc = _count_cycles(c_all, cap_mwh)
    logger.info("  连续模拟完成: %d 槽, 总循环 %.2f× (日均 %.2f×)",
                len(c_all), total_cyc, total_cyc / n_days)

    # ── 按日切片，用真实电价评估 ──
    rows = []
    for i, date in enumerate(valid):
        s, e = i * 96, (i + 1) * 96
        c_day   = c_all[s:e]
        d_day   = d_all[s:e]
        soc_day = soc_all[s:e]

        actual_96 = actual_df.loc[date].values.astype(float)
        actual_clean = _sanitize_96(actual_96)
        if actual_clean is None:
            continue
        actual_96 = actual_clean

        rev = _eval_with_comp(c_day, d_day, actual_96, cap_comp)
        cyc = _count_cycles(c_day, cap_mwh)

        pf_net = pf_lookup.get(date, 0.0)

        rows.append({
            "date": date,
            "charge_mwh":    rev["charge_mwh"],
            "discharge_mwh": rev["discharge_mwh"],
            "cycles":        round(cyc, 2),
            "arb_net":       rev["net"],
            "cap_comp":      rev["cap_comp"],
            "net_total":     rev["net_total"],
            "pf_net_total":  pf_net,
            "_actual": actual_96.tolist(),
            "_pred":   [300.0] * 96,
            "_c":      c_day.tolist(),
            "_d":      d_day.tolist(),
            "_soc":    soc_day.tolist(),
        })

        if (i + 1) % 10 == 0 or i + 1 == n_days:
            logger.info("  [%s] (%d/%d) 套利=%.1fk 补偿=%.1fk 合计=%.1fk 循环=%.1f×",
                        date, i + 1, n_days,
                        rev["net"] / 1e3, rev["cap_comp"] / 1e3,
                        rev["net_total"] / 1e3, cyc)

    df = pd.DataFrame(rows)
    if df.empty:
        logger.warning("[%s] 无有效天数", label)
        return df

    plot_cols = [col for col in df.columns if col.startswith("_")]
    df.drop(columns=plot_cols).to_csv(out_dir / "daily.csv", index=False)
    df.to_pickle(out_dir / "daily_full.pkl")

    n = len(df)
    net_sum     = float(df["net_total"].sum())
    pf_sum      = float(df["pf_net_total"].sum())
    arb_sum     = float(df["arb_net"].sum())
    comp_sum    = float(df["cap_comp"].sum())
    realization = net_sum / pf_sum if pf_sum != 0 else 0.0

    summary_lines = [
        "=" * 70,
        f"场景: {label}",
        "=" * 70,
        f"日期范围  : {df['date'].iloc[0]} ~ {df['date'].iloc[-1]} ({n} 天)",
        f"电站容量  : {cap_mwh} MWh",
        f"容量补偿  : {cap_comp} 元/MWh  (跨日连续贪心循环，不择时)",
        f"日均循环  : {df['cycles'].mean():.2f} 次",
        "",
        f"  累计充电    : {df['charge_mwh'].sum():>12,.1f} MWh",
        f"  累计放电    : {df['discharge_mwh'].sum():>12,.1f} MWh",
        f"  套利净收益  : {arb_sum:>12,.0f} 元 = {arb_sum/1e4:>8.1f} 万元",
        f"  容量补偿    : {comp_sum:>12,.0f} 元 = {comp_sum/1e4:>8.1f} 万元",
        f"  总净收益    : {net_sum:>12,.0f} 元 = {net_sum/1e4:>8.1f} 万元",
        f"  PF 总净收益 : {pf_sum:>12,.0f} 元 = {pf_sum/1e4:>8.1f} 万元",
        f"  兑现率      : {realization*100:>12.1f} %",
        "=" * 70,
    ]
    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)
    with open(out_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)
    pd.Series({"n_days": n, "cap_mwh": cap_mwh, "cap_comp": cap_comp,
               "cap_comp_in_obj": 2,
               "cycles_avg": df["cycles"].mean(),
               "arb_net": arb_sum, "comp_total": comp_sum,
               "net_total": net_sum, "pf_net_total": pf_sum,
               "realization": realization}).to_csv(out_dir / "summary.csv")

    plot_weekly_scenario(df, out_dir / "plots", label, cap_mwh)
    logger.info("[%s] 完成。结果: %s", label, out_dir)
    return df


def _eval_with_comp(c, d, actual_96, comp):
    rev = eval_day_revenue_15min(c, d, actual_96)
    cap_comp = rev["discharge_mwh"] * comp
    rev["cap_comp"] = round(cap_comp, 0)
    rev["net_total"] = round(rev["net"] + cap_comp, 0)
    return rev


def _week_label(date_str: str) -> str:
    ts = pd.Timestamp(date_str)
    iso = ts.isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


# ── 朴素预测生成 ─────────────────────────────────────────────────────────────

def _sanitize_96(arr: np.ndarray) -> np.ndarray | None:
    """对 96 维价格向量做 NaN/inf 清洗：
       - inf → NaN
       - 用日内非空均值兜底；若全空则返回 None
    """
    arr = np.asarray(arr, dtype=float).copy()
    arr[~np.isfinite(arr)] = np.nan
    if np.isnan(arr).all():
        return None
    if np.isnan(arr).any():
        fill = float(np.nanmean(arr))
        arr = np.where(np.isnan(arr), fill, arr)
    return arr


def build_naive_pred(actual_df: pd.DataFrame, dates: list[str],
                     mode: str) -> dict[str, np.ndarray]:
    """生成朴素预测 dict[date_str -> 96-dim ndarray]，带 NaN/inf 兜底。

    mode:
      'lag1'  : 前 1 天实际价格
      'lag3'  : 前 3 天平均（用 nanmean，避免单点 NaN 污染整条向量）
      'weekday': 前 1 周同星期
    """
    all_dates = sorted(actual_df.index.tolist())
    date2idx = {d: i for i, d in enumerate(all_dates)}
    result = {}
    for d in dates:
        if d not in date2idx:
            continue
        idx = date2idx[d]
        if mode == "lag1":
            if idx < 1:
                continue
            arr = actual_df.loc[all_dates[idx - 1]].values.astype(float)
        elif mode == "lag3":
            if idx < 3:
                continue
            vals = np.stack([actual_df.loc[all_dates[idx - k]].values
                             for k in range(1, 4)], axis=0).astype(float)
            arr = np.nanmean(vals, axis=0)
        elif mode == "weekday":
            if idx < 7:
                continue
            arr = actual_df.loc[all_dates[idx - 7]].values.astype(float)
        else:
            raise ValueError(f"未知 mode: {mode}")

        cleaned = _sanitize_96(arr)
        if cleaned is None:
            logger.warning("  [%s|%s] 源数据全为 NaN，跳过该日", mode, d)
            continue
        result[d] = cleaned
    return result


# ── 核心求解循环 ─────────────────────────────────────────────────────────────

def run_scenario(
    label: str,
    cap_mwh: float,
    max_charge_mwh: float,
    cap_comp: float,
    pred_dict: dict[str, np.ndarray],
    actual_df: pd.DataFrame,
    dates: list[str],
    out_dir: Path,
    *,
    cap_comp_in_obj: bool = True,
):
    """对单一场景跑完所有天：求解 + PF + 评估 + 保存 + 绘图。

    Args:
        cap_comp_in_obj: True  → cap_comp 进入 MILP 目标函数（A/B/C/D/E）
                         False → MILP 求解时不带补偿，仅事后按 cap_comp
                                 加在评估总收益上（场景 F）
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    comp_in_milp = cap_comp if cap_comp_in_obj else 0.0
    logger.info("=" * 70)
    logger.info("[%s] %d 天 | cap=%g MWh | 补偿=%g 元/MWh (%s)",
                label, len(dates), cap_mwh, cap_comp,
                "入MILP目标" if cap_comp_in_obj else "仅事后计入")
    logger.info("=" * 70)

    soc_pred = 0.0
    soc_pf   = 0.0
    rows = []

    for i, date in enumerate(dates):
        if date not in actual_df.index or date not in pred_dict:
            continue
        pred_96   = _expand_to_96(pred_dict[date])
        actual_96 = actual_df.loc[date].values.astype(float)

        # ── NaN/inf 守卫（避免 PuLP 抛错） ──────────────────────────────
        actual_clean = _sanitize_96(actual_96)
        pred_clean   = _sanitize_96(pred_96)
        if actual_clean is None or pred_clean is None:
            logger.warning("  [%s] 价格全空，跳过", date)
            continue
        actual_96 = actual_clean
        pred_96   = pred_clean

        is_last = (i == len(dates) - 1)
        force_end = is_last
        # 次日预测均价（跨日终端价值）
        if not is_last:
            nxt = dates[i + 1]
            if nxt in pred_dict:
                nxt_arr = _sanitize_96(_expand_to_96(pred_dict[nxt]))
                next_avg = float(np.mean(nxt_arr)) if nxt_arr is not None else 0.0
            else:
                next_avg = 0.0
        else:
            next_avg = 0.0

        common = dict(cap_mwh=cap_mwh, max_charge_mwh=max_charge_mwh,
                      force_zero_end=force_end, cap_comp_per_mwh=comp_in_milp,
                      time_limit=TIME_LIMIT_SEC)

        c, d, soc = solve_day_milp_15min(
            pred_96, soc_init=soc_pred, next_day_avg_price=next_avg, **common)
        c_pf, d_pf, soc_pf_arr = solve_pf_day_15min(
            actual_96, soc_init=soc_pf,
            next_day_avg_price=float(np.mean(actual_96)), **common)

        soc_pred = float(soc[-1])
        soc_pf   = float(soc_pf_arr[-1])

        # 事后评估：cap_comp 始终按用户配置加上（与 MILP 是否含补偿无关）
        rev    = _eval_with_comp(c, d, actual_96, cap_comp)
        rev_pf = _eval_with_comp(c_pf, d_pf, actual_96, cap_comp)
        cyc    = _count_cycles(c, cap_mwh)

        rows.append({
            "date": date,
            "charge_mwh":    rev["charge_mwh"],
            "discharge_mwh": rev["discharge_mwh"],
            "cycles":        round(cyc, 2),
            "arb_net":       rev["net"],
            "cap_comp":      rev["cap_comp"],
            "net_total":     rev["net_total"],
            "pf_net_total":  rev_pf["net_total"],
            # 绘图数据
            "_actual": actual_96.tolist(),
            "_pred":   pred_96.tolist(),
            "_c":      c.tolist(),
            "_d":      d.tolist(),
            "_soc":    soc.tolist(),
        })

        if (i + 1) % 10 == 0 or i + 1 == len(dates):
            logger.info("  [%s] (%d/%d) 套利=%.1fk 补偿=%.1fk 合计=%.1fk 循环=%.1f×",
                        date, i + 1, len(dates),
                        rev["net"] / 1e3, rev["cap_comp"] / 1e3,
                        rev["net_total"] / 1e3, cyc)

    df = pd.DataFrame(rows)
    if df.empty:
        logger.warning("[%s] 无有效天数", label)
        return df

    # 保存
    plot_cols = [c for c in df.columns if c.startswith("_")]
    df.drop(columns=plot_cols).to_csv(out_dir / "daily.csv", index=False)
    df.to_pickle(out_dir / "daily_full.pkl")

    # 汇总
    n = len(df)
    net_sum    = float(df["net_total"].sum())
    pf_sum     = float(df["pf_net_total"].sum())
    arb_sum    = float(df["arb_net"].sum())
    comp_sum   = float(df["cap_comp"].sum())
    realization = net_sum / pf_sum if pf_sum != 0 else 0.0

    summary_lines = [
        "=" * 70,
        f"场景: {label}",
        "=" * 70,
        f"日期范围  : {df['date'].iloc[0]} ~ {df['date'].iloc[-1]} ({n} 天)",
        f"电站容量  : {cap_mwh} MWh",
        f"容量补偿  : {cap_comp} 元/MWh"
        f"  ({'入MILP目标' if cap_comp_in_obj else '仅事后计入'})",
        f"日均循环  : {df['cycles'].mean():.2f} 次",
        "",
        f"  累计充电    : {df['charge_mwh'].sum():>12,.1f} MWh",
        f"  累计放电    : {df['discharge_mwh'].sum():>12,.1f} MWh",
        f"  套利净收益  : {arb_sum:>12,.0f} 元 = {arb_sum/1e4:>8.1f} 万元",
        f"  容量补偿    : {comp_sum:>12,.0f} 元 = {comp_sum/1e4:>8.1f} 万元",
        f"  总净收益    : {net_sum:>12,.0f} 元 = {net_sum/1e4:>8.1f} 万元",
        f"  PF 总净收益 : {pf_sum:>12,.0f} 元 = {pf_sum/1e4:>8.1f} 万元",
        f"  兑现率      : {realization*100:>12.1f} %",
        "=" * 70,
    ]
    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)
    with open(out_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)
    pd.Series({"n_days": n, "cap_mwh": cap_mwh, "cap_comp": cap_comp,
               "cap_comp_in_obj": int(cap_comp_in_obj),
               "cycles_avg": df["cycles"].mean(),
               "arb_net": arb_sum, "comp_total": comp_sum,
               "net_total": net_sum, "pf_net_total": pf_sum,
               "realization": realization}).to_csv(out_dir / "summary.csv")

    # 绘图
    plot_weekly_scenario(df, out_dir / "plots", label, cap_mwh)
    logger.info("[%s] 完成。结果: %s", label, out_dir)
    return df


# ── 绘图 ────────────────────────────────────────────────────────────────────

def plot_weekly_scenario(df: pd.DataFrame, out_dir: Path,
                         label: str, cap_mwh: float):
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
    t_axis = np.arange(96) * 15 / 60

    for wk, wdf in df.groupby("week", sort=False):
        days = wdf["date"].tolist()
        nd = len(days)
        fig, axes = plt.subplots(nd, 1, figsize=(16, 3.6 * nd),
                                 constrained_layout=True)
        if nd == 1:
            axes = [axes]
        fig.suptitle(f"{label}  {wk}", fontsize=11, fontweight="bold")

        for ax, (_, r) in zip(axes, wdf.iterrows()):
            actual_96 = np.array(r["_actual"])
            pred_96   = np.array(r["_pred"])
            c_arr     = np.array(r["_c"])
            d_arr     = np.array(r["_d"])
            soc_arr   = np.array(r["_soc"])

            for t in range(96):
                x0, x1 = t * 15 / 60, (t + 1) * 15 / 60
                if c_arr[t] > 0.5:
                    ax.axvspan(x0, x1, color="#BBDEFB", alpha=0.75, zorder=0)
                if d_arr[t] > 0.5:
                    ax.axvspan(x0, x1, color="#FFCDD2", alpha=0.75, zorder=0)

            mid = t_axis + 15 / 120
            ax.plot(mid, actual_96, color="#1565C0", lw=1.5,
                    label="实际价(15min)", alpha=0.9, zorder=3)
            ax.plot(mid, pred_96, color="#E53935", lw=1.2, ls="--",
                    label="预测/输入价", alpha=0.8, zorder=3)

            ax2 = ax.twinx()
            bar_w = 14 / 60
            ax2.bar(mid, c_arr,  width=bar_w, color="#1565C0", alpha=0.3,
                    label="充电(MW)", zorder=2)
            ax2.bar(mid, -d_arr, width=bar_w, color="#C62828", alpha=0.3,
                    label="放电(MW)", zorder=2)
            if soc_arr.max() > 1:
                ax2.plot(mid, soc_arr, color="#2E7D32", lw=1.2, ls=":",
                         alpha=0.85, label="SOC(MWh)", zorder=4)
            ax2.set_ylim(-280, max(280, cap_mwh * 1.1))
            ax2.set_ylabel("功率(MW) / SOC(MWh)", fontsize=FS - 1, color="#888")
            ax2.tick_params(labelsize=FS - 1, colors="#888")
            ax2.axhline(0, color="#aaa", lw=0.5, ls=":")

            cyc = r["cycles"]
            net_w = r["net_total"] / 1e4
            comp_w = r["cap_comp"] / 1e4
            arb_w = r["arb_net"] / 1e4
            soc_end = float(soc_arr[-1])
            ax.set_title(
                f"{r['date']}  充{r['charge_mwh']:.0f} 放{r['discharge_mwh']:.0f}MWh "
                f"{cyc:.1f}×  套利{arb_w:+.1f}万 补偿{comp_w:.1f}万 "
                f"合计{net_w:+.1f}万  SOC末{soc_end:.0f}",
                fontsize=FS, loc="left", pad=3)

            ax.set_xlim(0, 24)
            ax.set_xticks(range(0, 25, 2))
            ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 2)],
                               fontsize=FS - 1)
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
                          loc="upper right", fontsize=FS - 1,
                          framealpha=0.85, ncol=4)

        out_path = out_dir / f"{wk}.png"
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        plt.close(fig)

    logger.info("  [%s] 绘图完成: %s", label, out_dir)


# ── 对比汇总 ────────────────────────────────────────────────────────────────

def compare_all(base_dir: Path, scenario_dirs: dict[str, Path] | None = None):
    """读取 base_dir 下所有子目录的 summary.csv，合并输出对比表。

    若 scenario_dirs 为空，则自动扫描 base_dir 下所有含 summary.csv 的子目录，
    保证场景增量重跑后也能合并出完整对比。
    """
    rows = []
    # 统一用目录名做 key，避免与 main() 注入的别名重复
    discovered: dict[str, Path] = {}
    if scenario_dirs:
        for _, p in scenario_dirs.items():
            if p.exists():
                discovered[p.name] = p
    # 自动扫描补充
    for sub in sorted(base_dir.iterdir()):
        if sub.is_dir() and (sub / "summary.csv").exists():
            discovered.setdefault(sub.name, sub)

    for name, d in discovered.items():
        csv = d / "summary.csv"
        if not csv.exists():
            continue
        try:
            s = pd.read_csv(csv, index_col=0, header=None).squeeze()
        except Exception as e:
            logger.warning("读取 %s 失败: %s", csv, e)
            continue
        s["scenario"] = name
        rows.append(s)

    if not rows:
        return
    cmp = pd.DataFrame(rows).set_index("scenario")
    cmp_path = base_dir / "comparison.csv"
    cmp.to_csv(cmp_path)

    lines = ["=" * 88, "多场景对比汇总", "=" * 88]
    for _, r in cmp.iterrows():
        raw = r.get("cap_comp_in_obj", 1)
        in_obj = 1 if (pd.isna(raw) or raw == "") else int(float(raw))
        tag = "入目标" if in_obj else "事后"
        lines.append(
            f"  {r.name:28s}  cap={float(r.get('cap_mwh',0)):>5.0f}MWh  "
            f"补偿[{tag}]  净收益={float(r.get('net_total',0))/1e4:>8.1f}万  "
            f"PF={float(r.get('pf_net_total',0))/1e4:>8.1f}万  "
            f"兑现率={float(r.get('realization',0))*100:>5.1f}%  "
            f"循环={float(r.get('cycles_avg',0)):>4.2f}×"
        )
    lines.append("=" * 88)
    text = "\n".join(lines)
    print("\n" + text)
    with open(base_dir / "comparison.txt", "w", encoding="utf-8") as f:
        f.write(text)
    logger.info("对比表保存: %s", cmp_path)


# ── 主入口 ──────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")

    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-csv",
                    default=str(ROOT / "output/experiments/v8.0-12m-sudun500"
                                     "/test_predictions_hourly.csv"))
    ap.add_argument("--actual-xlsx",
                    default=str(ROOT / "source_data/日清算结果查询电厂侧(1)_副本.xlsx"))
    ap.add_argument("--base-dir",
                    default=str(ROOT / "output/experiments/new-constraints"))
    ap.add_argument("--start", default="2026-01-27")
    ap.add_argument("--end",   default="2026-04-17")
    ap.add_argument("--scenarios", default="A,B,C,D,E,F,G",
                    help="要跑的场景，逗号分隔 (A/B/C/D/E/F/G)")
    args = ap.parse_args()

    base_dir = Path(args.base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据
    actual_df = load_actual_15min(Path(args.actual_xlsx))
    all_actual_dates = sorted(actual_df.index.tolist())

    pred_raw = pd.read_csv(args.pred_csv, parse_dates=["ts"])
    pred_raw["date"] = pred_raw["ts"].dt.date.astype(str)
    pred_raw = pred_raw[(pred_raw["date"] >= args.start) &
                        (pred_raw["date"] <= args.end)]
    v8_dates = sorted(pred_raw["date"].unique())

    # 构建 V8 预测 dict
    v8_pred = {}
    for d in v8_dates:
        dp = pred_raw[pred_raw["date"] == d].sort_values("ts")
        if len(dp) >= 24:
            v8_pred[d] = np.repeat(dp["pred"].values[:24].astype(float), 4)

    # 限定有实际价格的天
    valid_dates = [d for d in v8_dates if d in actual_df.index]

    scenarios_to_run = [s.strip().upper() for s in args.scenarios.split(",")]
    scenario_dirs: dict[str, Path] = {}

    def _safe_run(key: str, runner_fn):
        """单场景错误隔离：任一场景崩溃不影响后续。"""
        try:
            runner_fn()
        except Exception as exc:
            logger.exception("[%s] 场景失败：%s（继续下一场景）", key, exc)

    # A) 400 MWh + 补贴(入目标) + V8 预测
    if "A" in scenarios_to_run:
        d = base_dir / "A_400mwh_v8pred"
        scenario_dirs["A_400MWh_V8pred"] = d
        _safe_run("A", lambda: run_scenario(
            "A: 400MWh + 补贴(入目标) + V8预测", 400.0, 2400.0,
            CAP_COMP_PER_MWH, v8_pred, actual_df, valid_dates, d))

    # B) 800 MWh + 补贴(入目标) + V8 预测
    if "B" in scenarios_to_run:
        d = base_dir / "B_800mwh_v8pred"
        scenario_dirs["B_800MWh_V8pred"] = d
        _safe_run("B", lambda: run_scenario(
            "B: 800MWh + 补贴(入目标) + V8预测", 800.0, 4800.0,
            CAP_COMP_PER_MWH, v8_pred, actual_df, valid_dates, d))

    # C) 400 MWh + 补贴(入目标) + 前1天实际
    if "C" in scenarios_to_run:
        lag1 = build_naive_pred(actual_df, valid_dates, "lag1")
        lag1_dates = [d for d in valid_dates if d in lag1]
        d = base_dir / "C_400mwh_lag1"
        scenario_dirs["C_400MWh_lag1"] = d
        _safe_run("C", lambda: run_scenario(
            "C: 400MWh + 补贴(入目标) + 前1天电价", 400.0, 2400.0,
            CAP_COMP_PER_MWH, lag1, actual_df, lag1_dates, d))

    # D) 400 MWh + 补贴(入目标) + 前3天平均
    if "D" in scenarios_to_run:
        lag3 = build_naive_pred(actual_df, valid_dates, "lag3")
        lag3_dates = [d for d in valid_dates if d in lag3]
        d = base_dir / "D_400mwh_lag3"
        scenario_dirs["D_400MWh_lag3"] = d
        _safe_run("D", lambda: run_scenario(
            "D: 400MWh + 补贴(入目标) + 前3天均价", 400.0, 2400.0,
            CAP_COMP_PER_MWH, lag3, actual_df, lag3_dates, d))

    # E) 400 MWh + 补贴(入目标) + 前1周同星期
    if "E" in scenarios_to_run:
        wkday = build_naive_pred(actual_df, valid_dates, "weekday")
        wkday_dates = [d for d in valid_dates if d in wkday]
        d = base_dir / "E_400mwh_weekday"
        scenario_dirs["E_400MWh_weekday"] = d
        _safe_run("E", lambda: run_scenario(
            "E: 400MWh + 补贴(入目标) + 前周同天电价", 400.0, 2400.0,
            CAP_COMP_PER_MWH, wkday, actual_df, wkday_dates, d))

    # F) 400 MWh + 补贴仅事后计入 + V8 预测
    if "F" in scenarios_to_run:
        d = base_dir / "F_400mwh_v8pred_compAfter"
        scenario_dirs["F_400MWh_V8pred_compAfter"] = d
        _safe_run("F", lambda: run_scenario(
            "F: 400MWh + 补贴(仅事后) + V8预测", 400.0, 2400.0,
            CAP_COMP_PER_MWH, v8_pred, actual_df, valid_dates, d,
            cap_comp_in_obj=False))

    # G) 400 MWh + 纯补贴循环（贪心模拟，不择时）
    if "G" in scenarios_to_run:
        d = base_dir / "G_400mwh_pure_cycling"
        scenario_dirs["G_400MWh_pure_cycling"] = d
        _safe_run("G", lambda: run_scenario_greedy_cycling(
            "G: 400MWh + 纯补贴循环(贪心模拟)", 400.0,
            CAP_COMP_PER_MWH, actual_df, valid_dates, d))

    # 对比汇总（自动扫描所有已完成场景，包含上一轮已有结果）
    compare_all(base_dir, scenario_dirs)
    logger.info("全部场景完成！")


if __name__ == "__main__":
    main()
