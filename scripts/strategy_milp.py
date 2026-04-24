"""
MILP 日前充放电调度策略
======================
使用混合整数线性规划（MILP）基于预测电价确定最优充放电计划，
在实际电价上评估收益，与旧启发式方案对比。

变量（共 4×24 = 96 个）：
  c[t]   : 第 t 小时向电网购电（充电功率，MW），连续，[0, P_MAX]
  d[t]   : 第 t 小时向电网售电（放电功率，MW），连续，[0, P_MAX]
  u[t]   : 二值变量（1=充电模式，0=放电/待机模式），防止同时充放
  soc[t] : 第 t 小时末荷电状态（MWh），连续，[0, CAP_MWH]

约束：
  ① SOC 动态方程：soc[t] = soc[t-1] + η_c·c[t-1] - d[t-1]/η_d
  ② SOC 边界：soc[0]=0（初始为空），soc[24]=0（日末放空）
  ③ 不同时充放：c[t] ≤ P_MAX·u[t]，d[t] ≤ P_MAX·(1-u[t])

目标（最大化预测净收益）：
  maximize Σ pred_price[t]·(d[t] - c[t])

参数（来自实测数据分析）：
  P_MAX    = 195 MW   （实测满功率，而非额定200 MW）
  CAP_MWH  = 800 MWh  （电池容量）
  ETA_RT   = 0.910    （双程效率91%，由实测充放电量推算）
  ETA_1WAY = √0.91 ≈ 0.9539  （单程效率，充放对称拆分）
  AUX_MWH  = 13.03 MWh/天  （辅助用电，正常日中位数）
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix, csc_matrix

# ── 全局参数 ────────────────────────────────────────────────────────────────
P_MAX_MW  = 195.0
CAP_MWH   = 800.0
ETA_RT    = 0.910
ETA_1WAY  = float(np.sqrt(ETA_RT))   # ≈ 0.9539
AUX_MWH   = 13.03


# ── MILP 求解核心 ───────────────────────────────────────────────────────────
def _build_milp(prices_24h: np.ndarray,
                eta_c: float = ETA_1WAY,
                eta_d: float = ETA_1WAY,
                cap_mwh: float = CAP_MWH,
                p_max: float = P_MAX_MW):
    """
    构建并求解日前充放电 MILP。
    返回 (c, d, soc) 各长度为24的数组，或在失败时返回零数组。
    """
    T = 24
    N = 4 * T          # 总变量数
    IC = 0             # c[t] 起始下标
    ID = T             # d[t] 起始下标
    IU = 2 * T         # u[t] 起始下标（二值）
    IS = 3 * T         # soc[t] 起始下标（t=1..24）

    prices = np.asarray(prices_24h, dtype=float)

    # ── 目标函数（最小化负收益）────────────────────────────────────────────
    obj = np.zeros(N)
    obj[IC:IC + T] = +prices      # 充电成本（正，需最小化）
    obj[ID:ID + T] = -prices      # 放电收入（负，最小化等于最大化收入）

    # ── 变量上下界 ──────────────────────────────────────────────────────────
    lb = np.zeros(N)
    ub = np.zeros(N)
    ub[IC:IC + T] = p_max     # c[t] ∈ [0, p_max]
    ub[ID:ID + T] = p_max     # d[t] ∈ [0, p_max]
    ub[IU:IU + T] = 1.0       # u[t] ∈ {0,1}
    ub[IS:IS + T] = cap_mwh   # soc[t] ∈ [0, cap_mwh]
    # soc[24]（索引 IS+T-1）必须为 0：日末放空
    ub[IS + T - 1] = 0.0
    lb[IS + T - 1] = 0.0

    # ── 整数性：仅 u[t] 为整数 ──────────────────────────────────────────────
    integ = np.zeros(N)
    integ[IU:IU + T] = 1

    # ── 约束矩阵（3T 行 × N 列）────────────────────────────────────────────
    n_con = 3 * T
    A = lil_matrix((n_con, N), dtype=float)
    lb_c = np.full(n_con, -np.inf)
    ub_c = np.zeros(n_con)

    # ① SOC 动态方程（等式约束：lb_c = ub_c = 0）
    #    eta_c·c[t] - d[t]/eta_d + soc[t-1] - soc[t] = 0
    #    （t=0 时 soc[-1] = soc[0]_init = 0，不加前项）
    for t in range(T):
        row = t
        A[row, IC + t] = eta_c
        A[row, ID + t] = -1.0 / eta_d
        if t > 0:
            A[row, IS + t - 1] = 1.0   # +soc[t]（上一小时末）
        A[row, IS + t] = -1.0           # -soc[t+1]（本小时末）
        lb_c[row] = 0.0
        ub_c[row] = 0.0

    # ② c[t] ≤ p_max·u[t]  →  c[t] - p_max·u[t] ≤ 0
    for t in range(T):
        row = T + t
        A[row, IC + t] = 1.0
        A[row, IU + t] = -p_max
        lb_c[row] = -np.inf
        ub_c[row] = 0.0

    # ③ d[t] ≤ p_max·(1-u[t])  →  d[t] + p_max·u[t] ≤ p_max
    for t in range(T):
        row = 2 * T + t
        A[row, ID + t] = 1.0
        A[row, IU + t] = p_max
        lb_c[row] = -np.inf
        ub_c[row] = p_max

    # ── 求解 ────────────────────────────────────────────────────────────────
    A_csc = csc_matrix(A)
    res = milp(
        obj,
        constraints=LinearConstraint(A_csc, lb_c, ub_c),
        integrality=integ,
        bounds=Bounds(lb, ub),
        options={"disp": False, "time_limit": 30.0},
    )

    if res.status not in (0, 3):   # 0=optimal, 3=solution found (may be suboptimal)
        return np.zeros(T), np.zeros(T), np.zeros(T)

    x = res.x
    return x[IC:IC + T], x[ID:ID + T], x[IS:IS + T]


def solve_day_milp(prices_24h: np.ndarray, **kwargs):
    """基于预测电价求解最优充放电计划。"""
    return _build_milp(np.asarray(prices_24h, dtype=float), **kwargs)


def solve_pf_day(actual_24h: np.ndarray, **kwargs):
    """完全预知基准：用实际电价求解同一 MILP。"""
    return _build_milp(np.asarray(actual_24h, dtype=float), **kwargs)


# ── 收益评估 ────────────────────────────────────────────────────────────────
def eval_day_revenue(c: np.ndarray,
                     d: np.ndarray,
                     actual_24h: np.ndarray,
                     aux_mwh: float = AUX_MWH) -> dict:
    """
    用实际节点电价计算当日净收益。
    c, d 均为 MW（每小时），乘以 1h 得 MWh。
    """
    prices = np.asarray(actual_24h, dtype=float)
    c = np.asarray(c, dtype=float)
    d = np.asarray(d, dtype=float)

    discharge_rev = float(np.nansum(prices * d))
    charge_cost   = float(np.nansum(prices * c))
    gross         = discharge_rev - charge_cost

    avg_price = float(np.nanmean(prices))
    aux_cost  = avg_price * aux_mwh
    net       = gross - aux_cost

    return {
        "charge_mwh":    round(float(c.sum()), 2),
        "discharge_mwh": round(float(d.sum()), 2),
        "charge_cost":   round(charge_cost, 0),
        "discharge_rev": round(discharge_rev, 0),
        "gross":         round(gross, 0),
        "aux_cost":      round(aux_cost, 0),
        "net":           round(net, 0),
    }


# ── 主流程 ──────────────────────────────────────────────────────────────────
def run(pred_csv: Path, out_csv: Path | None, label: str = "MILP策略"):
    pred = pd.read_csv(pred_csv, parse_dates=["ts"])
    pred["date"] = pred["ts"].dt.date.astype(str)

    dates = sorted(pred["date"].unique())
    rows  = []

    for date in dates:
        day = pred[pred["date"] == date].sort_values("ts")
        if len(day) < 24:
            continue

        actual_24h = day["actual"].values.astype(float)
        pred_24h   = day["pred"].values.astype(float)

        # ── MILP 求解（预测电价）──────────────────────────────────────────
        c, d, soc = solve_day_milp(pred_24h)

        # ── 完全预知基准（实际电价）──────────────────────────────────────
        c_pf, d_pf, _ = solve_pf_day(actual_24h)

        # ── 收益评估（实际电价）──────────────────────────────────────────
        rev      = eval_day_revenue(c,    d,    actual_24h)
        rev_pf   = eval_day_revenue(c_pf, d_pf, actual_24h)

        # ── 充放电时段摘要 ────────────────────────────────────────────────
        chg_hours = [t for t in range(24) if c[t] > 1.0]
        dis_hours = [t for t in range(24) if d[t] > 1.0]
        chg_str = (f"{chg_hours[0]:02d}–{chg_hours[-1]+1:02d}"
                   if chg_hours else "–")
        dis_str = (f"{dis_hours[0]:02d}–{dis_hours[-1]+1:02d}"
                   if dis_hours else "–")

        rows.append({
            "date":           date,
            "charge_window":  chg_str,
            "discharge_window": dis_str,
            "charge_mwh":     rev["charge_mwh"],
            "discharge_mwh":  rev["discharge_mwh"],
            "gross":          rev["gross"],
            "aux_cost":       rev["aux_cost"],
            "net":            rev["net"],
            "pf_gross":       rev_pf["gross"],
            "pf_aux_cost":    rev_pf["aux_cost"],
            "pf_net":         rev_pf["net"],
        })

    df = pd.DataFrame(rows)

    if out_csv:
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"结果已保存：{out_csv}")

    print_report(df, label)
    return df


# ── 报告打印 ────────────────────────────────────────────────────────────────
def _week_label(date_str: str) -> str:
    ts  = pd.Timestamp(date_str)
    iso = ts.isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def _fmt(v: float) -> str:
    return f"{v / 1e4:+.2f}万"


def print_report(df: pd.DataFrame, label: str = "MILP策略"):
    print(f"\n{'='*76}")
    print(f" {label}")
    print(f"{'='*76}")

    hdr = (f"{'日期':^12} {'充':^8} {'放':^8} "
           f"{'充(MWh)':>8} {'放(MWh)':>8} "
           f"{'净收益(万)':>10} {'PF净(万)':>10} {'兑现率':>7}")
    print(hdr)
    print("-" * 76)

    df = df.copy()
    df["week"] = df["date"].apply(_week_label)
    week_groups = df.groupby("week", sort=False)

    for wk, wdf in week_groups:
        for _, r in wdf.iterrows():
            ratio = r["net"] / r["pf_net"] if abs(r["pf_net"]) > 1 else float("nan")
            print(f"{r['date']:^12} {r['charge_window']:^8} {r['discharge_window']:^8} "
                  f"{r['charge_mwh']:>8.0f} {r['discharge_mwh']:>8.0f} "
                  f"{_fmt(r['net']):>10} {_fmt(r['pf_net']):>10} "
                  f"{ratio:>7.1%}" if not np.isnan(ratio) else
                  f"{r['date']:^12} {r['charge_window']:^8} {r['discharge_window']:^8} "
                  f"{r['charge_mwh']:>8.0f} {r['discharge_mwh']:>8.0f} "
                  f"{_fmt(r['net']):>10} {_fmt(r['pf_net']):>10} {'–':>7}")

        s = wdf[["charge_mwh","discharge_mwh","gross","aux_cost","net","pf_net"]].sum()
        wk_ratio = s["net"] / s["pf_net"] if abs(s["pf_net"]) > 1 else float("nan")
        print(f"  >> 周小计 {wk}: 净={_fmt(s['net'])}  PF净={_fmt(s['pf_net'])}  "
              f"兑现率={wk_ratio:.1%}")
        print()

    print("=" * 76)
    n   = len(df)
    tot = df[["charge_mwh","discharge_mwh","gross","aux_cost","net",
              "pf_gross","pf_aux_cost","pf_net"]].sum()

    print(f"测试期共 {n} 天\n")
    print(f"  {'项目':20}  {'MILP策略':>12}  {'完全预知PF':>12}  {'兑现率':>8}")
    print(f"  {'充电量':20}  {tot['charge_mwh']:>10.1f}MWh  {'–':>12}")
    print(f"  {'放电量':20}  {tot['discharge_mwh']:>10.1f}MWh  {'–':>12}")
    print(f"  {'毛收益':20}  {tot['gross']/1e4:>10.1f}万  {tot['pf_gross']/1e4:>10.1f}万")
    print(f"  {'辅助用电成本':20}  {tot['aux_cost']/1e4:>10.1f}万  {tot['pf_aux_cost']/1e4:>10.1f}万")

    ratio = tot["net"] / tot["pf_net"] if abs(tot["pf_net"]) > 1 else float("nan")
    print(f"  {'净收益':20}  {tot['net']/1e4:>10.1f}万  {tot['pf_net']/1e4:>10.1f}万  "
          f"{ratio:>7.1%}")

    ann_strat = tot["net"]    / n * 365
    ann_pf    = tot["pf_net"] / n * 365
    print(f"\n  全年外推（线性） 策略: {ann_strat/1e8:.3f}亿元  "
          f"PF: {ann_pf/1e8:.3f}亿元")

    print("\n  参考值（旧启发式方案B，η=89.9%，辅=13.03 MWh）:")
    print("    jan25版81天净收益: 924.80万  兑现率: 61.6%  全年外推: 0.416亿")
    print("    实际现货净收益:    807.32万  兑现率: 53.7%")


# ── CLI ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="MILP 充放电调度策略")
    ap.add_argument("--pred",  required=True, help="test_predictions_hourly.csv 路径")
    ap.add_argument("--out",   default=None,  help="输出结果 CSV 路径（可选）")
    ap.add_argument("--label", default="MILP策略（不跨日，η=91%）")
    ap.add_argument("--start", default=None,  help="筛选起始日期，如 2026-01-27")
    ap.add_argument("--end",   default=None,  help="筛选截止日期，如 2026-04-17")
    args = ap.parse_args()

    pred_csv = Path(args.pred)
    out_csv  = Path(args.out) if args.out else None

    # 支持日期范围筛选
    if args.start or args.end:
        pred = pd.read_csv(pred_csv, parse_dates=["ts"])
        pred["date"] = pred["ts"].dt.date.astype(str)
        if args.start:
            pred = pred[pred["date"] >= args.start]
        if args.end:
            pred = pred[pred["date"] <= args.end]
        import tempfile, os
        tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        pred.to_csv(tmp.name, index=False)
        tmp.close()
        run(Path(tmp.name), out_csv, args.label)
        os.unlink(tmp.name)
    else:
        run(pred_csv, out_csv, args.label)


if __name__ == "__main__":
    main()
