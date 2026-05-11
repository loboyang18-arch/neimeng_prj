"""
MILP 日前充放电调度策略（15分钟粒度）
=====================================
在小时级 MILP 基础上升级为 15 分钟粒度，含全套物理约束：

约束列表：
  ① SOC 动态方程：soc[t] = soc[t-1] + η_c·c[t]·0.25 - d[t]·0.25/η_d
  ② SOC 边界：soc[0]=0，soc[95]=0（日末放空）
  ③ 不同时充放：y_c[t] + y_d[t] ≤ 1
  ④ 功率上限：c[t] ≤ P_MAX·y_c[t]，d[t] ≤ P_MAX·y_d[t]
  ⑤ 爬坡约束：|c[t]-c[t-1]| ≤ ΔP_RAMP，|d[t]-d[t-1]| ≤ ΔP_RAMP
  ⑥ 切换间隔：y_c[t]+y_d[t+1] ≤ 1，y_d[t]+y_c[t+1] ≤ 1（充放各需 ≥1 时段空档）
  ⑦ 最小连续运行：单次充/放电必须连续 ≥ L_MIN 个时段（≥1 小时）
     线性化：若 y[t]=1,y[t-1]=0（本时段启动），则 y[t+k]=1 对 k=1..L_MIN-1
     等价约束：-y[t+k] + y[t] - y[t-1] ≤ 0
  ⑧ 日充电量上限：Σ c[t]·DT ≤ MAX_CHARGE_MWH（≤ 1.5×容量 = 1200 MWh）

效率与成本建模：
  - 电池效率：在 SOC 动态方程中体现（η_c·c[t]·DT 充入，d[t]·DT/η_d 消耗）
    充 800 MWh → 存 762.8 MWh → 放回网侧 727.6 MWh，双程效率 ≈ 91% ✓
  - 辅助用电成本：评估阶段固定扣除 avg_price × 13.03 MWh/天，不影响调度决策 ✓

数据说明：
  预测电价  : test_predictions_hourly.csv（小时 24 点 → 每点重复 4 次为 96 点）
            或 test_predictions_15min.csv（已有 96 点/天，逐槽用于 MILP，不做上采样）
  实际电价  : 日清算结果查询电厂侧(1)_副本.xlsx → `一期_省内实时节点电价`（真实15分钟粒度）

参数（来自实测数据）：
  P_MAX          = 195 MW
  CAP_MWH        = 800 MWh
  ETA_RT         = 0.910（双程效率 91%）
  ETA_1WAY       = √0.91 ≈ 0.9539（单程效率）
  ΔP_RAMP        = 65 MW/15min（约3时段内从0爬坡至满功率）
  L_MIN          = 4 槽（1小时，最小连续运行时长）
  MAX_CHARGE_MWH = 1200 MWh/天（日充电上限，1.5×容量）
  AUX_MWH        = 13.03 MWh/天（辅助用电）
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix, csc_matrix

# ── 全局参数 ────────────────────────────────────────────────────────────────
P_MAX_MW         = 195.0
CAP_MWH          = 800.0
ETA_RT           = 0.910
ETA_1WAY         = float(np.sqrt(ETA_RT))   # ≈ 0.9539
DT               = 0.25                      # 每时段小时数
DP_RAMP          = 65.0                      # MW/15min 爬坡速率上限
L_MIN            = 4                         # 最小连续充/放电时段数（1小时 = 4×15min）
MAX_CHARGE_MWH   = 1200.0                    # 日最大充电量（MWh），1.5×容量
AUX_MWH          = 13.03                     # 日辅助用电（MWh/天）
TERMINAL_DISCOUNT = 0.95                     # 跨日剩余电量的终端价值折扣

# ── 核心 MILP 求解 ──────────────────────────────────────────────────────────
class MILPSolverFailure(RuntimeError):
    """MILP 求解器在时限内未找到可行解。"""
    pass


def _build_milp_15min(prices_96: np.ndarray,
                      eta_c: float = ETA_1WAY,
                      eta_d: float = ETA_1WAY,
                      cap_mwh: float = CAP_MWH,
                      p_max: float = P_MAX_MW,
                      dp_ramp: float = DP_RAMP,
                      l_min: int = L_MIN,
                      max_charge_mwh: float = MAX_CHARGE_MWH,
                      min_charge_mwh: float = 0.0,
                      soc_init: float = 0.0,
                      force_zero_end: bool = True,
                      next_day_avg_price: float = 0.0,
                      prices_discharge_96: np.ndarray | None = None,
                      cap_comp_per_mwh: float = 0.0,
                      enforce_switch_gap: bool = True,
                      time_limit: float = 60.0,
                      raise_on_failure: bool = False):
    """
    构建并求解 15 分钟粒度日前充放电 MILP。

    Args:
        prices_96          : 价格序列；缺省时也用作放电价格（即标准 MILP）。
        prices_discharge_96: 鲁棒模式下放电时假设的价格（通常用 P10 - 偏低）；
                             此时 prices_96 视作充电时假设的价格（通常用 P90 - 偏高）。
        soc_init           : 日初 SOC（MWh），跨日时传入前一天日末 SOC。
        force_zero_end     : 若 True，强制 soc[T-1]=0（日末放空）。
        next_day_avg_price : 次日预测平均电价，用于跨日终端价值。

    Returns:
        (c, d, soc)：各长度 96 的 ndarray；求解失败时返回零数组。
    """
    T  = 96                # 时段数
    N  = 5 * T             # 变量总数
    IC  = 0                # c[0..95]
    ID  = T                # d[0..95]
    IYC = 2 * T            # y_c[0..95]（充电二值）
    IYD = 3 * T            # y_d[0..95]（放电二值）
    IS  = 4 * T            # soc[0..95]

    p_charge = np.asarray(prices_96, dtype=float)
    p_discharge = p_charge if prices_discharge_96 is None \
        else np.asarray(prices_discharge_96, dtype=float)

    # ── 目标函数 ─────────────────────────────────────────────────────────────
    # minimize Σ p_charge·c - Σ (p_discharge + cap_comp)·d  (等价于 maximize 净收益)
    # 收益单位：元，各时段能量 = power × DT（MWh）
    # cap_comp_per_mwh：放电容量补偿（元/MWh），与电价并列计入放电收入
    # 系数缩放（OBJ_SCALE）：HiGHS 对绝对系数尺度敏感，缩放后 MIP gap 收敛更稳；
    # 仅缩放目标，约束系数与可行域不变，最优解相同。
    OBJ_SCALE = 1000.0
    obj = np.zeros(N)
    obj[IC:IC + T] = +p_charge * DT / OBJ_SCALE                          # 充电成本
    obj[ID:ID + T] = -(p_discharge + cap_comp_per_mwh) * DT / OBJ_SCALE  # 放电收入 + 补偿
    # 跨日终端价值：持有 1 MWh 至次日的净收益预期 = next_day_avg × eta_d × discount
    if not force_zero_end and next_day_avg_price > 0:
        terminal_value_per_mwh = next_day_avg_price * eta_d * TERMINAL_DISCOUNT
        obj[IS + T - 1] = -terminal_value_per_mwh / OBJ_SCALE

    # ── 变量界 ───────────────────────────────────────────────────────────────
    lb = np.zeros(N)
    ub = np.zeros(N)
    ub[IC:IC + T] = p_max     # c ∈ [0, p_max]
    ub[ID:ID + T] = p_max     # d ∈ [0, p_max]
    ub[IYC:IYC + T] = 1.0    # y_c ∈ {0,1}
    ub[IYD:IYD + T] = 1.0    # y_d ∈ {0,1}
    ub[IS:IS + T] = cap_mwh   # soc ∈ [0, cap_mwh]
    if force_zero_end:
        # 强制日末放空（默认行为，或最后一天）
        ub[IS + T - 1] = 0.0
        lb[IS + T - 1] = 0.0
    # else: soc[T-1] ∈ [0, cap_mwh]，由终端价值项引导

    # ── 整数性 ───────────────────────────────────────────────────────────────
    integ = np.zeros(N)
    integ[IYC:IYC + T] = 1    # y_c 为整数
    integ[IYD:IYD + T] = 1    # y_d 为整数

    # ── 约束计数 ─────────────────────────────────────────────────────────────
    # ①  SOC 动态              T   行
    # ②  c ≤ p_max·y_c         T   行
    # ③  d ≤ p_max·y_d         T   行
    # ④  y_c+y_d ≤ 1           T   行
    # ⑤  y_c[t]+y_d[t+1] ≤ 1  T-1 行  充→放 间隔
    # ⑥  y_d[t]+y_c[t+1] ≤ 1  T-1 行  放→充 间隔
    # ⑦  c ramp-up             T-1 行
    # ⑧  c ramp-down           T-1 行
    # ⑨  d ramp-up             T-1 行
    # ⑩  d ramp-down           T-1 行
    # ⑪  最小连续充电 y_c      按 t×k 展开（≤ T×(l_min-1)）行
    # ⑫  最小连续放电 y_d      同上
    # ⑬  日充电量上限           1   行
    # ⑭  日充电量下限（可选）   1   行（仅当 min_charge_mwh > 0 时启用）
    def _min_run_count(TT, LM):
        """统计最小连续运行约束行数：t∈[0,TT-1], k∈[1,LM-1], t+k<TT"""
        cnt = 0
        for t in range(TT):
            for k in range(1, LM):
                if t + k < TT:
                    cnt += 1
        return cnt

    n_minrun = _min_run_count(T, l_min)  # 每种二值变量的最小运行约束数
    has_min_charge = float(min_charge_mwh) > 0
    # 充放电切换间隔约束 ⑤⑥（共 2*(T-1) 行）；取消时省去 2*(T-1) 行
    n_switch_rows = 2 * (T - 1) if enforce_switch_gap else 0
    n_con = (4 * T + 4 * (T - 1) + n_switch_rows + 2 * n_minrun + 1
             + (1 if has_min_charge else 0))
    A     = lil_matrix((n_con, N), dtype=float)
    lb_c  = np.full(n_con, -np.inf)
    ub_c  = np.zeros(n_con)

    row = 0

    # ① SOC 动态方程（等式）
    #   soc[t] = soc[t-1] + eta_c·c[t]·DT - d[t]·DT/eta_d
    #   t=0: soc[0] = soc_init + eta_c·c[0]·DT - d[0]·DT/eta_d
    #        → lhs - soc[0] = -soc_init
    for t in range(T):
        A[row, IC + t]  = +eta_c * DT
        A[row, ID + t]  = -DT / eta_d
        if t > 0:
            A[row, IS + t - 1] = +1.0
        A[row, IS + t] = -1.0
        lb_c[row] = ub_c[row] = (-soc_init if t == 0 else 0.0)
        row += 1

    # ② c[t] ≤ p_max·y_c[t]  →  c[t] - p_max·y_c[t] ≤ 0
    for t in range(T):
        A[row, IC  + t] = +1.0
        A[row, IYC + t] = -p_max
        ub_c[row] = 0.0
        row += 1

    # ③ d[t] ≤ p_max·y_d[t]  →  d[t] - p_max·y_d[t] ≤ 0
    for t in range(T):
        A[row, ID  + t] = +1.0
        A[row, IYD + t] = -p_max
        ub_c[row] = 0.0
        row += 1

    # ④ y_c[t] + y_d[t] ≤ 1（不同时充放）
    for t in range(T):
        A[row, IYC + t] = 1.0
        A[row, IYD + t] = 1.0
        ub_c[row] = 1.0
        row += 1

    # ⑤ 充→放间隔（可选）：y_c[t] + y_d[t+1] ≤ 1
    # ⑥ 放→充间隔（可选）：y_d[t] + y_c[t+1] ≤ 1
    # 取消该约束时整数搜索空间显著缩小，HiGHS 收敛更稳；
    # 物理上等价于"充放电切换无需 15min 间隔"。
    if enforce_switch_gap:
        for t in range(T - 1):
            A[row, IYC + t]     = 1.0
            A[row, IYD + t + 1] = 1.0
            ub_c[row] = 1.0
            row += 1
        for t in range(T - 1):
            A[row, IYD + t]     = 1.0
            A[row, IYC + t + 1] = 1.0
            ub_c[row] = 1.0
            row += 1

    # ⑦ c 爬坡上升：c[t] - c[t-1] ≤ dp_ramp
    for t in range(1, T):
        A[row, IC + t]     = +1.0
        A[row, IC + t - 1] = -1.0
        ub_c[row] = dp_ramp
        row += 1

    # ⑧ c 爬坡下降：c[t-1] - c[t] ≤ dp_ramp
    for t in range(1, T):
        A[row, IC + t - 1] = +1.0
        A[row, IC + t]     = -1.0
        ub_c[row] = dp_ramp
        row += 1

    # ⑨ d 爬坡上升
    for t in range(1, T):
        A[row, ID + t]     = +1.0
        A[row, ID + t - 1] = -1.0
        ub_c[row] = dp_ramp
        row += 1

    # ⑩ d 爬坡下降
    for t in range(1, T):
        A[row, ID + t - 1] = +1.0
        A[row, ID + t]     = -1.0
        ub_c[row] = dp_ramp
        row += 1

    # ⑪ 最小连续充电 ≥ l_min 时段
    # 若 y_c[t]=1 且 y_c[t-1]=0（启动时段），则 y_c[t+k]=1 对 k=1..l_min-1
    # 线性化（无需辅助变量）：-y_c[t+k] + y_c[t] - y_c[t-1] ≤ 0
    # t=0 时 y_c[-1]=0（隐含），约束退化为：-y_c[k] + y_c[0] ≤ 0
    for t in range(T):
        for k in range(1, l_min):
            if t + k >= T:
                break
            A[row, IYC + t + k] = -1.0   # -y_c[t+k]
            A[row, IYC + t]     = +1.0   # +y_c[t]
            if t > 0:
                A[row, IYC + t - 1] = -1.0  # -y_c[t-1]
            ub_c[row] = 0.0
            row += 1

    # ⑫ 最小连续放电 ≥ l_min 时段（与 ⑪ 对称）
    for t in range(T):
        for k in range(1, l_min):
            if t + k >= T:
                break
            A[row, IYD + t + k] = -1.0
            A[row, IYD + t]     = +1.0
            if t > 0:
                A[row, IYD + t - 1] = -1.0
            ub_c[row] = 0.0
            row += 1

    # ⑬ 日充电量上限：Σ c[t]·DT ≤ max_charge_mwh
    for t in range(T):
        A[row, IC + t] = DT
    ub_c[row] = max_charge_mwh
    row += 1

    # ⑭ 日充电量下限（可选）：Σ c[t]·DT ≥ min_charge_mwh
    #    用于在大额放电补偿（如 350 元/MWh）下，强制 MILP 做满循环避免遗失补偿。
    if has_min_charge:
        for t in range(T):
            A[row, IC + t] = -DT  # -Σ c·DT ≤ -min_charge_mwh ⇔ Σ c·DT ≥ min_charge_mwh
        ub_c[row] = -float(min_charge_mwh)
        row += 1

    assert row == n_con, f"约束行计数错误: {row} != {n_con}"

    # ── 求解 ─────────────────────────────────────────────────────────────────
    # 仅在大额容量补偿场景下放宽 mip_rel_gap，保证旧约束/无补偿场景与历史一致。
    if cap_comp_per_mwh > 0:
        solver_opts = {
            "disp": False,
            "time_limit": float(time_limit),
            "mip_rel_gap": 1e-3,
        }
    else:
        solver_opts = {
            "disp": False,
            "time_limit": float(time_limit),
        }
    res = milp(
        obj,
        constraints=LinearConstraint(csc_matrix(A), lb_c, ub_c),
        integrality=integ,
        bounds=Bounds(lb, ub),
        options=solver_opts,
    )

    # status 0=最优, 3=超时但有可行解；其他=失败
    if res.status not in (0, 3) or res.x is None:
        msg = (f"MILP 求解未找到可行解 (status={res.status}, "
               f"time_limit={time_limit}s, cap_comp={cap_comp_per_mwh})")
        if raise_on_failure:
            raise MILPSolverFailure(msg)
        import warnings
        warnings.warn(msg + "，返回零方案", RuntimeWarning, stacklevel=2)
        return np.zeros(T), np.zeros(T), np.zeros(T)

    x = res.x
    return x[IC:IC + T], x[ID:ID + T], x[IS:IS + T]


def _build_milp_cbc(prices_96: np.ndarray,
                    eta_c: float = ETA_1WAY,
                    eta_d: float = ETA_1WAY,
                    cap_mwh: float = CAP_MWH,
                    p_max: float = P_MAX_MW,
                    dp_ramp: float = DP_RAMP,
                    l_min: int = L_MIN,
                    max_charge_mwh: float = MAX_CHARGE_MWH,
                    soc_init: float = 0.0,
                    force_zero_end: bool = True,
                    next_day_avg_price: float = 0.0,
                    prices_discharge_96: np.ndarray | None = None,
                    cap_comp_per_mwh: float = 0.0,
                    enforce_switch_gap: bool = True,
                    time_limit: float = 120.0,
                    raise_on_failure: bool = False):
    """PuLP + CBC 版本的 15 分钟 MILP 求解。

    与 _build_milp_15min 完全相同的变量/约束/目标结构，但使用 CBC 求解器
    （对 192 个二值变量 + 容量补偿目标函数远比 HiGHS 稳定）。

    cap_comp_per_mwh 直接进入目标函数：每放电 1 MWh 额外获得 cap_comp 元，
    求解器自动决定最优循环次数——无需 min_charge 硬约束。
    """
    import pulp

    T = 96
    p_charge = np.asarray(prices_96, dtype=float)
    p_discharge = p_charge if prices_discharge_96 is None \
        else np.asarray(prices_discharge_96, dtype=float)

    prob = pulp.LpProblem("battery_dispatch_15min", pulp.LpMaximize)

    c   = [pulp.LpVariable(f"c_{t}",   0, p_max) for t in range(T)]
    d   = [pulp.LpVariable(f"d_{t}",   0, p_max) for t in range(T)]
    y_c = [pulp.LpVariable(f"yc_{t}",  0, 1, cat=pulp.LpBinary) for t in range(T)]
    y_d = [pulp.LpVariable(f"yd_{t}",  0, 1, cat=pulp.LpBinary) for t in range(T)]

    soc_ub = 0.0 if force_zero_end else cap_mwh
    soc = [pulp.LpVariable(f"soc_{t}", 0, cap_mwh) for t in range(T)]
    if force_zero_end:
        soc[T - 1].upBound = 0.0

    # ── 目标函数：maximize 放电收入(含补偿) - 充电成本 + 终端价值 ──
    revenue = pulp.lpSum(
        (p_discharge[t] + cap_comp_per_mwh) * d[t] * DT
        - p_charge[t] * c[t] * DT
        for t in range(T)
    )
    if not force_zero_end and next_day_avg_price > 0:
        terminal = next_day_avg_price * eta_d * TERMINAL_DISCOUNT * soc[T - 1]
        revenue += terminal
    prob += revenue

    # ── 约束 ──

    # ① SOC 动态方程
    for t in range(T):
        prev_soc = soc_init if t == 0 else soc[t - 1]
        prob += soc[t] == prev_soc + eta_c * c[t] * DT - d[t] * DT / eta_d

    # ② c[t] ≤ p_max·y_c[t]
    for t in range(T):
        prob += c[t] <= p_max * y_c[t]

    # ③ d[t] ≤ p_max·y_d[t]
    for t in range(T):
        prob += d[t] <= p_max * y_d[t]

    # ④ y_c[t] + y_d[t] ≤ 1
    for t in range(T):
        prob += y_c[t] + y_d[t] <= 1

    # ⑤⑥ 切换间隔（可选）
    if enforce_switch_gap:
        for t in range(T - 1):
            prob += y_c[t] + y_d[t + 1] <= 1
            prob += y_d[t] + y_c[t + 1] <= 1

    # ⑦⑧ 充电爬坡
    for t in range(1, T):
        prob += c[t] - c[t - 1] <= dp_ramp
        prob += c[t - 1] - c[t] <= dp_ramp

    # ⑨⑩ 放电爬坡
    for t in range(1, T):
        prob += d[t] - d[t - 1] <= dp_ramp
        prob += d[t - 1] - d[t] <= dp_ramp

    # ⑪ 最小连续充电 ≥ l_min 时段
    for t in range(T):
        prev = 0 if t == 0 else y_c[t - 1]
        for k in range(1, l_min):
            if t + k < T:
                prob += y_c[t + k] >= y_c[t] - prev

    # ⑫ 最小连续放电 ≥ l_min 时段
    for t in range(T):
        prev = 0 if t == 0 else y_d[t - 1]
        for k in range(1, l_min):
            if t + k < T:
                prob += y_d[t + k] >= y_d[t] - prev

    # ⑬ 日充电量上限
    prob += pulp.lpSum(c[t] * DT for t in range(T)) <= max_charge_mwh

    # ── 求解 ──
    solver = pulp.PULP_CBC_CMD(
        msg=0,
        timeLimit=int(time_limit),
        gapRel=0.05,
    )
    prob.solve(solver)

    # CBC status: 1=Optimal, 0=Not Solved (可能超时但有可行解), -1=Infeasible, -2=Unbounded
    has_solution = (prob.status == pulp.constants.LpStatusOptimal or
                    (prob.status == pulp.constants.LpStatusNotSolved
                     and prob.sol_status == pulp.constants.LpSolutionFeasible))
    if not has_solution:
        msg = f"CBC 求解失败 (status={prob.status}: {pulp.LpStatus[prob.status]}, time_limit={time_limit}s)"
        if raise_on_failure:
            raise MILPSolverFailure(msg)
        import warnings
        warnings.warn(msg + "，返回零方案", RuntimeWarning, stacklevel=2)
        return np.zeros(T), np.zeros(T), np.zeros(T)

    c_out   = np.array([v.varValue or 0.0 for v in c])
    d_out   = np.array([v.varValue or 0.0 for v in d])
    soc_out = np.array([v.varValue or 0.0 for v in soc])
    return c_out, d_out, soc_out


def solve_day_milp_15min(prices_96: np.ndarray, **kwargs):
    """基于预测电价（15分钟展开）求解最优充放电计划。默认使用 CBC 求解器。"""
    return _build_milp_cbc(np.asarray(prices_96, dtype=float), **kwargs)


def solve_pf_day_15min(actual_96: np.ndarray, **kwargs):
    """完全预知基准：用真实 15 分钟电价求解同一 MILP。默认使用 CBC 求解器。"""
    return _build_milp_cbc(np.asarray(actual_96, dtype=float), **kwargs)


def solve_day_milp_15min_robust(p10_96: np.ndarray,
                                 p50_96: np.ndarray,
                                 p90_96: np.ndarray,
                                 alpha: float = 0.5,
                                 **kwargs):
    """鲁棒 MILP：基于分位数预测求解充放电计划。

    思路（pessimistic minmax over P10–P90）：
      - 充电成本按 (1-α)·P50 + α·P90 估计（假设充电时电价偏高）
      - 放电收入按 (1-α)·P50 + α·P10 估计（假设放电时电价偏低）
      - α=0  退化为标准 MILP（用 P50）
      - α=1  完全保守
      - 推荐 α ∈ [0.3, 0.7]

    Args:
        p10_96, p50_96, p90_96: 各 96 维分位数序列（15 分钟粒度，元/MWh）
        alpha:  鲁棒强度，∈ [0, 1]
        kwargs: 透传给 `_build_milp_15min`（如 soc_init / force_zero_end / next_day_avg_price）

    Returns:
        (c, d, soc)：与 solve_day_milp_15min 一致
    """
    p10 = np.asarray(p10_96, dtype=float)
    p50 = np.asarray(p50_96, dtype=float)
    p90 = np.asarray(p90_96, dtype=float)

    p_charge = (1.0 - alpha) * p50 + alpha * p90
    p_discharge = (1.0 - alpha) * p50 + alpha * p10

    return _build_milp_15min(
        prices_96=p_charge,
        prices_discharge_96=p_discharge,
        **kwargs,
    )


# ── 收益评估 ────────────────────────────────────────────────────────────────
def eval_day_revenue_15min(c: np.ndarray,
                           d: np.ndarray,
                           actual_96: np.ndarray,
                           aux_mwh: float = AUX_MWH) -> dict:
    """用真实 15 分钟节点电价计算当日净收益。"""
    prices = np.asarray(actual_96, dtype=float)
    c = np.asarray(c, dtype=float)
    d = np.asarray(d, dtype=float)

    # 各时段能量（MWh）= 功率(MW) × 0.25h
    e_c = c * DT
    e_d = d * DT

    discharge_rev = float(np.nansum(prices * e_d))
    charge_cost   = float(np.nansum(prices * e_c))
    gross         = discharge_rev - charge_cost

    avg_price = float(np.nanmean(prices))
    aux_cost  = avg_price * aux_mwh
    net       = gross - aux_cost

    return {
        "charge_mwh":    round(float(e_c.sum()), 2),
        "discharge_mwh": round(float(e_d.sum()), 2),
        "charge_cost":   round(charge_cost, 0),
        "discharge_rev": round(discharge_rev, 0),
        "gross":         round(gross, 0),
        "aux_cost":      round(aux_cost, 0),
        "net":           round(net, 0),
    }


# ── 数据加载 ────────────────────────────────────────────────────────────────
def load_actual_15min(xlsx_path: Path) -> pd.DataFrame:
    """
    从日清算副本提取每天 96 个 15 分钟实际节点电价（一期_省内实时节点电价）。
    返回 DataFrame，index=date_str，columns=[slot_0..slot_95]。
    """
    df = pd.read_excel(xlsx_path, header=0)
    df["查询日期"] = pd.to_datetime(df["查询日期"]).dt.date.astype(str)
    df["price_15m"] = pd.to_numeric(df["一期_省内实时节点电价"], errors="coerce")

    # 每天按行顺序取 96 个值
    result = {}
    for date, grp in df.groupby("查询日期", sort=True):
        vals = grp["price_15m"].values
        if len(vals) >= 96:
            result[date] = vals[:96].astype(float)

    out = pd.DataFrame(result).T   # (n_days, 96)
    out.index.name = "date"
    return out


# ── 主流程 ──────────────────────────────────────────────────────────────────
def run(pred_csv: Path,
        actual_xlsx: Path,
        out_csv: Path | None,
        label: str = "MILP-15min策略",
        start: str | None = None,
        end: str | None = None,
        carry_soc: bool = False):
    """
    Args:
        carry_soc: 若 True，启用跨日 SOC 传递——前一天日末剩余电量作为次日初始 SOC，
                   并用次日预测均价计算终端价值（引导合理存留）。
                   若 False（默认），每天独立，强制日末 SOC=0。
    """
    # ── 加载预测：支持小时级（24 点/天）或原生 15 分钟（96 点/天）────────────
    pred = pd.read_csv(pred_csv, parse_dates=["ts"])
    pred["date"] = pred["ts"].dt.date.astype(str)
    if start:
        pred = pred[pred["date"] >= start]
    if end:
        pred = pred[pred["date"] <= end]

    # ── 加载实际 15 分钟价格 ──────────────────────────────────────────────
    actual_df = load_actual_15min(actual_xlsx)

    dates = sorted(pred["date"].unique())
    rows  = []
    mode_str = "跨日SOC" if carry_soc else "日清零"
    print(f"共 {len(dates)} 天，逐日求解 15min MILP（{mode_str}模式）…")

    # 跨日 SOC 状态
    soc_carry     = 0.0   # 策略侧
    soc_carry_pf  = 0.0   # 完全预知侧

    for i, date in enumerate(dates):
        if date not in actual_df.index:
            print(f"  [{date}] 缺少实际15分钟价格，跳过")
            continue

        day_pred = pred[pred["date"] == date].sort_values("ts")
        n = len(day_pred)
        if n == 96:
            pred_96 = day_pred["pred"].values.astype(float)
            pred_native_15m = True
        elif n >= 90:
            pred_96 = day_pred["pred"].values[:96].astype(float)
            pred_native_15m = True
        else:
            if n < 24:
                continue
            pred_hourly = day_pred["pred"].values.astype(float)
            pred_96     = np.repeat(pred_hourly, 4)  # (96,)
            pred_native_15m = False

        # 实际 15 分钟价格
        actual_96 = actual_df.loc[date].values.astype(float)

        # ── 跨日参数 ────────────────────────────────────────────────────
        is_last = (i == len(dates) - 1)
        if carry_soc:
            force_end = is_last  # 仅最后一天强制放空
            # 次日预测均价（用于终端价值计算）
            if not is_last and dates[i + 1] in pred["date"].values:
                next_pred_h = pred[pred["date"] == dates[i + 1]]["pred"].values
                next_avg    = float(np.mean(next_pred_h)) if len(next_pred_h) > 0 else 0.0
            else:
                next_avg = 0.0
        else:
            force_end = True
            next_avg  = 0.0

        # MILP 求解（用预测价格）
        c, d, soc = solve_day_milp_15min(
            pred_96,
            soc_init=soc_carry,
            force_zero_end=force_end,
            next_day_avg_price=next_avg,
        )

        # 完全预知基准（用实际价格，也跟踪 carry_soc_pf）
        c_pf, d_pf, soc_pf = solve_pf_day_15min(
            actual_96,
            soc_init=soc_carry_pf,
            force_zero_end=force_end,
            next_day_avg_price=float(np.mean(actual_96)),  # PF 已知次日均价
        )

        # 更新跨日 SOC（取日末时段值）
        soc_end    = float(soc[-1])    if soc.sum()    > 0 else 0.0
        soc_end_pf = float(soc_pf[-1]) if soc_pf.sum() > 0 else 0.0
        if carry_soc:
            soc_carry    = soc_end
            soc_carry_pf = soc_end_pf
        else:
            soc_carry = soc_carry_pf = 0.0

        # 收益评估（均用实际价格）
        rev    = eval_day_revenue_15min(c,    d,    actual_96)
        rev_pf = eval_day_revenue_15min(c_pf, d_pf, actual_96)

        # 充放电时段摘要（15min 索引 → 小时：槽 t = 分钟/15 → t/4 小时）
        chg_slots = [t for t in range(96) if c[t] > 0.5]
        dis_slots = [t for t in range(96) if d[t] > 0.5]
        chg_str = (f"{chg_slots[0]*15//60:02d}:{chg_slots[0]*15%60:02d}–"
                   f"{(chg_slots[-1]+1)*15//60:02d}:{(chg_slots[-1]+1)*15%60:02d}"
                   if chg_slots else "–")
        dis_str = (f"{dis_slots[0]*15//60:02d}:{dis_slots[0]*15%60:02d}–"
                   f"{(dis_slots[-1]+1)*15//60:02d}:{(dis_slots[-1]+1)*15%60:02d}"
                   if dis_slots else "–")

        rows.append({
            "date":            date,
            "pred_native_15m": pred_native_15m,
            "charge_window":   chg_str,
            "discharge_window": dis_str,
            "charge_mwh":      rev["charge_mwh"],
            "discharge_mwh":   rev["discharge_mwh"],
            "soc_end":         round(soc_end, 1),
            "gross":           rev["gross"],
            "aux_cost":        rev["aux_cost"],
            "net":             rev["net"],
            "pf_gross":        rev_pf["gross"],
            "pf_aux_cost":     rev_pf["aux_cost"],
            "pf_net":          rev_pf["net"],
            # 保存逐时段数据供绘图
            "_c":      c.tolist(),
            "_d":      d.tolist(),
            "_actual": actual_96.tolist(),
            "_pred":   pred_96.tolist(),
            "_soc":    soc.tolist(),
        })

    df = pd.DataFrame(rows)
    plot_cols = ["_c", "_d", "_actual", "_pred", "_soc"]  # pred_native_15m 保留在 CSV

    # 保存（不含绘图数组列）
    if out_csv:
        df.drop(columns=plot_cols).to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"结果已保存：{out_csv}")

    print_report(df, label, carry_soc=carry_soc)
    return df


# ── 报告 ────────────────────────────────────────────────────────────────────
def _week_label(date_str: str) -> str:
    ts = pd.Timestamp(date_str)
    iso = ts.isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def _fmt(v: float) -> str:
    return f"{v / 1e4:+.2f}万"


def print_report(df: pd.DataFrame, label: str = "MILP-15min策略",
                 carry_soc: bool = False):
    print(f"\n{'='*82}")
    print(f" {label}{'  [跨日SOC模式]' if carry_soc else ''}")
    print(f"{'='*82}")

    soc_col = "  SOC末" if carry_soc else ""
    hdr = (f"{'日期':^12} {'充':^12} {'放':^12} "
           f"{'充(MWh)':>8} {'放(MWh)':>8}"
           f"{soc_col}"
           f"  {'净收益(万)':>10} {'PF净(万)':>10} {'兑现率':>7}")
    print(hdr)
    print("-" * 82)

    df = df.copy()
    df["week"] = df["date"].apply(_week_label)

    for wk, wdf in df.groupby("week", sort=False):
        for _, r in wdf.iterrows():
            ratio = r["net"] / r["pf_net"] if abs(r["pf_net"]) > 1 else float("nan")
            ratio_str = f"{ratio:>7.1%}" if not np.isnan(ratio) else f"{'–':>7}"
            soc_str = f"  {r.get('soc_end', 0):>6.0f}" if carry_soc else ""
            print(f"{r['date']:^12} {r['charge_window']:^12} {r['discharge_window']:^12} "
                  f"{r['charge_mwh']:>8.0f} {r['discharge_mwh']:>8.0f}"
                  f"{soc_str}"
                  f"  {_fmt(r['net']):>10} {_fmt(r['pf_net']):>10} {ratio_str}")

        s = wdf[["charge_mwh","discharge_mwh","gross","aux_cost","net","pf_net"]].sum()
        wk_ratio = s["net"] / s["pf_net"] if abs(s["pf_net"]) > 1 else float("nan")
        print(f"  >> 周小计 {wk}: 净={_fmt(s['net'])}  PF净={_fmt(s['pf_net'])}  "
              f"兑现率={wk_ratio:.1%}")
        print()

    print("=" * 82)
    n   = len(df)
    tot = df[["charge_mwh","discharge_mwh","gross","aux_cost","net",
              "pf_gross","pf_aux_cost","pf_net"]].sum()

    print(f"测试期共 {n} 天\n")
    print(f"  {'项目':20}  {'MILP-15min策略':>14}  {'完全预知PF':>12}  {'兑现率':>8}")
    print(f"  {'充电量':20}  {tot['charge_mwh']:>12.1f}MWh  {'–':>12}")
    print(f"  {'放电量':20}  {tot['discharge_mwh']:>12.1f}MWh  {'–':>12}")
    print(f"  {'毛收益':20}  {tot['gross']/1e4:>12.1f}万  {tot['pf_gross']/1e4:>10.1f}万")
    print(f"  {'辅助用电成本':20}  {tot['aux_cost']/1e4:>12.1f}万  {tot['pf_aux_cost']/1e4:>10.1f}万")

    ratio = tot["net"] / tot["pf_net"] if abs(tot["pf_net"]) > 1 else float("nan")
    print(f"  {'净收益':20}  {tot['net']/1e4:>12.1f}万  {tot['pf_net']/1e4:>10.1f}万  "
          f"{ratio:>7.1%}")

    ann_strat = tot["net"]    / n * 365
    ann_pf    = tot["pf_net"] / n * 365
    print(f"\n  全年外推 策略: {ann_strat/1e8:.3f}亿元  PF: {ann_pf/1e8:.3f}亿元")

    print("\n  对比参考（小时级 MILP，jan25版 81 天）:")
    print("    净收益: 987.3万  兑现率: 45.1%  全年外推: 0.445亿")
    print("  对比参考（旧启发式方案B）:")
    print("    净收益: 924.8万  兑现率: 61.6%  全年外推: 0.416亿")
    print("  实际现货净收益: 807.3万  全年外推: 0.364亿")


# ── 绘图 ────────────────────────────────────────────────────────────────────
def plot_weekly(df: pd.DataFrame, out_dir: Path, show_pred_price: bool = True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.font_manager as fm

    font_path = "/usr/share/fonts/wqy-microhei/wqy-microhei.ttc"
    fm.fontManager.addfont(font_path)
    prop = fm.FontProperties(fname=font_path)
    plt.rcParams["font.family"] = prop.get_name()
    plt.rcParams["axes.unicode_minus"] = False
    FS = 8.5

    out_dir.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df["week"] = df["date"].apply(_week_label)

    t_axis = np.arange(96) * 15 / 60   # 小时轴，0~24

    for wk, wdf in df.groupby("week", sort=False):
        days = wdf["date"].tolist()
        n = len(days)
        fig, axes = plt.subplots(n, 1, figsize=(15, 3.4 * n), constrained_layout=True)
        if n == 1:
            axes = [axes]
        title_suffix = "" if show_pred_price else "（仅实际价）"
        fig.suptitle(f"MILP-15min 充放电决策与电价对比{title_suffix}  {wk}",
                     fontsize=11, fontweight="bold")

        has_soc = "_soc" in wdf.columns

        for ax, (_, r) in zip(axes, wdf.iterrows()):
            actual_96 = np.array(r["_actual"])
            pred_96   = np.array(r["_pred"]) if show_pred_price else None
            c         = np.array(r["_c"])
            d         = np.array(r["_d"])
            soc_96    = np.array(r["_soc"]) if has_soc else None

            # 背景色块（15分钟粒度）
            for t in range(96):
                x0 = t * 15 / 60
                x1 = (t + 1) * 15 / 60
                if c[t] > 0.5:
                    ax.axvspan(x0, x1, color="#BBDEFB", alpha=0.75, zorder=0)
                if d[t] > 0.5:
                    ax.axvspan(x0, x1, color="#FFCDD2", alpha=0.75, zorder=0)

            # 价格曲线（15分钟实际；可选叠加小时预测阶梯）
            ax.plot(t_axis + 15/120, actual_96,
                    color="#1565C0", lw=1.5, label="实际价(15min)", alpha=0.9, zorder=3)
            if show_pred_price:
                native = bool(r.get("pred_native_15m", False))
                if native:
                    ax.plot(
                        t_axis + 15 / 120,
                        pred_96,
                        color="#E53935",
                        lw=1.3,
                        ls="--",
                        label="预测价(15min)",
                        zorder=3,
                        alpha=0.88,
                    )
                else:
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

            # 充放电功率次轴（柱状）+ SOC 曲线
            ax2 = ax.twinx()
            bar_w = 14 / 60   # ≈ 0.23h（窄于15min格）
            ax2.bar(t_axis + 15/120, c,  width=bar_w, color="#1565C0", alpha=0.3, label="充电(MW)", zorder=2)
            ax2.bar(t_axis + 15/120, -d, width=bar_w, color="#C62828", alpha=0.3, label="放电(MW)", zorder=2)
            if soc_96 is not None and soc_96.max() > 1:
                # SOC 叠加在功率轴，用绿色虚线，右侧刻度覆盖 0~800 MWh
                ax2.plot(t_axis + 15/120, soc_96, color="#2E7D32", lw=1.2, ls=":",
                         alpha=0.85, label="SOC(MWh)", zorder=4)
            ax2.set_ylim(-280, 280)
            ax2.set_ylabel("功率(MW) / SOC(MWh×0.35)", fontsize=FS - 1, color="#888888")
            ax2.tick_params(labelsize=FS - 1, colors="#888888")
            ax2.axhline(0, color="#aaaaaa", lw=0.5, ls=":")

            # 标题
            soc_end_str = f"  SOC末:{r.get('soc_end', 0):.0f}MWh" if has_soc else ""
            ax.set_title(
                f"{r['date']}   充: {r['charge_window']} {r['charge_mwh']:.0f}MWh  "
                f"放: {r['discharge_window']} {r['discharge_mwh']:.0f}MWh  "
                f"净收益: {r['net']/1e4:+.2f}万{soc_end_str}",
                fontsize=FS, loc="left", pad=3)

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
                ncol_leg = 4 if show_pred_price else 3
                ax.legend(handles=h1 + [patch_c, patch_d],
                          labels=l1 + ["充电时段", "放电时段"],
                          loc="upper right", fontsize=FS - 1, framealpha=0.85, ncol=ncol_leg)

        out_path = out_dir / f"{wk}.png"
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  {out_path.name}")


# ── CLI ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="MILP 15分钟充放电调度策略")
    ap.add_argument(
        "--pred", required=True,
        help="预测 CSV：test_predictions_hourly.csv（24 点/天）"
             " 或 test_predictions_15min.csv（96 点/天）",
    )
    ap.add_argument("--actual_xlsx", required=True,
                    help="日清算结果查询电厂侧(1)_副本.xlsx")
    ap.add_argument("--out",   default=None,  help="输出 CSV 路径（可选）")
    ap.add_argument("--plots", default=None,  help="绘图输出目录（可选）")
    ap.add_argument("--label", default="MILP-15min策略（爬坡+间隔约束）")
    ap.add_argument("--start", default=None,  help="起始日期 如 2026-01-27")
    ap.add_argument("--end",   default=None,  help="截止日期 如 2026-04-17")
    ap.add_argument("--carry_soc", action="store_true",
                    help="启用跨日SOC传递（默认关闭，日末强制放空）")
    ap.add_argument("--hide-pred-in-plots", action="store_true",
                    help="周图中不绘制预测价曲线（仍用预测求解 MILP）")
    args = ap.parse_args()

    df = run(
        pred_csv=Path(args.pred),
        actual_xlsx=Path(args.actual_xlsx),
        out_csv=Path(args.out) if args.out else None,
        label=args.label,
        start=args.start,
        end=args.end,
        carry_soc=args.carry_soc,
    )

    if args.plots:
        print("\n开始绘图 …")
        plot_weekly(df, Path(args.plots), show_pred_price=not args.hide_pred_in_plots)
        print("绘图完成！")


if __name__ == "__main__":
    main()
