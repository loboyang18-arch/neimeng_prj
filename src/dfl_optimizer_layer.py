"""
可微 LP 松弛层 — 将 MILP 充放电约束翻译为 CVXPY 参数化 LP
=============================================================
LP 松弛：将 y_c, y_d ∈ {0,1} 松弛为 [0,1]，其余约束保持不变。
这使得 KKT 条件可微，梯度可以穿过优化层回传到 V8 预测模型。

训练时用此 LP 近似 MILP 获得梯度；推理时仍用完整 MILP 做最终决策。

约束清单（同 strategy_milp_15min.py）：
  ① SOC 动态  ② SOC 边界  ③ 不同时充放  ④ 功率上限
  ⑤ 爬坡  ⑥ 切换间隔  ⑦ 最小连续运行  ⑧ 日充电量上限
"""
from __future__ import annotations

import cvxpy as cp
import numpy as np
import torch
from cvxpylayers.torch import CvxpyLayer

T = 96
P_MAX = 195.0
CAP_MWH = 800.0
ETA_RT = 0.910
ETA_1WAY = float(np.sqrt(ETA_RT))
DT = 0.25
DP_RAMP = 65.0
L_MIN = 4
MAX_CHARGE_MWH = 1200.0


def build_lp_layer(solver_args: dict | None = None) -> CvxpyLayer:
    """
    构建一个 CvxpyLayer，输入为 96 维预测电价（parameter），
    输出为 (c, d) 各 96 维的充放电功率（LP 松弛最优解）。
    """
    price = cp.Parameter(T, name="price")

    c = cp.Variable(T, nonneg=True, name="c")
    d = cp.Variable(T, nonneg=True, name="d")
    yc = cp.Variable(T, name="yc")
    yd = cp.Variable(T, name="yd")
    soc = cp.Variable(T, name="soc")

    constraints = []

    # 变量边界
    constraints += [c <= P_MAX, d <= P_MAX]
    constraints += [yc >= 0, yc <= 1, yd >= 0, yd <= 1]
    constraints += [soc >= 0, soc <= CAP_MWH]

    # ① SOC 动态方程
    # t=0: soc[0] = 0 + eta_c * c[0] * DT - d[0] * DT / eta_d
    constraints.append(soc[0] == ETA_1WAY * c[0] * DT - d[0] * DT / ETA_1WAY)
    # t>0: soc[t] = soc[t-1] + eta_c * c[t] * DT - d[t] * DT / eta_d
    for t in range(1, T):
        constraints.append(
            soc[t] == soc[t-1] + ETA_1WAY * c[t] * DT - d[t] * DT / ETA_1WAY
        )

    # ② SOC 日末放空
    constraints.append(soc[T-1] == 0)

    # ③ 不同时充放: yc[t] + yd[t] <= 1
    constraints.append(yc + yd <= 1)

    # ④ 功率上限: c[t] <= P_MAX * yc[t], d[t] <= P_MAX * yd[t]
    constraints.append(c <= P_MAX * yc)
    constraints.append(d <= P_MAX * yd)

    # ⑤ 爬坡约束
    for t in range(1, T):
        constraints.append(c[t] - c[t-1] <= DP_RAMP)
        constraints.append(c[t-1] - c[t] <= DP_RAMP)
        constraints.append(d[t] - d[t-1] <= DP_RAMP)
        constraints.append(d[t-1] - d[t] <= DP_RAMP)

    # ⑥ 切换间隔: yc[t]+yd[t+1] <= 1, yd[t]+yc[t+1] <= 1
    for t in range(T - 1):
        constraints.append(yc[t] + yd[t+1] <= 1)
        constraints.append(yd[t] + yc[t+1] <= 1)

    # ⑦ 最小连续运行 (LP 松弛版本)
    # -yc[t+k] + yc[t] - yc[t-1] <= 0  (t=0 时 yc[-1]=0)
    for t in range(T):
        for k in range(1, L_MIN):
            if t + k >= T:
                break
            if t > 0:
                constraints.append(-yc[t+k] + yc[t] - yc[t-1] <= 0)
            else:
                constraints.append(-yc[t+k] + yc[t] <= 0)
    for t in range(T):
        for k in range(1, L_MIN):
            if t + k >= T:
                break
            if t > 0:
                constraints.append(-yd[t+k] + yd[t] - yd[t-1] <= 0)
            else:
                constraints.append(-yd[t+k] + yd[t] <= 0)

    # ⑧ 日充电量上限
    constraints.append(cp.sum(c) * DT <= MAX_CHARGE_MWH)

    # 目标：maximize Σ price * (d - c) * DT
    objective = cp.Maximize(price @ (d - c) * DT)
    problem = cp.Problem(objective, constraints)

    layer = CvxpyLayer(
        problem,
        parameters=[price],
        variables=[c, d],
    )
    return layer


class DiffDispatchLP(torch.nn.Module):
    """
    封装 CvxpyLayer 为 PyTorch Module。
    输入: price_pred (B, 96) — 预测电价
    输出: (c, d) 各 (B, 96) — LP 松弛最优充放电功率
    """
    def __init__(self):
        super().__init__()
        self._layer = build_lp_layer()

    def forward(self, price_pred: torch.Tensor):
        """
        Args:
            price_pred: (B, 96) 预测电价
        Returns:
            c: (B, 96) 充电功率
            d: (B, 96) 放电功率
        """
        batch_size = price_pred.shape[0]
        cs, ds = [], []
        for i in range(batch_size):
            try:
                c_i, d_i = self._layer(
                    price_pred[i],
                    solver_args={"solve_method": "SCS", "max_iters": 5000, "eps": 1e-5},
                )
                cs.append(c_i.unsqueeze(0))
                ds.append(d_i.unsqueeze(0))
            except Exception:
                cs.append(torch.zeros(1, T, dtype=price_pred.dtype, device=price_pred.device))
                ds.append(torch.zeros(1, T, dtype=price_pred.dtype, device=price_pred.device))

        return torch.cat(cs, dim=0), torch.cat(ds, dim=0)


def compute_revenue(c: torch.Tensor, d: torch.Tensor,
                    actual_price: torch.Tensor) -> torch.Tensor:
    """
    用实际电价计算策略收益。
    Args:
        c, d: (B, 96) 充放电功率
        actual_price: (B, 96) 实际电价
    Returns:
        revenue: (B,) 各天净收益（元）
    """
    return ((d - c) * actual_price * DT).sum(dim=1)
