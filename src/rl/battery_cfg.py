"""
电池物理参数与特征列配置
========================
参数来源：scripts/strategy_milp_15min.py（与 MILP 约束保持一致）
特征来源：docs/dws_ingest_and_v8.md（与 V8 模型特征体系对齐）
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List

# ═══════════════════════════════════════════════════════════════════════════════
# 电池物理参数
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BatteryConfig:
    p_max_mw: float = 195.0
    cap_mwh: float = 800.0
    eta_rt: float = 0.910
    dt: float = 0.25           # 每时段 0.25 小时 (15 min)
    dp_ramp_mw: float = 65.0   # MW/15min 爬坡上限
    l_min: int = 4             # 最小连续充/放时段数 (1 h)
    max_charge_mwh: float = 1200.0  # 日充电量上限
    aux_mwh: float = 13.03    # 日辅助用电 (MWh/天)

    @property
    def eta_c(self) -> float:
        return math.sqrt(self.eta_rt)

    @property
    def eta_d(self) -> float:
        return math.sqrt(self.eta_rt)

    T: int = 96  # 每日时段数

DEFAULT_CFG = BatteryConfig()

# ═══════════════════════════════════════════════════════════════════════════════
# DWS 特征列配置
# ═══════════════════════════════════════════════════════════════════════════════

# B 组：当日市场特征（仅日前已发布的预测/计划类，对应 V8 LAG0）
# 出清电价（price_unified 等）和节点电价均为事后结算数据，不可用于当日决策
MARKET_SNAPSHOT_COLS: List[str] = [
    "load_forecast",
    "renewable_forecast",
    "wind_forecast",
    "solar_forecast",
    "east_send_forecast",
    "reserve_pos_capacity",
    "reserve_neg_capacity",
    "price_dayahead_preclear_energy",
]

# 结算用节点电价（仅用于奖励计算，不进入观测）
NODAL_PRICE_COL = "price_sudun_500kv1m_nodal"

# C 组滞后电价：D-1 / D-7 日均（出清价隔日发布，D-1 可用）
LAG_PRICE_COLS: List[str] = [
    "price_unified",
    "price_hbd",
    "price_hbx",
    "price_sudun_500kv1m_nodal",
]

# C 组滞后实测：D-2 / D-8 日均（实测值发布更晚，V8 从 D-2 开始）
LAG_ACTUAL_COLS: List[str] = [
    "load_actual",
    "renewable_actual",
    "wind_actual",
    "solar_actual",
]

# 所有需要从 DWS 加载的列（去重）
ALL_DWS_COLS: List[str] = sorted(set(
    MARKET_SNAPSHOT_COLS
    + [NODAL_PRICE_COL]
    + LAG_PRICE_COLS
    + LAG_ACTUAL_COLS
))

# 观测维度计算
N_BATTERY_STATE = 7                              # A 组
N_MARKET_SNAPSHOT = len(MARKET_SNAPSHOT_COLS)     # B 组（仅日前特征，无实时价格）
N_LAG = len(LAG_PRICE_COLS) * 2 + len(LAG_ACTUAL_COLS) * 2  # C 组: (D-1+D-7)价格 + (D-2+D-8)实测
OBS_DIM = N_BATTERY_STATE + N_MARKET_SNAPSHOT + N_LAG

# 数据划分日期
TRAIN_START = "2024-12-14"
VAL_START   = "2025-10-01"
TEST_START  = "2026-01-25"
DATA_END    = "2026-04-18"
