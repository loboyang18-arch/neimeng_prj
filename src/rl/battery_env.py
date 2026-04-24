"""
Gymnasium 储能充放电环境
========================
一天 = 一个 episode（96 步 × 15 min），日初 SOC=0，日末引导放空。

两种观测模式：
- flat 模式 (use_v8=False): Box(31,) — 兼容旧版纯 MLP 策略
- v8 模式 (use_v8=True): Dict 观测 — 供 V8 编码器 + RL 策略使用
    "battery_state": Box(7,)
    "market_flat": Box(24,)
    "v8_grid": Box(C_TOTAL, H_SLOTS, LOOKBACK_DAYS)
    "v8_target": Box(1,)   — 当前时段真实价格（辅助监督用，训练时可用）

动作空间 [-1, 1] 映射到 [-P_MAX, +P_MAX] MW（正=放电，负=充电）。
"""
from __future__ import annotations

import datetime
import math
from typing import Any, Dict, List, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .battery_cfg import (
    DEFAULT_CFG,
    LAG_ACTUAL_COLS,
    LAG_PRICE_COLS,
    MARKET_SNAPSHOT_COLS,
    OBS_DIM,
    BatteryConfig,
)


class BatteryEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        episodes: List[dict],
        norm: Optional[dict] = None,
        cfg: Optional[BatteryConfig] = None,
        soc_end_penalty: float = 50.0,
        reward_scale: float = 1e4,
        v8_grid_cache=None,
    ):
        super().__init__()
        self.episodes = episodes
        self.cfg = cfg or DEFAULT_CFG
        self.soc_end_penalty = soc_end_penalty
        self.reward_scale = reward_scale
        self._v8_cache = v8_grid_cache
        self.use_v8 = v8_grid_cache is not None

        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

        if self.use_v8:
            from src.model_v8_multitask import C_TOTAL, H_SLOTS, LOOKBACK_DAYS
            self.observation_space = spaces.Dict({
                "battery_state": spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
                "market_flat": spaces.Box(-np.inf, np.inf, shape=(len(MARKET_SNAPSHOT_COLS) + len(LAG_PRICE_COLS)*2 + len(LAG_ACTUAL_COLS)*2,), dtype=np.float32),
                "v8_grid": spaces.Box(-np.inf, np.inf, shape=(C_TOTAL, H_SLOTS, LOOKBACK_DAYS), dtype=np.float32),
                "v8_target": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
            })
        else:
            self.observation_space = spaces.Box(
                -np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32
            )

        if norm:
            self._feat_norm = np.array(norm["features_p99"], dtype=np.float32)
            self._lag_p_norm = np.array(norm["lag_price_p99"], dtype=np.float32)
            self._lag_a_norm = np.array(norm["lag_actual_p99"], dtype=np.float32)
        else:
            self._feat_norm = np.ones(len(MARKET_SNAPSHOT_COLS), dtype=np.float32)
            self._lag_p_norm = np.ones(len(LAG_PRICE_COLS), dtype=np.float32)
            self._lag_a_norm = np.ones(len(LAG_ACTUAL_COLS), dtype=np.float32)

        self._ep: Optional[dict] = None
        self.slot = 0
        self.soc = 0.0
        self.last_power = 0.0
        self.mode = 0
        self.mode_slots = 0
        self.day_charge = 0.0
        self.day_discharge = 0.0

    # ─── reset ────────────────────────────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        idx = self.np_random.integers(len(self.episodes))
        self._ep = self.episodes[idx]

        self.slot = 0
        self.soc = 0.0
        self.last_power = 0.0
        self.mode = 0
        self.mode_slots = 0
        self.day_charge = 0.0
        self.day_discharge = 0.0

        return self._obs(), {}

    # ─── observation ──────────────────────────────────────────────────────────
    def _battery_state(self) -> np.ndarray:
        t = min(self.slot, self.cfg.T - 1)
        c = self.cfg
        hour = t * c.dt
        return np.array([
            self.soc / c.cap_mwh,
            t / c.T,
            math.sin(2 * math.pi * hour / 24),
            math.cos(2 * math.pi * hour / 24),
            self.last_power / c.p_max_mw,
            float(self.mode),
            self.mode_slots / c.T,
        ], dtype=np.float32)

    def _market_flat(self) -> np.ndarray:
        t = min(self.slot, self.cfg.T - 1)
        b_group = self._ep["features_96"][t] / self._feat_norm
        c_group = np.concatenate([
            self._ep["lag1_price"] / self._lag_p_norm,
            self._ep["lag2_actual"] / self._lag_a_norm,
            self._ep["lag7_price"] / self._lag_p_norm,
            self._ep["lag8_actual"] / self._lag_a_norm,
        ])
        return np.nan_to_num(
            np.concatenate([b_group, c_group]),
            nan=0.0, posinf=0.0, neginf=0.0,
        )

    def _obs(self):
        if self.use_v8:
            return self._obs_v8()
        return self._obs_flat()

    def _obs_flat(self) -> np.ndarray:
        a = self._battery_state()
        m = self._market_flat()
        return np.nan_to_num(
            np.concatenate([a, m]), nan=0.0, posinf=0.0, neginf=0.0
        )

    def _obs_v8(self) -> Dict[str, np.ndarray]:
        t = min(self.slot, self.cfg.T - 1)
        date_str = self._ep["date"]
        if isinstance(date_str, str):
            date = datetime.date.fromisoformat(date_str)
        else:
            date = date_str

        grid = self._v8_cache.get_grid(date, t)
        target_price = float(self._ep["nodal_price_96"][t])

        return {
            "battery_state": self._battery_state(),
            "market_flat": self._market_flat(),
            "v8_grid": grid,
            "v8_target": np.array([target_price / 500.0], dtype=np.float32),
        }

    # ─── step ─────────────────────────────────────────────────────────────────
    def step(self, action):
        raw_power = float(np.clip(action[0], -1.0, 1.0)) * self.cfg.p_max_mw
        power = self._apply_constraints(raw_power)

        c = self.cfg
        if power < -0.5:
            charge_mw = -power
            self.soc += charge_mw * c.dt * c.eta_c
            self.day_charge += charge_mw * c.dt
            new_mode = 1
        elif power > 0.5:
            discharge_mw = power
            self.soc -= discharge_mw * c.dt / c.eta_d
            self.day_discharge += discharge_mw * c.dt
            new_mode = -1
        else:
            power = 0.0
            new_mode = 0

        if new_mode == self.mode and new_mode != 0:
            self.mode_slots += 1
        elif new_mode != 0:
            self.mode = new_mode
            self.mode_slots = 1
        else:
            if self.mode != 0:
                self.mode = 0
                self.mode_slots = 0

        self.last_power = power

        price = self._ep["nodal_price_96"][self.slot]
        revenue = power * c.dt * price

        self.slot += 1
        terminated = self.slot >= c.T

        soc_penalty = 0.0
        if terminated and self.soc > 1.0:
            soc_penalty = -self.soc * self.soc_end_penalty

        reward = float((revenue + soc_penalty) / self.reward_scale)

        info: Dict[str, Any] = {"revenue": revenue, "soc": self.soc}
        if terminated:
            info["day_revenue"] = 0.0
            info["soc_end"] = self.soc

        return self._obs(), reward, terminated, False, info

    # ─── 约束裁剪 ─────────────────────────────────────────────────────────────
    def _apply_constraints(self, raw_power: float) -> float:
        c = self.cfg
        power = raw_power

        power = np.clip(power, self.last_power - c.dp_ramp_mw,
                         self.last_power + c.dp_ramp_mw)

        if 0 < self.mode_slots < c.l_min:
            if self.mode == 1:
                power = min(power, -1.0)
            elif self.mode == -1:
                power = max(power, 1.0)

        if self.mode_slots >= c.l_min and self.mode != 0:
            if self.mode == 1 and power > 0.5:
                power = 0.0
            elif self.mode == -1 and power < -0.5:
                power = 0.0

        if power < -0.5:
            charge_mw = -power
            max_charge = (c.cap_mwh - self.soc) / (c.eta_c * c.dt)
            max_charge = max(0.0, min(c.p_max_mw, max_charge))
            charge_mw = min(charge_mw, max_charge)
            remaining = c.max_charge_mwh - self.day_charge
            if remaining <= 0:
                charge_mw = 0.0
            else:
                charge_mw = min(charge_mw, remaining / c.dt)
            power = -charge_mw
        elif power > 0.5:
            discharge_mw = power
            max_discharge = self.soc * c.eta_d / c.dt
            max_discharge = max(0.0, min(c.p_max_mw, max_discharge))
            discharge_mw = min(discharge_mw, max_discharge)
            power = discharge_mw

        return power
