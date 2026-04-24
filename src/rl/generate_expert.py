"""
用 PF-MILP 为训练集生成专家轨迹
================================
对每天调用 strategy_milp_15min._build_milp_15min（完全预知）求解最优解，
在 BatteryEnv 中回放专家动作，记录 (obs, action) pair 供 BC 使用。

支持两种模式：
- flat: 传统 31 维 obs
- v8:   Dict obs（battery_state + market_flat + v8_grid + v8_target）
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.rl.battery_cfg import DEFAULT_CFG
from src.rl.battery_env import BatteryEnv
from src.rl.data_loader import load_and_split


def solve_pf_day(prices_96: np.ndarray) -> tuple:
    """调用 MILP 求解器获取 PF 最优解。"""
    from scripts.strategy_milp_15min import _build_milp_15min
    c, d, soc = _build_milp_15min(prices_96)
    return c, d, soc


def generate_expert_data(
    episodes: list[dict],
    norm: dict,
    out_path: Path,
    v8_grid_cache=None,
    verbose: bool = True,
):
    """
    对 episodes 中每天求解 PF，回放到环境，收集 (obs, action)。

    存储为 .npz:
        flat 模式: obs(N, 31), actions(N, 1), revenues(n_days,)
        v8 模式: battery_state(N, 7), market_flat(N, 24),
                 v8_grid(N, 22, 12, 7), v8_target(N, 1),
                 actions(N, 1), revenues(n_days,)
    """
    cfg = DEFAULT_CFG
    use_v8 = v8_grid_cache is not None
    env = BatteryEnv(episodes, norm=norm, v8_grid_cache=v8_grid_cache)

    if use_v8:
        all_batt, all_mkt, all_grid, all_tgt = [], [], [], []
    else:
        all_obs = []
    all_actions = []
    day_revenues = []

    for i, ep in enumerate(episodes):
        prices = ep["nodal_price_96"]
        c_pf, d_pf, soc_pf = solve_pf_day(prices)
        expert_power = d_pf - c_pf

        env._ep = ep
        env.slot = 0
        env.soc = 0.0
        env.last_power = 0.0
        env.mode = 0
        env.mode_slots = 0
        env.day_charge = 0.0
        env.day_discharge = 0.0

        ep_revenue = 0.0
        for t in range(cfg.T):
            obs = env._obs()
            action_normalized = np.clip(expert_power[t] / cfg.p_max_mw, -1.0, 1.0)

            if use_v8:
                all_batt.append(obs["battery_state"])
                all_mkt.append(obs["market_flat"])
                all_grid.append(obs["v8_grid"])
                all_tgt.append(obs["v8_target"])
            else:
                all_obs.append(obs)
            all_actions.append([action_normalized])

            _, reward, terminated, _, info = env.step(
                np.array([action_normalized], dtype=np.float32)
            )
            ep_revenue += info["revenue"]

        day_revenues.append(ep_revenue)

        if verbose and (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(episodes)}] 最近日收益: {ep_revenue/1e4:.2f} 万")

    act_arr = np.array(all_actions, dtype=np.float32)
    rev_arr = np.array(day_revenues, dtype=np.float32)

    if use_v8:
        np.savez_compressed(
            out_path,
            battery_state=np.array(all_batt, dtype=np.float32),
            market_flat=np.array(all_mkt, dtype=np.float32),
            v8_grid=np.array(all_grid, dtype=np.float32),
            v8_target=np.array(all_tgt, dtype=np.float32),
            actions=act_arr,
            revenues=rev_arr,
        )
    else:
        np.savez_compressed(
            out_path,
            obs=np.array(all_obs, dtype=np.float32),
            actions=act_arr,
            revenues=rev_arr,
        )

    if verbose:
        print(f"\n专家数据已保存: {out_path}")
        if use_v8:
            print(f"  battery_state: {np.array(all_batt).shape}")
            print(f"  v8_grid:       {np.array(all_grid).shape}")
        else:
            print(f"  obs shape:     {np.array(all_obs).shape}")
        print(f"  actions shape: {act_arr.shape}")
        print(f"  PF 日均收益:   {rev_arr.mean()/1e4:.2f} 万/天")
        print(f"  PF 总收益:     {rev_arr.sum()/1e4:.1f} 万 ({len(rev_arr)} 天)")


def main():
    out_dir = ROOT / "output" / "rl"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "expert_data_v8.npz"

    print("加载数据（含 V8 daily_arrays）…")
    train, val, test, norm, v8_ctx = load_and_split(with_v8=True)
    print(f"训练集: {len(train)} 天")

    from src.rl.v8_data_builder import V8GridCache
    grid_cache = V8GridCache(
        v8_ctx["day_lag0"], v8_ctx["day_lag1"], v8_ctx["day_lag2"],
        v8_ctx["norm_mean"], v8_ctx["norm_std"],
    )

    print("生成 PF 专家轨迹（V8 模式）…")
    generate_expert_data(train, norm, out_path, v8_grid_cache=grid_cache)


if __name__ == "__main__":
    main()
