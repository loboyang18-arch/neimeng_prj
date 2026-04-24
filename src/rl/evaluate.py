"""
测试集评估：RL vs V8+MILP vs PF 对比
======================================
支持两种模式：
- flat: 旧版 31 维 MLP 策略
- v8:   V8 编码器端到端策略
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from stable_baselines3 import SAC

from src.rl.battery_cfg import DEFAULT_CFG
from src.rl.battery_env import BatteryEnv
from src.rl.data_loader import load_and_split

OUT_DIR = ROOT / "output" / "rl"
MILP_RESULT_PATH = (
    ROOT / "output" / "experiments" / "v8.0-jan25-sudun500"
    / "strategy_milp_15min_carry_soc_result.csv"
)


def evaluate_rl_on_test(
    model: SAC,
    test_episodes: list[dict],
    norm: dict,
    v8_grid_cache=None,
) -> pd.DataFrame:
    """逐日在测试集上运行 RL 策略。"""
    cfg = DEFAULT_CFG
    env = BatteryEnv(test_episodes, norm=norm, v8_grid_cache=v8_grid_cache)

    records = []
    for ep in test_episodes:
        env._ep = ep
        env.slot = 0
        env.soc = 0.0
        env.last_power = 0.0
        env.mode = 0
        env.mode_slots = 0
        env.day_charge = 0.0
        env.day_discharge = 0.0

        obs = env._obs()
        gross_revenue = 0.0

        for t in range(cfg.T):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, _, info = env.step(action)
            gross_revenue += info["revenue"]

        avg_price = np.mean(ep["nodal_price_96"])
        aux_cost = cfg.aux_mwh * avg_price

        records.append({
            "date": ep["date"],
            "rl_charge_mwh": round(env.day_charge, 2),
            "rl_discharge_mwh": round(env.day_discharge, 2),
            "rl_gross": round(gross_revenue, 0),
            "rl_aux_cost": round(aux_cost, 0),
            "rl_net": round(gross_revenue - aux_cost, 0),
            "rl_soc_end": round(env.soc, 1),
        })

    return pd.DataFrame(records)


def build_comparison(rl_df: pd.DataFrame) -> pd.DataFrame:
    """合并 RL、V8+MILP、PF 结果。"""
    if not MILP_RESULT_PATH.exists():
        return rl_df

    milp_df = pd.read_csv(MILP_RESULT_PATH)
    milp_df = milp_df.rename(columns={
        "net": "milp_net",
        "gross": "milp_gross",
        "pf_net": "pf_net",
        "pf_gross": "pf_gross",
    })

    merged = pd.merge(
        rl_df,
        milp_df[["date", "milp_net", "milp_gross", "pf_net", "pf_gross"]],
        on="date",
        how="left",
    )
    return merged


def print_summary(comp: pd.DataFrame):
    n_days = len(comp)
    cap = DEFAULT_CFG.cap_mwh

    rl_total = comp["rl_net"].sum()
    milp_total = comp["milp_net"].sum() if "milp_net" in comp else np.nan
    pf_total = comp["pf_net"].sum() if "pf_net" in comp else np.nan

    print(f"\n{'='*60}")
    print(f"测试集评估结果 ({n_days} 天)")
    print(f"{'='*60}")
    print(f"{'策略':<20} {'总净收益(万元)':>14} {'日均(万/天)':>12} {'兑现率':>8}")
    print(f"{'-'*60}")

    def row(name, total, pf_ref):
        daily = total / n_days / 1e4
        total_w = total / 1e4
        rate = total / pf_ref * 100 if pf_ref and not np.isnan(pf_ref) else np.nan
        r_str = f"{rate:.1f}%" if not np.isnan(rate) else "N/A"
        print(f"{name:<20} {total_w:>14.1f} {daily:>12.2f} {r_str:>8}")

    row("RL-V8 (本次训练)", rl_total, pf_total)
    if not np.isnan(milp_total):
        row("V8+15min MILP", milp_total, pf_total)
    if not np.isnan(pf_total):
        row("PF 理论最优", pf_total, pf_total)

    print(f"\n等效循环数:")
    rl_cycles = comp["rl_discharge_mwh"].sum() / cap
    print(f"  RL-V8: {rl_cycles:.1f}")
    print()


def main():
    model_path = OUT_DIR / "best_model.zip"
    if not model_path.exists():
        model_path = OUT_DIR / "sac_v8_final.zip"
    if not model_path.exists():
        model_path = OUT_DIR / "sac_battery_final.zip"
    if not model_path.exists():
        print(f"错误: 找不到训练好的模型 {model_path}")
        sys.exit(1)

    print(f"加载模型: {model_path}")

    print("加载数据（含 V8）…")
    train, val, test, norm, v8_ctx = load_and_split(with_v8=True)
    print(f"测试集: {len(test)} 天 ({test[0]['date']} ~ {test[-1]['date']})")

    from src.rl.v8_data_builder import V8GridCache
    from src.rl.v8_policy import V8FeaturesExtractor
    grid_cache = V8GridCache(
        v8_ctx["day_lag0"], v8_ctx["day_lag1"], v8_ctx["day_lag2"],
        v8_ctx["norm_mean"], v8_ctx["norm_std"],
    )

    custom_objects = {
        "policy_kwargs": dict(
            features_extractor_class=V8FeaturesExtractor,
            features_extractor_kwargs=dict(
                v8_weights_path=None,
                freeze_encoder=False,
            ),
            net_arch=[256, 256],
            share_features_extractor=False,
        ),
    }
    model = SAC.load(str(model_path), custom_objects=custom_objects)

    print("评估 RL-V8 策略…")
    rl_df = evaluate_rl_on_test(model, test, norm, v8_grid_cache=grid_cache)

    comp = build_comparison(rl_df)
    out_csv = OUT_DIR / "test_comparison_v8.csv"
    comp.to_csv(out_csv, index=False)
    print(f"逐日对比已保存: {out_csv}")

    print_summary(comp)
    return comp


if __name__ == "__main__":
    main()
