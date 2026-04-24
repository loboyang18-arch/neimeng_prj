"""
SAC 训练主脚本（V8 编码器端到端版）
====================================
支持两种模式：
- flat: 旧版 31 维 MLP 策略
- v8:   V8 编码器 + RL 策略，Dict 观测空间，辅助价格预测损失

三阶段策略：
- S1: BC 热启动（编码器冻结，策略头训练）— 由 pretrain_bc.py 完成
- S2: SAC 端到端微调（编码器用小 LR，策略正常 LR）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from src.rl.battery_cfg import OBS_DIM, DEFAULT_CFG
from src.rl.battery_env import BatteryEnv
from src.rl.data_loader import load_and_split

OUT_DIR = ROOT / "output" / "rl"
V8_WEIGHTS = (ROOT / "output" / "experiments"
              / "v8.0-rl-pretrain" / "model_weights.pt")


def _make_env(episodes, norm, v8_grid_cache=None, seed=0):
    def _init():
        env = BatteryEnv(episodes, norm=norm, v8_grid_cache=v8_grid_cache)
        env.reset(seed=seed)
        return env
    return _init


def _make_v8_cache(v8_ctx):
    from src.rl.v8_data_builder import V8GridCache
    return V8GridCache(
        v8_ctx["day_lag0"], v8_ctx["day_lag1"], v8_ctx["day_lag2"],
        v8_ctx["norm_mean"], v8_ctx["norm_std"],
    )


def _load_bc_into_sac_v8(sac_model: SAC, bc_path: Path):
    """将 V8 BC 预训练权重注入 SAC 的 features_extractor + actor。"""
    from src.rl.v8_policy import V8RLPolicy
    bc = V8RLPolicy()
    bc.load_state_dict(
        torch.load(bc_path, map_location="cpu", weights_only=True)
    )

    fe = sac_model.policy.actor.features_extractor
    fe.v8_encoder.load_state_dict(bc.v8_encoder.state_dict())
    fe.batt_mlp.load_state_dict(bc.batt_mlp.state_dict())
    fe.mkt_mlp.load_state_dict(bc.mkt_mlp.state_dict())

    for target_fe in [sac_model.policy.critic.features_extractor,
                      sac_model.policy.critic_target.features_extractor]:
        target_fe.v8_encoder.load_state_dict(bc.v8_encoder.state_dict())
        target_fe.batt_mlp.load_state_dict(bc.batt_mlp.state_dict())
        target_fe.mkt_mlp.load_state_dict(bc.mkt_mlp.state_dict())

    bc_pi_linears = [m for m in bc.policy_head if isinstance(m, torch.nn.Linear)]
    sac_pi_linears = []
    for m in sac_model.policy.actor.latent_pi.modules():
        if isinstance(m, torch.nn.Linear):
            sac_pi_linears.append(m)
    sac_pi_linears.append(sac_model.policy.actor.mu)

    if len(bc_pi_linears) == len(sac_pi_linears):
        with torch.no_grad():
            for bc_l, sac_l in zip(bc_pi_linears, sac_pi_linears):
                sac_l.weight.copy_(bc_l.weight)
                sac_l.bias.copy_(bc_l.bias)
        print("BC 策略头权重已注入 SAC actor")
    else:
        print(f"警告: BC 层数 {len(bc_pi_linears)} != SAC 层数 {len(sac_pi_linears)}，"
              f"仅注入特征提取器")

    print("V8 BC 权重已注入 SAC 特征提取器")


def main(
    total_timesteps: int = 500_000,
    n_envs: int = 4,
    eval_freq: int = 5000,
    n_eval_episodes: int = 50,
    use_bc: bool = True,
    encoder_lr: float = 1e-5,
    policy_lr: float = 3e-4,
):
    bc_path = OUT_DIR / "bc_v8_pretrained.pt"
    log_dir = OUT_DIR / "sac_v8_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    print("加载数据（含 V8 daily_arrays）…")
    train, val, test, norm, v8_ctx = load_and_split(with_v8=True)
    print(f"训练集: {len(train)} 天, 验证集: {len(val)} 天")

    norm_path = OUT_DIR / "norm_stats.json"
    with open(norm_path, "w") as f:
        json.dump(norm, f)

    grid_cache = _make_v8_cache(v8_ctx)

    print(f"创建 {n_envs} 个并行训练环境（V8 模式）…")
    train_envs = DummyVecEnv(
        [_make_env(train, norm, v8_grid_cache=grid_cache, seed=i)
         for i in range(n_envs)]
    )
    eval_env = DummyVecEnv(
        [_make_env(val, norm, v8_grid_cache=grid_cache, seed=42)]
    )

    from src.rl.v8_policy import V8FeaturesExtractor, FEATURES_DIM
    policy_kwargs = dict(
        features_extractor_class=V8FeaturesExtractor,
        features_extractor_kwargs=dict(
            v8_weights_path=V8_WEIGHTS,
            freeze_encoder=False,
        ),
        net_arch=[256, 256],
        share_features_extractor=False,
    )

    print("初始化 SAC（MultiInputPolicy + V8FeaturesExtractor）…")
    model = SAC(
        "MultiInputPolicy",
        train_envs,
        learning_rate=policy_lr,
        buffer_size=200_000,
        batch_size=256,
        gamma=0.99,
        ent_coef="auto",
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=str(log_dir / "tb"),
        seed=42,
    )

    if use_bc and bc_path.exists():
        _load_bc_into_sac_v8(model, bc_path)
    else:
        print("未找到 V8 BC 权重或 use_bc=False，从 V8 预训练编码器 + 随机策略头开始")

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(OUT_DIR),
        log_path=str(log_dir),
        eval_freq=max(eval_freq // n_envs, 1),
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        verbose=1,
    )

    print(f"\n开始 SAC V8 端到端训练: {total_timesteps} 步")
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        progress_bar=True,
    )

    final_path = OUT_DIR / "sac_v8_final"
    model.save(str(final_path))
    print(f"\n最终模型已保存: {final_path}")
    print(f"最佳模型: {OUT_DIR / 'best_model.zip'}")

    train_envs.close()
    eval_env.close()


if __name__ == "__main__":
    main()
