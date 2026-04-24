"""
V8 编码器 + RL 策略网络
========================
将 V8 的 Conv2D 编码器嵌入 SB3 的策略框架：
- V8Encoder: 从 Conv2dMultiTaskNet 提取 block1/2/3 + flatten
- V8FeaturesExtractor: SB3 BaseFeaturesExtractor 子类，处理 Dict obs
- 辅助价格预测头：训练时附加监督损失
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from src.model_v8_multitask import (
    C_TOTAL,
    LOOKBACK_DAYS,
    H_SLOTS,
    Conv2dMultiTaskNet,
)

V8_PRETRAIN_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "output" / "experiments" / "v8.0-rl-pretrain"
)

ENCODER_OUT_DIM = 320   # 64 * h_out(1) * w_out(5)
BATTERY_STATE_DIM = 7
MARKET_FLAT_DIM = 24    # 8 market + 16 lag
BATT_EMBED = 32
MKT_EMBED = 32
FEATURES_DIM = ENCODER_OUT_DIM + BATT_EMBED + MKT_EMBED  # 384


class V8Encoder(nn.Module):
    """V8 Conv2D 编码器（block1/2/3 + flatten），不含预测头。"""

    def __init__(self, h_slots: int = H_SLOTS):
        super().__init__()
        net = Conv2dMultiTaskNet(c_in=C_TOTAL, h_slots=h_slots)
        self.block1 = net.block1
        self.block2 = net.block2
        self.block3 = net.block3
        self.flatten = net.flatten

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.flatten(x)

    @staticmethod
    def load_pretrained(
        weights_path: Optional[Path] = None,
        h_slots: int = H_SLOTS,
    ) -> "V8Encoder":
        """从 V8 完整模型权重中加载编码器部分。"""
        encoder = V8Encoder(h_slots=h_slots)
        wp = weights_path or (V8_PRETRAIN_DIR / "model_weights.pt")
        full_state = torch.load(wp, map_location="cpu", weights_only=True)
        encoder_keys = {}
        for k, v in full_state.items():
            if k.startswith(("block1.", "block2.", "block3.", "flatten.")):
                encoder_keys[k] = v
        encoder.load_state_dict(encoder_keys, strict=True)
        return encoder


class PricePredictionHead(nn.Module):
    """辅助价格预测头（与 V8 原版对齐）。"""

    def __init__(self, feat_dim: int = ENCODER_OUT_DIM):
        super().__init__()
        self.reg_head = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.reg_head(feat).squeeze(-1)

    @staticmethod
    def load_pretrained(
        weights_path: Optional[Path] = None,
    ) -> "PricePredictionHead":
        head = PricePredictionHead()
        wp = weights_path or (V8_PRETRAIN_DIR / "model_weights.pt")
        full_state = torch.load(wp, map_location="cpu", weights_only=True)
        head_keys = {}
        for k, v in full_state.items():
            if k.startswith("reg_head."):
                head_keys[k] = v
        head.load_state_dict(head_keys, strict=True)
        return head


class V8FeaturesExtractor(BaseFeaturesExtractor):
    """SB3 自定义特征提取器：V8 编码器 + 电池/市场嵌入。

    输入 Dict obs:
        "v8_grid": (B, C_TOTAL, H_SLOTS, 7)
        "battery_state": (B, 7)
        "market_flat": (B, 24)

    输出: (B, 384) 拼接特征
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        v8_weights_path: Optional[Path] = None,
        freeze_encoder: bool = False,
    ):
        super().__init__(observation_space, features_dim=FEATURES_DIM)

        if v8_weights_path:
            self.v8_encoder = V8Encoder.load_pretrained(v8_weights_path)
        else:
            self.v8_encoder = V8Encoder()

        if freeze_encoder:
            for p in self.v8_encoder.parameters():
                p.requires_grad = False

        self.batt_mlp = nn.Sequential(
            nn.Linear(BATTERY_STATE_DIM, BATT_EMBED),
            nn.GELU(),
        )
        self.mkt_mlp = nn.Sequential(
            nn.Linear(MARKET_FLAT_DIM, MKT_EMBED),
            nn.GELU(),
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        grid_feat = self.v8_encoder(observations["v8_grid"])
        batt_feat = self.batt_mlp(observations["battery_state"])
        mkt_feat = self.mkt_mlp(observations["market_flat"])
        return torch.cat([grid_feat, batt_feat, mkt_feat], dim=-1)


class V8RLPolicy(nn.Module):
    """独立的 V8+RL 策略网络（用于 BC 预训练，不依赖 SB3）。

    结构:
        V8Encoder(320) + batt_mlp(32) + mkt_mlp(32) → 384
        → policy_mlp(384 → 256 → 256 → 1 → Tanh)
    """

    def __init__(
        self,
        v8_weights_path: Optional[Path] = None,
        freeze_encoder: bool = False,
        hidden: list[int] | None = None,
    ):
        super().__init__()
        self.v8_encoder = (
            V8Encoder.load_pretrained(v8_weights_path)
            if v8_weights_path
            else V8Encoder()
        )
        if freeze_encoder:
            for p in self.v8_encoder.parameters():
                p.requires_grad = False

        self.batt_mlp = nn.Sequential(
            nn.Linear(BATTERY_STATE_DIM, BATT_EMBED),
            nn.GELU(),
        )
        self.mkt_mlp = nn.Sequential(
            nn.Linear(MARKET_FLAT_DIM, MKT_EMBED),
            nn.GELU(),
        )

        hidden = hidden or [256, 256]
        layers = []
        in_dim = FEATURES_DIM
        for h in hidden:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU()])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.Tanh())
        self.policy_head = nn.Sequential(*layers)

        self.price_head = PricePredictionHead(ENCODER_OUT_DIM)

    def forward(
        self, obs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        grid_feat = self.v8_encoder(obs["v8_grid"])
        batt_feat = self.batt_mlp(obs["battery_state"])
        mkt_feat = self.mkt_mlp(obs["market_flat"])
        combined = torch.cat([grid_feat, batt_feat, mkt_feat], dim=-1)
        return self.policy_head(combined)

    def forward_with_aux(
        self, obs: Dict[str, torch.Tensor],
    ):
        """返回 (action, predicted_price) 用于联合训练。"""
        grid_feat = self.v8_encoder(obs["v8_grid"])
        batt_feat = self.batt_mlp(obs["battery_state"])
        mkt_feat = self.mkt_mlp(obs["market_flat"])
        combined = torch.cat([grid_feat, batt_feat, mkt_feat], dim=-1)
        action = self.policy_head(combined)
        price = self.price_head(grid_feat)
        return action, price
