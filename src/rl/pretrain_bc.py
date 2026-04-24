"""
Behavioral Cloning 预训练
=========================
支持两种模式：
- flat: 传统 MLP 策略，31 维 obs
- v8:   V8 编码器 + RL 策略头，Dict obs
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.rl.battery_cfg import OBS_DIM


class BCPolicy(nn.Module):
    """与 SB3 SAC 的 MlpPolicy actor 结构对齐的策略网络（flat 模式）。"""
    def __init__(self, obs_dim: int = OBS_DIM, act_dim: int = 1,
                 hidden: list[int] | None = None):
        super().__init__()
        hidden = hidden or [256, 256]
        layers = []
        in_dim = obs_dim
        for h in hidden:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU()])
            in_dim = h
        layers.append(nn.Linear(in_dim, act_dim))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class V8ExpertDataset(torch.utils.data.Dataset):
    """加载 V8 格式专家数据的 Dataset。"""

    def __init__(self, npz_path: Path):
        data = np.load(npz_path)
        self.batt = torch.from_numpy(data["battery_state"]).float()
        self.mkt = torch.from_numpy(data["market_flat"]).float()
        self.grid = torch.from_numpy(data["v8_grid"]).float()
        self.tgt = torch.from_numpy(data["v8_target"]).float()
        self.actions = torch.from_numpy(data["actions"]).float()

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        return {
            "battery_state": self.batt[idx],
            "market_flat": self.mkt[idx],
            "v8_grid": self.grid[idx],
            "v8_target": self.tgt[idx],
            "action": self.actions[idx],
        }


def train_bc_v8(
    expert_path: Path,
    out_path: Path,
    v8_weights_path: Path,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 3e-4,
    aux_weight: float = 0.1,
    freeze_encoder: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    verbose: bool = True,
):
    """V8 模式 BC 预训练：冻结编码器，训练策略头 + 轻量辅助损失。"""
    from src.rl.v8_policy import V8RLPolicy

    dataset = V8ExpertDataset(expert_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=0, pin_memory=True)

    model = V8RLPolicy(
        v8_weights_path=v8_weights_path,
        freeze_encoder=freeze_encoder,
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  V8RLPolicy: {total} params, {trainable} trainable"
          f" (encoder {'frozen' if freeze_encoder else 'trainable'})")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    policy_loss_fn = nn.MSELoss()
    price_loss_fn = nn.L1Loss()

    for epoch in range(epochs):
        model.train()
        total_policy_loss = 0.0
        total_price_loss = 0.0
        n_batches = 0

        for batch in loader:
            obs = {
                "battery_state": batch["battery_state"].to(device),
                "market_flat": batch["market_flat"].to(device),
                "v8_grid": batch["v8_grid"].to(device),
                "v8_target": batch["v8_target"].to(device),
            }
            target_action = batch["action"].to(device)

            pred_action, pred_price = model.forward_with_aux(obs)
            policy_loss = policy_loss_fn(pred_action, target_action)
            price_loss = price_loss_fn(
                pred_price, obs["v8_target"].squeeze(-1)
            )
            loss = policy_loss + aux_weight * price_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_price_loss += price_loss.item()
            n_batches += 1

        if verbose and ((epoch + 1) % 5 == 0 or epoch == 0):
            avg_p = total_policy_loss / n_batches
            avg_a = total_price_loss / n_batches
            print(f"  Epoch {epoch+1:3d}/{epochs}: "
                  f"policy_loss={avg_p:.6f}  aux_price={avg_a:.4f}")

    torch.save(model.state_dict(), out_path)
    if verbose:
        print(f"\nV8 BC 权重已保存: {out_path}")

    return model


def train_bc(
    expert_path: Path,
    out_path: Path,
    epochs: int = 50,
    batch_size: int = 512,
    lr: float = 3e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    verbose: bool = True,
):
    """原版 flat BC 预训练（保留兼容）。"""
    data = np.load(expert_path)
    obs = torch.from_numpy(data["obs"]).float()
    actions = torch.from_numpy(data["actions"]).float()

    dataset = TensorDataset(obs, actions)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=0, pin_memory=True)

    model = BCPolicy().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        for batch_obs, batch_act in loader:
            batch_obs = batch_obs.to(device)
            batch_act = batch_act.to(device)

            pred = model(batch_obs)
            loss = loss_fn(pred, batch_act)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}: loss = {avg_loss:.6f}")

    torch.save(model.state_dict(), out_path)
    if verbose:
        print(f"\nBC 权重已保存: {out_path}")

    return model


def main():
    out_dir = ROOT / "output" / "rl"
    expert_path = out_dir / "expert_data_v8.npz"
    bc_path = out_dir / "bc_v8_pretrained.pt"
    v8_weights = (ROOT / "output" / "experiments"
                  / "v8.0-rl-pretrain" / "model_weights.pt")

    print("V8 模式 BC 预训练…")
    train_bc_v8(
        expert_path, bc_path, v8_weights,
        epochs=50, freeze_encoder=True, aux_weight=0.1,
    )


if __name__ == "__main__":
    main()
