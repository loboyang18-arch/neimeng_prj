"""
V10-Quantile — 基于 V10 (Pure Transformer) 的分位数预测版本

输出每个小时 5 个分位数（P10/P30/P50/P70/P90），用 Pinball Loss 训练。
下游 MILP 可基于 (P10, P50, P90) 做鲁棒优化。

模型结构与 V10 相同（input_proj → pos_enc → Transformer encoder），
只把回归头从 (B,24) → (B,24,Q)；方向头与 V10 一致用于训练稳定性。
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import OUTPUT_DIR
from .model_v10_joint import (
    C_TOTAL, H_SLOTS_V10, LOOKBACK_DAYS, DIR_CLASSES, DEVICE,
    HourPositionalEncoding, DailyJointDataset,
    _seed, _load_dws, _build_daily_arrays, _compute_norm,
    pairwise_rank_loss,
)

logger = logging.getLogger(__name__)

# ── 分位数级别 ──────────────────────────────────────────────────
QUANTILES = (0.1, 0.3, 0.5, 0.7, 0.9)
N_Q = len(QUANTILES)
Q_TENSOR = torch.tensor(QUANTILES, dtype=torch.float32).view(1, 1, N_Q)


# ── 模型 ────────────────────────────────────────────────────────

class V10QuantileNet(nn.Module):
    """V10 + 分位数输出头。

    输入: (B, 24, C, 4, 7) 同 V10
    输出:
      quantiles: (B, 24, Q) — 每个小时 Q 个分位数（标准化空间）
      dir_logits: (B, 24, 3) — 涨/平/跌方向分类（与 V10 相同，仅辅助训练）
    """

    def __init__(
        self,
        c_in: int = C_TOTAL,
        h_slots: int = H_SLOTS_V10,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        n_quantiles: int = N_Q,
    ):
        super().__init__()
        self.n_quantiles = n_quantiles
        self._flat_dim = c_in * h_slots * LOOKBACK_DAYS

        self.input_proj = nn.Sequential(
            nn.Linear(self._flat_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pos_enc = HourPositionalEncoding(d_model, max_len=24)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=dim_feedforward, dropout=dropout,
            activation="gelu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # 分位数回归头：输出 (P50, Δ_p70, Δ_p90, Δ_-p30, Δ_-p10)
        # 用累积 softplus 保证单调性：q_k = median ± Σ softplus(...)
        self.q_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_quantiles),  # 直接输出 Q 个分位数（不强制单调，后处理 sort）
        )

        # 方向头与 V10 一致
        self.dir_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, DIR_CLASSES),
        )

    def forward(self, x: torch.Tensor):
        B, T = x.shape[0], x.shape[1]
        feat = x.reshape(B, T, -1)
        feat = self.input_proj(feat)
        feat = self.pos_enc(feat)
        feat = self.transformer(feat)               # (B, 24, D)

        quantiles = self.q_head(feat)               # (B, 24, Q)
        dir_logits = self.dir_head(feat)            # (B, 24, 3)
        return quantiles, dir_logits


# ── Pinball Loss ────────────────────────────────────────────────

def pinball_loss(quantiles: torch.Tensor, target: torch.Tensor,
                 q_levels: torch.Tensor | None = None) -> torch.Tensor:
    """
    quantiles: (B, T, Q) — 模型输出的 Q 个分位数预测
    target:    (B, T)    — 真实值
    q_levels:  (1, 1, Q) — 各分位数等级 ∈ (0, 1)；缺省用 QUANTILES
    """
    if q_levels is None:
        q_levels = Q_TENSOR.to(quantiles.device)
    err = target.unsqueeze(-1) - quantiles          # (B, T, Q)
    loss = torch.maximum(q_levels * err, (q_levels - 1.0) * err)
    return loss.mean()


def quantile_crossing_penalty(quantiles: torch.Tensor) -> torch.Tensor:
    """惩罚分位数交叉：q_{k+1} 应 ≥ q_k。"""
    diff = quantiles[..., 1:] - quantiles[..., :-1]
    return F.relu(-diff).mean()


# ── Evaluation ──────────────────────────────────────────────────

def _eval_v10q(model, loader: DataLoader, y_mean: float, y_std: float):
    """评估：MAE/RMSE 用 P50；同时输出 quantile coverage 与 Pinball loss。"""
    model.eval()
    all_q, all_actual = [], []

    with torch.no_grad():
        for grids, tgt_norm, dir_labels, tgt_raw in loader:
            grids = grids.to(DEVICE)
            q_norm, _ = model(grids)
            q_raw = q_norm.cpu().numpy() * y_std + y_mean   # (B, 24, Q)
            all_q.append(q_raw.reshape(-1, q_raw.shape[-1]))
            all_actual.append(tgt_raw.numpy().reshape(-1))

    q_pred = np.concatenate(all_q, axis=0)  # (N, Q)
    actual = np.concatenate(all_actual)
    mask = ~np.isnan(actual)
    q_pred, actual = q_pred[mask], actual[mask]

    # 后处理：保证分位数单调（取每行的排序结果）
    q_pred_sorted = np.sort(q_pred, axis=-1)

    # P50 索引
    p50_idx = N_Q // 2
    p50 = q_pred_sorted[:, p50_idx]
    mae = float(np.mean(np.abs(p50 - actual)))
    rmse = float(np.sqrt(np.mean((p50 - actual) ** 2)))

    # 覆盖率：多少比例落在 P10–P90 区间
    p10_idx = 0
    p90_idx = N_Q - 1
    coverage = float(np.mean((actual >= q_pred_sorted[:, p10_idx]) &
                             (actual <= q_pred_sorted[:, p90_idx])))

    # 平均区间宽度
    width = float(np.mean(q_pred_sorted[:, p90_idx] - q_pred_sorted[:, p10_idx]))

    # Pinball loss（标准化空间外）
    q_levels = np.array(QUANTILES)
    err = actual[:, None] - q_pred_sorted
    pinball = float(np.mean(np.maximum(q_levels * err, (q_levels - 1.0) * err)))

    return {
        "mae_p50": mae,
        "rmse_p50": rmse,
        "coverage_80": coverage,
        "interval_width": width,
        "pinball": pinball,
    }


# ── Training ────────────────────────────────────────────────────

def train_v10_quantile(
    max_epochs: int = 200,
    lr: float = 5e-4,
    lambda_dir: float = 0.3,
    lambda_rank: float = 0.1,
    lambda_cross: float = 0.5,
    d_model: int = 256,
    n_heads: int = 4,
    n_layers: int = 3,
    test_start: str = "2026-01-27",
    test_end: str = "2026-04-17",
    batch_size: int = 8,
    eval_every: int = 10,
    warmup_epochs: int = 10,
    out_tag: str = "",
):
    out_dir = OUTPUT_DIR / "experiments" / (
        f"v10.0-quantile{out_tag}" if out_tag else "v10.0-quantile"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("V10-Quantile (Pinball loss, %d quantiles)", N_Q)
    logger.info("  分位数: %s", QUANTILES)
    logger.info("  输出目录: %s", out_dir)
    logger.info("=" * 60)

    _seed(42)

    dws = _load_dws()
    valid_dates, day_lag0, day_lag1, day_lag2, day_targets = _build_daily_arrays(dws)

    test_dt = pd.Timestamp(test_start).date()
    test_end_dt = pd.Timestamp(test_end).date()
    train_days = [d for d in valid_dates if d < test_dt]
    test_days = [d for d in valid_dates if test_dt <= d <= test_end_dt]
    logger.info("  训练: %d 天, 测试: %d 天 (%s ~ %s)",
                len(train_days), len(test_days), test_start, test_end)

    norm_mean, norm_std = _compute_norm(day_lag0, day_lag1, day_lag2, train_days)
    tgt_stack = np.stack([day_targets[d] for d in train_days if d in day_targets])
    y_mean = float(tgt_stack.mean())
    y_std = float(tgt_stack.std()) + 1e-8

    ds_kwargs = dict(
        day_lag0=day_lag0, day_lag1=day_lag1, day_lag2=day_lag2,
        day_targets=day_targets,
        norm_mean=norm_mean, norm_std=norm_std,
        y_mean=y_mean, y_std=y_std,
    )
    train_ds = DailyJointDataset(sample_dates=train_days, **ds_kwargs)
    test_ds = DailyJointDataset(sample_dates=test_days, **ds_kwargs)
    logger.info("  train_ds=%d 天, test_ds=%d 天", len(train_ds), len(test_ds))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(
        test_ds, batch_size=min(batch_size, max(len(test_ds), 1)), shuffle=False)

    model = V10QuantileNet(
        c_in=C_TOTAL, h_slots=H_SLOTS_V10,
        d_model=d_model, n_heads=n_heads, n_layers=n_layers,
        n_quantiles=N_Q,
    ).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("  模型参数总量: %d", total_params)

    eval_log = []

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    warmup = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=0.1, total_iters=warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max_epochs - warmup_epochs, eta_min=1e-6)
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt, schedulers=[warmup, cosine], milestones=[warmup_epochs])

    logger.info("=" * 60)
    logger.info("训练 %d epochs, lr=%e, λ_dir=%.2f, λ_rank=%.2f, λ_cross=%.2f",
                max_epochs, lr, lambda_dir, lambda_rank, lambda_cross)
    logger.info("=" * 60)

    for epoch in range(max_epochs):
        model.train()
        ep_pin, ep_ce, ep_rank, ep_cross, nb = 0.0, 0.0, 0.0, 0.0, 0

        for grids, tgt_norm, dir_labels, tgt_raw in train_loader:
            grids = grids.to(DEVICE)
            tgt_norm = tgt_norm.to(DEVICE)
            dir_labels = dir_labels.to(DEVICE)

            opt.zero_grad()
            quantiles, dir_logits = model(grids)         # (B,24,Q), (B,24,3)

            pin = pinball_loss(quantiles, tgt_norm)
            ce = F.cross_entropy(
                dir_logits.reshape(-1, DIR_CLASSES), dir_labels.reshape(-1)
            )
            # rank loss 应用在 P50 上（中位数预测的排序）
            p50_norm = quantiles[..., N_Q // 2]            # (B, 24)
            rank = pairwise_rank_loss(p50_norm, tgt_norm)
            cross = quantile_crossing_penalty(quantiles)

            loss = pin + lambda_dir * ce + lambda_rank * rank + lambda_cross * cross

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            ep_pin += pin.item()
            ep_ce += ce.item()
            ep_rank += rank.item()
            ep_cross += cross.item()
            nb += 1

        sched.step()

        do_eval = (epoch + 1) % eval_every == 0 or epoch == max_epochs - 1
        if do_eval:
            metrics = _eval_v10q(model, test_loader, y_mean, y_std)
            cur_lr = opt.param_groups[0]["lr"]
            logger.info(
                "  ep%3d  Pin=%.4f CE=%.3f Rank=%.4f Cross=%.4f"
                " | MAE_P50=%.1f RMSE_P50=%.1f cov80=%.3f width=%.1f"
                " pinball=%.1f  lr=%.1e",
                epoch + 1,
                ep_pin / max(nb, 1), ep_ce / max(nb, 1),
                ep_rank / max(nb, 1), ep_cross / max(nb, 1),
                metrics["mae_p50"], metrics["rmse_p50"],
                metrics["coverage_80"], metrics["interval_width"],
                metrics["pinball"], cur_lr,
            )
            eval_log.append({"epoch": epoch + 1, **metrics})

    # ── 保存模型 ──
    torch.save(model.state_dict(), out_dir / "model_weights.pt")
    np.save(out_dir / "norm_mean.npy", norm_mean)
    np.save(out_dir / "norm_std.npy", norm_std)
    np.savez(out_dir / "target_stats.npz", y_mean=y_mean, y_std=y_std)
    np.save(out_dir / "quantile_levels.npy", np.array(QUANTILES))

    # ── 生成预测 CSV：long 格式（每小时 Q 行）────────────────────
    model.eval()
    rows_long = []
    with torch.no_grad():
        idx = 0
        for grids, tgt_norm, dir_labels, tgt_raw in DataLoader(
            test_ds, batch_size=1, shuffle=False
        ):
            grids = grids.to(DEVICE)
            quantiles, _ = model(grids)
            q_24q = quantiles.cpu().numpy()[0]                 # (24, Q)
            q_24q = np.sort(q_24q, axis=-1)                    # 强制单调
            q_24q = q_24q * y_std + y_mean
            actual_24 = tgt_raw.numpy()[0]
            d = test_ds.dates[idx]

            for h in range(24):
                row = {
                    "ts": pd.Timestamp(d) + pd.Timedelta(hours=h),
                    "actual": float(actual_24[h]),
                }
                for q_i, q_lvl in enumerate(QUANTILES):
                    row[f"p{int(q_lvl * 100):02d}"] = float(q_24q[h, q_i])
                rows_long.append(row)
            idx += 1

    pred_df = pd.DataFrame(rows_long)
    pred_df = pred_df.set_index("ts").sort_index()
    pred_df.to_csv(out_dir / "test_predictions_quantile.csv")

    # 同时保存兼容 V10 的 P50-only 文件，便于复用 strategy_milp_15min
    pred_df_p50 = pred_df[["actual", "p50"]].rename(columns={"p50": "pred"})
    pred_df_p50.to_csv(out_dir / "test_predictions_hourly.csv")

    logger.info("分位数预测已保存: %s", out_dir / "test_predictions_quantile.csv")

    # ── 最终评估 ──
    final_metrics = _eval_v10q(model, test_loader, y_mean, y_std)
    logger.info("=" * 60)
    logger.info("V10-Quantile RESULTS")
    logger.info("  MAE (P50):       %.2f", final_metrics["mae_p50"])
    logger.info("  RMSE (P50):      %.2f", final_metrics["rmse_p50"])
    logger.info("  Coverage 80%%:    %.3f (target: 0.80)", final_metrics["coverage_80"])
    logger.info("  Interval Width:  %.1f", final_metrics["interval_width"])
    logger.info("  Pinball Loss:    %.2f", final_metrics["pinball"])
    logger.info("  Output:          %s", out_dir)
    logger.info("=" * 60)

    eval_df = pd.DataFrame(eval_log)
    eval_df.to_csv(out_dir / "eval_log.csv", index=False)

    return {
        "predictions": pred_df,
        "metrics": final_metrics,
        "model": model,
        "y_mean": y_mean,
        "y_std": y_std,
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    train_v10_quantile(
        max_epochs=200,
        lr=5e-4,
        lambda_dir=0.3,
        lambda_rank=0.1,
        lambda_cross=0.5,
        eval_every=10,
        out_tag="",
    )
