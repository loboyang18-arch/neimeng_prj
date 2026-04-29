"""
V10.0 — 全日联合预测：Pure Transformer

每小时的原始特征 (C×4×7) 直接展平 → Linear 投影 → Transformer Encoder 建模全天
24 小时间关系 → 联合输出 24 个价格预测。

无 Conv2D 编码器，完全依赖 Transformer 学习时序模式。
"""

import logging
import math
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

# ── 环境变量（在任何依赖它们的模块导入前设置）──────────────────
NODAL_COL = "price_sudun_500kv1m_nodal"
os.environ.setdefault("NM_V8_TARGET", NODAL_COL)
os.environ.setdefault("NM_V8_EXTRA_LAG1",
                       "price_sudun_500kv1m_energy,price_sudun_500kv1m_cong")
os.environ.setdefault("NM_V8_HOURLY_AGG", "mean4")
os.environ.setdefault("NM_MIN_FEATURE_DATE", "2024-12-14")

from .config import OUTPUT_DIR
from .fill_sudun_dws_gaps import fill_sudun_price_columns
from .experiment.splits import MIN_FEATURE_CALENDAR_DATE

# ── 通道定义（与 V8 保持一致）─────────────────────────────────
LAG0_COLS = (
    "load_forecast", "renewable_forecast", "wind_forecast",
    "solar_forecast", "east_send_forecast",
    "reserve_pos_capacity", "reserve_neg_capacity",
    "price_dayahead_preclear_energy",
)
LAG2_COLS = ("load_actual", "renewable_actual", "wind_actual", "solar_actual")
TARGET_COL = os.environ.get("NM_V8_TARGET", "price_unified")
HOURLY_AGG = os.environ.get("NM_V8_HOURLY_AGG", "slot0").lower()

_DEFAULT_LAG1 = ("price_unified", "price_hbd", "price_hbx")
_extra = [c.strip() for c in os.environ.get("NM_V8_EXTRA_LAG1", "").split(",") if c.strip()]
_lag1 = [TARGET_COL]
for c in _DEFAULT_LAG1:
    if c not in _lag1:
        _lag1.append(c)
for c in _extra:
    if c not in _lag1:
        _lag1.append(c)
LAG1_COLS = tuple(_lag1)

C_LAG0 = len(LAG0_COLS) + 4  # +4 for time encoding
C_LAG1 = len(LAG1_COLS)
C_LAG2 = len(LAG2_COLS)
C_TOTAL = C_LAG0 + C_LAG1 + C_LAG2

LOOKBACK_DAYS = 7
SLOTS_PER_DAY = 96
SLOTS_PER_HOUR = 4
H_SLOTS_V10 = 4  # V10: 每小时仅取自身 4 个 15min 槽
DIR_CLASSES = 3

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if getattr(torch.backends, "mps", None)
             and torch.backends.mps.is_available()
    else "cpu"
)

V10_DIR = OUTPUT_DIR / "experiments" / "v10.0-joint"


# ── 工具函数 ──────────────────────────────────────────────────

def _seed(s=42):
    np.random.seed(s)
    torch.manual_seed(s)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(s)


def _load_dws() -> pd.DataFrame:
    path = OUTPUT_DIR / "dws_15min_features.csv"
    df = pd.read_csv(path, parse_dates=["ts"], index_col="ts").sort_index()
    df = fill_sudun_price_columns(df)
    logger.info("DWS: %d rows x %d cols (%s ~ %s)",
                len(df), len(df.columns), df.index.min(), df.index.max())
    return df


def _build_daily_arrays(df: pd.DataFrame):
    """按日构建 lag0/lag1/lag2 特征和目标。与 V8 共用相同的通道定义。"""
    start_date = df.index.min().normalize().date()
    end_date = df.index.max().date()
    date_range = pd.date_range(start_date, end_date, freq="D")
    day_lag0: Dict = {}
    day_lag1: Dict = {}
    day_lag2: Dict = {}
    day_targets: Dict = {}
    valid: List = []

    for d_ts in date_range:
        d = d_ts.date()
        grid = pd.date_range(pd.Timestamp(d), periods=96, freq="15min")
        raw = df.reindex(grid).copy()
        if raw.isna().to_numpy().all():
            continue

        l0 = raw[list(LAG0_COLS)].values.astype(np.float32)
        steps = np.arange(96, dtype=np.float32)
        dow = float(pd.Timestamp(d).dayofweek)
        te = np.column_stack([
            np.sin(2 * np.pi * steps / 96),
            np.cos(2 * np.pi * steps / 96),
            np.full(96, np.sin(2 * np.pi * dow / 7), dtype=np.float32),
            np.full(96, np.cos(2 * np.pi * dow / 7), dtype=np.float32),
        ])
        day_lag0[d] = np.concatenate([l0, te], axis=1).astype(np.float32)  # (96, C_LAG0)
        day_lag1[d] = raw[list(LAG1_COLS)].values.astype(np.float32)      # (96, C_LAG1)
        day_lag2[d] = raw[list(LAG2_COLS)].values.astype(np.float32)      # (96, C_LAG2)

        tgt_96 = raw[TARGET_COL].values.astype(np.float32)

        if HOURLY_AGG == "mean4":
            hourly_y = tgt_96.reshape(24, 4).mean(axis=1).astype(np.float32)
        else:
            hourly_y = tgt_96[np.arange(0, 96, 4)]

        if len(hourly_y) == 24 and np.isfinite(hourly_y).all():
            day_targets[d] = hourly_y
            if d >= MIN_FEATURE_CALENDAR_DATE:
                valid.append(d)

    valid = sorted(valid)
    logger.info("Daily arrays: %d days, %d with target", len(day_lag0), len(valid))
    return valid, day_lag0, day_lag1, day_lag2, day_targets


def _compute_norm(day_lag0, day_lag1, day_lag2, train_days):
    rows = []
    for d in train_days:
        if d in day_lag0 and d in day_lag1 and d in day_lag2:
            row = np.concatenate([day_lag0[d], day_lag1[d], day_lag2[d]], axis=1)
            rows.append(row)
    stack = np.concatenate(rows, axis=0)
    mean = np.nanmean(stack, axis=0).astype(np.float32)
    std = np.nanstd(stack, axis=0).astype(np.float32) + 1e-8
    return mean, std


def _get_hour_own_slots(day_arrays: Dict, d, h: int):
    """获取日 d 第 h 小时自身的 4 个 15min 切片 → (4, C)。不扩展前后小时。"""
    arr = day_arrays[d]  # (96, C)
    s = h * SLOTS_PER_HOUR
    return arr[s:s + SLOTS_PER_HOUR].copy()


# ── V10 Model ─────────────────────────────────────────────────────

class HourPositionalEncoding(nn.Module):
    """固定正弦位置编码，24 个位置。"""

    def __init__(self, d_model: int, max_len: int = 24):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, 24, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class V10JointNet(nn.Module):
    """
    全日联合预测网络 — Pure Transformer

    输入: (B, 24, C, 4, 7)  — 24 小时，每小时 4 个 15min × 7 天
      → 展平为 (B, 24, C*4*7)
      → Linear 投影 → (B, 24, d_model)
      → + 位置编码
      → Transformer Encoder (L 层)
      → (B, 24, d_model)
      ├→ 回归头 → (B, 24) 价格
      └→ 方向头 → (B, 24, 3) 涨/平/跌
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
    ):
        super().__init__()
        self._flat_dim = c_in * h_slots * LOOKBACK_DAYS  # 22 * 4 * 7 = 616

        # ── 输入投影（展平 → d_model）──
        self.input_proj = nn.Sequential(
            nn.Linear(self._flat_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pos_enc = HourPositionalEncoding(d_model, max_len=24)

        # ── Transformer Encoder ──
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers,
        )

        # ── 回归头（逐小时共享权重）──
        self.reg_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        # ── 方向分类头 ──
        self.dir_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, DIR_CLASSES),
        )

    def forward(self, x: torch.Tensor):
        """
        x: (B, 24, C, 4, 7)
        Returns:
          prices: (B, 24)
          directions: (B, 24, 3)
        """
        B, T = x.shape[0], x.shape[1]
        feat = x.reshape(B, T, -1)                 # (B, 24, C*4*7)
        feat = self.input_proj(feat)                # (B, 24, d_model)
        feat = self.pos_enc(feat)                   # + positional encoding
        feat = self.transformer(feat)               # (B, 24, d_model)

        prices = self.reg_head(feat).squeeze(-1)    # (B, 24)
        directions = self.dir_head(feat)            # (B, 24, 3)
        return prices, directions


# ── V10 Dataset ───────────────────────────────────────────────────

class DailyJointDataset(Dataset):
    """每日 1 个样本：24 × (C, 4, 7) grid 栈 + 24 维目标 + 24 维方向标签。

    每小时仅取自身 4 个 15min 槽，不扩展前后小时。
    """

    def __init__(
        self,
        sample_dates: List,
        day_lag0: Dict, day_lag1: Dict, day_lag2: Dict,
        day_targets: Dict,
        norm_mean: np.ndarray, norm_std: np.ndarray,
        y_mean: float, y_std: float,
    ):
        self.items = []
        self.dates = []

        for d in sample_dates:
            if d not in day_targets:
                continue

            dates0 = [(pd.Timestamp(d) - pd.Timedelta(days=off)).date()
                      for off in range(LOOKBACK_DAYS - 1, -1, -1)]
            dates1 = [(pd.Timestamp(d) - pd.Timedelta(days=off)).date()
                      for off in range(LOOKBACK_DAYS, 0, -1)]
            dates2 = [(pd.Timestamp(d) - pd.Timedelta(days=off)).date()
                      for off in range(LOOKBACK_DAYS + 1, 1, -1)]

            a0, a1, a2 = set(day_lag0), set(day_lag1), set(day_lag2)
            if not (all(dd in a0 for dd in dates0) and
                    all(dd in a1 for dd in dates1) and
                    all(dd in a2 for dd in dates2)):
                continue

            grids_24 = []
            for h in range(24):
                layers = []
                for k in range(LOOKBACK_DAYS):
                    s0 = _get_hour_own_slots(day_lag0, dates0[k], h)  # (4, C_LAG0)
                    s1 = _get_hour_own_slots(day_lag1, dates1[k], h)  # (4, C_LAG1)
                    s2 = _get_hour_own_slots(day_lag2, dates2[k], h)  # (4, C_LAG2)
                    layers.append(np.concatenate([s0, s1, s2], axis=1))  # (4, C_TOTAL)
                grid = np.stack(layers, axis=-1)      # (4, C_TOTAL, 7)
                grid = grid.transpose(1, 0, 2)        # (C_TOTAL, 4, 7)
                grid = np.nan_to_num(grid, nan=0.0)
                grid = ((grid - norm_mean.reshape(C_TOTAL, 1, 1))
                        / norm_std.reshape(C_TOTAL, 1, 1)).astype(np.float32)
                grids_24.append(grid)

            grids = np.stack(grids_24, axis=0)  # (24, C, 4, 7)

            targets = day_targets[d]  # (24,)
            tgt_norm = ((targets - y_mean) / y_std).astype(np.float32)

            d_prev = (pd.Timestamp(d) - pd.Timedelta(days=1)).date()
            dir_labels = np.zeros(24, dtype=np.int64)
            for h in range(24):
                if h > 0:
                    diff = targets[h] - targets[h - 1]
                elif d_prev in day_targets:
                    diff = targets[0] - day_targets[d_prev][23]
                else:
                    diff = 0.0
                dir_labels[h] = 2 if diff > 0 else (0 if diff < 0 else 1)

            self.items.append((grids, tgt_norm, dir_labels, targets))
            self.dates.append(d)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        grids, tgt_norm, dir_labels, targets_raw = self.items[idx]
        return (
            torch.from_numpy(grids),
            torch.from_numpy(tgt_norm),
            torch.from_numpy(dir_labels),
            torch.from_numpy(targets_raw.astype(np.float32)),
        )


# ── Loss Functions ────────────────────────────────────────────────

def pairwise_rank_loss(pred: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
    """排序损失：鼓励预测和实际具有相同的高低价排列。

    对所有 (i, j) 对: 如果 actual[i] > actual[j]，则 pred[i] 应 > pred[j]。
    """
    diff_pred = pred.unsqueeze(-1) - pred.unsqueeze(-2)      # (B, 24, 24)
    diff_actual = actual.unsqueeze(-1) - actual.unsqueeze(-2)  # (B, 24, 24)
    sign = torch.sign(diff_actual)
    margin = 1.0
    loss = F.relu(margin - sign * diff_pred)
    mask = (diff_actual.abs() > 1e-6).float()
    n = mask.sum() + 1e-8
    return (loss * mask).sum() / n


def spread_loss(pred: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
    """峰谷价差保真损失：保证预测的日内价格振幅与实际一致。"""
    pred_spread = pred.max(dim=-1).values - pred.min(dim=-1).values   # (B,)
    actual_spread = actual.max(dim=-1).values - actual.min(dim=-1).values
    return F.l1_loss(pred_spread, actual_spread)


def weighted_l1_loss(pred: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
    """加权 L1：日内高价/低价时段给予更高权重。

    每个样本内，按实际价格排名分配权重：top/bottom 各 4 小时权重 ×3，其余 ×1。
    """
    B, T = actual.shape
    ranks = actual.argsort(dim=-1).argsort(dim=-1).float()  # (B, 24)
    weights = torch.ones_like(ranks)
    k = 4
    weights[ranks < k] = 3.0       # 最低 4 小时
    weights[ranks >= T - k] = 3.0   # 最高 4 小时
    weights = weights / weights.mean(dim=-1, keepdim=True)  # 归一化使总权重 = T
    return (weights * (pred - actual).abs()).mean()


# ── Evaluation ────────────────────────────────────────────────────

def _eval_v10(model, loader: DataLoader, y_mean: float, y_std: float):
    """评估 V10 模型：返回 MAE, RMSE, 方向准确率, 排序相关。"""
    model.eval()
    all_pred, all_actual = [], []
    dir_correct, dir_total = 0, 0

    with torch.no_grad():
        for grids, tgt_norm, dir_labels, tgt_raw in loader:
            grids = grids.to(DEVICE)
            prices, dir_logits = model(grids)

            pred_raw = prices.cpu().numpy() * y_std + y_mean
            all_pred.append(pred_raw.reshape(-1))
            all_actual.append(tgt_raw.numpy().reshape(-1))

            pred_dir = dir_logits.argmax(dim=-1)
            dir_correct += (pred_dir.cpu() == dir_labels).sum().item()
            dir_total += dir_labels.numel()

    pred = np.concatenate(all_pred)
    actual = np.concatenate(all_actual)
    mask = ~(np.isnan(pred) | np.isnan(actual))
    pred, actual = pred[mask], actual[mask]

    mae = float(np.mean(np.abs(pred - actual)))
    rmse = float(np.sqrt(np.mean((pred - actual) ** 2)))
    dir_acc = dir_correct / max(dir_total, 1)

    rank_corr = float(np.nan)
    if len(pred) >= 24:
        from scipy.stats import spearmanr
        daily_corrs = []
        n_days = len(pred) // 24
        for i in range(n_days):
            p24 = pred[i*24:(i+1)*24]
            a24 = actual[i*24:(i+1)*24]
            if np.std(p24) > 1e-6 and np.std(a24) > 1e-6:
                daily_corrs.append(spearmanr(p24, a24).correlation)
        if daily_corrs:
            rank_corr = float(np.mean(daily_corrs))

    return {"mae": mae, "rmse": rmse, "dir_acc": dir_acc, "rank_corr": rank_corr}


# ── Training ──────────────────────────────────────────────────────

def train_v10(
    max_epochs: int = 200,
    lr: float = 5e-4,
    lambda_dir: float = 0.3,
    lambda_rank: float = 0.1,
    lambda_spread: float = 0.0,
    use_weighted_l1: bool = False,
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
    out_dir = OUTPUT_DIR / "experiments" / (f"v10.0-joint{out_tag}" if out_tag else "v10.0-joint")
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("V10.0 — 全日联合预测 (Pure Transformer)")
    logger.info("  每小时特征展平后直接投影入 Transformer，无 Conv2D")
    logger.info("  输出目录: %s", out_dir)
    logger.info("=" * 60)

    _seed(42)

    # ── 加载数据 ──
    dws = _load_dws()
    valid_dates, day_lag0, day_lag1, day_lag2, day_targets = _build_daily_arrays(dws)

    test_dt = pd.Timestamp(test_start).date()
    test_end_dt = pd.Timestamp(test_end).date()
    train_days = [d for d in valid_dates if d < test_dt]
    test_days = [d for d in valid_dates if test_dt <= d <= test_end_dt]
    logger.info("  训练: %d 天, 测试: %d 天 (%s ~ %s)",
                len(train_days), len(test_days), test_start, test_end)

    # ── 归一化 ──
    norm_mean, norm_std = _compute_norm(day_lag0, day_lag1, day_lag2, train_days)
    tgt_stack = np.stack([day_targets[d] for d in train_days if d in day_targets])
    y_mean = float(tgt_stack.mean())
    y_std = float(tgt_stack.std()) + 1e-8

    # ── 构建数据集 ──
    ds_kwargs = dict(
        day_lag0=day_lag0, day_lag1=day_lag1, day_lag2=day_lag2,
        day_targets=day_targets,
        norm_mean=norm_mean, norm_std=norm_std,
        y_mean=y_mean, y_std=y_std,
    )
    train_ds = DailyJointDataset(sample_dates=train_days, **ds_kwargs)
    test_ds = DailyJointDataset(sample_dates=test_days, **ds_kwargs)
    logger.info("  train_ds=%d 天, test_ds=%d 天", len(train_ds), len(test_ds))
    logger.info("  每小时输入维度: %d × %d × %d = %d (展平)",
                C_TOTAL, H_SLOTS_V10, LOOKBACK_DAYS,
                C_TOTAL * H_SLOTS_V10 * LOOKBACK_DAYS)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=min(batch_size, max(len(test_ds), 1)),
                             shuffle=False)

    # ── 创建 V10 模型 ──
    model = V10JointNet(
        c_in=C_TOTAL, h_slots=H_SLOTS_V10,
        d_model=d_model, n_heads=n_heads, n_layers=n_layers,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info("  V10 参数总量: %d", total_params)

    eval_log = []

    # ── 单阶段训练 ──
    logger.info("=" * 60)
    logger.info("训练: %d epochs, lr=%e, λ_dir=%.2f, λ_rank=%.2f, λ_spread=%.2f, weighted_l1=%s",
                max_epochs, lr, lambda_dir, lambda_rank, lambda_spread, use_weighted_l1)
    logger.info("=" * 60)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    warmup = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=0.1, total_iters=warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max_epochs - warmup_epochs, eta_min=1e-6)
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt, schedulers=[warmup, cosine], milestones=[warmup_epochs])

    for epoch in range(max_epochs):
        model.train()
        ep_l1, ep_ce, ep_rank, ep_sprd, nb = 0.0, 0.0, 0.0, 0.0, 0

        for grids, tgt_norm, dir_labels, tgt_raw in train_loader:
            grids = grids.to(DEVICE)
            tgt_norm = tgt_norm.to(DEVICE)
            dir_labels = dir_labels.to(DEVICE)

            opt.zero_grad()
            prices, dir_logits = model(grids)

            if use_weighted_l1:
                l1 = weighted_l1_loss(prices, tgt_norm)
            else:
                l1 = F.l1_loss(prices, tgt_norm)
            ce = F.cross_entropy(
                dir_logits.reshape(-1, DIR_CLASSES), dir_labels.reshape(-1)
            )
            rank = pairwise_rank_loss(prices, tgt_norm)

            loss = l1 + lambda_dir * ce + lambda_rank * rank

            if lambda_spread > 0:
                sprd = spread_loss(prices, tgt_norm)
                loss = loss + lambda_spread * sprd
                ep_sprd += sprd.item()

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            ep_l1 += l1.item()
            ep_ce += ce.item()
            ep_rank += rank.item()
            nb += 1

        sched.step()

        do_eval = (epoch + 1) % eval_every == 0 or epoch == max_epochs - 1
        if do_eval:
            metrics = _eval_v10(model, test_loader, y_mean, y_std)
            cur_lr = opt.param_groups[0]["lr"]

            extra = ""
            if lambda_spread > 0:
                extra = f" Spread=%.4f" % (ep_sprd / max(nb, 1))

            logger.info(
                "  ep%3d  L1=%.4f CE=%.3f Rank=%.4f%s"
                " | test MAE=%.1f RMSE=%.1f dir=%.3f rank_r=%.3f  lr=%.1e",
                epoch + 1, ep_l1/max(nb,1), ep_ce/max(nb,1), ep_rank/max(nb,1),
                extra,
                metrics["mae"], metrics["rmse"], metrics["dir_acc"],
                metrics["rank_corr"], cur_lr,
            )
            eval_log.append({"epoch": epoch+1, **metrics})

    # ── 直接使用最终 epoch 的模型（不做测试集选优）──
    logger.info("使用最终 epoch %d 的模型权重", max_epochs)

    torch.save(model.state_dict(), out_dir / "model_weights.pt")
    np.save(out_dir / "norm_mean.npy", norm_mean)
    np.save(out_dir / "norm_std.npy", norm_std)
    np.savez(out_dir / "target_stats.npz", y_mean=y_mean, y_std=y_std)

    # ── 生成预测 CSV ──
    model.eval()
    all_rows = []
    day_preds_v10 = {}
    with torch.no_grad():
        idx = 0
        for grids, tgt_norm, dir_labels, tgt_raw in DataLoader(
            test_ds, batch_size=1, shuffle=False
        ):
            grids = grids.to(DEVICE)
            prices, _ = model(grids)
            pred_24 = prices.cpu().numpy()[0] * y_std + y_mean
            actual_24 = tgt_raw.numpy()[0]
            d = test_ds.dates[idx]
            day_preds_v10[d] = pred_24

            for h in range(24):
                all_rows.append({
                    "ts": pd.Timestamp(d) + pd.Timedelta(hours=h),
                    "actual": float(actual_24[h]),
                    "pred": float(pred_24[h]),
                })
            idx += 1

    result_df = pd.DataFrame(all_rows)
    if "ts" in result_df.columns:
        result_df = result_df.set_index("ts").sort_index()
    result_df.to_csv(out_dir / "test_predictions_hourly.csv")
    logger.info("预测结果已保存: %s", out_dir / "test_predictions_hourly.csv")

    # ── 绘图 ──
    _plot_v10(result_df, out_dir)

    # ── 最终报告 ──
    final_metrics = _eval_v10(model, test_loader, y_mean, y_std)
    logger.info("=" * 60)
    logger.info("V10.0 RESULTS")
    logger.info("  MAE:       %.2f", final_metrics["mae"])
    logger.info("  RMSE:      %.2f", final_metrics["rmse"])
    logger.info("  Dir Acc:   %.3f", final_metrics["dir_acc"])
    logger.info("  Rank Corr: %.3f", final_metrics["rank_corr"])
    logger.info("  Output:    %s", out_dir)
    logger.info("=" * 60)

    eval_df = pd.DataFrame(eval_log)
    eval_df.to_csv(out_dir / "eval_log.csv", index=False)

    return {
        "predictions": result_df,
        "day_preds": day_preds_v10,
        "metrics": final_metrics,
        "model": model,
        "y_mean": y_mean,
        "y_std": y_std,
        "norm_mean": norm_mean,
        "norm_std": norm_std,
    }


def _plot_v10(result_df: pd.DataFrame, out_dir: Path):
    """按周绘图。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm

    _cn_candidates = (
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/wqy-microhei/wqy-microhei.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/PingFang.ttc",
    )
    for p in _cn_candidates:
        if os.path.exists(p):
            fm.fontManager.addfont(p)
            name = fm.FontProperties(fname=p).get_name()
            matplotlib.rcParams["font.family"] = "sans-serif"
            matplotlib.rcParams["font.sans-serif"] = [name]
            break
    matplotlib.rcParams["axes.unicode_minus"] = False

    if len(result_df) == 0:
        return

    dates = sorted(set(result_df.index.date))
    weeks, week = [], [0]
    for i in range(1, len(dates)):
        if (dates[i] - dates[week[0]]).days >= 7:
            weeks.append(week)
            week = [i]
        else:
            week.append(i)
    if week:
        weeks.append(week)

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    for wi, w_idxs in enumerate(weeks):
        fig, ax = plt.subplots(figsize=(18, 5))
        a_all, p_all, ticks, labels, pos = [], [], [], [], 0
        for di in w_idxs:
            d = dates[di]
            chunk = result_df.loc[result_df.index.date == d].sort_index()
            if len(chunk) == 0:
                continue
            a_all.extend(chunk["actual"].values)
            p_all.extend(chunk["pred"].values)
            if pos > 0:
                ax.axvline(pos, color="gray", ls="--", alpha=0.3, lw=0.8)
            ticks.append(pos + 12)
            a24 = chunk["actual"].values
            p24 = chunk["pred"].values
            if np.std(a24) > 1e-6 and np.std(p24) > 1e-6:
                labels.append(f"{d}\nr={np.corrcoef(a24, p24)[0,1]:.2f}")
            else:
                labels.append(str(d))
            pos += len(chunk)

        ax.plot(np.arange(len(a_all)), a_all, "k-", lw=1.8, label="实际", zorder=3)
        ax.plot(np.arange(len(p_all)), p_all, "#1976D2", lw=1.3, alpha=0.85,
                label="V10.0-Joint")
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("元/MWh")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.25)
        ax.set_title(f"V10.0 Joint — 第{wi+1}周", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(plots_dir / f"da_week{wi+1}.png", dpi=120, bbox_inches="tight")
        plt.close()
        logger.info("Saved: da_week%d.png", wi + 1)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    train_v10(
        lambda_rank=0.4,
        lambda_spread=0.2,
        use_weighted_l1=True,
        out_tag="-rank",
    )
