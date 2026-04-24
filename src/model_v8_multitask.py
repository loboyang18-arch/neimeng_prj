"""
V8.0 — Conv2D Multi-Task：小时级回归 + 方向分类辅助头

基于 V6.0 Conv2D 骨干，增加方向分类辅助头：
  - 回归头：预测价格（同 V6）
  - 方向头：预测相比前一小时的涨/跌/平（3 分类）
  - 联合损失 = L1 + λ·CrossEntropy(direction)
  - 直接对齐 direction_acc 指标

优势：保持小时级样本，标准 shuffle 批次，无需日级聚合。

可选训练增强（对齐重庆 V18）：
  NM_TRAIN_OVERSAMPLE：逻辑样本倍数 K（默认 1）
  NM_OVERSAMPLE_RESID_SCALE：rep>0 时在归一化目标上加残差池抽样
  NM_RESIDUAL_MC：1 开启 rep=0 的残差 MC 目标抖动
  NM_RESIDUAL_MC_P / NM_RESIDUAL_MC_SCALE / NM_RESIDUAL_MC_NPASS

可选 15min 粒度预测（NM_V8_15MIN=1）：
  每个样本预测 1 个 15min 时段；局部窗口为 SLOT_WINDOW 个 15min × 7 天
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .config import OUTPUT_DIR
from .experiment.splits import MIN_FEATURE_CALENDAR_DATE
from .fill_sudun_dws_gaps import fill_sudun_price_columns
from .model_lgbm import build_rolling_test_windows

logger = logging.getLogger(__name__)

# ── Channel definitions (同 V6, 支持 env 配置目标列) ───────────────
LAG0_COLS = (
    "load_forecast", "renewable_forecast", "wind_forecast",
    "solar_forecast", "east_send_forecast",
    "reserve_pos_capacity", "reserve_neg_capacity",
    "price_dayahead_preclear_energy",
)
LAG2_COLS = ("load_actual", "renewable_actual", "wind_actual", "solar_actual")

TARGET_COL = os.environ.get("NM_V8_TARGET", "price_unified")

# hourly_agg: 'slot0' = 首槽 15min 值; 'mean4' = 每小时 4 个 15min 均值
HOURLY_AGG = os.environ.get("NM_V8_HOURLY_AGG", "slot0").lower()
assert HOURLY_AGG in ("slot0", "mean4")

# 15min 粒度预测模式
PREDICT_15MIN = int(os.environ.get("NM_V8_15MIN", "0"))
SLOTS_BEFORE = int(os.environ.get("NM_V8_SLOTS_BEFORE", "7"))
SLOTS_AFTER  = int(os.environ.get("NM_V8_SLOTS_AFTER",  "4"))
SLOT_WINDOW  = SLOTS_BEFORE + 1 + SLOTS_AFTER  # default 7+1+4 = 12

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

C_LAG0 = len(LAG0_COLS) + 4
C_LAG1 = len(LAG1_COLS)
C_LAG2 = len(LAG2_COLS)
C_TOTAL = C_LAG0 + C_LAG1 + C_LAG2

LOOKBACK_DAYS = 7
SLOTS_PER_HOUR = 4
SLOTS_PER_DAY = 96
CONTEXT_BEFORE = int(os.environ.get("NM_CTX_BEFORE", "1"))
CONTEXT_AFTER  = int(os.environ.get("NM_CTX_AFTER",  "1"))
CONTEXT_HOURS  = CONTEXT_BEFORE + 1 + CONTEXT_AFTER
H_SLOTS = CONTEXT_HOURS * SLOTS_PER_HOUR

# ── Device ────────────────────────────────────────────────────────
DEVICE = torch.device(
    "mps" if getattr(torch.backends, "mps", None)
             and torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

# ── Hyper-parameters ──────────────────────────────────────────────
MAX_EPOCHS     = int(os.environ.get("NM_TF_EPOCHS", "200"))
BATCH_SIZE     = 64
LR             = 1e-3
WARMUP_EPOCHS  = 10
TRAIN_MONTHS   = int(os.environ.get("NM_TRAIN_MONTHS", "6"))

LAMBDA_DIR = 0.3
DIR_CLASSES = 3

_v8_tag = os.environ.get("NM_V8_TAG", "v8.0")
V8_DIR = OUTPUT_DIR / "experiments" / _v8_tag

NM_V8_SINGLE_SPLIT = os.environ.get("NM_V8_SINGLE_SPLIT", "")
NM_V8_TEST_WEEKS = int(os.environ.get("NM_V8_TEST_WEEKS", "0"))
NM_V8_TEST_START = os.environ.get("NM_V8_TEST_START", "")

NM_TRAIN_OVERSAMPLE = max(1, int(os.environ.get("NM_TRAIN_OVERSAMPLE", "1")))
NM_OVERSAMPLE_RESID_SCALE = float(os.environ.get("NM_OVERSAMPLE_RESID_SCALE", "0.28"))
NM_RESIDUAL_MC = int(os.environ.get("NM_RESIDUAL_MC", "0"))
NM_RESIDUAL_MC_P = float(os.environ.get("NM_RESIDUAL_MC_P", "0.28"))
NM_RESIDUAL_MC_SCALE = float(os.environ.get("NM_RESIDUAL_MC_SCALE", "0.35"))
NM_RESIDUAL_MC_NPASS = int(os.environ.get("NM_RESIDUAL_MC_NPASS", "1"))


# ── Helpers ───────────────────────────────────────────────────────
def _log_device():
    logger.info("PyTorch %s | device=%s", torch.__version__, DEVICE)


def _seed(s=42):
    np.random.seed(s)
    torch.manual_seed(s)
    if DEVICE.type == "mps":
        torch.mps.manual_seed(s)
    elif DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(s)


def _load_dws() -> pd.DataFrame:
    path = OUTPUT_DIR / "dws_15min_features.csv"
    df = pd.read_csv(path, parse_dates=["ts"], index_col="ts")
    df = df.sort_index()
    df = fill_sudun_price_columns(df)
    logger.info("DWS: %d rows x %d cols (%s ~ %s)",
                len(df), len(df.columns), df.index.min(), df.index.max())
    return df


def _build_daily_arrays(df: pd.DataFrame):
    """将入库宽表（DWS）按日切成 96×15min：当日 00:00 … 23:45，并组 lag0/lag1/lag2 特征张量。

    回归标签 ``day_targets`` 只来自 DWS 中的 ``TARGET_COL`` 列（默认 ``price_unified``；
    苏敦节点见模块首段示例），即「从入库事实表读目标列」；**不是**从 lag0/lag1/lag2 任何通道。
    """
    start_date = df.index.min().normalize().date()
    end_date = df.index.max().date()
    date_range = pd.date_range(start_date, end_date, freq="D")
    day_lag0: Dict = {}
    day_lag1: Dict = {}
    day_lag2: Dict = {}
    day_targets: Dict = {}
    valid: List = []

    _hour_idx = np.arange(0, SLOTS_PER_DAY, 4, dtype=np.int64)

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
        day_lag0[d] = np.concatenate([l0, te], axis=1).astype(np.float32)
        day_lag1[d] = raw[list(LAG1_COLS)].values.astype(np.float32)
        day_lag2[d] = raw[list(LAG2_COLS)].values.astype(np.float32)

        tgt_96 = raw[TARGET_COL].values.astype(np.float32)

        # 15min 模式：整日 96 点全有限值即入选
        if PREDICT_15MIN:
            if len(tgt_96) == SLOTS_PER_DAY and np.isfinite(tgt_96).all():
                day_targets[d] = tgt_96
                if d >= MIN_FEATURE_CALENDAR_DATE:
                    valid.append(d)
            continue

        # 小时级模式
        hourly_y = None
        if HOURLY_AGG == "mean4":
            hourly_y = tgt_96.reshape(24, 4).mean(axis=1).astype(np.float32)
        else:
            hourly_y = tgt_96[_hour_idx]

        if hourly_y is not None and len(hourly_y) == 24 and np.isfinite(hourly_y).all():
            day_targets[d] = hourly_y
            if d >= MIN_FEATURE_CALENDAR_DATE:
                valid.append(d)

    valid = sorted(valid)
    logger.info("Daily arrays (15min): %d days, %d with target",
                len(day_lag0), len(valid))
    return valid, day_lag0, day_lag1, day_lag2, day_targets


def _hour_four_slots(arr, hh):
    """单日内日历小时 hh 的 4 个 15min 点 (4, C)；DWS 为 00:00 起的完整四档。"""
    s = 4 * hh
    return arr[s:s + 4].copy()


def _get_hour_slots(day_arrays: Dict, d, h: int,
                    ctx_before: int = CONTEXT_BEFORE,
                    ctx_after: int = CONTEXT_AFTER):
    """获取日 d 第 h 小时 [h-ctx_before, h+ctx_after] 的 15min 切片（每日历小时 4 槽）。"""
    n_slots = (ctx_before + 1 + ctx_after) * 4
    arr = day_arrays[d]
    C = arr.shape[1]

    start_slot = (h - ctx_before) * 4
    end_slot = (h + ctx_after + 1) * 4

    if 0 <= start_slot and end_slot <= 96:
        return arr[start_slot:end_slot]

    result = np.zeros((n_slots, C), dtype=np.float32)
    out_idx = 0
    for hh in range(h - ctx_before, h + ctx_after + 1):
        cur_d = d
        cur_h = hh
        if cur_h < 0:
            cur_d = (pd.Timestamp(d) - pd.Timedelta(days=1)).date()
            cur_h += 24
        elif cur_h >= 24:
            cur_d = (pd.Timestamp(d) + pd.Timedelta(days=1)).date()
            cur_h -= 24
        if cur_d in day_arrays:
            result[out_idx:out_idx + 4] = _hour_four_slots(day_arrays[cur_d], cur_h)
        else:
            result[out_idx:out_idx + 4] = arr[0:4] if hh < 0 else arr[92:96]
        out_idx += 4
    return result


def _get_context_slots(day_arrays: Dict, center_d, slot_idx: int):
    """预测日 center_d、当日 slot_idx∈[0,95] 时，取时间轴上连续 SLOT_WINDOW 个 15min。
    默认 SLOTS_BEFORE=7, SLOTS_AFTER=4 → [slot_idx-7, …, slot_idx, …, slot_idx+4]。
    LAG0 为日前预测，全天可用；跨自然日时向前/后借邻日数据。
    """
    arr0 = day_arrays[center_d]
    C = arr0.shape[1]
    out = np.zeros((SLOT_WINDOW, C), dtype=np.float32)

    for j, rel in enumerate(range(-SLOTS_BEFORE, SLOTS_AFTER + 1)):
        idx = slot_idx + rel
        cur_d = center_d
        while idx < 0:
            cur_d = (pd.Timestamp(cur_d) - pd.Timedelta(days=1)).date()
            idx += SLOTS_PER_DAY
        while idx >= SLOTS_PER_DAY:
            cur_d = (pd.Timestamp(cur_d) + pd.Timedelta(days=1)).date()
            idx -= SLOTS_PER_DAY
        if cur_d in day_arrays:
            out[j] = day_arrays[cur_d][idx]
        else:
            out[j] = np.nan
    return out


def _build_residual_mc_pool(
    train_days: List,
    day_targets: Dict,
    y_mean: float,
    y_std: float,
) -> np.ndarray:
    """训练集 (日×24) 按小时去均值后的归一化目标残差，展平供过采样 / MC 抽样。"""
    rows = []
    for d in train_days:
        if d not in day_targets:
            continue
        y = day_targets[d].astype(np.float64)
        rows.append((y - y_mean) / y_std)
    if not rows:
        return np.array([], dtype=np.float32)
    mat = np.stack(rows, axis=0)
    hour_mean = mat.mean(axis=0, keepdims=True)
    resid = (mat - hour_mean).reshape(-1).astype(np.float32)
    return resid


# ── Dataset (增加方向标签 + 15min 模式) ──────────────────────────
class HourlyMultiTaskDataset(Dataset):
    """每小时或每 15min 一个样本 → (C, H_SLOTS, 7) + 归一化价格 + 方向标签；可选 V18 式过采样。"""

    def __init__(
        self,
        sample_dates: List,
        day_lag0: Dict, day_lag1: Dict, day_lag2: Dict,
        day_targets: Dict,
        norm_mean: np.ndarray, norm_std: np.ndarray,
        y_mean: float, y_std: float,
        residual_mc_pool: Optional[np.ndarray] = None,
        residual_mc_prob: float = 0.0,
        residual_mc_scale: float = 0.35,
        residual_mc_npass: int = 1,
        train_oversample: int = 1,
        oversample_resid_scale: float = 0.28,
    ):
        a0 = set(day_lag0.keys())
        a1 = set(day_lag1.keys())
        a2 = set(day_lag2.keys())

        self.items = []
        self.meta = []
        self._predict_15min = bool(PREDICT_15MIN)

        for d in sample_dates:
            if d not in day_targets:
                continue

            dates0 = [(pd.Timestamp(d) - pd.Timedelta(days=off)).date()
                      for off in range(LOOKBACK_DAYS - 1, -1, -1)]
            dates1 = [(pd.Timestamp(d) - pd.Timedelta(days=off)).date()
                      for off in range(LOOKBACK_DAYS, 0, -1)]
            dates2 = [(pd.Timestamp(d) - pd.Timedelta(days=off)).date()
                      for off in range(LOOKBACK_DAYS + 1, 1, -1)]

            ok = (all(dd in a0 for dd in dates0)
                  and all(dd in a1 for dd in dates1)
                  and all(dd in a2 for dd in dates2))
            if not ok:
                continue

            d_prev = (pd.Timestamp(d) - pd.Timedelta(days=1)).date()

            if self._predict_15min:
                # 15min 粒度：每天 96 个样本
                n_slot = day_lag0[d].shape[0]
                for s in range(n_slot):
                    layers = []
                    for k in range(LOOKBACK_DAYS):
                        s0 = _get_context_slots(day_lag0, dates0[k], s)
                        s1 = _get_context_slots(day_lag1, dates1[k], s)
                        s2 = _get_context_slots(day_lag2, dates2[k], s)
                        layers.append(np.concatenate([s0, s1, s2], axis=1))
                    grid = np.stack(layers, axis=-1).transpose(1, 0, 2)  # (C, SLOT_WINDOW, 7)
                    grid = np.nan_to_num(grid, nan=0.0)
                    grid = ((grid - norm_mean.reshape(C_TOTAL, 1, 1))
                            / norm_std.reshape(C_TOTAL, 1, 1)).astype(np.float32)

                    tgt = np.float32((day_targets[d][s] - y_mean) / y_std)

                    if s > 0:
                        diff = day_targets[d][s] - day_targets[d][s - 1]
                    elif d_prev in day_targets:
                        diff = day_targets[d][0] - day_targets[d_prev][-1]
                    else:
                        diff = 0.0

                    dir_label = 2 if diff > 0 else (0 if diff < 0 else 1)
                    self.items.append((grid, tgt, dir_label))
                    self.meta.append((d, s))
            else:
                # 小时级
                for h in range(24):
                    layers = []
                    for k in range(LOOKBACK_DAYS):
                        s0 = _get_hour_slots(day_lag0, dates0[k], h)
                        s1 = _get_hour_slots(day_lag1, dates1[k], h)
                        s2 = _get_hour_slots(day_lag2, dates2[k], h)
                        layers.append(np.concatenate([s0, s1, s2], axis=1))

                    grid = np.stack(layers, axis=-1).transpose(1, 0, 2)
                    grid = np.nan_to_num(grid, nan=0.0)
                    grid = ((grid - norm_mean.reshape(C_TOTAL, 1, 1))
                            / norm_std.reshape(C_TOTAL, 1, 1)).astype(np.float32)

                    tgt = np.float32((day_targets[d][h] - y_mean) / y_std)

                    if h > 0:
                        diff = day_targets[d][h] - day_targets[d][h - 1]
                    elif d_prev in day_targets:
                        diff = day_targets[d][0] - day_targets[d_prev][23]
                    else:
                        diff = 0.0

                    dir_label = 2 if diff > 0 else (0 if diff < 0 else 1)
                    self.items.append((grid, tgt, dir_label))
                    self.meta.append((d, h))

        self._n_orig = len(self.items)
        self._train_oversample = max(1, int(train_oversample))
        self._oversample_resid_scale = float(oversample_resid_scale)
        self._residual_pool = residual_mc_pool
        self._residual_mc_prob = float(residual_mc_prob)
        self._residual_mc_scale = float(residual_mc_scale)
        self._residual_mc_npass = max(1, int(residual_mc_npass))

    def __len__(self):
        return self._n_orig * self._train_oversample

    def __getitem__(self, idx):
        base = idx % self._n_orig
        rep = idx // self._n_orig
        grid, tgt, dl = self.items[base]
        grid = np.copy(grid)
        tgt = np.float32(tgt)
        pool = self._residual_pool

        if rep > 0 and pool is not None and len(pool) > 0:
            tgt = np.float32(
                tgt + self._oversample_resid_scale * float(np.random.choice(pool))
            )
        elif rep == 0 and pool is not None and len(pool) > 0 and self._residual_mc_prob > 0.0:
            for _ in range(self._residual_mc_npass):
                if np.random.random() < self._residual_mc_prob:
                    r = float(np.random.choice(pool))
                    tgt = np.float32(tgt + self._residual_mc_scale * r)
        return (
            torch.from_numpy(grid),
            torch.tensor(tgt),
            torch.tensor(dl, dtype=torch.long),
        )


# ── Model (Multi-Task) ───────────────────────────────────────────
class Conv2dMultiTaskNet(nn.Module):
    """
    (B, C, H_SLOTS, 7) → Conv2d×3 → 共享特征
      ├→ 回归头 → (B,)  价格预测
      └→ 方向头 → (B,3) 涨/平/跌分类
    自动适配不同 H_SLOTS。
    """

    def __init__(self, c_in=C_TOTAL, h_slots=H_SLOTS):
        super().__init__()
        k_h = min(3, h_slots)
        self.block1 = nn.Sequential(
            nn.Conv2d(c_in, 64, kernel_size=(k_h, 3), padding=(k_h // 2, 1)),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        h_out = h_slots // 2 // 2 - 2
        w_out = LOOKBACK_DAYS - 2
        fc_in = 64 * h_out * w_out
        self.flatten = nn.Flatten()

        self.reg_head = nn.Sequential(
            nn.Linear(fc_in, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )
        self.dir_head = nn.Sequential(
            nn.Linear(fc_in, 32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, DIR_CLASSES),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        feat = self.flatten(x)
        price = self.reg_head(feat).squeeze(-1)
        direction = self.dir_head(feat)
        return price, direction


# ── Training helpers ──────────────────────────────────────────────
def _eval_mae_hourly(model, loader, y_mean, y_std):
    model.eval()
    ps, ts = [], []
    with torch.no_grad():
        for batch in loader:
            grid, tgt = batch[0], batch[1]
            price, _ = model(grid.to(DEVICE))
            ps.append(price.cpu().numpy())
            ts.append(tgt.numpy())
    p = np.concatenate(ps) * y_std + y_mean
    t = np.concatenate(ts) * y_std + y_mean
    return float(np.mean(np.abs(p - t)))


def _eval_dir_acc(model, loader):
    """方向分类准确率。"""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in loader:
            grid, _, dir_label = batch[0], batch[1], batch[2]
            _, dir_logits = model(grid.to(DEVICE))
            pred_cls = dir_logits.argmax(dim=1)
            correct += (pred_cls.cpu() == dir_label).sum().item()
            total += dir_label.numel()
    return correct / max(total, 1)


def _eval_loss(model, loader):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            grid, tgt = batch[0], batch[1]
            price, _ = model(grid.to(DEVICE))
            total += F.l1_loss(price, tgt.to(DEVICE), reduction="sum").item()
            n += tgt.numel()
    return total / max(n, 1)


def _train_fold(model, train_ds, val_ds, y_mean, y_std, fold, test_ds=None):
    tl = DataLoader(train_ds, BATCH_SIZE, shuffle=True, drop_last=True)
    train_eval_l = DataLoader(train_ds, min(512, len(train_ds)), shuffle=False)
    val_l = DataLoader(val_ds, min(512, max(len(val_ds), 1)), shuffle=False)
    test_l = (DataLoader(test_ds, min(512, max(len(test_ds), 1)), shuffle=False)
              if test_ds and len(test_ds) > 0 else None)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPOCHS)
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=MAX_EPOCHS - WARMUP_EPOCHS, eta_min=1e-6)
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt, schedulers=[warmup_sched, cosine_sched],
        milestones=[WARMUP_EPOCHS])

    best_mae = float("inf")

    for ep in range(MAX_EPOCHS):
        model.train()
        ep_l1, ep_ce, ep_dir_ok, ep_dir_n, nb = 0., 0., 0, 0, 0

        for grid, tgt, dir_label in tl:
            grid = grid.to(DEVICE)
            tgt = tgt.to(DEVICE)
            dir_label = dir_label.to(DEVICE)

            opt.zero_grad()
            price, dir_logits = model(grid)

            l1 = F.l1_loss(price, tgt)
            ce = F.cross_entropy(dir_logits, dir_label)
            loss = l1 + LAMBDA_DIR * ce

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            ep_l1 += l1.item()
            ep_ce += ce.item()
            ep_dir_ok += (dir_logits.argmax(1) == dir_label).sum().item()
            ep_dir_n += dir_label.numel()
            nb += 1

        sched.step()
        cur_lr = opt.param_groups[0]["lr"]
        batch_dir_acc = ep_dir_ok / max(ep_dir_n, 1)

        log_interval = 5 if MAX_EPOCHS > 50 else 10
        do_log = ep % log_interval == 0 or ep == MAX_EPOCHS - 1

        if do_log:
            train_mae = _eval_mae_hourly(model, train_eval_l, y_mean, y_std)
            val_mae = _eval_mae_hourly(model, val_l, y_mean, y_std)
            test_mae = (_eval_mae_hourly(model, test_l, y_mean, y_std)
                        if test_l else float("nan"))
            val_dir = _eval_dir_acc(model, val_l)

            if val_mae < best_mae:
                best_mae = val_mae

            logger.info(
                "  fold%02d ep%3d  L1=%.4f CE=%.3f dir_acc=%.3f"
                " | train=%.1f val=%.1f test=%.1f best=%.1f"
                " | v_dir=%.3f  lr=%.1e",
                fold, ep, ep_l1 / max(nb, 1), ep_ce / max(nb, 1),
                batch_dir_acc,
                train_mae, val_mae, test_mae, best_mae,
                val_dir, cur_lr,
            )
        else:
            val_mae = _eval_mae_hourly(model, val_l, y_mean, y_std)
            if val_mae < best_mae:
                best_mae = val_mae

    return best_mae


def _train_no_val(model, train_ds, y_mean, y_std):
    """无验证集训练，跑满 MAX_EPOCHS，取最终模型。"""
    tl = DataLoader(train_ds, BATCH_SIZE, shuffle=True, drop_last=True)
    train_eval_l = DataLoader(train_ds, min(512, len(train_ds)), shuffle=False)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPOCHS)
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=MAX_EPOCHS - WARMUP_EPOCHS, eta_min=1e-6)
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt, schedulers=[warmup_sched, cosine_sched],
        milestones=[WARMUP_EPOCHS])

    for ep in range(MAX_EPOCHS):
        model.train()
        ep_l1, ep_ce, ep_dir_ok, ep_dir_n, nb = 0., 0., 0, 0, 0

        for grid, tgt, dir_label in tl:
            grid = grid.to(DEVICE)
            tgt = tgt.to(DEVICE)
            dir_label = dir_label.to(DEVICE)

            opt.zero_grad()
            price, dir_logits = model(grid)
            l1 = F.l1_loss(price, tgt)
            ce = F.cross_entropy(dir_logits, dir_label)
            loss = l1 + LAMBDA_DIR * ce
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            ep_l1 += l1.item()
            ep_ce += ce.item()
            ep_dir_ok += (dir_logits.argmax(1) == dir_label).sum().item()
            ep_dir_n += dir_label.numel()
            nb += 1

        sched.step()
        cur_lr = opt.param_groups[0]["lr"]
        batch_dir_acc = ep_dir_ok / max(ep_dir_n, 1)

        log_interval = 5 if MAX_EPOCHS > 50 else 10
        if ep % log_interval == 0 or ep == MAX_EPOCHS - 1:
            train_mae = _eval_mae_hourly(model, train_eval_l, y_mean, y_std)
            logger.info(
                "  single ep%3d  L1=%.4f CE=%.3f dir_acc=%.3f"
                " | train_mae=%.1f  lr=%.1e",
                ep, ep_l1 / max(nb, 1), ep_ce / max(nb, 1),
                batch_dir_acc, train_mae, cur_lr,
            )


def _predict_day(model, ds, y_mean, y_std):
    if len(ds) == 0:
        return {}, []
    loader = DataLoader(ds, min(512, len(ds)), shuffle=False)
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in loader:
            grid = batch[0]
            price, _ = model(grid.to(DEVICE))
            all_preds.append(price.cpu().numpy())
    preds_flat = np.concatenate(all_preds) * y_std + y_mean

    day_preds: Dict = {}
    n_out = 96 if getattr(ds, "_predict_15min", False) else 24
    for i, (d, idx) in enumerate(ds.meta):
        if d not in day_preds:
            day_preds[d] = np.full(n_out, np.nan)
        day_preds[d][idx] = preds_flat[i]
    return day_preds, sorted(day_preds.keys())


def _compute_norm(day_lag0, day_lag1, day_lag2, train_days):
    rows = []
    for d in train_days:
        if d in day_lag0 and d in day_lag1 and d in day_lag2:
            row = np.concatenate([day_lag0[d], day_lag1[d], day_lag2[d]], axis=1)
            rows.append(row)
    stack = np.concatenate(rows, axis=0)
    mean = np.nanmean(stack, axis=0).astype(np.float32)
    std  = np.nanstd(stack, axis=0).astype(np.float32) + 1e-8
    return mean, std


# ── Plotting ──────────────────────────────────────────────────────
def _plot_weekly(result_df, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    _cn_font_candidates = (
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/PingFang.ttc",
    )
    for p in _cn_font_candidates:
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
            weeks.append(week); week = [i]
        else:
            week.append(i)
    if week:
        weeks.append(week)
    plots_dir = out_dir / "plots"; plots_dir.mkdir(exist_ok=True)
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
            a24, p24 = chunk["actual"].values, chunk["pred"].values
            if np.std(a24) > 1e-6 and np.std(p24) > 1e-6:
                labels.append(f"{d}\nr={np.corrcoef(a24, p24)[0,1]:.2f}")
            else:
                labels.append(str(d))
            pos += len(chunk)
        ax.plot(np.arange(len(a_all)), a_all, "k-", lw=1.8, label="实际", zorder=3)
        ax.plot(np.arange(len(p_all)), p_all, "#E53935", lw=1.3, alpha=0.85,
                label="V8.0-MultiTask")
        ax.set_xticks(ticks); ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("元/MWh"); ax.legend(fontsize=10)
        ax.grid(True, alpha=0.25)
        ax.set_title(f"V8.0 MultiTask — 第{wi+1}周", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(plots_dir / f"da_week{wi+1}.png", dpi=120, bbox_inches="tight")
        plt.close()
        logger.info("Saved: da_week%d.png", wi + 1)


# ── Main ──────────────────────────────────────────────────────────
def run_v8_multitask():
    V8_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("V8.0 Conv2D Multi-Task (regression + direction)")
    logger.info("  target=%s", TARGET_COL)
    if PREDICT_15MIN:
        logger.info("  predict_15min=1  (96 点/日；局部 7+1 个 15min × 7 日 → %d×%d，无未来槽)",
                     SLOT_WINDOW, LOOKBACK_DAYS)
    logger.info("  hourly_agg=%s (slot0=首槽 15min, mean4=每小时 4×15min 均值)", HOURLY_AGG)
    logger.info("  LAG1=%s", LAG1_COLS)
    logger.info("  input=(%d ch, %d slots, %d days)",
                C_TOTAL, SLOT_WINDOW if PREDICT_15MIN else H_SLOTS, LOOKBACK_DAYS)
    logger.info("  reg_head: →64→1,  dir_head: →32→3")
    logger.info("  λ_dir=%.2f,  epochs=%d,  bs=%d", LAMBDA_DIR, MAX_EPOCHS, BATCH_SIZE)
    logger.info("  LR=%.1e (warmup %d ep → Cosine), train_window=%dM",
                LR, WARMUP_EPOCHS, TRAIN_MONTHS)
    logger.info(
        "  augment(V18-style): oversample=%d resid_scale=%.3f | residual_mc=%d "
        "p=%.3f scale=%.3f npass=%d",
        NM_TRAIN_OVERSAMPLE, NM_OVERSAMPLE_RESID_SCALE,
        NM_RESIDUAL_MC, NM_RESIDUAL_MC_P, NM_RESIDUAL_MC_SCALE, NM_RESIDUAL_MC_NPASS,
    )

    _log_device()
    _seed(42)

    dws = _load_dws()
    valid_dates, day_lag0, day_lag1, day_lag2, day_targets = _build_daily_arrays(dws)
    logger.info("Valid days: %d", len(valid_dates))

    model_tmp = Conv2dMultiTaskNet(
        h_slots=SLOT_WINDOW if PREDICT_15MIN else H_SLOTS
    )
    n_params = sum(p.numel() for p in model_tmp.parameters() if p.requires_grad)
    logger.info("Model params: %d", n_params)

    # ── Single-split 模式 ──
    if NM_V8_SINGLE_SPLIT or NM_V8_TEST_START:
        if NM_V8_TEST_START:
            _dt = pd.Timestamp(NM_V8_TEST_START).date()
        elif NM_V8_TEST_WEEKS > 0:
            test_n_days = NM_V8_TEST_WEEKS * 7
            _dt = valid_dates[-test_n_days] if test_n_days < len(valid_dates) else valid_dates[0]
        else:
            _dt = valid_dates[-7]
        logger.info("Single-split mode: test_start=%s, no val set", _dt)

        train_days = [d for d in valid_dates if d < _dt]
        test_days  = [d for d in valid_dates if d >= _dt]

        norm_mean, norm_std = _compute_norm(day_lag0, day_lag1, day_lag2, train_days)
        tgt_stack = np.stack([day_targets[d] for d in train_days if d in day_targets])
        y_mean = float(tgt_stack.mean())
        y_std = float(tgt_stack.std()) + 1e-8

        ds_base = dict(
            day_lag0=day_lag0, day_lag1=day_lag1, day_lag2=day_lag2,
            day_targets=day_targets,
            norm_mean=norm_mean, norm_std=norm_std,
            y_mean=y_mean, y_std=y_std,
        )

        use_resid_mc = NM_RESIDUAL_MC == 1
        need_resid_pool = use_resid_mc or (NM_TRAIN_OVERSAMPLE > 1)
        resid_pool = _build_residual_mc_pool(train_days, day_targets, y_mean, y_std) if need_resid_pool else None

        train_ds = HourlyMultiTaskDataset(
            sample_dates=train_days,
            residual_mc_pool=resid_pool if need_resid_pool else None,
            residual_mc_prob=NM_RESIDUAL_MC_P if use_resid_mc else 0.0,
            residual_mc_scale=NM_RESIDUAL_MC_SCALE,
            residual_mc_npass=NM_RESIDUAL_MC_NPASS if use_resid_mc else 1,
            train_oversample=NM_TRAIN_OVERSAMPLE,
            oversample_resid_scale=NM_OVERSAMPLE_RESID_SCALE,
            **ds_base,
        )
        test_ds = HourlyMultiTaskDataset(
            sample_dates=test_days,
            residual_mc_pool=None, residual_mc_prob=0.0,
            train_oversample=1, **ds_base,
        )

        logger.info("  train=%d  test=%d (%s ~ %s)",
                     len(train_ds), len(test_ds),
                     test_days[0] if test_days else "?",
                     test_days[-1] if test_days else "?")

        _seed(42)
        model = Conv2dMultiTaskNet(
            h_slots=SLOT_WINDOW if PREDICT_15MIN else H_SLOTS
        ).to(DEVICE)
        _train_no_val(model, train_ds, y_mean, y_std)

        # 保存模型权重和归一化参数（供 RL 集成使用）
        weights_path = V8_DIR / "model_weights.pt"
        torch.save(model.state_dict(), weights_path)
        np.save(V8_DIR / "norm_mean.npy", norm_mean)
        np.save(V8_DIR / "norm_std.npy", norm_std)
        np.savez(V8_DIR / "target_stats.npz", y_mean=y_mean, y_std=y_std)
        logger.info("Saved: model_weights.pt, norm_mean/std.npy, target_stats.npz")

        day_preds, pred_dates = _predict_day(model, test_ds, y_mean, y_std)

        all_rows = []
        for d in pred_dates:
            if d not in day_targets:
                continue
            actual_h = day_targets[d]
            pred_h = day_preds[d]
            if PREDICT_15MIN:
                for s in range(96):
                    if not np.isnan(pred_h[s]):
                        all_rows.append({
                            "ts": pd.Timestamp(d) + pd.Timedelta(minutes=15 * s),
                            "actual": actual_h[s], "pred": pred_h[s],
                        })
            else:
                for h in range(24):
                    if not np.isnan(pred_h[h]):
                        all_rows.append({
                            "ts": pd.Timestamp(d) + pd.Timedelta(hours=h),
                            "actual": actual_h[h], "pred": pred_h[h],
                        })

        result_df = pd.DataFrame(all_rows)
        if "ts" in result_df.columns:
            result_df = result_df.set_index("ts").sort_index()
            result_df.index.name = "ts"

        pred_csv = "test_predictions_15min.csv" if PREDICT_15MIN else "test_predictions_hourly.csv"
        result_df.to_csv(V8_DIR / pred_csv)

        if len(result_df) > 0:
            a = result_df["actual"].values
            p = result_df["pred"].values
            mask = ~(np.isnan(a) | np.isnan(p))
            a, p = a[mask], p[mask]
            overall_mae  = float(np.mean(np.abs(a - p)))
            overall_rmse = float(np.sqrt(np.mean((a - p) ** 2)))
        else:
            overall_mae = overall_rmse = float("nan")

        test_dir = _eval_dir_acc(model, DataLoader(
            test_ds, min(512, max(len(test_ds), 1)), shuffle=False))

        logger.info("Single-split | train=%d  test_MAE=%.1f  test_RMSE=%.1f  dir=%.3f",
                     len(train_ds), overall_mae, overall_rmse, test_dir)
        logger.info("V8.0 SINGLE-SPLIT RESULTS  MAE=%.2f  RMSE=%.2f", overall_mae, overall_rmse)
        logger.info("  Output: %s", V8_DIR)

        _plot_weekly(result_df, V8_DIR)
        return {"predictions": result_df,
                "overall_mae": overall_mae, "overall_rmse": overall_rmse,
                "model": model, "norm_mean": norm_mean, "norm_std": norm_std,
                "y_mean": y_mean, "y_std": y_std}

    # ── Rolling-fold 模式 ──
    windows = build_rolling_test_windows()
    fold_start = int(os.environ.get("NM_TF_FOLD_START", "1"))
    max_folds = int(os.environ.get("NM_TF_FOLDS", "0"))
    if fold_start > 1:
        windows = [w for w in windows if w.fold >= fold_start]
    if max_folds > 0:
        windows = windows[:max_folds]
    logger.info("Rolling folds: %d (fold %d ~ %d)",
                len(windows),
                windows[0].fold if windows else 0,
                windows[-1].fold if windows else 0)

    all_rows = []
    fold_records = []

    for w in windows:
        train_start = (w.val_start - pd.DateOffset(months=TRAIN_MONTHS)).date()
        train_days = [d for d in valid_dates if train_start <= d < w.val_start.date()]
        val_days = [d for d in valid_dates if w.val_start.date() <= d < w.test_start.date()]
        test_days = [d for d in valid_dates if w.test_start.date() <= d <= w.test_end.date()]

        if not train_days or not val_days or not test_days:
            logger.warning("Fold %d skipped (empty split)", w.fold); continue
        if not all(d in day_targets for d in test_days):
            logger.warning("Fold %d skipped (missing test targets)", w.fold); continue

        norm_mean, norm_std = _compute_norm(day_lag0, day_lag1, day_lag2, train_days)
        tgt_stack = np.stack([day_targets[d] for d in train_days if d in day_targets])
        y_mean = float(tgt_stack.mean())
        y_std = float(tgt_stack.std()) + 1e-8

        ds_base = dict(
            day_lag0=day_lag0, day_lag1=day_lag1, day_lag2=day_lag2,
            day_targets=day_targets,
            norm_mean=norm_mean, norm_std=norm_std,
            y_mean=y_mean, y_std=y_std,
        )
        use_resid_mc = NM_RESIDUAL_MC == 1
        need_resid_pool = use_resid_mc or (NM_TRAIN_OVERSAMPLE > 1)
        resid_pool = None
        if need_resid_pool:
            resid_pool = _build_residual_mc_pool(train_days, day_targets, y_mean, y_std)
            logger.info(
                "  fold %d residual pool: size=%d (MC=%s, oversample=%d, virt_scale=%.3f)",
                w.fold, len(resid_pool), "on" if use_resid_mc else "off",
                NM_TRAIN_OVERSAMPLE, NM_OVERSAMPLE_RESID_SCALE,
            )

        train_ds = HourlyMultiTaskDataset(
            sample_dates=train_days,
            residual_mc_pool=resid_pool if need_resid_pool else None,
            residual_mc_prob=NM_RESIDUAL_MC_P if use_resid_mc else 0.0,
            residual_mc_scale=NM_RESIDUAL_MC_SCALE,
            residual_mc_npass=NM_RESIDUAL_MC_NPASS if use_resid_mc else 1,
            train_oversample=NM_TRAIN_OVERSAMPLE,
            oversample_resid_scale=NM_OVERSAMPLE_RESID_SCALE,
            **ds_base,
        )
        val_ds = HourlyMultiTaskDataset(
            sample_dates=val_days,
            residual_mc_pool=None, residual_mc_prob=0.0,
            train_oversample=1, **ds_base,
        )
        test_ds = HourlyMultiTaskDataset(
            sample_dates=test_days,
            residual_mc_pool=None, residual_mc_prob=0.0,
            train_oversample=1, **ds_base,
        )

        if len(train_ds) == 0:
            logger.warning("Fold %d skipped (no valid samples)", w.fold); continue

        logger.info(
            "Fold %2d | train=%d base (%d logical) (%s~%s) val=%d test=%d | test %s ~ %s",
            w.fold,
            getattr(train_ds, "_n_orig", len(train_ds)),
            len(train_ds),
            train_start, w.val_start.date(),
            len(val_ds), len(test_ds), w.test_start.date(), w.test_end.date(),
        )

        _seed(42)
        model = Conv2dMultiTaskNet(
            h_slots=SLOT_WINDOW if PREDICT_15MIN else H_SLOTS
        ).to(DEVICE)
        best_val = _train_fold(model, train_ds, val_ds, y_mean, y_std, w.fold, test_ds)

        day_preds, pred_dates = _predict_day(model, test_ds, y_mean, y_std)
        for d in pred_dates:
            if d not in day_targets:
                continue
            actual_h = day_targets[d]
            pred_h = day_preds[d]
            if PREDICT_15MIN:
                for s in range(96):
                    if not np.isnan(pred_h[s]):
                        all_rows.append({
                            "ts": pd.Timestamp(d) + pd.Timedelta(minutes=15 * s),
                            "actual": actual_h[s], "pred": pred_h[s],
                        })
            else:
                for h in range(24):
                    if not np.isnan(pred_h[h]):
                        all_rows.append({
                            "ts": pd.Timestamp(d) + pd.Timedelta(hours=h),
                            "actual": actual_h[h], "pred": pred_h[h],
                        })

        test_act_list, test_pred_list = [], []
        for d in pred_dates:
            if d in day_targets:
                test_act_list.append(day_targets[d])
                test_pred_list.append(day_preds[d])
        if test_act_list:
            test_act  = np.stack(test_act_list)
            test_pred = np.stack(test_pred_list)
            test_mae  = float(np.nanmean(np.abs(test_act - test_pred)))
            test_rmse = float(np.sqrt(np.nanmean((test_act - test_pred) ** 2)))
        else:
            test_mae = test_rmse = float("nan")

        test_dir = _eval_dir_acc(model, DataLoader(
            test_ds, min(512, max(len(test_ds), 1)), shuffle=False))

        fold_records.append({
            "fold": w.fold,
            "train_start": str(train_start),
            "test_start": str(w.test_start.date()),
            "test_end": str(w.test_end.date()),
            "train_n": len(train_ds), "val_n": len(val_ds), "test_n": len(test_ds),
            "val_MAE": round(best_val, 2),
            "test_MAE": round(test_mae, 2),
            "test_RMSE": round(test_rmse, 2),
            "test_dir_acc": round(test_dir, 4),
        })
        logger.info("Fold %2d done | val=%.1f  test_MAE=%.1f  test_RMSE=%.1f  dir=%.3f",
                     w.fold, best_val, test_mae, test_rmse, test_dir)

    result_df = pd.DataFrame(all_rows)
    if "ts" in result_df.columns:
        result_df = result_df.set_index("ts").sort_index()
        result_df.index.name = "ts"

    pred_csv = "test_predictions_15min.csv" if PREDICT_15MIN else "test_predictions_hourly.csv"
    result_df.to_csv(V8_DIR / pred_csv)

    metrics_df = pd.DataFrame(fold_records)
    metrics_df.to_csv(V8_DIR / "rolling_metrics.csv", index=False)

    if len(result_df) > 0:
        a = result_df["actual"].values
        p = result_df["pred"].values
        mask = ~(np.isnan(a) | np.isnan(p))
        a, p = a[mask], p[mask]
        overall_mae  = float(np.mean(np.abs(a - p)))
        overall_rmse = float(np.sqrt(np.mean((a - p) ** 2)))
    else:
        overall_mae = overall_rmse = float("nan")

    logger.info("=" * 60)
    logger.info("V8.0 RESULTS")
    logger.info("  MAE:  %.2f", overall_mae)
    logger.info("  RMSE: %.2f", overall_rmse)
    if PREDICT_15MIN:
        logger.info("  Predictions: %d 15min points (~%d days)",
                     len(result_df), len(result_df) // 96 if len(result_df) > 0 else 0)
    else:
        logger.info("  Predictions: %d hours (%d days)",
                     len(result_df), len(result_df) // 24 if len(result_df) > 0 else 0)
    logger.info("  Output: %s", V8_DIR)
    logger.info("=" * 60)

    _plot_weekly(result_df, V8_DIR)

    return {"predictions": result_df, "fold_metrics": metrics_df,
            "overall_mae": overall_mae, "overall_rmse": overall_rmse}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    run_v8_multitask()
