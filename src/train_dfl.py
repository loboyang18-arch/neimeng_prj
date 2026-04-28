"""
Decision-Focused Learning (DFL) 训练脚本
==========================================
两阶段训练策略：
  阶段 1 (MSE warm-up)：用标准 L1+方向分类 loss 训练 V8 至收敛（复用现有逻辑）
  阶段 2 (DFL fine-tune)：用 Regret Loss 微调 V8，让预测直接优化 MILP 决策质量

Regret = PF_Revenue(actual_price) - LP_Revenue(lp_dispatch(pred_price), actual_price)

梯度穿过 cvxpylayers 的可微 LP 层，回传到 V8 的 Conv2D 参数。
推理时：DFL 微调后的 V8 输出预测 → 送入完整 MILP（非松弛）做最终调度。
"""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

# 环境变量必须在导入 model_v8_multitask 之前设置
NODAL_COL = "price_sudun_500kv1m_nodal"
os.environ.setdefault("NM_V8_TARGET", NODAL_COL)
os.environ.setdefault("NM_V8_EXTRA_LAG1",
                       "price_sudun_500kv1m_energy,price_sudun_500kv1m_cong")
os.environ.setdefault("NM_V8_HOURLY_AGG", "mean4")
os.environ.setdefault("NM_MIN_FEATURE_DATE", "2023-06-01")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.model_v8_multitask import (
    Conv2dMultiTaskNet,
    _load_dws,
    _build_daily_arrays,
    _compute_norm,
    _get_hour_slots,
    _seed,
    HourlyMultiTaskDataset,
    _train_no_val,
    _predict_day,
    _eval_mae_hourly,
    LAG0_COLS, LAG1_COLS, LAG2_COLS,
    C_TOTAL, H_SLOTS, LOOKBACK_DAYS,
    CONTEXT_BEFORE, CONTEXT_AFTER,
    DEVICE, MAX_EPOCHS, BATCH_SIZE, LR,
)
from src.dfl_optimizer_layer import DiffDispatchLP, compute_revenue, T as LP_T

logger = logging.getLogger(__name__)

OUTPUT_DIR = ROOT / "output"
DFL_DIR = OUTPUT_DIR / "experiments" / "v8.0-dfl"


class DailyDFLDataset(torch.utils.data.Dataset):
    """每日一个样本：24 个 V8 输入 grid + 96 个实际节点电价。"""

    def __init__(
        self,
        sample_dates: List,
        day_lag0: Dict, day_lag1: Dict, day_lag2: Dict,
        day_targets: Dict,
        day_nodal_96: Dict,
        norm_mean: np.ndarray, norm_std: np.ndarray,
        y_mean: float, y_std: float,
        pf_revenues: Dict,
    ):
        self.items = []

        for d in sample_dates:
            if d not in day_targets or d not in day_nodal_96:
                continue

            nodal = day_nodal_96[d]
            if not np.isfinite(nodal).all():
                continue

            dates0 = [(pd.Timestamp(d) - pd.Timedelta(days=off)).date()
                      for off in range(LOOKBACK_DAYS - 1, -1, -1)]
            dates1 = [(pd.Timestamp(d) - pd.Timedelta(days=off)).date()
                      for off in range(LOOKBACK_DAYS, 0, -1)]
            dates2 = [(pd.Timestamp(d) - pd.Timedelta(days=off)).date()
                      for off in range(LOOKBACK_DAYS + 1, 1, -1)]

            a0 = set(day_lag0.keys())
            a1 = set(day_lag1.keys())
            a2 = set(day_lag2.keys())
            ok = (all(dd in a0 for dd in dates0)
                  and all(dd in a1 for dd in dates1)
                  and all(dd in a2 for dd in dates2))
            if not ok:
                continue

            grids_24 = []
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
                grids_24.append(grid)

            grids_stack = np.stack(grids_24, axis=0)  # (24, C, H_SLOTS, 7)
            pf_rev = pf_revenues.get(d, 0.0)

            self.items.append({
                "grids": grids_stack,
                "actual_96": nodal.astype(np.float32),
                "pf_revenue": np.float32(pf_rev),
                "y_mean": np.float32(y_mean),
                "y_std": np.float32(y_std),
                "date": str(d),
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        return (
            torch.from_numpy(item["grids"]),
            torch.from_numpy(item["actual_96"]),
            torch.tensor(item["pf_revenue"]),
            torch.tensor(item["y_mean"]),
            torch.tensor(item["y_std"]),
        )


def _precompute_pf_revenues(day_nodal_96: Dict, dates: List) -> Dict:
    """用完整 MILP 求解每天的 PF 收益。"""
    from scripts.strategy_milp_15min import solve_pf_day_15min, eval_day_revenue_15min

    pf = {}
    for d in dates:
        ds = str(d)
        if d not in day_nodal_96:
            continue
        actual = day_nodal_96[d]
        if not np.isfinite(actual).all():
            continue
        c_pf, d_pf, _ = solve_pf_day_15min(actual)
        rev = eval_day_revenue_15min(c_pf, d_pf, actual, aux_mwh=0.0)
        pf[d] = rev["gross"]
    return pf


def _build_nodal_96(dws: pd.DataFrame) -> Dict:
    """从 DWS 提取每天 96 个 15 分钟的节点电价。"""
    start_date = dws.index.min().normalize().date()
    end_date = dws.index.max().date()
    date_range = pd.date_range(start_date, end_date, freq="D")
    result = {}
    for d_ts in date_range:
        d = d_ts.date()
        grid = pd.date_range(pd.Timestamp(d), periods=96, freq="15min")
        vals = dws.reindex(grid)[NODAL_COL].values.astype(np.float32)
        if len(vals) == 96 and np.isfinite(vals).all():
            result[d] = vals
    return result


def train_dfl(
    dfl_epochs: int = 30,
    dfl_lr: float = 3e-5,
    mse_pretrain: bool = True,
    mse_epochs: int = 100,
    test_start: str = "2026-01-25",
    min_feature_date: str = "2023-06-01",
    alpha_start: float = 0.3,
    alpha_end: float = 0.8,
):
    """
    DFL 训练主流程：
    1. 加载数据，构建日级数据集
    2. (可选) MSE 预训练
    3. DFL 微调
    4. 保存预测结果
    """
    DFL_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Decision-Focused Learning (DFL) for V8")
    logger.info("=" * 60)

    _seed(42)

    # ── 加载数据 ──
    logger.info("加载 DWS 数据...")
    dws = _load_dws()
    valid_dates, day_lag0, day_lag1, day_lag2, day_targets = _build_daily_arrays(dws)
    logger.info("  有效天数: %d", len(valid_dates))

    # 提取节点电价
    day_nodal_96 = _build_nodal_96(dws)
    logger.info("  节点电价天数: %d", len(day_nodal_96))

    # ── 划分数据 ──
    test_dt = pd.Timestamp(test_start).date()
    train_days = [d for d in valid_dates if d < test_dt]
    test_days = [d for d in valid_dates if d >= test_dt]
    logger.info("  训练: %d 天, 测试: %d 天 (%s ~ %s)",
                len(train_days), len(test_days),
                test_days[0] if test_days else "?",
                test_days[-1] if test_days else "?")

    # ── 归一化 ──
    norm_mean, norm_std = _compute_norm(day_lag0, day_lag1, day_lag2, train_days)
    tgt_stack = np.stack([day_targets[d] for d in train_days if d in day_targets])
    y_mean = float(tgt_stack.mean())
    y_std = float(tgt_stack.std()) + 1e-8

    # ── 阶段 1：MSE 预训练 ──
    model = Conv2dMultiTaskNet(h_slots=H_SLOTS).to(DEVICE)

    pretrained_path = DFL_DIR / "v8_mse_pretrained.pt"
    if mse_pretrain:
        existing_weights = (OUTPUT_DIR / "experiments" / "v8.0-rl-pretrain" / "model_weights.pt")
        if existing_weights.exists():
            logger.info("加载已有 V8 预训练权重: %s", existing_weights)
            model.load_state_dict(
                torch.load(existing_weights, map_location=DEVICE, weights_only=True)
            )
        else:
            logger.info("阶段 1: MSE 预训练 (%d epochs)...", mse_epochs)
            ds_base = dict(
                day_lag0=day_lag0, day_lag1=day_lag1, day_lag2=day_lag2,
                day_targets=day_targets,
                norm_mean=norm_mean, norm_std=norm_std,
                y_mean=y_mean, y_std=y_std,
            )
            train_ds = HourlyMultiTaskDataset(
                sample_dates=train_days,
                residual_mc_pool=None, residual_mc_prob=0.0,
                train_oversample=1, **ds_base,
            )
            global MAX_EPOCHS
            old_epochs = MAX_EPOCHS
            # Temporarily override for pre-training
            import src.model_v8_multitask as v8mod
            v8mod.MAX_EPOCHS = mse_epochs
            _train_no_val(model, train_ds, y_mean, y_std)
            v8mod.MAX_EPOCHS = old_epochs

        torch.save(model.state_dict(), pretrained_path)
        logger.info("MSE 预训练权重保存: %s", pretrained_path)

    # MSE baseline 预测
    logger.info("生成 MSE baseline 预测...")
    ds_base = dict(
        day_lag0=day_lag0, day_lag1=day_lag1, day_lag2=day_lag2,
        day_targets=day_targets,
        norm_mean=norm_mean, norm_std=norm_std,
        y_mean=y_mean, y_std=y_std,
    )
    test_ds = HourlyMultiTaskDataset(
        sample_dates=test_days,
        residual_mc_pool=None, residual_mc_prob=0.0,
        train_oversample=1, **ds_base,
    )
    mse_preds, _ = _predict_day(model, test_ds, y_mean, y_std)

    # ── 预计算 PF 收益 ──
    logger.info("预计算 PF 收益（训练集）...")
    train_with_nodal = [d for d in train_days if d in day_nodal_96]
    pf_revenues = _precompute_pf_revenues(day_nodal_96, train_with_nodal)
    logger.info("  PF 收益计算完成: %d 天", len(pf_revenues))

    # ── 构建 DFL 数据集 ──
    dfl_train_dates = [d for d in train_days if d in pf_revenues]
    dfl_dataset = DailyDFLDataset(
        sample_dates=dfl_train_dates,
        day_lag0=day_lag0, day_lag1=day_lag1, day_lag2=day_lag2,
        day_targets=day_targets,
        day_nodal_96=day_nodal_96,
        norm_mean=norm_mean, norm_std=norm_std,
        y_mean=y_mean, y_std=y_std,
        pf_revenues=pf_revenues,
    )
    logger.info("DFL 训练样本: %d 天", len(dfl_dataset))

    if len(dfl_dataset) == 0:
        logger.error("DFL 训练集为空，退出")
        return

    # ── 阶段 2：DFL 微调 (混合损失: α·regret + (1-α)·MSE) ──
    # 整个管道在 CPU 上端到端可微：V8 → 上采样 → LP 松弛 → revenue → regret
    logger.info("=" * 60)
    logger.info("阶段 2: DFL 微调 (%d epochs, lr=%e, α=%.2f→%.2f)",
                dfl_epochs, dfl_lr, alpha_start, alpha_end)
    logger.info("=" * 60)

    lp_layer = DiffDispatchLP()
    mse_loss_fn = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=dfl_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=dfl_epochs, eta_min=dfl_lr * 0.01,
    )

    loader = torch.utils.data.DataLoader(
        dfl_dataset, batch_size=1, shuffle=True, num_workers=0,
    )

    best_regret = float("inf")
    best_state = None

    for epoch in range(dfl_epochs):
        alpha = alpha_start + (alpha_end - alpha_start) * epoch / max(dfl_epochs - 1, 1)

        model.train()
        total_regret = 0.0
        total_mse = 0.0
        total_pf = 0.0
        total_decision = 0.0
        n_days = 0
        n_fail = 0
        t_epoch = time.time()

        for batch in loader:
            grids_24, actual_96, pf_rev, ym, ys = batch

            grids_24 = grids_24.squeeze(0).to(DEVICE)  # (24, C, H, 7)
            actual_96 = actual_96.squeeze(0)             # (96,)
            pf_val = pf_rev.item()
            y_m = ym.item()
            y_s = ys.item()

            # V8 forward → 24 hourly predictions
            model_out, _ = model(grids_24)             # (24,) normalized
            pred_hourly = model_out * y_s + y_m        # denormalized

            # MSE term: L1 vs actual hourly means
            actual_hourly = actual_96.reshape(24, 4).mean(dim=1).to(DEVICE)
            mse_term = mse_loss_fn(pred_hourly, actual_hourly)

            # Upsample 24→96, LP solve, revenue — all end-to-end differentiable
            pred_96 = pred_hourly.repeat_interleave(4).unsqueeze(0)  # (1,96)
            try:
                c_lp, d_lp = lp_layer(pred_96)       # each (1,96)
            except Exception:
                n_fail += 1
                optimizer.zero_grad()
                mse_term.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_mse += mse_term.item()
                continue

            decision_rev = compute_revenue(c_lp, d_lp, actual_96.unsqueeze(0))
            decision_rev = decision_rev.squeeze(0)  # scalar

            # 归一化 regret: regret / |pf| — 使量级与 MSE (~30-100) 可比
            pf_abs = max(abs(pf_val), 1.0)
            regret_norm = (pf_val - decision_rev) / pf_abs

            # 混合损失
            loss = alpha * regret_norm + (1 - alpha) * mse_term

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_regret += (pf_val - decision_rev.item())
            total_mse += mse_term.item()
            total_pf += pf_val
            total_decision += decision_rev.item()
            n_days += 1

        scheduler.step()
        elapsed = time.time() - t_epoch

        if n_days > 0:
            avg_regret = total_regret / n_days
            realization = total_decision / total_pf if total_pf > 0 else 0
            lr_now = optimizer.param_groups[0]["lr"]

            if avg_regret < best_regret:
                best_regret = avg_regret
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

            logger.info(
                "  Epoch %2d/%d  α=%.2f  regret=%.0f  mse=%.1f  decision=%.0f  pf=%.0f  "
                "realization=%.1f%%  fail=%d  lr=%.1e  (%.1fs)",
                epoch + 1, dfl_epochs, alpha, avg_regret, total_mse / n_days,
                total_decision / n_days,
                total_pf / n_days, realization * 100, n_fail, lr_now, elapsed,
            )

    # 恢复最佳权重
    if best_state is not None:
        model.load_state_dict(best_state)
        logger.info("加载最佳 DFL 权重 (regret=%.0f)", best_regret)

    # ── 保存 DFL 权重 ──
    dfl_weights_path = DFL_DIR / "model_weights_dfl.pt"
    torch.save(model.state_dict(), dfl_weights_path)
    logger.info("DFL 权重保存: %s", dfl_weights_path)

    # ── 保存归一化参数 ──
    np.save(DFL_DIR / "norm_mean.npy", norm_mean)
    np.save(DFL_DIR / "norm_std.npy", norm_std)
    np.savez(DFL_DIR / "target_stats.npz", y_mean=y_mean, y_std=y_std)

    # ── DFL 预测（测试集）──
    logger.info("生成 DFL 预测...")
    dfl_preds, pred_dates = _predict_day(model, test_ds, y_mean, y_std)

    # ── 保存预测 CSV ──
    all_rows = []
    for d in pred_dates:
        if d not in day_targets:
            continue
        actual_h = day_targets[d]
        pred_h = dfl_preds[d]
        for h in range(24):
            if not np.isnan(pred_h[h]):
                all_rows.append({
                    "ts": pd.Timestamp(d) + pd.Timedelta(hours=h),
                    "actual": actual_h[h],
                    "pred": pred_h[h],
                })
    result_df = pd.DataFrame(all_rows)
    if "ts" in result_df.columns:
        result_df = result_df.set_index("ts").sort_index()

    result_df.to_csv(DFL_DIR / "test_predictions_hourly.csv")
    logger.info("DFL 预测保存: %s", DFL_DIR / "test_predictions_hourly.csv")

    # 同时保存 MSE baseline 预测
    mse_rows = []
    for d in pred_dates:
        if d not in day_targets or d not in mse_preds:
            continue
        actual_h = day_targets[d]
        pred_h = mse_preds[d]
        for h in range(24):
            if not np.isnan(pred_h[h]):
                mse_rows.append({
                    "ts": pd.Timestamp(d) + pd.Timedelta(hours=h),
                    "actual": actual_h[h],
                    "pred": pred_h[h],
                })
    mse_df = pd.DataFrame(mse_rows)
    if "ts" in mse_df.columns:
        mse_df = mse_df.set_index("ts").sort_index()
    mse_df.to_csv(DFL_DIR / "test_predictions_hourly_mse.csv")

    # ── 简要预测质量对比 ──
    if len(result_df) > 0:
        a = result_df["actual"].values
        p = result_df["pred"].values
        mask = ~(np.isnan(a) | np.isnan(p))
        dfl_mae = float(np.mean(np.abs(a[mask] - p[mask])))
    else:
        dfl_mae = float("nan")

    if len(mse_df) > 0:
        a = mse_df["actual"].values
        p = mse_df["pred"].values
        mask = ~(np.isnan(a) | np.isnan(p))
        mse_mae = float(np.mean(np.abs(a[mask] - p[mask])))
    else:
        mse_mae = float("nan")

    logger.info("=" * 60)
    logger.info("预测 MAE 对比:")
    logger.info("  MSE baseline: %.1f 元/MWh", mse_mae)
    logger.info("  DFL fine-tune: %.1f 元/MWh", dfl_mae)
    logger.info("(MAE 可能变大，但决策质量应更优 — 需 MILP 评估验证)")
    logger.info("=" * 60)

    return {
        "model": model,
        "dfl_predictions": result_df,
        "mse_predictions": mse_df,
        "dfl_mae": dfl_mae,
        "mse_mae": mse_mae,
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    train_dfl(
        dfl_epochs=30,
        dfl_lr=1e-4,
        mse_pretrain=True,
        mse_epochs=100,
    )
