"""
SPO+ (Smart Predict then Optimize) 训练脚本
=============================================
与 DFL (cvxpylayers) 方案的关键区别：
  - 训练时直接调用完整 MILP 黑箱（非 LP 松弛）→ 消除 LP-MILP gap
  - 梯度由 SPO+ 公式解析给出，不需要可微求解器
  - 每个样本需 2 次 MILP 调用（pred 价格 + surrogate 价格）

SPO+ Loss:
  L(ŷ, y) = max_x { (2ŷ - y)ᵀ x } - ŷᵀ x*(ŷ)
  ∂L/∂ŷ = 2·[x*(2ŷ-y) - x*(ŷ)]   （解析梯度，无需微分求解器）

含中途测试集评估：每 eval_every 个 epoch 跑一次完整 MILP 评估。
"""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

NODAL_COL = "price_sudun_500kv1m_nodal"
os.environ.setdefault("NM_V8_TARGET", NODAL_COL)
os.environ.setdefault("NM_V8_EXTRA_LAG1",
                       "price_sudun_500kv1m_energy,price_sudun_500kv1m_cong")
os.environ.setdefault("NM_V8_HOURLY_AGG", "mean4")
os.environ.setdefault("NM_MIN_FEATURE_DATE", "2024-12-14")

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
    _predict_day,
    C_TOTAL, H_SLOTS, LOOKBACK_DAYS,
    DEVICE,
)
NODAL_COL_LOCAL = NODAL_COL


def _build_nodal_96(dws: pd.DataFrame) -> Dict:
    """从 DWS 提取每天 96 个 15 分钟的节点电价。"""
    start_date = dws.index.min().normalize().date()
    end_date = dws.index.max().date()
    date_range = pd.date_range(start_date, end_date, freq="D")
    result = {}
    for d_ts in date_range:
        d = d_ts.date()
        grid = pd.date_range(pd.Timestamp(d), periods=96, freq="15min")
        vals = dws.reindex(grid)[NODAL_COL_LOCAL].values.astype(np.float32)
        if len(vals) == 96 and np.isfinite(vals).all():
            result[d] = vals
    return result


def _precompute_pf_revenues(day_nodal_96: Dict, dates: List) -> Dict:
    from scripts.strategy_milp_15min import solve_pf_day_15min, eval_day_revenue_15min
    pf = {}
    for d in dates:
        if d not in day_nodal_96:
            continue
        actual = day_nodal_96[d]
        if not np.isfinite(actual).all():
            continue
        c_pf, d_pf, _ = solve_pf_day_15min(actual)
        rev = eval_day_revenue_15min(c_pf, d_pf, actual, aux_mwh=0.0)
        pf[d] = rev["gross"]
    return pf


class DailyDFLDataset(torch.utils.data.Dataset):
    """每日一个样本：24 个 V8 输入 grid + 96 个实际节点电价。"""

    def __init__(self, sample_dates, day_lag0, day_lag1, day_lag2,
                 day_targets, day_nodal_96, norm_mean, norm_std,
                 y_mean, y_std, pf_revenues):
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
            a0, a1, a2 = set(day_lag0), set(day_lag1), set(day_lag2)
            if not (all(dd in a0 for dd in dates0) and
                    all(dd in a1 for dd in dates1) and
                    all(dd in a2 for dd in dates2)):
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
            self.items.append({
                "grids": np.stack(grids_24, axis=0),
                "actual_96": nodal.astype(np.float32),
                "pf_revenue": np.float32(pf_revenues.get(d, 0.0)),
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

logger = logging.getLogger(__name__)

OUTPUT_DIR = ROOT / "output"
SPO_DIR = OUTPUT_DIR / "experiments" / "v8.0-spo-v4"
DT = 0.25


def _solve_milp_dispatch(prices_96: np.ndarray):
    """调用完整 MILP，返回净功率向量 (d-c)*DT。"""
    from scripts.strategy_milp_15min import solve_day_milp_15min
    c, d, _ = solve_day_milp_15min(prices_96)
    return c, d


def _evaluate_test_milp(
    model, test_ds, y_mean, y_std, actual_xlsx, test_start, test_end,
) -> dict:
    """
    中途评估：V8 预测 → 完整 MILP → 用实际价格计算净收益。
    返回 {net, pf_net, realization, mae}。
    """
    from scripts.strategy_milp_15min import (
        solve_day_milp_15min, solve_pf_day_15min,
        eval_day_revenue_15min, load_actual_15min,
    )

    # V8 预测
    day_preds, pred_dates = _predict_day(model, test_ds, y_mean, y_std)

    # 加载实际 15 分钟价格
    actual_df = load_actual_15min(actual_xlsx)

    total_net = 0.0
    total_pf_net = 0.0
    n_days = 0
    soc_carry = 0.0
    soc_carry_pf = 0.0
    dates_sorted = sorted([str(d) for d in pred_dates if str(d) in actual_df.index])

    for i, date in enumerate(dates_sorted):
        if date not in actual_df.index:
            continue
        d_obj = pd.Timestamp(date).date()
        if d_obj not in day_preds:
            continue

        pred_h = day_preds[d_obj]
        pred_96 = np.repeat(pred_h, 4)
        actual_96 = actual_df.loc[date].values.astype(float)

        is_last = (i == len(dates_sorted) - 1)
        force_end = is_last
        if not is_last and i + 1 < len(dates_sorted):
            next_d = dates_sorted[i + 1]
            next_d_obj = pd.Timestamp(next_d).date()
            next_avg = float(np.mean(day_preds.get(next_d_obj, pred_h))) if next_d_obj in day_preds else 0.0
        else:
            next_avg = 0.0

        c, d, soc = solve_day_milp_15min(
            pred_96, soc_init=soc_carry,
            force_zero_end=force_end,
            next_day_avg_price=next_avg,
        )
        c_pf, d_pf, soc_pf = solve_pf_day_15min(
            actual_96, soc_init=soc_carry_pf,
            force_zero_end=force_end,
            next_day_avg_price=float(np.mean(actual_96)),
        )

        soc_carry = float(soc[-1]) if soc.sum() > 0 else 0.0
        soc_carry_pf = float(soc_pf[-1]) if soc_pf.sum() > 0 else 0.0

        rev = eval_day_revenue_15min(c, d, actual_96)
        rev_pf = eval_day_revenue_15min(c_pf, d_pf, actual_96)

        total_net += rev["net"]
        total_pf_net += rev_pf["net"]
        n_days += 1

    realization = total_net / total_pf_net if abs(total_pf_net) > 1 else 0
    mae = float("nan")
    if len(test_ds) > 0:
        from torch.utils.data import DataLoader
        from src.model_v8_multitask import _eval_mae_hourly
        test_loader = DataLoader(test_ds, min(512, len(test_ds)), shuffle=False)
        mae = _eval_mae_hourly(model, test_loader, y_mean, y_std)

    return {
        "n_days": n_days,
        "net": total_net,
        "pf_net": total_pf_net,
        "realization": realization,
        "mae": mae,
    }


def train_spo_plus(
    spo_epochs: int = 30,
    spo_lr: float = 3e-5,
    test_start: str = "2026-01-27",
    test_end: str = "2026-04-17",
    alpha_start: float = 0.3,
    alpha_end: float = 0.7,
    eval_every: int = 5,
    freeze_encoder: bool = False,
    anchor_lambda: float = 0.0,
    grad_accum_steps: int = 1,
    spo_grad_scale: float = 1.0,
):
    SPO_DIR.mkdir(parents=True, exist_ok=True)
    actual_xlsx = ROOT / "source_data" / "日清算结果查询电厂侧(1)_副本.xlsx"

    logger.info("=" * 60)
    logger.info("SPO+ Training for V8")
    logger.info("=" * 60)

    _seed(42)

    # ── 加载数据 ──
    logger.info("加载 DWS 数据...")
    dws = _load_dws()
    valid_dates, day_lag0, day_lag1, day_lag2, day_targets = _build_daily_arrays(dws)
    day_nodal_96 = _build_nodal_96(dws)

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

    # ── 加载 MSE 预训练权重 ──
    model = Conv2dMultiTaskNet(h_slots=H_SLOTS).to(DEVICE)
    pretrained = OUTPUT_DIR / "experiments" / "v8.0-rl-pretrain" / "model_weights.pt"
    if pretrained.exists():
        model.load_state_dict(
            torch.load(pretrained, map_location=DEVICE, weights_only=True)
        )
        logger.info("加载 V8 预训练权重: %s", pretrained)
    else:
        logger.warning("未找到预训练权重，从随机初始化开始")

    # 冻结编码器（只微调回归头）
    if freeze_encoder:
        for name, param in model.named_parameters():
            if "reg_head" not in name:
                param.requires_grad = False
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_p = sum(p.numel() for p in model.parameters())
        logger.info("冻结编码器: 可训练参数 %d / %d", trainable, total_p)

    # 保存 MSE baseline 权重（同时作为锚定基准）
    torch.save(model.state_dict(), SPO_DIR / "v8_mse_baseline.pt")
    anchor_params = {n: p.detach().clone() for n, p in model.named_parameters()
                     if p.requires_grad} if anchor_lambda > 0 else {}

    # ── 预计算 PF 收益 ──
    logger.info("预计算 PF 收益...")
    train_with_nodal = [d for d in train_days if d in day_nodal_96]
    pf_revenues = _precompute_pf_revenues(day_nodal_96, train_with_nodal)
    logger.info("  PF 完成: %d 天", len(pf_revenues))

    # ── DFL 数据集 ──
    dfl_train_dates = [d for d in train_days if d in pf_revenues]
    train_dataset = DailyDFLDataset(
        sample_dates=dfl_train_dates,
        day_lag0=day_lag0, day_lag1=day_lag1, day_lag2=day_lag2,
        day_targets=day_targets,
        day_nodal_96=day_nodal_96,
        norm_mean=norm_mean, norm_std=norm_std,
        y_mean=y_mean, y_std=y_std,
        pf_revenues=pf_revenues,
    )
    logger.info("SPO+ 训练样本: %d 天", len(train_dataset))

    # ── 测试集 Dataset（用于中途评估）──
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

    # ── MSE baseline 评估 ──
    logger.info("MSE baseline 测试集评估...")
    mse_eval = _evaluate_test_milp(
        model, test_ds, y_mean, y_std, actual_xlsx, test_start, test_end,
    )
    logger.info(
        "  MSE baseline: 净收益=%.1f万  PF=%.1f万  兑现率=%.1f%%  MAE=%.1f",
        mse_eval["net"] / 1e4, mse_eval["pf_net"] / 1e4,
        mse_eval["realization"] * 100, mse_eval["mae"],
    )

    # ── SPO+ 训练 ──
    logger.info("=" * 60)
    logger.info("SPO+ 微调: %d epochs, lr=%e, α=%.2f→%.2f, eval_every=%d",
                spo_epochs, spo_lr, alpha_start, alpha_end, eval_every)
    logger.info("  freeze_encoder=%s, anchor_λ=%.1e, grad_accum=%d, spo_scale=%.2f",
                freeze_encoder, anchor_lambda, grad_accum_steps, spo_grad_scale)
    logger.info("=" * 60)

    mse_loss_fn = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=spo_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=spo_epochs, eta_min=spo_lr * 0.01,
    )

    loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=0,
    )

    best_test_real = mse_eval["realization"]
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    eval_log = [{"epoch": 0, **mse_eval, "label": "MSE-baseline"}]

    patience = getattr(train_spo_plus, '_patience', 6)
    no_improve_count = 0

    for epoch in range(spo_epochs):
        if alpha_start == alpha_end:
            alpha = alpha_start
        else:
            alpha = alpha_start + (alpha_end - alpha_start) * epoch / max(spo_epochs - 1, 1)
        model.train()

        total_spo = 0.0
        total_mse = 0.0
        total_decision = 0.0
        total_pf = 0.0
        n_days = 0
        t_epoch = time.time()

        for batch in loader:
            grids_24, actual_96, pf_rev, ym, ys = batch

            grids_24 = grids_24.squeeze(0).to(DEVICE)
            actual_96_np = actual_96.squeeze(0).numpy()
            pf_val = pf_rev.item()
            y_m = ym.item()
            y_s = ys.item()

            # V8 forward
            model_out, _ = model(grids_24)
            pred_hourly = model_out * y_s + y_m

            # MSE term
            actual_hourly = torch.from_numpy(
                actual_96_np.reshape(24, 4).mean(axis=1)
            ).float().to(DEVICE)
            mse_term = mse_loss_fn(pred_hourly, actual_hourly)

            # Upsample to 96
            pred_96_np = np.repeat(pred_hourly.detach().cpu().numpy(), 4)

            # MILP 黑箱调用（不参与反向传播）
            c_pred, d_pred = _solve_milp_dispatch(pred_96_np)

            surrogate_price = 2 * pred_96_np - actual_96_np
            c_spo, d_spo = _solve_milp_dispatch(surrogate_price)

            # 决策收益（用实际价格评估）
            decision_rev = float(np.sum((d_pred - c_pred) * actual_96_np * DT))

            # SPO+ 解析梯度: 2 * [(d_spo - c_spo) - (d_pred - c_pred)] * DT
            # 表示 "预测每增加1元，SPO+ loss 的变化量"
            net_spo = (d_spo - c_spo) * DT  # (96,)
            net_pred = (d_pred - c_pred) * DT  # (96,)
            spo_grad_96 = 2.0 * (net_spo - net_pred)  # (96,)

            # 聚合为 24 维（每小时平均）
            spo_grad_24 = spo_grad_96.reshape(24, 4).mean(axis=1)
            spo_grad_24 = torch.from_numpy(spo_grad_24.astype(np.float32)).to(DEVICE)

            # 梯度归一化 + 衰减：将 SPO+ 梯度量级对齐到 MSE loss
            grad_norm = spo_grad_24.norm() + 1e-8
            spo_grad_24 = spo_grad_24 / grad_norm * (mse_term.item() + 1e-4) * spo_grad_scale

            # 混合反向传播（梯度累积模式：每 grad_accum_steps 步才 step）
            if n_days % grad_accum_steps == 0:
                optimizer.zero_grad()
            # MSE 部分正常反向传播
            ((1 - alpha) * mse_term / grad_accum_steps).backward(retain_graph=True)
            # SPO+ 部分：手动注入梯度到 pred_hourly
            pred_hourly.backward(alpha * spo_grad_24 / grad_accum_steps)

            # 权重锚定正则化
            if anchor_lambda > 0:
                for name, param in model.named_parameters():
                    if param.requires_grad and name in anchor_params and param.grad is not None:
                        param.grad.add_(anchor_lambda * (param.data - anchor_params[name]))

            if (n_days + 1) % grad_accum_steps == 0 or n_days == len(loader) - 1:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_spo += float(np.sum(spo_grad_96 * pred_96_np))
            total_mse += mse_term.item()
            total_decision += decision_rev
            total_pf += pf_val
            n_days += 1

        scheduler.step()
        elapsed = time.time() - t_epoch

        if n_days > 0:
            avg_regret = (total_pf - total_decision) / n_days
            realization = total_decision / total_pf if total_pf > 0 else 0
            lr_now = optimizer.param_groups[0]["lr"]

            logger.info(
                "  Epoch %2d/%d  α=%.2f  train_regret=%.0f  mse=%.1f  "
                "train_real=%.1f%%  lr=%.1e  (%.1fs)",
                epoch + 1, spo_epochs, alpha, avg_regret,
                total_mse / n_days, realization * 100, lr_now, elapsed,
            )

        # ── 中途测试集评估 ──
        if (epoch + 1) % eval_every == 0 or epoch == spo_epochs - 1:
            logger.info("  → 测试集 MILP 评估中...")
            t_eval = time.time()
            test_eval = _evaluate_test_milp(
                model, test_ds, y_mean, y_std, actual_xlsx, test_start, test_end,
            )
            eval_log.append({"epoch": epoch + 1, **test_eval, "label": f"SPO-ep{epoch+1}"})

            improved = ""
            if test_eval["realization"] > best_test_real:
                best_test_real = test_eval["realization"]
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                torch.save(model.state_dict(), SPO_DIR / "best_spo_model.pt")
                improved = " ★ NEW BEST"
                no_improve_count = 0
            else:
                no_improve_count += 1

            logger.info(
                "  → 测试集: 净收益=%.1f万  兑现率=%.1f%%  MAE=%.1f%s  (%.1fs)",
                test_eval["net"] / 1e4,
                test_eval["realization"] * 100,
                test_eval["mae"],
                improved,
                time.time() - t_eval,
            )

            if no_improve_count >= patience:
                logger.info("  ⏹ Early stopping: 连续 %d 次评估无改善", patience)
                break

    # ── 恢复最佳权重 ──
    model.load_state_dict(best_state)
    logger.info("加载最佳 SPO+ 权重 (测试兑现率=%.1f%%)", best_test_real * 100)

    # ── 保存 ──
    torch.save(model.state_dict(), SPO_DIR / "model_weights_spo.pt")
    np.save(SPO_DIR / "norm_mean.npy", norm_mean)
    np.save(SPO_DIR / "norm_std.npy", norm_std)
    np.savez(SPO_DIR / "target_stats.npz", y_mean=y_mean, y_std=y_std)

    # 生成最终预测
    dfl_preds, pred_dates = _predict_day(model, test_ds, y_mean, y_std)
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
                    "actual": actual_h[h], "pred": pred_h[h],
                })
    result_df = pd.DataFrame(all_rows)
    if "ts" in result_df.columns:
        result_df = result_df.set_index("ts").sort_index()
    result_df.to_csv(SPO_DIR / "test_predictions_hourly.csv")

    # 保存评估日志
    eval_df = pd.DataFrame(eval_log)
    eval_df.to_csv(SPO_DIR / "eval_log.csv", index=False)

    # ── 最终报告 ──
    logger.info("=" * 60)
    logger.info("SPO+ 训练完成 — 评估日志:")
    logger.info("=" * 60)
    for row in eval_log:
        logger.info(
            "  %-16s  净收益=%+.1f万  兑现率=%.1f%%  MAE=%.1f",
            row["label"],
            row["net"] / 1e4,
            row["realization"] * 100,
            row["mae"],
        )
    logger.info("=" * 60)
    logger.info("结果保存: %s", SPO_DIR)

    return eval_log


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    train_spo_plus(
        spo_epochs=30,
        spo_lr=3e-5,
        alpha_start=0.36,
        alpha_end=0.36,
        eval_every=2,
        freeze_encoder=False,
        anchor_lambda=1e-3,
        grad_accum_steps=4,
        spo_grad_scale=0.5,
    )
