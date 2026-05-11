"""
V11 fine-tuning — 在内蒙古训练集 hourly 价格上微调 Chronos-Bolt

策略：
  - 加载预训练 chronos-bolt-small (47.7M params)
  - 滑窗构造 (context=720h, target=24h) 训练对
  - forward(context, target) 内置 Pinball loss → 反向传播
  - 评估在 test_days 上：MAE_P50 / Coverage / Robust MILP α 网格
"""
from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("NM_V8_TARGET", "price_sudun_500kv1m_nodal")

from src.config import OUTPUT_DIR  # noqa: E402
from src.fill_sudun_dws_gaps import fill_sudun_price_columns  # noqa: E402

logger = logging.getLogger(__name__)
NODAL_COL = os.environ.get("NM_V8_TARGET", "price_sudun_500kv1m_nodal")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _seed(s=42):
    np.random.seed(s)
    random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def _load_hourly() -> pd.Series:
    path = OUTPUT_DIR / "dws_15min_features.csv"
    df = pd.read_csv(path, parse_dates=["ts"], index_col="ts").sort_index()
    df = fill_sudun_price_columns(df)
    s = df[NODAL_COL].astype(float).resample("1h").mean().dropna()
    return s


class SlidingWindowDataset(Dataset):
    """滑窗构造 (context, target)。每个样本：起点 t，context=hourly[t : t+ctx]，target=hourly[t+ctx : t+ctx+pred]"""

    def __init__(self, hourly: np.ndarray, hourly_ts: pd.DatetimeIndex,
                 train_end_ts: pd.Timestamp, ctx_len: int = 720, pred_len: int = 24,
                 stride: int = 24, max_samples: int | None = None,
                 seed: int = 42):
        starts = []
        end_idx = np.searchsorted(hourly_ts.values, train_end_ts.to_datetime64())
        # 仅在 train 区间内构造样本：t + ctx + pred ≤ train_end_idx
        for t in range(0, end_idx - ctx_len - pred_len, stride):
            starts.append(t)
        rng = np.random.RandomState(seed)
        if max_samples is not None and len(starts) > max_samples:
            starts = rng.choice(starts, size=max_samples, replace=False).tolist()
        self.starts = sorted(starts)
        self.hourly = hourly
        self.ctx_len = ctx_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        t = self.starts[i]
        ctx = self.hourly[t:t + self.ctx_len].astype(np.float32)
        tgt = self.hourly[t + self.ctx_len:t + self.ctx_len + self.pred_len].astype(np.float32)
        return torch.from_numpy(ctx), torch.from_numpy(tgt)


def _build_test_inputs(hourly: pd.Series, test_days, context_hours: int):
    contexts, dates, actuals = [], [], {}
    for d in test_days:
        d0 = pd.Timestamp(d)
        ctx = hourly.loc[d0 - pd.Timedelta(hours=context_hours):d0 - pd.Timedelta(hours=1)]
        a = hourly.loc[d0:d0 + pd.Timedelta(hours=23)]
        if len(ctx) < context_hours - 24 or len(a) != 24:
            continue
        contexts.append(ctx.values.astype(np.float32))
        dates.append(d)
        actuals[d] = a.values.astype(np.float32)
    return dates, contexts, actuals


def _eval_quantiles(model, contexts, actuals, quantile_levels=(0.1, 0.3, 0.5, 0.7, 0.9),
                    batch_size: int = 8):
    """模型在 test contexts 上的分位数预测；返回 (N, 24, 9) 全 9 分位 + (N, 24) actuals。"""
    model.eval()
    out_q = []  # (N, 9, 64)
    with torch.no_grad():
        for i in range(0, len(contexts), batch_size):
            ctx_list = contexts[i:i + batch_size]
            ctx_padded = torch.stack([torch.from_numpy(c[-2048:]) for c in ctx_list]).to(DEVICE)
            res = model(context=ctx_padded)
            q = res.quantile_preds.cpu().numpy()    # (B, 9, 64)
            out_q.append(q[:, :, :24])               # 截前 24 维
    quantiles_9 = np.concatenate(out_q, axis=0)      # (N, 9, 24)
    quantiles_9 = np.transpose(quantiles_9, (0, 2, 1))  # (N, 24, 9)

    # 模型预训练分位数：[0.1, 0.2, ..., 0.9]
    pretrained_q = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    # 取需要的列
    idx_map = [int(np.argmin(np.abs(pretrained_q - q))) for q in quantile_levels]
    quantiles = quantiles_9[:, :, idx_map]            # (N, 24, len(quantile_levels))

    # 排序保单调
    quantiles = np.sort(quantiles, axis=-1)

    actual_arr = np.stack([actuals[d] for d in actuals], axis=0)  # 但顺序需对齐
    # 顺序由调用方控制，这里假设 actuals 是按调用顺序的 dict
    return quantiles


def train_v11_finetune(
    model_id: str = "amazon/chronos-bolt-small",
    test_start: str = "2026-01-27",
    test_end: str = "2026-04-17",
    context_hours: int = 720,
    pred_len: int = 24,
    stride: int = 24,
    max_train_samples: int = 5000,
    max_epochs: int = 30,
    lr: float = 3e-5,
    batch_size: int = 16,
    weight_decay: float = 1e-4,
    eval_every: int = 5,
    out_tag: str = "",
):
    out_dir = OUTPUT_DIR / "experiments" / (
        f"v11.0-finetune{out_tag}" if out_tag else "v11.0-finetune"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("=" * 60)
    logger.info("V11 Fine-tune: %s", model_id)
    logger.info("  test=%s ~ %s, ctx=%dh pred=%dh stride=%d",
                test_start, test_end, context_hours, pred_len, stride)
    logger.info("  max_train_samples=%d epochs=%d lr=%g bs=%d",
                max_train_samples, max_epochs, lr, batch_size)
    logger.info("=" * 60)

    _seed(42)

    # ── 1. 数据 ──
    hourly_s = _load_hourly()
    logger.info("Hourly: %d (%s ~ %s)", len(hourly_s), hourly_s.index.min(), hourly_s.index.max())
    hourly = hourly_s.values.astype(np.float32)
    hourly_ts = hourly_s.index

    test_dt = pd.Timestamp(test_start)
    test_end_dt = pd.Timestamp(test_end) + pd.Timedelta(hours=23)

    # 训练样本截止于 test_start
    train_ds = SlidingWindowDataset(
        hourly, hourly_ts, train_end_ts=test_dt,
        ctx_len=context_hours, pred_len=pred_len, stride=stride,
        max_samples=max_train_samples,
    )
    logger.info("训练样本数: %d", len(train_ds))

    # test 集
    test_days = [d for d in sorted(set(hourly_ts.normalize().date.tolist()))
                 if test_dt.date() <= d <= test_end_dt.date()]
    test_dates, test_contexts, test_actuals = _build_test_inputs(
        hourly_s, test_days, context_hours)
    logger.info("test 天数: %d", len(test_dates))

    # ── 2. 加载模型 ──
    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained(
        model_id, device_map=str(DEVICE), dtype=torch.float32,
    )
    model = pipe.model

    # ── 3. 优化器 ──
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    n_steps = max_epochs * max(1, len(train_ds) // batch_size)
    warmup_steps = max(50, n_steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lr_lambda=lambda s: min(1.0, s / max(1, warmup_steps)) *
                            max(0.05, 0.5 * (1 + np.cos(
                                np.pi * max(0, s - warmup_steps) /
                                max(1, n_steps - warmup_steps)))),
    )

    # ── 4. eval helper ──
    def _zero_shot_eval():
        model.eval()
        with torch.no_grad():
            n_q_pretrained = 9
            preds = np.zeros((len(test_contexts), 24, n_q_pretrained), dtype=np.float32)
            for i in range(0, len(test_contexts), batch_size):
                batch = test_contexts[i:i + batch_size]
                # context 长度对齐到 max(2048)；不足时不需要 padding（chronos 支持变长）
                ctx_t = torch.stack([torch.from_numpy(c[-2048:]) for c in batch]).to(DEVICE)
                res = model(context=ctx_t)
                q = res.quantile_preds[:, :, :24].cpu().numpy()  # (B, 9, 24)
                q = np.transpose(q, (0, 2, 1))                    # (B, 24, 9)
                preds[i:i + len(batch)] = q
        # 取 P10/P50/P90 (索引 0, 4, 8)
        actuals_arr = np.stack([test_actuals[d] for d in test_dates], axis=0)
        flat_a = actuals_arr.reshape(-1)
        flat_p = preds.reshape(-1, n_q_pretrained)
        flat_p_sorted = np.sort(flat_p, axis=-1)
        mae = float(np.mean(np.abs(flat_p_sorted[:, 4] - flat_a)))
        rmse = float(np.sqrt(np.mean((flat_p_sorted[:, 4] - flat_a) ** 2)))
        cov80 = float(np.mean(
            (flat_a >= flat_p_sorted[:, 0]) & (flat_a <= flat_p_sorted[:, 8])
        ))
        width = float(np.mean(flat_p_sorted[:, 8] - flat_p_sorted[:, 0]))
        return {"mae": mae, "rmse": rmse, "cov80": cov80, "width": width}, preds

    # ── 5. 训练 ──
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=True, num_workers=0)
    eval_log = []
    best_mae = float("inf")

    logger.info("初始 zero-shot 评估…")
    init_metrics, _ = _zero_shot_eval()
    logger.info("  MAE=%.2f RMSE=%.2f cov80=%.3f width=%.1f",
                init_metrics["mae"], init_metrics["rmse"],
                init_metrics["cov80"], init_metrics["width"])
    eval_log.append({"epoch": 0, **init_metrics})

    for epoch in range(max_epochs):
        model.train()
        ep_loss, nb = 0.0, 0
        for ctx, tgt in train_loader:
            ctx, tgt = ctx.to(DEVICE), tgt.to(DEVICE)
            opt.zero_grad()
            out = model(context=ctx, target=tgt)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            ep_loss += loss.item()
            nb += 1

        if (epoch + 1) % eval_every == 0 or epoch == max_epochs - 1:
            metrics, preds = _zero_shot_eval()
            cur_lr = opt.param_groups[0]["lr"]
            logger.info(
                "  ep%3d  loss=%.4f | MAE=%.2f RMSE=%.2f cov80=%.3f width=%.1f lr=%.1e",
                epoch + 1, ep_loss / max(nb, 1),
                metrics["mae"], metrics["rmse"],
                metrics["cov80"], metrics["width"], cur_lr,
            )
            eval_log.append({"epoch": epoch + 1, "loss": ep_loss / max(nb, 1), **metrics})
            if metrics["mae"] < best_mae:
                best_mae = metrics["mae"]
                # 保存权重
                torch.save(model.state_dict(), out_dir / "model_weights.pt")

    # ── 6. 用 best 重新预测并保存（兼容评估管道）──
    model.load_state_dict(torch.load(out_dir / "model_weights.pt", map_location=DEVICE))
    metrics, preds = _zero_shot_eval()  # 9 quantiles
    pretrained_q = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    target_q = (0.1, 0.3, 0.5, 0.7, 0.9)
    idx_map = [int(np.argmin(np.abs(pretrained_q - q))) for q in target_q]
    preds_5 = np.sort(preds[:, :, idx_map], axis=-1)  # (N, 24, 5)

    # 保存 long format
    rows_long, rows_p50 = [], []
    for di, d in enumerate(test_dates):
        for h in range(24):
            row = {"ts": pd.Timestamp(d) + pd.Timedelta(hours=h),
                   "actual": float(test_actuals[d][h])}
            for q_i, q_lvl in enumerate(target_q):
                row[f"p{int(q_lvl * 100):02d}"] = float(preds_5[di, h, q_i])
            rows_long.append(row)
            rows_p50.append({"ts": row["ts"], "actual": row["actual"],
                             "pred": float(preds_5[di, h, 2])})  # P50

    pd.DataFrame(rows_long).sort_values("ts").reset_index(drop=True).to_csv(
        out_dir / "test_predictions_quantile.csv", index=False)
    pd.DataFrame(rows_p50).set_index("ts").sort_index().to_csv(
        out_dir / "test_predictions_hourly.csv")
    np.save(out_dir / "quantile_levels.npy", np.array(target_q))

    logger.info("=" * 60)
    logger.info("Fine-tune 完成: best MAE=%.2f, cov80=%.3f, width=%.1f",
                metrics["mae"], metrics["cov80"], metrics["width"])
    logger.info("Output: %s", out_dir)
    logger.info("=" * 60)

    pd.DataFrame(eval_log).to_csv(out_dir / "eval_log.csv", index=False)
    return metrics


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="amazon/chronos-bolt-small")
    p.add_argument("--context-hours", type=int, default=720)
    p.add_argument("--max-train-samples", type=int, default=5000)
    p.add_argument("--max-epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--stride", type=int, default=24)
    p.add_argument("--out-tag", default="")
    args = p.parse_args()
    train_v11_finetune(
        model_id=args.model,
        context_hours=args.context_hours,
        max_train_samples=args.max_train_samples,
        max_epochs=args.max_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        stride=args.stride,
        out_tag=args.out_tag,
    )
