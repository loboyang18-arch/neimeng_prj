"""
V12 Fine-tune — 在红井 hourly 节点电价上微调 Moirai-1.1-R-small

策略：
  - 加载预训练 Moirai-1.1-R-small (~14M params)
  - finetune_pattern: head_only / freeze_ffn / full
  - 滑窗 (context=720h, pred=24h) 构造训练对
  - PackedNLLLoss + AdamW
  - 验证：测试期前 14 天的 P50 MAE

输入：DWS hourly 重采样 target（默认 price_hongjing_220kv1m_nodal）
输出：finetuned MoiraiModule weights → output/experiments/v12.1-moirai-finetune-{pattern}-{tag}/
"""
from __future__ import annotations

import argparse
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("NM_V8_TARGET", "price_hongjing_220kv1m_nodal")

from src.config import OUTPUT_DIR  # noqa: E402

logger = logging.getLogger(__name__)
NODAL_COL = os.environ.get("NM_V8_TARGET", "price_hongjing_220kv1m_nodal")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _seed(s: int = 42):
    np.random.seed(s)
    random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def _load_hourly() -> pd.Series:
    path = Path(os.environ.get(
        "NM_DWS_CSV",
        str(OUTPUT_DIR / "dws_15min_features_ext_data4_hongjing.csv"),
    ))
    df = pd.read_csv(path, parse_dates=["ts"], index_col="ts").sort_index()
    s = df[NODAL_COL].astype(float).resample("1h").mean().dropna()
    logger.info("Hourly target %s: %d rows (%s ~ %s)",
                NODAL_COL, len(s), s.index.min(), s.index.max())
    return s


def _make_packed_sample(
    context: np.ndarray,
    target: np.ndarray,
    patch_size: int = 32,
    max_patch: int = 128,
) -> dict:
    """把 (context_720, target_24) 转成 MoiraiModule.forward 的 packed 输入。

    关键：MultiInSizeLinear 要求每个 patch 的最后一维 = max(patch_sizes) = 128。
    实际值填前 patch_size 个，其余补 0；patch_size 张量值仍写实际值（用于 mask 选择）。

    - 历史段 patch 数：ceil(720/32) = 23 个
    - 预测段 patch 数：ceil(24/32) = 1 个
    - 总 seq_len = 24 patches
    """
    ctx_len = len(context)
    pred_len = len(target)
    n_ctx_patches = (ctx_len + patch_size - 1) // patch_size  # 23
    n_pred_patches = (pred_len + patch_size - 1) // patch_size  # 1
    seq_len = n_ctx_patches + n_pred_patches  # 24

    full = np.zeros((seq_len, max_patch), dtype=np.float32)
    observed = np.zeros((seq_len, max_patch), dtype=bool)

    ctx_padded_len = n_ctx_patches * patch_size  # 736
    ctx_pad_left = ctx_padded_len - ctx_len  # 16
    ctx_full = np.zeros(ctx_padded_len, dtype=np.float32)
    ctx_obs = np.zeros(ctx_padded_len, dtype=bool)
    ctx_full[ctx_pad_left:] = context
    ctx_obs[ctx_pad_left:] = True
    for i in range(n_ctx_patches):
        full[i, :patch_size] = ctx_full[i * patch_size:(i + 1) * patch_size]
        observed[i, :patch_size] = ctx_obs[i * patch_size:(i + 1) * patch_size]

    pred_full = np.zeros(n_pred_patches * patch_size, dtype=np.float32)
    pred_obs = np.zeros(n_pred_patches * patch_size, dtype=bool)
    pred_full[:pred_len] = target
    pred_obs[:pred_len] = True
    for j in range(n_pred_patches):
        full[n_ctx_patches + j, :patch_size] = pred_full[j * patch_size:(j + 1) * patch_size]
        observed[n_ctx_patches + j, :patch_size] = pred_obs[j * patch_size:(j + 1) * patch_size]

    time_id = np.arange(seq_len, dtype=np.int64)
    variate_id = np.zeros(seq_len, dtype=np.int64)
    prediction_mask = np.zeros(seq_len, dtype=bool)
    prediction_mask[n_ctx_patches:] = True
    patch_size_arr = np.full(seq_len, patch_size, dtype=np.int64)
    sample_id = np.ones(seq_len, dtype=np.int64)

    return {
        "target": full,
        "observed_mask": observed,
        "time_id": time_id,
        "variate_id": variate_id,
        "prediction_mask": prediction_mask,
        "patch_size": patch_size_arr,
        "sample_id": sample_id,
    }


class SlidingWindowDataset(Dataset):
    """滑窗：每个样本起点 t，context=hourly[t:t+ctx], target=hourly[t+ctx:t+ctx+pred]

    通过 (start_ts, end_ts) 限定 **target 段**所在时间范围。
    """

    def __init__(
        self,
        hourly: np.ndarray,
        hourly_ts: pd.DatetimeIndex,
        start_ts: pd.Timestamp,
        end_ts: pd.Timestamp,
        ctx_len: int = 720,
        pred_len: int = 24,
        stride: int = 1,
        patch_size: int = 32,
    ):
        self.hourly = hourly
        self.ctx_len = ctx_len
        self.pred_len = pred_len
        self.patch_size = patch_size
        starts = []
        for i, ts in enumerate(hourly_ts):
            end_i = i + ctx_len + pred_len
            if end_i > len(hourly_ts):
                break
            pred_start_ts = hourly_ts[i + ctx_len]
            pred_end_ts = hourly_ts[end_i - 1]
            if pred_start_ts < start_ts:
                continue
            if pred_end_ts >= end_ts:
                break
            starts.append(i)
        self.starts = starts[::stride]
        logger.info("SlidingWindow: %d samples in [%s, %s)",
                    len(self.starts), start_ts, end_ts)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        i = self.starts[idx]
        ctx = self.hourly[i:i + self.ctx_len]
        tgt = self.hourly[i + self.ctx_len:i + self.ctx_len + self.pred_len]
        packed = _make_packed_sample(ctx, tgt, patch_size=self.patch_size)
        return {k: torch.from_numpy(v) for k, v in packed.items()}


def _collate(batch):
    out = {}
    for k in batch[0].keys():
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    # sample_id 在 packed 语义中跨 batch 唯一；这里 batch 间相同也无所谓（不在同一个 packed sequence 内）
    return out


def _freeze_for_pattern(module: torch.nn.Module, pattern: str):
    if pattern == "full":
        for p in module.parameters():
            p.requires_grad = True
    elif pattern == "head_only":
        for pn, p in module.named_parameters():
            p.requires_grad = "param_proj" in pn
    elif pattern == "freeze_ffn":
        for pn, p in module.named_parameters():
            p.requires_grad = "ffn" not in pn
    else:
        raise ValueError(f"Unknown pattern: {pattern}")
    n_train = sum(p.numel() for p in module.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in module.parameters())
    logger.info("Pattern=%s trainable=%d / %d (%.2f%%)",
                pattern, n_train, n_total, 100 * n_train / n_total)


@torch.no_grad()
def _eval_p50_mae(
    module,
    forecast_cls,
    hourly: np.ndarray,
    hourly_ts: pd.DatetimeIndex,
    eval_days,
    context_hours: int = 720,
    patch_size: int = 32,
    num_samples: int = 100,
    batch_size: int = 8,
) -> tuple[float, np.ndarray, np.ndarray]:
    """用与 zero-shot 一致的 MoiraiForecast.create_predictor 推理路径评估 MAE。"""
    from gluonts.dataset.common import ListDataset

    items, actuals, dates = [], [], []
    for d in eval_days:
        d0 = pd.Timestamp(d)
        ctx_start = d0 - pd.Timedelta(hours=context_hours)
        ctx_end = d0 - pd.Timedelta(hours=1)
        fut_start = d0
        fut_end = d0 + pd.Timedelta(hours=23)

        ctx_mask = (hourly_ts >= ctx_start) & (hourly_ts <= ctx_end)
        fut_mask = (hourly_ts >= fut_start) & (hourly_ts <= fut_end)
        ctx = hourly[ctx_mask]
        fut = hourly[fut_mask]
        if len(ctx) < context_hours - 24 or len(fut) != 24:
            continue
        items.append({"target": ctx.astype(np.float32),
                      "start": pd.Period(hourly_ts[ctx_mask][0], freq="h")})
        actuals.append(fut.astype(np.float32))
        dates.append(d)

    if not items:
        return float("nan"), np.array([]), np.array([])
    ds = ListDataset(items, freq="h", one_dim_target=True)

    model = forecast_cls(
        module=module, prediction_length=24, context_length=context_hours,
        patch_size=patch_size, num_samples=num_samples, target_dim=1,
        feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0,
    )
    if torch.cuda.is_available():
        model = model.to("cuda")
    predictor = model.create_predictor(batch_size=batch_size)
    forecasts = list(predictor.predict(ds))

    p50 = np.stack([fc.quantile(0.5) for fc in forecasts], axis=0)
    act = np.stack(actuals, axis=0)
    mae = float(np.mean(np.abs(p50 - act)))
    return mae, p50, act


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Salesforce/moirai-1.1-R-small")
    p.add_argument("--pattern", default="head_only",
                   choices=["head_only", "freeze_ffn", "full"])
    p.add_argument("--test-start", default="2026-01-25")
    p.add_argument("--test-end", default="2026-05-09")
    p.add_argument("--val-days", type=int, default=14)
    p.add_argument("--context-hours", type=int, default=720)
    p.add_argument("--pred-hours", type=int, default=24)
    p.add_argument("--patch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--stride", type=int, default=24,
                   help="滑窗步长（小时）：1=每小时一窗，24=每日一窗")
    p.add_argument("--loss", default="nll", choices=["nll", "mae", "mse"],
                   help="训练损失：nll=PackedNLLLoss(原版)；mae/mse=用 distr.mean 做 point loss")
    p.add_argument("--train-start", default=None,
                   help="限定训练数据起点（YYYY-MM-DD），用于方案 C 的近端窗口实验")
    p.add_argument("--out-tag", default="hongjing")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    _seed(args.seed)

    out_dir = OUTPUT_DIR / "experiments" / f"v12.1-moirai-finetune-{args.pattern}-{args.out_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("=" * 60)
    logger.info("V12.1 Fine-tune: %s (%s)", args.model, args.pattern)
    logger.info("  device: %s", DEVICE)
    logger.info("  out:    %s", out_dir)
    logger.info("=" * 60)

    s = _load_hourly()
    hourly = s.values.astype(np.float32)
    hourly_ts = s.index

    test_start_ts = pd.Timestamp(args.test_start)
    test_end_ts = pd.Timestamp(args.test_end) + pd.Timedelta(days=1)
    val_start_ts = test_start_ts - pd.Timedelta(days=args.val_days)

    logger.info("Splits: train < %s | val [%s, %s) | test [%s, %s)",
                val_start_ts, val_start_ts, test_start_ts,
                test_start_ts, test_end_ts)

    default_train_start = hourly_ts[0] + pd.Timedelta(hours=args.context_hours)
    if args.train_start is not None:
        train_start_ts = max(pd.Timestamp(args.train_start), default_train_start)
    else:
        train_start_ts = default_train_start
    logger.info("Train target window: [%s, %s)", train_start_ts, val_start_ts)
    train_ds = SlidingWindowDataset(
        hourly, hourly_ts,
        start_ts=train_start_ts,
        end_ts=val_start_ts,
        ctx_len=args.context_hours, pred_len=args.pred_hours,
        stride=args.stride, patch_size=args.patch_size,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, collate_fn=_collate, drop_last=True,
    )

    val_days_list = sorted(set(
        pd.Timestamp(d).date()
        for d in pd.date_range(val_start_ts, test_start_ts - pd.Timedelta(days=1), freq="D")
    ))
    test_days_list = sorted(set(
        pd.Timestamp(d).date()
        for d in pd.date_range(test_start_ts, test_end_ts - pd.Timedelta(days=1), freq="D")
    ))

    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
    from uni2ts.loss.packed import PackedNLLLoss

    logger.info("加载预训练 %s …", args.model)
    module = MoiraiModule.from_pretrained(args.model)
    module = module.to(DEVICE)

    _freeze_for_pattern(module, args.pattern)
    nll_func = PackedNLLLoss()
    logger.info("Loss = %s", args.loss)

    optim = torch.optim.AdamW(
        [p for p in module.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )

    logger.info("Zero-shot baseline 评估 (val) …")
    val_mae_zs, _, _ = _eval_p50_mae(
        module, MoiraiForecast, hourly, hourly_ts, val_days_list,
        context_hours=args.context_hours, patch_size=args.patch_size,
    )
    logger.info("  baseline val MAE = %.3f", val_mae_zs)

    best_val_mae = float("inf")
    best_state = None
    for ep in range(1, args.epochs + 1):
        module.train()
        t0 = time.time()
        losses = []
        for batch in train_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            distr = module(
                target=batch["target"],
                observed_mask=batch["observed_mask"],
                sample_id=batch["sample_id"],
                time_id=batch["time_id"],
                variate_id=batch["variate_id"],
                prediction_mask=batch["prediction_mask"],
                patch_size=batch["patch_size"],
            )
            if args.loss == "nll":
                loss = nll_func(
                    pred=distr,
                    target=batch["target"],
                    prediction_mask=batch["prediction_mask"],
                    observed_mask=batch["observed_mask"],
                    sample_id=batch["sample_id"],
                    variate_id=batch["variate_id"],
                )
            else:
                # 用 distr.mean 做点估计 loss
                # distr.mean shape: [B, seq_len, max_patch] (= target shape)
                # mask: 只在 prediction patch 且 observed 的位置算
                pred_mean = distr.mean
                target = batch["target"]
                mask = batch["prediction_mask"].unsqueeze(-1) & batch["observed_mask"]
                if args.loss == "mae":
                    diff = (pred_mean - target).abs()
                else:  # mse
                    diff = (pred_mean - target).pow(2)
                denom = mask.sum().clamp(min=1).to(diff.dtype)
                loss = (diff * mask).sum() / denom
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in module.parameters() if p.requires_grad], 1.0,
            )
            optim.step()
            losses.append(float(loss.item()))
        train_loss = float(np.mean(losses)) if losses else float("nan")

        module.eval()
        val_mae, _, _ = _eval_p50_mae(
            module, MoiraiForecast, hourly, hourly_ts, val_days_list,
            context_hours=args.context_hours, patch_size=args.patch_size,
        )

        flag = ""
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
            flag = " *"
        logger.info("epoch %02d | train %s=%.4f | val MAE=%.3f%s | %.1fs",
                    ep, args.loss.upper(), train_loss, val_mae, flag,
                    time.time() - t0)

    logger.info("最佳 val MAE = %.3f (baseline zero-shot %.3f, 提升 %.1f%%)",
                best_val_mae, val_mae_zs,
                (val_mae_zs - best_val_mae) / val_mae_zs * 100)

    if best_state is not None:
        module.load_state_dict(best_state)

    logger.info("最佳 checkpoint 在 test 上评估 …")
    test_mae, p50, actual = _eval_p50_mae(
        module, MoiraiForecast, hourly, hourly_ts, test_days_list,
        context_hours=args.context_hours, patch_size=args.patch_size,
    )

    logger.info("=" * 60)
    logger.info("TEST 集结果:")
    logger.info("  baseline (zero-shot)   val MAE = %.3f", val_mae_zs)
    logger.info("  finetuned (%s)  val MAE = %.3f", args.pattern, best_val_mae)
    logger.info("  finetuned             test MAE = %.3f", test_mae)
    logger.info("=" * 60)

    torch.save(module.state_dict(), out_dir / "module_finetuned.pt")
    np.savez(out_dir / "test_eval.npz", p50=p50, actual=actual,
             test_mae=test_mae, baseline_val_mae=val_mae_zs,
             best_val_mae=best_val_mae)
    logger.info("Saved -> %s", out_dir)


if __name__ == "__main__":
    main()
