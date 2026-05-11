"""
V11 — 时序基础模型迁移 (Chronos-Bolt zero-shot / fine-tune)

把内蒙古每日 24h 节点电价序列拼成长 hourly 时间序列，
对每个测试日 D 用其前 N 小时作为 context，让 chronos-bolt 预测 D 的 24 小时价格。

输入：纯 univariate 历史电价（基础模型不直接用 covariates）。
输出：24 小时分位数 (P10/P30/P50/P70/P90)，与 V10-Quantile 完全兼容
       → 可直接喂入 robust MILP / strategy_milp_15min。
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("NM_V8_TARGET", "price_sudun_500kv1m_nodal")
os.environ.setdefault("NM_MIN_FEATURE_DATE", "2024-12-14")

from src.config import OUTPUT_DIR  # noqa: E402
from src.fill_sudun_dws_gaps import fill_sudun_price_columns  # noqa: E402

logger = logging.getLogger(__name__)

NODAL_COL = os.environ.get("NM_V8_TARGET", "price_sudun_500kv1m_nodal")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_MODEL = "amazon/chronos-bolt-small"
DEFAULT_QUANTILES = (0.1, 0.3, 0.5, 0.7, 0.9)


def _load_hourly_price_series() -> pd.Series:
    """加载内蒙古节点电价（hourly，4 个 15min 槽求均值），返回连续小时索引的 Series。"""
    path = OUTPUT_DIR / "dws_15min_features.csv"
    df = pd.read_csv(path, parse_dates=["ts"], index_col="ts").sort_index()
    df = fill_sudun_price_columns(df)
    s_15m = df[NODAL_COL].astype(float)

    # 重采样到 hourly（mean of 4 slots）
    s_h = s_15m.resample("1h").mean()
    s_h = s_h.dropna()
    logger.info("Hourly price series: %d hours (%s ~ %s)",
                len(s_h), s_h.index.min(), s_h.index.max())
    return s_h


def _build_test_inputs(hourly: pd.Series, test_days: List, context_hours: int):
    """对每个 test day D，提取前 context_hours 小时作为 context。

    Returns:
        dates: list of date
        contexts: list of (context_hours,) np arrays
        actuals: dict {date: (24,) np array}
    """
    contexts = []
    dates = []
    actuals = {}
    for d in test_days:
        d_start = pd.Timestamp(d)
        ctx_start = d_start - pd.Timedelta(hours=context_hours)
        ctx_end = d_start - pd.Timedelta(hours=1)
        ctx = hourly.loc[ctx_start:ctx_end]
        if len(ctx) < context_hours - 24:
            logger.warning("[%s] context 太短 (%d/%d)，跳过", d, len(ctx), context_hours)
            continue
        # 当日真实 24h
        d_end = d_start + pd.Timedelta(hours=23)
        a = hourly.loc[d_start:d_end]
        if len(a) != 24:
            logger.warning("[%s] actual 不足 24h (%d)，跳过", d, len(a))
            continue
        contexts.append(ctx.values.astype(np.float32))
        actuals[d] = a.values.astype(np.float32)
        dates.append(d)
    logger.info("有效 test 天数: %d / %d (context_hours=%d)",
                len(dates), len(test_days), context_hours)
    return dates, contexts, actuals


def _run_inference(
    model_id: str,
    contexts: List[np.ndarray],
    quantile_levels: List[float],
    prediction_length: int = 24,
    batch_size: int = 8,
):
    """批量调用 chronos-bolt 推理，返回 (N, T, Q)。"""
    from chronos import BaseChronosPipeline

    logger.info("加载基础模型: %s (device=%s)", model_id, DEVICE)
    pipe = BaseChronosPipeline.from_pretrained(
        model_id, device_map=str(DEVICE), dtype=torch.float32,
    )

    all_q, all_mean = [], []
    for i in range(0, len(contexts), batch_size):
        batch = contexts[i:i + batch_size]
        # chronos 接受 List[Tensor] 当各 context 长度不一致
        inputs = [torch.from_numpy(c.copy()) for c in batch]
        q, m = pipe.predict_quantiles(
            inputs=inputs,
            prediction_length=prediction_length,
            quantile_levels=list(quantile_levels),
        )
        all_q.append(q.cpu().numpy())   # (B, T, Q)
        all_mean.append(m.cpu().numpy())  # (B, T)
        if (i // batch_size) % 5 == 0:
            logger.info("  推理 %d/%d", i + len(batch), len(contexts))

    quantiles = np.concatenate(all_q, axis=0)
    means = np.concatenate(all_mean, axis=0)
    return quantiles, means


def run(
    model_id: str = DEFAULT_MODEL,
    test_start: str = "2026-01-27",
    test_end: str = "2026-04-17",
    context_hours: int = 720,         # 30 天历史
    quantile_levels: tuple = DEFAULT_QUANTILES,
    out_tag: str = "",
):
    out_dir = OUTPUT_DIR / "experiments" / (
        f"v11.0-foundation{out_tag}" if out_tag else "v11.0-foundation"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("V11 Foundation Model: %s", model_id)
    logger.info("  test_start=%s test_end=%s context_hours=%d",
                test_start, test_end, context_hours)
    logger.info("  quantile_levels=%s", quantile_levels)
    logger.info("=" * 60)

    # ── 1. 数据 ──
    hourly = _load_hourly_price_series()

    test_dt = pd.Timestamp(test_start).date()
    test_end_dt = pd.Timestamp(test_end).date()
    all_days = sorted(set(hourly.index.normalize().date.tolist()))
    test_days = [d for d in all_days if test_dt <= d <= test_end_dt]
    logger.info("test_days: %d", len(test_days))

    dates, contexts, actuals = _build_test_inputs(hourly, test_days, context_hours)

    # ── 2. 推理 ──
    quantiles, means = _run_inference(
        model_id, contexts, list(quantile_levels), prediction_length=24,
    )
    logger.info("推理完成: quantiles=%s mean=%s", quantiles.shape, means.shape)

    # ── 3. 保存预测（兼容 V10/V10-Quantile 评估管道）──
    n_q = len(quantile_levels)
    p50_idx = n_q // 2

    rows_long = []
    rows_p50 = []
    for di, d in enumerate(dates):
        q_24 = np.sort(quantiles[di], axis=-1)        # (24, Q) 强制单调
        a_24 = actuals[d]
        for h in range(24):
            row = {
                "ts": pd.Timestamp(d) + pd.Timedelta(hours=h),
                "actual": float(a_24[h]),
            }
            for q_i, q_lvl in enumerate(quantile_levels):
                row[f"p{int(q_lvl * 100):02d}"] = float(q_24[h, q_i])
            rows_long.append(row)

            rows_p50.append({
                "ts": row["ts"],
                "actual": float(a_24[h]),
                "pred": float(q_24[h, p50_idx]),
            })

    pred_q_df = pd.DataFrame(rows_long).sort_values("ts").reset_index(drop=True)
    pred_p50_df = pd.DataFrame(rows_p50).set_index("ts").sort_index()

    pred_q_df.to_csv(out_dir / "test_predictions_quantile.csv", index=False)
    pred_p50_df.to_csv(out_dir / "test_predictions_hourly.csv")
    np.save(out_dir / "quantile_levels.npy", np.array(quantile_levels))

    # ── 4. 简单评估指标 ──
    flat_actual = pred_p50_df["actual"].values
    flat_p50 = pred_p50_df["pred"].values
    mask = ~(np.isnan(flat_actual) | np.isnan(flat_p50))
    fa, fp = flat_actual[mask], flat_p50[mask]
    mae = float(np.mean(np.abs(fa - fp)))
    rmse = float(np.sqrt(np.mean((fa - fp) ** 2)))
    bias = float(np.mean(fp - fa))

    # 覆盖率（P10–P90）
    flat_q_mat = pred_q_df.set_index("ts")[
        [f"p{int(q * 100):02d}" for q in quantile_levels]
    ].values
    flat_a = pred_q_df["actual"].values
    cov80 = float(np.mean(
        (flat_a >= flat_q_mat[:, 0]) & (flat_a <= flat_q_mat[:, -1])
    ))
    width = float(np.mean(flat_q_mat[:, -1] - flat_q_mat[:, 0]))

    logger.info("=" * 60)
    logger.info("V11 Foundation 评估")
    logger.info("  MAE (P50):       %.2f", mae)
    logger.info("  RMSE (P50):      %.2f", rmse)
    logger.info("  Bias (P50):      %.2f", bias)
    logger.info("  Coverage 80%%:    %.3f (target 0.80)", cov80)
    logger.info("  Interval Width:  %.1f", width)
    logger.info("  保存: %s", out_dir)
    logger.info("=" * 60)

    metrics = {
        "model_id": model_id, "context_hours": context_hours,
        "n_test_days": len(dates),
        "mae": mae, "rmse": rmse, "bias": bias,
        "coverage_80": cov80, "interval_width": width,
    }
    pd.Series(metrics).to_csv(out_dir / "metrics.csv")

    return metrics


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="HuggingFace model id (chronos-bolt-tiny/mini/small/base)")
    p.add_argument("--context-hours", type=int, default=720,
                   help="历史 context 长度（小时），chronos-bolt 上限 ~2048")
    p.add_argument("--out-tag", default="",
                   help="输出目录后缀，例：'-base' → v11.0-foundation-base")
    args = p.parse_args()
    run(model_id=args.model, context_hours=args.context_hours, out_tag=args.out_tag)
