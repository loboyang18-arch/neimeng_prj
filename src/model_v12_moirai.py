"""
V12 — Moirai-1.1-R covariate-aware 时序基础模型

支持两个模式：
  1. univariate：仅 hourly 节点电价 → 与 Chronos zero-shot 对比
  2. covariate-aware：target=price_sudun，feat_dynamic_real=[wind/load/reserve_neg/preclear]
                     四个 D-day 已知的预测特征作 future covariate

输出每日 24h 的 5 个分位数 (P10/P30/P50/P70/P90)，与 V10/V11 评估管道兼容。
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

DEFAULT_COVARIATES = (
    "wind_forecast",
    "load_forecast",
    "reserve_neg_capacity",
    "price_dayahead_preclear_energy",
)
DEFAULT_QUANTILES = (0.1, 0.3, 0.5, 0.7, 0.9)


def _load_hourly_with_covariates(covariates: List[str]) -> pd.DataFrame:
    """加载 hourly target + covariates。"""
    path = OUTPUT_DIR / "dws_15min_features.csv"
    df15 = pd.read_csv(path, parse_dates=["ts"], index_col="ts").sort_index()
    df15 = fill_sudun_price_columns(df15)
    cols = [NODAL_COL] + list(covariates)
    df15 = df15[cols].astype(float)
    df_h = df15.resample("1h").mean().dropna()
    df_h = df_h.rename(columns={NODAL_COL: "target"})
    logger.info("Hourly: %d rows, cols=%s", len(df_h), df_h.columns.tolist())
    return df_h


def _build_test_inputs(
    df_h: pd.DataFrame, test_days: List, context_hours: int, covariates: List[str],
):
    """对每个 D：构造 history (target+cov 全部) + future covariates 24h。"""
    samples = []
    for d in test_days:
        d0 = pd.Timestamp(d)
        ctx_start = d0 - pd.Timedelta(hours=context_hours)
        ctx_end = d0 - pd.Timedelta(hours=1)
        fut_start = d0
        fut_end = d0 + pd.Timedelta(hours=23)

        history = df_h.loc[ctx_start:ctx_end]
        future = df_h.loc[fut_start:fut_end]
        if len(history) < context_hours - 24 or len(future) != 24:
            continue
        actual_24 = future["target"].values.astype(np.float32)
        # 历史 target
        target_hist = history["target"].values.astype(np.float32)
        # 历史 + 未来 covariate（dynamic real）
        cov_full = pd.concat(
            [history[covariates], future[covariates]], axis=0
        ).values.astype(np.float32).T  # (num_cov, T_total)

        samples.append({
            "date": d,
            "start": history.index[0],
            "target_hist": target_hist,
            "cov_full": cov_full,    # (num_cov, ctx_len + 24)
            "actual": actual_24,
        })
    logger.info("有效 test 天数: %d / %d (context_hours=%d)",
                len(samples), len(test_days), context_hours)
    return samples


def _make_gluonts_dataset(samples: List[dict], use_covariates: bool):
    """把 samples 列表转成 GluonTS ListDataset 风格 dict 列表。

    每条记录：
      target: (T_history,) — 历史 target
      start:  起始时间戳
      feat_dynamic_real: (num_cov, T_history + T_future) — 全段 covariate
    """
    from gluonts.dataset.common import ListDataset
    items = []
    for s in samples:
        item = {
            "target": s["target_hist"],
            "start": pd.Period(s["start"], freq="h"),
        }
        if use_covariates:
            item["feat_dynamic_real"] = s["cov_full"]
        items.append(item)
    freq = "h"
    one_dim_target = True
    return ListDataset(items, freq=freq, one_dim_target=one_dim_target)


def run(
    model_id: str = "Salesforce/moirai-1.1-R-small",
    test_start: str = "2026-01-27",
    test_end: str = "2026-04-17",
    context_hours: int = 720,
    use_covariates: bool = False,
    covariates: tuple = DEFAULT_COVARIATES,
    patch_size: int = 32,
    num_samples: int = 100,
    quantile_levels: tuple = DEFAULT_QUANTILES,
    out_tag: str = "",
    batch_size: int = 8,
):
    suffix = "-cov" if use_covariates else "-uni"
    out_dir = OUTPUT_DIR / "experiments" / f"v12.0-moirai{suffix}{out_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("=" * 60)
    logger.info("V12 Moirai: %s", model_id)
    logger.info("  test=%s ~ %s, context=%dh, covariates=%s",
                test_start, test_end, context_hours,
                covariates if use_covariates else "(univariate)")
    logger.info("  patch_size=%d num_samples=%d", patch_size, num_samples)
    logger.info("=" * 60)

    # ── 1. 数据 ──
    df_h = _load_hourly_with_covariates(list(covariates) if use_covariates else [])

    test_dt = pd.Timestamp(test_start).date()
    test_end_dt = pd.Timestamp(test_end).date()
    all_days = sorted(set(df_h.index.normalize().date.tolist()))
    test_days = [d for d in all_days if test_dt <= d <= test_end_dt]

    samples = _build_test_inputs(
        df_h, test_days,
        context_hours=context_hours,
        covariates=list(covariates) if use_covariates else [],
    )
    if not samples:
        raise RuntimeError("无有效 test 样本")
    actual_arr = np.stack([s["actual"] for s in samples], axis=0)  # (N, 24)

    # ── 2. 加载 Moirai ──
    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
    logger.info("加载 %s …", model_id)
    module = MoiraiModule.from_pretrained(model_id)
    n_params = sum(p.numel() for p in module.parameters())
    logger.info("  params: %d", n_params)

    feat_dim = len(covariates) if use_covariates else 0
    model = MoiraiForecast(
        module=module,
        prediction_length=24,
        context_length=context_hours,
        patch_size=patch_size,
        num_samples=num_samples,
        target_dim=1,
        feat_dynamic_real_dim=feat_dim,
        past_feat_dynamic_real_dim=0,
    )
    if torch.cuda.is_available():
        model = model.to("cuda")

    # ── 3. 推理 ──
    ds = _make_gluonts_dataset(samples, use_covariates=use_covariates)
    predictor = model.create_predictor(batch_size=batch_size)

    logger.info("开始推理 %d 天 …", len(samples))
    forecasts = list(predictor.predict(ds))
    logger.info("推理完成: %d forecasts", len(forecasts))

    # ── 4. 提取分位数 ──
    quantiles_arr = np.zeros((len(samples), 24, len(quantile_levels)), dtype=np.float32)
    means_arr = np.zeros((len(samples), 24), dtype=np.float32)
    for i, fc in enumerate(forecasts):
        for q_i, q in enumerate(quantile_levels):
            quantiles_arr[i, :, q_i] = fc.quantile(q)
        means_arr[i] = fc.mean
    quantiles_arr = np.sort(quantiles_arr, axis=-1)

    # ── 5. 保存 ──
    rows_long, rows_p50 = [], []
    p50_idx = len(quantile_levels) // 2
    for i, s in enumerate(samples):
        d = s["date"]
        for h in range(24):
            row = {"ts": pd.Timestamp(d) + pd.Timedelta(hours=h),
                   "actual": float(actual_arr[i, h])}
            for q_i, q in enumerate(quantile_levels):
                row[f"p{int(q * 100):02d}"] = float(quantiles_arr[i, h, q_i])
            rows_long.append(row)
            rows_p50.append({
                "ts": row["ts"], "actual": row["actual"],
                "pred": float(quantiles_arr[i, h, p50_idx]),
            })
    pd.DataFrame(rows_long).sort_values("ts").reset_index(drop=True).to_csv(
        out_dir / "test_predictions_quantile.csv", index=False)
    pd.DataFrame(rows_p50).set_index("ts").sort_index().to_csv(
        out_dir / "test_predictions_hourly.csv")
    np.save(out_dir / "quantile_levels.npy", np.array(quantile_levels))

    # ── 6. 评估 ──
    flat_a = actual_arr.reshape(-1)
    flat_q = quantiles_arr.reshape(-1, len(quantile_levels))
    mask = ~np.isnan(flat_a)
    fa, fq = flat_a[mask], flat_q[mask]
    p50 = fq[:, p50_idx]
    mae = float(np.mean(np.abs(p50 - fa)))
    rmse = float(np.sqrt(np.mean((p50 - fa) ** 2)))
    bias = float(np.mean(p50 - fa))
    cov80 = float(np.mean((fa >= fq[:, 0]) & (fa <= fq[:, -1])))
    width = float(np.mean(fq[:, -1] - fq[:, 0]))

    metrics = {
        "model_id": model_id, "use_covariates": use_covariates,
        "covariates": list(covariates) if use_covariates else [],
        "context_hours": context_hours, "patch_size": patch_size,
        "num_samples": num_samples, "n_test_days": len(samples),
        "mae": round(mae, 2), "rmse": round(rmse, 2), "bias": round(bias, 2),
        "coverage_80": round(cov80, 3), "interval_width": round(width, 1),
    }
    pd.Series(metrics).to_csv(out_dir / "metrics.csv")

    logger.info("=" * 60)
    logger.info("V12 Moirai (%s) 评估", "cov" if use_covariates else "univariate")
    logger.info("  MAE (P50):       %.2f", mae)
    logger.info("  RMSE (P50):      %.2f", rmse)
    logger.info("  Bias (P50):      %.2f", bias)
    logger.info("  Coverage 80%%:    %.3f", cov80)
    logger.info("  Interval Width:  %.1f", width)
    logger.info("  保存: %s", out_dir)
    logger.info("=" * 60)
    return metrics


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Salesforce/moirai-1.1-R-small",
                   help="Moirai HF id; 也可 small/base/large")
    p.add_argument("--context-hours", type=int, default=720)
    p.add_argument("--use-covariates", action="store_true")
    p.add_argument("--patch-size", type=int, default=32)
    p.add_argument("--num-samples", type=int, default=100)
    p.add_argument("--out-tag", default="")
    p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args()
    run(
        model_id=args.model,
        context_hours=args.context_hours,
        use_covariates=args.use_covariates,
        patch_size=args.patch_size,
        num_samples=args.num_samples,
        out_tag=args.out_tag,
        batch_size=args.batch_size,
    )
