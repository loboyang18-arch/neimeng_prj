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
# V8 lag 通道定义（与 model_v8_multitask.py 对齐）
V8_LAG0_COLS = (
    "load_forecast", "renewable_forecast", "wind_forecast",
    "solar_forecast", "east_send_forecast",
    "reserve_pos_capacity", "reserve_neg_capacity",
    "price_dayahead_preclear_energy",
)
V8_LAG1_PRICE_COLS = (
    "price_unified", "price_hbd", "price_hbx",
)
V8_LAG2_COLS = (
    "load_actual", "renewable_actual", "wind_actual", "solar_actual",
)
DEFAULT_QUANTILES = (0.1, 0.3, 0.5, 0.7, 0.9)

# freq → (GluonTS freq, 默认 context 步数, 默认 prediction 步数, 每小时的 slot 数)
FREQ_PRESETS = {
    "1h": {"gluonts_freq": "h", "context_steps": 720, "pred_steps": 24, "sph": 1},
    "15min": {"gluonts_freq": "15min", "context_steps": 720, "pred_steps": 96, "sph": 4},
}


def _columns_for_lag_mode(lag_mode: str, base_covariates: List[str]) -> List[str]:
    """返回从 DWS 需要读取的列（含 lag 源列）。"""
    cols = {NODAL_COL, *base_covariates}
    if lag_mode in ("v8_lag1", "v8_lag1_lag2"):
        cols.update(V8_LAG1_PRICE_COLS)
    if lag_mode == "v8_lag1_lag2":
        cols.update(V8_LAG2_COLS)
    return list(cols)


def _add_lag_features(df_h: pd.DataFrame, lag_mode: str) -> List[str]:
    """按 V8 语义派生 lag 列；返回新增 covariate 列名列表。"""
    lag_covs: List[str] = []
    if lag_mode == "none":
        return lag_covs

    if lag_mode in ("target_lag1d", "target_only"):
        df_h["target_lag1d"] = df_h["target"].shift(24)
        lag_covs.append("target_lag1d")
        if lag_mode == "target_only":
            df_h["target_lag2d"] = df_h["target"].shift(48)
            lag_covs.append("target_lag2d")
        return lag_covs

    if lag_mode in ("v8_lag1", "v8_lag1_lag2"):
        df_h["target_lag1d"] = df_h["target"].shift(24)
        lag_covs.append("target_lag1d")
        for col in V8_LAG1_PRICE_COLS:
            if col not in df_h.columns:
                logger.warning("LAG1 源列 %s 缺失，跳过", col)
                continue
            out = f"{col}_lag1d"
            df_h[out] = df_h[col].shift(24)
            lag_covs.append(out)
        if lag_mode == "v8_lag1_lag2":
            for col in V8_LAG2_COLS:
                if col not in df_h.columns:
                    logger.warning("LAG2 源列 %s 缺失，跳过", col)
                    continue
                out = f"{col}_lag2d"
                df_h[out] = df_h[col].shift(48)
                lag_covs.append(out)
    return lag_covs


def _mean4_to_hourly(arr: np.ndarray) -> np.ndarray:
    """(96,) / (N,96) / (N,96,Q) → 对应小时 mean4。"""
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 1:
        if a.shape[0] != 96:
            raise ValueError(f"mean4 需要 96 slots，got {a.shape}")
        return a.reshape(24, 4).mean(axis=1).astype(np.float32)
    if a.ndim == 2 and a.shape[1] == 96:
        return a.reshape(a.shape[0], 24, 4).mean(axis=2).astype(np.float32)
    if a.ndim == 3 and a.shape[1] == 96:
        return a.reshape(a.shape[0], 24, 4, a.shape[2]).mean(axis=2).astype(np.float32)
    raise ValueError(f"mean4 不支持的 shape: {a.shape}")


def _load_series_with_covariates(
    covariates: List[str],
    dws_csv: str | Path | None = None,
    lag_mode: str = "none",
    freq: str = "1h",
) -> tuple[pd.DataFrame, List[str]]:
    """加载 target（+ 可选 cov / lag）。freq=1h 时重采样到小时；15min 保持原生步长。"""
    if freq not in FREQ_PRESETS:
        raise ValueError(f"未知 freq={freq}，可选 {list(FREQ_PRESETS)}")
    path = Path(dws_csv) if dws_csv else Path(
        os.environ.get("NM_DWS_CSV", str(OUTPUT_DIR / "dws_15min_features.csv"))
    )
    df15 = pd.read_csv(path, parse_dates=["ts"], index_col="ts").sort_index()
    if NODAL_COL.startswith("price_sudun"):
        df15 = fill_sudun_price_columns(df15)

    load_cols = _columns_for_lag_mode(lag_mode, covariates)
    missing = [c for c in load_cols if c not in df15.columns]
    if missing:
        logger.warning("DWS 中缺少列 %s，跳过", missing)
        load_cols = [c for c in load_cols if c in df15.columns]

    df15 = df15[load_cols].astype(float)
    if freq == "1h":
        df = df15.resample("1h").mean()
    else:
        if lag_mode != "none":
            raise NotImplementedError("15min 模式暂仅支持 uni（lag_mode=none）")
        df = df15
    df = df.rename(columns={NODAL_COL: "target"})

    if freq == "1h":
        lag_covs = _add_lag_features(df, lag_mode)
    else:
        lag_covs = []
    all_covs = list(covariates) + lag_covs
    df = df[["target"] + all_covs].dropna()

    logger.info(
        "%s: %d rows from %s | lag_mode=%s | covs=%s",
        freq, len(df), path.name, lag_mode, all_covs,
    )
    return df, all_covs


def _load_hourly_with_covariates(
    covariates: List[str],
    dws_csv: str | Path | None = None,
    lag_mode: str = "none",
) -> tuple[pd.DataFrame, List[str]]:
    """兼容旧调用：等价于 freq=1h。"""
    return _load_series_with_covariates(
        covariates, dws_csv=dws_csv, lag_mode=lag_mode, freq="1h",
    )


def _build_test_inputs(
    df: pd.DataFrame,
    test_days: List,
    context_steps: int,
    pred_steps: int,
    covariates: List[str],
    freq: str = "1h",
):
    """对每个自然日 D：构造 history + future（步长由 freq 决定）。"""
    step_min = 60 if freq == "1h" else 15
    samples = []
    for d in test_days:
        d0 = pd.Timestamp(d)
        if freq == "1h":
            ctx_start = d0 - pd.Timedelta(hours=context_steps)
            ctx_end = d0 - pd.Timedelta(hours=1)
            fut_start = d0
            fut_end = d0 + pd.Timedelta(hours=23)
        else:
            ctx_start = d0 - pd.Timedelta(minutes=step_min * context_steps)
            ctx_end = d0 - pd.Timedelta(minutes=step_min)
            fut_start = d0
            fut_end = d0 + pd.Timedelta(hours=23, minutes=45)

        history = df.loc[ctx_start:ctx_end]
        future = df.loc[fut_start:fut_end]
        min_ctx = context_steps - (24 if freq == "1h" else 96)
        if len(history) < min_ctx or len(future) != pred_steps:
            continue
        actual_fut = future["target"].values.astype(np.float32)
        target_hist = history["target"].values.astype(np.float32)
        cov_full = None
        if covariates:
            cov_full = pd.concat(
                [history[covariates], future[covariates]], axis=0,
            ).values.astype(np.float32).T

        samples.append({
            "date": d,
            "start": history.index[0],
            "target_hist": target_hist,
            "cov_full": cov_full,
            "actual_fut": actual_fut,
        })
    logger.info(
        "有效 test 天数: %d / %d (freq=%s context_steps=%d pred_steps=%d)",
        len(samples), len(test_days), freq, context_steps, pred_steps,
    )
    return samples


def _make_gluonts_dataset(
    samples: List[dict], use_covariates: bool, gluonts_freq: str,
):
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
            "start": pd.Period(s["start"], freq=gluonts_freq),
        }
        if use_covariates:
            item["feat_dynamic_real"] = s["cov_full"]
        items.append(item)
    return ListDataset(items, freq=gluonts_freq, one_dim_target=True)


def run(
    model_id: str = "Salesforce/moirai-1.1-R-small",
    test_start: str = "2026-01-27",
    test_end: str = "2026-04-17",
    context_hours: int = 720,
    use_covariates: bool = False,
    covariates: tuple = DEFAULT_COVARIATES,
    lag_mode: str = "none",
    freq: str = "1h",
    context_length: int | None = None,
    prediction_length: int | None = None,
    patch_size: int = 32,
    num_samples: int = 100,
    quantile_levels: tuple = DEFAULT_QUANTILES,
    out_tag: str = "",
    batch_size: int = 8,
):
    preset = FREQ_PRESETS[freq]
    ctx_steps = context_length if context_length is not None else preset["context_steps"]
    pred_steps = prediction_length if prediction_length is not None else preset["pred_steps"]
    gluonts_freq = preset["gluonts_freq"]

    base_covs = list(covariates) if use_covariates else []
    use_cov = use_covariates or lag_mode != "none"
    suffix = "-cov" if use_cov else "-uni"
    out_dir = OUTPUT_DIR / "experiments" / f"v12.0-moirai{suffix}{out_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("=" * 60)
    logger.info("V12 Moirai: %s", model_id)
    logger.info("  test=%s ~ %s | freq=%s | context=%d | pred=%d | lag_mode=%s",
                test_start, test_end, freq, ctx_steps, pred_steps, lag_mode)
    logger.info("  covariates=%s",
                base_covs if use_cov else "(univariate)")
    logger.info("  patch_size=%d num_samples=%d", patch_size, num_samples)
    logger.info("=" * 60)

    # ── 1. 数据 ──
    dws_csv = os.environ.get("NM_DWS_CSV")
    df, cov_list = _load_series_with_covariates(
        base_covs, dws_csv=dws_csv, lag_mode=lag_mode, freq=freq,
    )
    if use_cov:
        logger.info("  effective covariates (%d): %s", len(cov_list), cov_list)

    test_dt = pd.Timestamp(test_start).date()
    test_end_dt = pd.Timestamp(test_end).date()
    all_days = sorted(set(df.index.normalize().date.tolist()))
    test_days = [d for d in all_days if test_dt <= d <= test_end_dt]

    samples = _build_test_inputs(
        df, test_days,
        context_steps=ctx_steps,
        pred_steps=pred_steps,
        covariates=cov_list if use_cov else [],
        freq=freq,
    )
    if not samples:
        raise RuntimeError("无有效 test 样本")
    actual_fut = np.stack([s["actual_fut"] for s in samples], axis=0)

    # ── 2. 加载 Moirai ──
    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
    logger.info("加载 %s …", model_id)
    module = MoiraiModule.from_pretrained(model_id)
    n_params = sum(p.numel() for p in module.parameters())
    logger.info("  params: %d", n_params)

    feat_dim = len(cov_list) if use_cov else 0
    model = MoiraiForecast(
        module=module,
        prediction_length=pred_steps,
        context_length=ctx_steps,
        patch_size=patch_size,
        num_samples=num_samples,
        target_dim=1,
        feat_dynamic_real_dim=feat_dim,
        past_feat_dynamic_real_dim=0,
    )
    if torch.cuda.is_available():
        model = model.to("cuda")

    # ── 3. 推理 ──
    ds = _make_gluonts_dataset(samples, use_covariates=use_cov, gluonts_freq=gluonts_freq)
    predictor = model.create_predictor(batch_size=batch_size)

    logger.info("开始推理 %d 天 …", len(samples))
    forecasts = list(predictor.predict(ds))
    logger.info("推理完成: %d forecasts", len(forecasts))

    # ── 4. 提取分位数（原生步长） ──
    quantiles_native = np.zeros(
        (len(samples), pred_steps, len(quantile_levels)), dtype=np.float32,
    )
    for i, fc in enumerate(forecasts):
        for q_i, q in enumerate(quantile_levels):
            quantiles_native[i, :, q_i] = fc.quantile(q)
    quantiles_native = np.sort(quantiles_native, axis=-1)

    # ── 5. 保存与评估用数组 ──
    p50_idx = len(quantile_levels) // 2
    if freq == "15min":
        actual_arr = _mean4_to_hourly(actual_fut)
        quantiles_arr = _mean4_to_hourly(quantiles_native)
        eval_note = "hourly_mean4_from_96x15min"
        native_label = "15min"
        native_steps = pred_steps
    else:
        actual_arr = actual_fut
        quantiles_arr = quantiles_native
        eval_note = "native_hourly"
        native_label = "1h"
        native_steps = pred_steps

    rows_long, rows_p50 = [], []
    rows_native = []
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
        if freq == "15min":
            for slot in range(native_steps):
                ts = pd.Timestamp(d) + pd.Timedelta(minutes=15 * slot)
                row_n = {"ts": ts, "actual": float(actual_fut[i, slot])}
                for q_i, q in enumerate(quantile_levels):
                    row_n[f"p{int(q * 100):02d}"] = float(
                        quantiles_native[i, slot, q_i],
                    )
                rows_native.append(row_n)

    pd.DataFrame(rows_long).sort_values("ts").reset_index(drop=True).to_csv(
        out_dir / "test_predictions_quantile.csv", index=False)
    pd.DataFrame(rows_p50).set_index("ts").sort_index().to_csv(
        out_dir / "test_predictions_hourly.csv")
    if freq == "15min":
        pd.DataFrame(rows_native).sort_values("ts").reset_index(drop=True).to_csv(
            out_dir / "test_predictions_15min.csv", index=False)
    np.save(out_dir / "quantile_levels.npy", np.array(quantile_levels))

    # ── 6. 评估（小时口径；15min 模式为 mean4 聚合后） ──
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
        "model_id": model_id, "use_covariates": use_cov,
        "freq": freq, "eval_granularity": eval_note,
        "lag_mode": lag_mode,
        "covariates": cov_list if use_cov else [],
        "context_steps": ctx_steps, "prediction_steps_native": pred_steps,
        "patch_size": patch_size,
        "num_samples": num_samples, "n_test_days": len(samples),
        "mae": round(mae, 2), "rmse": round(rmse, 2), "bias": round(bias, 2),
        "coverage_80": round(cov80, 3), "interval_width": round(width, 1),
    }
    pd.Series(metrics).to_csv(out_dir / "metrics.csv")

    logger.info("=" * 60)
    logger.info("V12 Moirai (%s) 评估 [%s]", "cov" if use_cov else "univariate", eval_note)
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
    p.add_argument("--context-hours", type=int, default=720,
                   help="1h 模式下等价于 context 步数；15min 请用 --context-length")
    p.add_argument("--freq", default="1h", choices=list(FREQ_PRESETS),
                   help="时间步长：1h 或 15min（720/96 步 ≈ 7.5天 context + 1天预测）")
    p.add_argument("--context-length", type=int, default=None,
                   help="覆盖默认 context 步数（1h 默认 720，15min 默认 720）")
    p.add_argument("--prediction-length", type=int, default=None,
                   help="覆盖默认 prediction 步数（1h 默认 24，15min 默认 96）")
    p.add_argument("--use-covariates", action="store_true")
    p.add_argument("--covariates", default=None,
                   help="逗号分隔的 covariate 列名（D-day 已知的 forecast/计划值）；"
                        "默认使用 DEFAULT_COVARIATES。需配合 --use-covariates 才生效。")
    p.add_argument("--lag-mode", default="none",
                   choices=["none", "target_lag1d", "target_only",
                            "v8_lag1", "v8_lag1_lag2"],
                   help="lag 特征模式（参考 V8 LAG1/LAG2）；非 none 时自动启用 covariate")
    p.add_argument("--patch-size", type=int, default=32)
    p.add_argument("--num-samples", type=int, default=100)
    p.add_argument("--test-start", default="2026-01-27")
    p.add_argument("--test-end", default="2026-04-17")
    p.add_argument("--out-tag", default="")
    p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args()

    cov_tuple = DEFAULT_COVARIATES
    if args.covariates:
        cov_tuple = tuple(c.strip() for c in args.covariates.split(",") if c.strip())

    ctx_len = args.context_length
    if ctx_len is None and args.freq == "1h":
        ctx_len = args.context_hours

    run(
        model_id=args.model,
        test_start=args.test_start,
        test_end=args.test_end,
        context_hours=args.context_hours,
        use_covariates=args.use_covariates,
        covariates=cov_tuple,
        lag_mode=args.lag_mode,
        freq=args.freq,
        context_length=ctx_len,
        prediction_length=args.prediction_length,
        patch_size=args.patch_size,
        num_samples=args.num_samples,
        out_tag=args.out_tag,
        batch_size=args.batch_size,
    )
