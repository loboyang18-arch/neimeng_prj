"""
V10-Quantile + Split Conformal Calibration + 鲁棒 MILP

流程：
  1. 加载已训练的 V10-quantile 模型权重
  2. 在 calibration set（训练集尾部 N 天）上推理出分位数 → 算残差
  3. 对每个 quantile 等级加经验残差 → conformalized quantiles（覆盖率有理论保证）
  4. 对 test 预测应用相同的加性校正
  5. 跑鲁棒 MILP α 网格
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

os.environ.setdefault("NM_V8_TARGET", "price_sudun_500kv1m_nodal")
os.environ.setdefault("NM_V8_EXTRA_LAG1",
                       "price_sudun_500kv1m_energy,price_sudun_500kv1m_cong")
os.environ.setdefault("NM_V8_HOURLY_AGG", "mean4")
os.environ.setdefault("NM_MIN_FEATURE_DATE", "2024-12-14")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import OUTPUT_DIR  # noqa: E402
from src.model_v10_joint import (  # noqa: E402
    DailyJointDataset, _load_dws, _build_daily_arrays, _compute_norm,
    DEVICE, C_TOTAL, H_SLOTS_V10,
)
from src.model_v10_quantile import (  # noqa: E402
    V10QuantileNet, QUANTILES, N_Q,
)
from src.eval_v10_quantile_robust import evaluate_alpha  # noqa: E402
from scripts.strategy_milp_15min import load_actual_15min  # noqa: E402

logger = logging.getLogger(__name__)
ACTUAL_XLSX = ROOT / "source_data" / "日清算结果查询电厂侧(1)_副本.xlsx"


def _load_model(exp_dir: Path):
    """加载已训练的 V10-quantile 模型。"""
    weights = exp_dir / "model_weights.pt"
    norm_mean = np.load(exp_dir / "norm_mean.npy")
    norm_std = np.load(exp_dir / "norm_std.npy")
    stats = np.load(exp_dir / "target_stats.npz")
    y_mean = float(stats["y_mean"])
    y_std = float(stats["y_std"])
    q_levels = np.load(exp_dir / "quantile_levels.npy")

    model = V10QuantileNet(
        c_in=C_TOTAL, h_slots=H_SLOTS_V10,
        n_quantiles=len(q_levels),
    ).to(DEVICE)
    model.load_state_dict(torch.load(weights, map_location=DEVICE))
    model.eval()
    return model, norm_mean, norm_std, y_mean, y_std, q_levels


def _predict(model, ds: DailyJointDataset, y_mean: float, y_std: float):
    """返回 (dates, quantiles_DxTxQ, actuals_DxT)，T=24。"""
    loader = DataLoader(ds, batch_size=8, shuffle=False)
    all_q, all_actual = [], []
    with torch.no_grad():
        for grids, _, _, tgt_raw in loader:
            grids = grids.to(DEVICE)
            q_norm, _ = model(grids)
            q_raw = q_norm.cpu().numpy() * y_std + y_mean   # (B, 24, Q)
            q_raw = np.sort(q_raw, axis=-1)                  # 保单调
            all_q.append(q_raw)
            all_actual.append(tgt_raw.numpy())
    quantiles = np.concatenate(all_q, axis=0)
    actuals = np.concatenate(all_actual, axis=0)
    return ds.dates, quantiles, actuals


def conformalize(
    cal_quantiles: np.ndarray,    # (N_cal, T, Q)
    cal_actuals: np.ndarray,      # (N_cal, T)
    q_levels: np.ndarray,         # (Q,)
):
    """
    Split Conformal: 对每个 quantile 等级 q_k 计算校准残差分位数。

    标准 split conformal for quantile regression (CQR-style):
      score_i_k = max(ŷ_q^{lower}_i - y_i, y_i - ŷ_q^{upper}_i) 用于每对 (lower, upper)

    简化版（每个 quantile 独立校准）：
      对 q_k > 0.5 (上分位): δ_k = quantile_q_k(y_i - ŷ_k_i)（残差的 q_k 分位数）
                              校正后 ŷ'_k = ŷ_k + δ_k
      对 q_k < 0.5 (下分位): δ_k = quantile_q_k(y_i - ŷ_k_i)（残差的 q_k 分位数，会是负值）
                              校正后 ŷ'_k = ŷ_k + δ_k
      对 q_k = 0.5 (中位数): δ_50 = median(y_i - ŷ_50_i)，bias correction
    Returns:
      delta: (T, Q) 校正常数
    """
    N, T, Q = cal_quantiles.shape
    delta = np.zeros((T, Q), dtype=float)
    for t in range(T):
        for q_i in range(Q):
            residual = cal_actuals[:, t] - cal_quantiles[:, t, q_i]
            mask = ~np.isnan(residual)
            if not mask.any():
                continue
            r = residual[mask]
            # 取该 q level 对应的经验分位数
            delta[t, q_i] = float(np.quantile(r, q_levels[q_i]))
    return delta


def apply_conformal(quantiles: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """对预测分位数加 (T, Q) 校正项。返回单调排序后的结果。"""
    out = quantiles + delta[np.newaxis, :, :]    # broadcast
    out = np.sort(out, axis=-1)
    return out


def coverage_metrics(quantiles: np.ndarray, actuals: np.ndarray, q_levels: np.ndarray):
    p10_idx = 0
    p50_idx = N_Q // 2
    p90_idx = N_Q - 1
    flat_q = quantiles.reshape(-1, quantiles.shape[-1])
    flat_a = actuals.reshape(-1)
    mask = ~np.isnan(flat_a)
    flat_q, flat_a = flat_q[mask], flat_a[mask]

    cov80 = float(np.mean(
        (flat_a >= flat_q[:, p10_idx]) & (flat_a <= flat_q[:, p90_idx])
    ))
    width = float(np.mean(flat_q[:, p90_idx] - flat_q[:, p10_idx]))
    mae_p50 = float(np.mean(np.abs(flat_q[:, p50_idx] - flat_a)))
    err = flat_a[:, None] - flat_q
    pinball = float(np.mean(np.maximum(q_levels * err, (q_levels - 1.0) * err)))
    return {"coverage_80": cov80, "interval_width": width,
            "mae_p50": mae_p50, "pinball": pinball}


def make_pred_df(dates, quantiles, actuals, q_levels):
    """把 quantile 数组转成与 test_predictions_quantile.csv 相同的 long DataFrame。"""
    rows = []
    for di, d in enumerate(dates):
        for h in range(24):
            row = {"ts": pd.Timestamp(d) + pd.Timedelta(hours=h),
                   "actual": float(actuals[di, h])}
            for q_i, q_lvl in enumerate(q_levels):
                row[f"p{int(q_lvl * 100):02d}"] = float(quantiles[di, h, q_i])
            rows.append(row)
    df = pd.DataFrame(rows)
    df["date"] = df["ts"].dt.date.astype(str)
    return df.sort_values("ts").reset_index(drop=True)


def run(
    exp_dir: str = "v10.0-quantile",
    cal_days: int = 30,
    test_start: str = "2026-01-27",
    test_end: str = "2026-04-17",
    alphas: tuple = (0.0, 0.3, 0.5, 0.7, 1.0),
):
    work_dir = OUTPUT_DIR / "experiments" / exp_dir
    out_dir = work_dir / "robust_milp_conformal"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Conformal Calibration + 鲁棒 MILP")
    logger.info("  实验: %s, calibration_days=%d", exp_dir, cal_days)
    logger.info("=" * 60)

    # ── 1. 加载模型 ──
    model, norm_mean, norm_std, y_mean, y_std, q_levels = _load_model(work_dir)
    logger.info("已加载模型，分位数: %s", q_levels.tolist())

    # ── 2. 重建数据集 ──
    dws = _load_dws()
    valid, day_lag0, day_lag1, day_lag2, day_targets = _build_daily_arrays(dws)
    test_dt = pd.Timestamp(test_start).date()
    test_end_dt = pd.Timestamp(test_end).date()
    train_days = [d for d in valid if d < test_dt]
    test_days = [d for d in valid if test_dt <= d <= test_end_dt]

    cal_days = min(cal_days, len(train_days))
    cal_set = train_days[-cal_days:]
    logger.info("  train=%d, calibration=%d (尾部), test=%d",
                len(train_days), cal_days, len(test_days))
    logger.info("  calibration date range: %s ~ %s", cal_set[0], cal_set[-1])

    ds_kwargs = dict(
        day_lag0=day_lag0, day_lag1=day_lag1, day_lag2=day_lag2,
        day_targets=day_targets,
        norm_mean=norm_mean, norm_std=norm_std,
        y_mean=y_mean, y_std=y_std,
    )
    cal_ds = DailyJointDataset(sample_dates=cal_set, **ds_kwargs)
    test_ds = DailyJointDataset(sample_dates=test_days, **ds_kwargs)

    # ── 3. 推理：calibration & test ──
    _, cal_q, cal_a = _predict(model, cal_ds, y_mean, y_std)
    test_dates, test_q, test_a = _predict(model, test_ds, y_mean, y_std)
    logger.info("  cal shapes: q=%s a=%s", cal_q.shape, cal_a.shape)
    logger.info("  test shapes: q=%s a=%s", test_q.shape, test_a.shape)

    # ── 4. Conformal 校准 ──
    delta = conformalize(cal_q, cal_a, q_levels)
    logger.info("  delta range per q-level (T avg): %s",
                delta.mean(axis=0).round(2).tolist())

    test_q_conf = apply_conformal(test_q, delta)
    cal_q_conf = apply_conformal(cal_q, delta)

    # ── 5. 覆盖率对比 ──
    raw_metrics = coverage_metrics(test_q, test_a, q_levels)
    conf_metrics = coverage_metrics(test_q_conf, test_a, q_levels)
    logger.info("=" * 60)
    logger.info("Test set metrics:")
    logger.info("  RAW:        coverage_80=%.3f width=%.1f MAE_P50=%.1f pinball=%.1f",
                raw_metrics["coverage_80"], raw_metrics["interval_width"],
                raw_metrics["mae_p50"], raw_metrics["pinball"])
    logger.info("  CONFORMAL:  coverage_80=%.3f width=%.1f MAE_P50=%.1f pinball=%.1f",
                conf_metrics["coverage_80"], conf_metrics["interval_width"],
                conf_metrics["mae_p50"], conf_metrics["pinball"])

    cal_raw_metrics = coverage_metrics(cal_q, cal_a, q_levels)
    cal_conf_metrics = coverage_metrics(cal_q_conf, cal_a, q_levels)
    logger.info("Calibration set metrics:")
    logger.info("  RAW:        coverage_80=%.3f width=%.1f",
                cal_raw_metrics["coverage_80"], cal_raw_metrics["interval_width"])
    logger.info("  CONFORMAL:  coverage_80=%.3f width=%.1f",
                cal_conf_metrics["coverage_80"], cal_conf_metrics["interval_width"])

    # ── 6. 保存 conformalized 预测 ──
    pred_df_conf = make_pred_df(test_dates, test_q_conf, test_a, q_levels)
    pred_df_conf.to_csv(out_dir / "test_predictions_quantile_conformal.csv", index=False)

    # 同时保存 conformalized P50 兼容文件
    p50_idx = N_Q // 2
    rows_p50 = []
    for di, d in enumerate(test_dates):
        for h in range(24):
            rows_p50.append({
                "ts": pd.Timestamp(d) + pd.Timedelta(hours=h),
                "actual": float(test_a[di, h]),
                "pred": float(test_q_conf[di, h, p50_idx]),
            })
    pd.DataFrame(rows_p50).set_index("ts").to_csv(
        out_dir / "test_predictions_p50_conformal.csv")

    # ── 7. α 网格扫描（conformal 版） ──
    actual_df = load_actual_15min(ACTUAL_XLSX)
    summaries = []

    for alpha in alphas:
        logger.info("─" * 50)
        logger.info("[CONFORMAL] α = %.2f", alpha)
        df = evaluate_alpha(pred_df_conf, actual_df, alpha=alpha, carry_soc=True)
        df.to_csv(out_dir / f"daily_alpha_{alpha:.2f}_conformal.csv", index=False)

        net_total = df["net"].sum()
        pf_total = df["pf_net"].sum()
        n_loss = (df["net"] < 0).sum()
        avg_width = df["interval_width"].mean()
        ratio = net_total / pf_total if abs(pf_total) > 1 else np.nan

        logger.info("  累计净=%.2f万 (PF=%.2f万) 兑现率=%.1f%% 亏损天=%d 区间宽=%.1f",
                    net_total / 1e4, pf_total / 1e4,
                    ratio * 100 if not np.isnan(ratio) else 0,
                    n_loss, avg_width)
        summaries.append({
            "alpha": alpha,
            "n_days": len(df),
            "n_loss_days": int(n_loss),
            "net_total_wan": round(net_total / 1e4, 2),
            "pf_total_wan": round(pf_total / 1e4, 2),
            "realization_rate": round(ratio, 4) if not np.isnan(ratio) else None,
            "avg_interval_width": round(avg_width, 2),
        })

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(out_dir / "summary_alpha_grid_conformal.csv", index=False)

    print("\n" + "=" * 80)
    print(f" V10-Quantile + Conformal + 鲁棒 MILP α 网格（cal_days={cal_days}）")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print("=" * 80)

    return summary_df


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    p = argparse.ArgumentParser()
    p.add_argument("--exp", default="v10.0-quantile")
    p.add_argument("--cal-days", type=int, default=30)
    p.add_argument("--alphas", default="0.0,0.3,0.5,0.7,1.0")
    args = p.parse_args()
    alphas = tuple(float(x) for x in args.alphas.split(","))
    run(exp_dir=args.exp, cal_days=args.cal_days, alphas=alphas)
