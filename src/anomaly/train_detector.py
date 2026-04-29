"""
反向日检测器训练
================
LightGBM 二分类，时序 5-fold CV。
输出测试期每天的检测结果（含概率），并通过最大化 F1 选择阈值。
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_recall_curve, roc_auc_score)
from sklearn.model_selection import TimeSeriesSplit

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.config import OUTPUT_DIR

logger = logging.getLogger(__name__)

ANOMALY_DIR = OUTPUT_DIR / "experiments" / "anomaly-detector"


def _load_data(test_start: str = "2026-01-27"):
    df = pd.read_csv(ANOMALY_DIR / "day_features.csv", parse_dates=["date"])
    df["date"] = df["date"].dt.date
    df = df.sort_values("date").reset_index(drop=True)

    test_dt = pd.Timestamp(test_start).date()
    train_df = df[df["date"] < test_dt].reset_index(drop=True)
    test_df = df[df["date"] >= test_dt].reset_index(drop=True)

    drop_cols = {"date", "shape_corr", "category", "is_reverse"}
    feature_cols = [c for c in df.columns if c not in drop_cols]

    return train_df, test_df, feature_cols


def _train_lgbm(X_train, y_train, X_val=None, y_val=None,
                pos_weight: float = 1.0, num_boost_round: int = 500):
    params = dict(
        objective="binary",
        metric="binary_logloss",
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        min_data_in_leaf=20,
        scale_pos_weight=pos_weight,
        verbosity=-1,
        seed=42,
    )
    train_set = lgb.Dataset(X_train, label=y_train)
    callbacks = [lgb.log_evaluation(period=0)]
    valid_sets = None
    if X_val is not None:
        valid_sets = [lgb.Dataset(X_val, label=y_val, reference=train_set)]
        callbacks.append(lgb.early_stopping(stopping_rounds=30, verbose=False))
    model = lgb.train(
        params, train_set,
        num_boost_round=num_boost_round,
        valid_sets=valid_sets,
        callbacks=callbacks,
    )
    return model


def main(test_start: str = "2026-01-27", n_splits: int = 5):
    ANOMALY_DIR.mkdir(parents=True, exist_ok=True)

    train_df, test_df, feature_cols = _load_data(test_start)
    logger.info("训练日: %d, 测试日: %d, 特征数: %d",
                len(train_df), len(test_df), len(feature_cols))
    logger.info("训练集反向日: %d / %d (%.1f%%)",
                int(train_df["is_reverse"].sum()), len(train_df),
                train_df["is_reverse"].mean() * 100)

    X = train_df[feature_cols].astype(float).values
    y = train_df["is_reverse"].values.astype(int)

    pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)
    logger.info("scale_pos_weight = %.2f", pos_weight)

    tss = TimeSeriesSplit(n_splits=n_splits)
    cv_oof = np.zeros(len(X))
    cv_metrics = []
    for fold, (tr, vl) in enumerate(tss.split(X)):
        model_f = _train_lgbm(X[tr], y[tr], X[vl], y[vl], pos_weight=pos_weight)
        cv_oof[vl] = model_f.predict(X[vl], num_iteration=model_f.best_iteration)
        if y[vl].sum() > 0:
            auc = roc_auc_score(y[vl], cv_oof[vl])
            ap = average_precision_score(y[vl], cv_oof[vl])
            cv_metrics.append({"fold": fold + 1, "n_val": len(vl),
                               "n_pos": int(y[vl].sum()), "auc": auc, "ap": ap})
            logger.info("Fold %d  n_val=%d  pos=%d  AUC=%.3f  AP=%.3f",
                        fold + 1, len(vl), int(y[vl].sum()), auc, ap)
        else:
            logger.info("Fold %d  n_val=%d  pos=0  (skip metric)", fold + 1, len(vl))

    valid_mask = cv_oof > 0
    if valid_mask.any() and y[valid_mask].sum() > 0:
        oof_auc = roc_auc_score(y[valid_mask], cv_oof[valid_mask])
        oof_ap = average_precision_score(y[valid_mask], cv_oof[valid_mask])
        logger.info("OOF AUC=%.3f  AP=%.3f", oof_auc, oof_ap)

    final_model = _train_lgbm(X, y, pos_weight=pos_weight, num_boost_round=300)
    final_model.save_model(str(ANOMALY_DIR / "detector_lgbm.txt"))

    test_X = test_df[feature_cols].astype(float).values
    test_proba = final_model.predict(test_X)

    if valid_mask.any():
        prec, rec, ths = precision_recall_curve(y[valid_mask], cv_oof[valid_mask])
        f1s = 2 * prec * rec / (prec + rec + 1e-9)
        best_idx = int(np.argmax(f1s[:-1])) if len(ths) > 0 else 0
        best_th = float(ths[best_idx]) if len(ths) > 0 else 0.5
        best_f1 = float(f1s[best_idx])
        logger.info("最佳阈值（max F1）= %.3f  F1=%.3f  P=%.3f  R=%.3f",
                    best_th, best_f1, prec[best_idx], rec[best_idx])
    else:
        best_th = 0.5
        best_f1 = float("nan")

    high_prec_th = best_th
    if valid_mask.any() and len(ths) > 0:
        precision_target = 0.6
        for i in range(len(ths) - 1, -1, -1):
            if prec[i] >= precision_target and rec[i] >= 0.3:
                high_prec_th = float(ths[i])
                logger.info("高精确率阈值 (P>=%.1f, R>=0.3) = %.3f  P=%.3f  R=%.3f",
                            precision_target, high_prec_th, prec[i], rec[i])
                break

    test_df_out = test_df[["date", "shape_corr", "category", "is_reverse"]].copy()
    test_df_out["prob_reverse"] = test_proba
    test_df_out["pred_reverse_default"] = (test_proba >= 0.5).astype(int)
    test_df_out["pred_reverse_f1"] = (test_proba >= best_th).astype(int)
    test_df_out["pred_reverse_highprec"] = (test_proba >= high_prec_th).astype(int)

    csv_path = ANOMALY_DIR / "test_predictions.csv"
    test_df_out.to_csv(csv_path, index=False)
    logger.info("测试期检测结果已保存: %s", csv_path)

    n_pos = int(test_df_out["is_reverse"].sum())
    for col, name in [
        ("pred_reverse_default", "默认 (0.5)"),
        ("pred_reverse_f1", f"F1 ({best_th:.3f})"),
        ("pred_reverse_highprec", f"高精确率 ({high_prec_th:.3f})"),
    ]:
        tp = int(((test_df_out[col] == 1) & (test_df_out["is_reverse"] == 1)).sum())
        fp = int(((test_df_out[col] == 1) & (test_df_out["is_reverse"] == 0)).sum())
        fn = int(((test_df_out[col] == 0) & (test_df_out["is_reverse"] == 1)).sum())
        prec_v = tp / max(tp + fp, 1)
        rec_v = tp / max(tp + fn, 1)
        f1_v = 2 * prec_v * rec_v / max(prec_v + rec_v, 1e-6)
        logger.info("测试集 %s  TP=%d  FP=%d  FN=%d  P=%.2f R=%.2f F1=%.2f",
                    name, tp, fp, fn, prec_v, rec_v, f1_v)

    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": final_model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)
    importance.to_csv(ANOMALY_DIR / "feature_importance.csv", index=False)
    logger.info("Top 15 重要特征:")
    for _, r in importance.head(15).iterrows():
        logger.info("  %-50s %.0f", r["feature"], r["importance"])

    summary = {
        "test_start": test_start,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "n_features": len(feature_cols),
        "train_pos_rate": float(train_df["is_reverse"].mean()),
        "best_threshold_f1": best_th,
        "high_precision_threshold": high_prec_th,
        "cv_metrics": cv_metrics,
    }
    with open(ANOMALY_DIR / "detector_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=float)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
