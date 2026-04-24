"""LightGBM 滚动预测 — 仅保留 v4 Transformer 需要的窗口生成逻辑。"""
import logging
from dataclasses import dataclass

import pandas as pd

from .experiment.splits import (
    TEST_END, TEST_START, TEST_STEP_DAYS,
    TRAIN_WINDOW_MONTHS, VAL_WINDOW_DAYS,
)

logger = logging.getLogger(__name__)

_15MIN = pd.Timedelta("15min")


@dataclass
class RollingWindow:
    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def build_rolling_test_windows():
    windows = []
    tw_start = TEST_START
    fold = 0

    while tw_start + pd.Timedelta(days=TEST_STEP_DAYS) - _15MIN <= TEST_END:
        fold += 1
        tw_end = tw_start + pd.Timedelta(days=TEST_STEP_DAYS) - _15MIN
        val_end = tw_start
        val_start = tw_start - pd.Timedelta(days=VAL_WINDOW_DAYS)
        train_end = val_start
        train_start = val_start - pd.DateOffset(months=TRAIN_WINDOW_MONTHS)

        windows.append(RollingWindow(
            fold=fold,
            train_start=pd.Timestamp(train_start),
            train_end=train_end,
            val_start=val_start,
            val_end=val_end,
            test_start=tw_start,
            test_end=tw_end,
        ))

        tw_start = tw_start + pd.Timedelta(days=TEST_STEP_DAYS)

    return windows
