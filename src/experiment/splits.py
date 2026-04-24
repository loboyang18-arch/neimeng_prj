"""
时间切分单一真源（15 分钟粒度）。

建模用特征/目标：仅使用 **MIN_FEATURE_CALENDAR_DATE（含）之后** 的日历日；更早日期不参与
（源 CSV 可仍保留历史，训练侧忽略）。可用环境变量 ``NM_MIN_FEATURE_DATE`` 覆盖，格式 ``YYYY-MM-DD``，
默认 ``2024-12-14``。

滚动周测试方案（与 ``model_lgbm.build_rolling_test_windows`` 一致）：
  从 TEST_START 起每周一步，共 **NUM_ROLLING_TEST_FOLDS** 折：
    [训练 TRAIN_WINDOW_MONTHS] → [验证 VAL_WINDOW_DAYS] → [预测 1 周]
  验证窗 = 测试周起点前 14 天；训练窗终点 = 验证窗起点（不含）。

当前配置：**16 折**，最后一折测试窗结束于 **TEST_END**（当日 23:45 格）。
TEST_START 由 TEST_END 与折数反推，保证恰好 16 周、不丢不增。
"""

import os

import pandas as pd


def _min_feature_calendar_date():
    raw = os.environ.get("NM_MIN_FEATURE_DATE", "2024-12-14").strip()
    return pd.Timestamp(raw).date()


MIN_FEATURE_CALENDAR_DATE = _min_feature_calendar_date()

RAW_START = pd.Timestamp("2022-06-01 00:15:00")

RAW_END = pd.Timestamp("2026-04-18 23:45:00")

FEATURE_ENGINEERING_START = pd.Timestamp("2022-08-01 00:00:00")

VAL_WINDOW_DAYS = 14
TEST_STEP_DAYS = 7
TRAIN_WINDOW_MONTHS = 6

NUM_ROLLING_TEST_FOLDS = 16
_15MIN = pd.Timedelta("15min")
TEST_END = pd.Timestamp("2026-04-18 23:45:00")
TEST_START = TEST_END - pd.Timedelta(days=NUM_ROLLING_TEST_FOLDS * TEST_STEP_DAYS) + _15MIN
