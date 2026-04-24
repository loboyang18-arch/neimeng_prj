"""
苏敦站 500kV.1M 三列（``price_sudun_500kv1m_*``）在 DWS 中的缺测补齐规则（建模读入宽表时应用）。

**规则（按顺序执行）**

1. **同小时、同列、临近档位（非整日全缺时）**
   对每个日历日、每个小时内的 4 个 15min 点（同一列独立处理）：
   先对该小时内序列做线性插值（``limit_direction='both'``），
   再 ffill/bfill 补齐首尾 NaN。

2. **整日全缺时，用 price_unified 代理节点电价**
   若某日苏敦节点电价全部为 NaN，则用 ``price_unified`` 列值填充。
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import pandas as pd

SUDUN_COLS: Tuple[str, str, str] = (
    "price_sudun_500kv1m_nodal",
    "price_sudun_500kv1m_energy",
    "price_sudun_500kv1m_cong",
)
UNIFIED_COL = "price_unified"

logger = logging.getLogger(__name__)


def _fill_one_hour_series(s: pd.Series) -> pd.Series:
    """同一列、同一日历小时内的 4 个 15min 点。"""
    x = s.copy()
    if x.notna().sum() > 0:
        x = x.interpolate(method="linear", limit_direction="both")
        x = x.ffill().bfill()
    return x


def fill_sudun_price_columns(
    df: pd.DataFrame,
    log: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    返回副本；对 ``SUDUN_COLS`` 应用上述文档中的两步补齐。
    """
    log = log or logger
    cols_list = list(SUDUN_COLS)
    if not any(c in df.columns for c in cols_list):
        return df.copy()

    out = df.copy()
    before = int(out[cols_list].isna().to_numpy().sum())

    # Step 1: 同小时内线性插值 + ffill/bfill
    for col in cols_list:
        if col not in out.columns:
            continue
        out[col] = (
            out.groupby(pd.DatetimeIndex(out.index).normalize())[col]
            .transform(
                lambda g: g.groupby(pd.DatetimeIndex(g.index).hour)
                .transform(_fill_one_hour_series)
            )
        )

    # Step 2: 整日全缺 → 用 price_unified 代理
    nodal = cols_list[0]  # price_sudun_500kv1m_nodal
    if nodal in out.columns and UNIFIED_COL in out.columns:
        days = pd.DatetimeIndex(out.index).normalize().unique()
        n_day_proxy = 0
        for d in days:
            m = pd.DatetimeIndex(out.index).normalize() == d
            sub_idx = out.index[m]
            if out.loc[sub_idx, nodal].isna().all():
                u = out.loc[sub_idx, UNIFIED_COL]
                if u.notna().any():
                    for c in cols_list:
                        if c in out.columns:
                            out.loc[sub_idx, c] = u
                    n_day_proxy += 1
        after = int(out[cols_list].isna().to_numpy().sum())
        filled = before - after
        log.info(
            "苏敦缺测补齐: 15min 格点 NaN %d -> %d（填补 %d）；整日用 %s 共 %d 日",
            before, after, filled, UNIFIED_COL, n_day_proxy,
        )
    else:
        after = int(out[cols_list].isna().to_numpy().sum())
        filled = before - after
        if UNIFIED_COL not in out.columns:
            log.warning("fill_sudun: 无 %s，跳过整日补齐", UNIFIED_COL)

    return out
