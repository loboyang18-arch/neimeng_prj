"""按自然日对稀疏电价列做 ffill + bfill，使当日 DWS 已有行在「至少一个观测」时被填满。"""
from __future__ import annotations

from typing import Iterable

import pandas as pd


def intraday_ffill_bfill(
    df: pd.DataFrame,
    col_names: Iterable[str],
    *,
    ts_col: str = "ts",
) -> None:
    """就地修改 `df`：对每个 `col`，按 `ts` 的日历日分组，组内先 ffill 再 bfill。"""
    if ts_col not in df.columns:
        return
    day = pd.to_datetime(df[ts_col]).dt.normalize()
    for c in col_names:
        if c not in df.columns:
            continue
        df[c] = df.groupby(day, sort=False)[c].transform(lambda s: s.ffill().bfill())
