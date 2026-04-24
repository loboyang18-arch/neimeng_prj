"""
内蒙时间轴约定：源 CSV 为「查询日」上的 00:15…24:00（96 点），
入库到 DWS 时整体平移 −15 分钟，落到同一查询日的 00:00…23:45（仍是 96 点、标准 datetime）。

数学：ts_dws = ts_source_wall − 15min；其中源侧 24:00 先按「次日 00:00」解析再平移 → 当日 23:45。
"""
from __future__ import annotations

import pandas as pd

# 源 00:15 对齐到 DWS 00:00；源 24:00（次日 0 点）对齐到 DWS 23:45
GRID_SHIFT = pd.Timedelta(minutes=15)


def parse_homepage_query_clock(dates: pd.Series, times: pd.Series) -> pd.Series:
    """首页表：查询日期 + 时点 → 源语义墙钟时间（24:00 = 次日 00:00）。"""
    base = pd.to_datetime(dates, errors="coerce").dt.normalize()
    tm = times.astype(str).str.strip()
    parts = tm.str.split(":", expand=True)
    h = pd.to_numeric(parts[0], errors="coerce").fillna(0).astype(int)
    if parts.shape[1] > 1:
        mi = pd.to_numeric(parts[1], errors="coerce").fillna(0).astype(int)
    else:
        mi = pd.Series(0, index=h.index, dtype=int)

    mask_24 = tm.isin(("24:00", "24:0", "24")) | ((h == 24) & (mi == 0))
    out = pd.Series(pd.NaT, index=base.index, dtype="datetime64[ns]")
    out.loc[mask_24] = base.loc[mask_24] + pd.Timedelta(days=1)
    ok = ~mask_24
    out.loc[ok] = base.loc[ok] + pd.to_timedelta(h.loc[ok] * 60 + mi.loc[ok], unit="m")
    return out


def shift_source_ts_to_dws_grid(ts: pd.Series) -> pd.Series:
    """将源墙钟时间平移到 DWS 索引（同一物理时刻，日历日落在 查询日 的 00:00–23:45）。"""
    return ts - GRID_SHIFT
