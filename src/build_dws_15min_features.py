"""
内蒙 15min 特征宽表入库：source_data/首页/*.csv → output/dws_15min_features.csv

源 CSV 每个「查询日期」为 96 点：00:15 … 24:00（24:00 先解析为次日 00:00 再参与计算）。
入库时按 neimeng_ts 约定整体平移 −15 分钟，使 DWS 中该日数据落在 **当日 00:00 … 23:45**
（与 v4–v8 使用的 `date_range(当日0点, periods=96)` 一致）。

合并方式：各表按 ts 外连接；缺测为 NaN。首页四类出清/预出清电价在合并后按**自然日**
组内 `ffill` + `bfill`，使当日 96 个 15 分钟格在存在观测时被自动补齐（与小时报价在日内常数段一致）。
完成后若存在 source_data/内蒙苏敦站.csv，则调用 merge_sudun_prices_into_dws.merge_into_dws 追加苏敦列。
"""
from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .config import OUTPUT_DIR, SOURCE_DIR
from .dws_intraday_price_fill import intraday_ffill_bfill
from .neimeng_ts import parse_homepage_query_clock, shift_source_ts_to_dws_grid

logger = logging.getLogger(__name__)

DWS_PATH = OUTPUT_DIR / "dws_15min_features.csv"
DWS_LEGACY_BACKUP = OUTPUT_DIR / "dws_15min_features.csv.bak_before_reingest"

# 首页源文件 → DWS 列名（与现有模型 / 旧宽表一致）
SOURCES: List[Dict] = [
    {
        "path": SOURCE_DIR / "首页/负荷预测.csv",
        "rename": {
            "统调负荷预测": "load_forecast",
            "统调负荷实测": "load_actual",
        },
    },
    {
        "path": SOURCE_DIR / "首页/新能源预测.csv",
        "rename": {
            "新能源出力预测": "renewable_forecast",
            "新能源出力实测": "renewable_actual",
            "风电出力预测": "wind_forecast",
            "风电出力实测": "wind_actual",
            "光伏出力预测": "solar_forecast",
            "光伏出力实测": "solar_actual",
        },
    },
    {
        "path": SOURCE_DIR / "首页/东送计划.csv",
        "rename": {"东送计划预测": "east_send_forecast"},
    },
    {
        "path": SOURCE_DIR / "首页/正负备用容量.csv",
        "rename": {
            "正备用容量": "reserve_pos_capacity",
            "负备用容量": "reserve_neg_capacity",
        },
    },
    {
        "path": SOURCE_DIR / "首页/电价曲线.csv",
        "rename": {
            "全网统一出清电价": "price_unified",
            "呼包东统一出清电价": "price_hbd",
            "呼包西统一出清电价": "price_hbx",
            "日前预出清电能价格": "price_dayahead_preclear_energy",
        },
    },
]

# 输出列顺序（与 model_v4–v8 及旧表一致；不含苏敦，苏敦由 merge 脚本追加）
BASE_COL_ORDER = [
    "ts",
    "east_send_forecast",
    "load_actual",
    "load_forecast",
    "price_hbd",
    "price_hbx",
    "price_dayahead_preclear_energy",
    "price_unified",
    "renewable_actual",
    "renewable_forecast",
    "reserve_neg_capacity",
    "reserve_pos_capacity",
    "solar_actual",
    "solar_forecast",
    "wind_actual",
    "wind_forecast",
]

# 源表多为小时级时点，入库对齐到 15min 后稀疏；喂模型前在日历日内补齐到满格
HOMEPAGE_PRICE_FILL_COLS = [
    "price_hbd",
    "price_hbx",
    "price_dayahead_preclear_energy",
    "price_unified",
]


def _load_one_homepage(meta: Dict) -> Optional[pd.DataFrame]:
    path: Path = meta["path"]
    rename: Dict[str, str] = meta["rename"]
    if not path.is_file():
        logger.warning("跳过（文件不存在）: %s", path)
        return None
    df = pd.read_csv(path, encoding="utf-8-sig")
    need = {"查询日期", "时点"} | set(rename.keys())
    miss = need - set(df.columns)
    if miss:
        logger.error("列缺失 %s: %s", path, miss)
        return None
    df = df.copy()
    raw_ts = parse_homepage_query_clock(df["查询日期"], df["时点"])
    df["ts"] = shift_source_ts_to_dws_grid(raw_ts)
    df = df.drop(columns=[c for c in ("查询日期", "时点") if c in df.columns])
    df = df.rename(columns=rename)
    df = df.dropna(subset=["ts"])
    dup = df["ts"].duplicated().sum()
    if dup:
        logger.warning("%s: 丢弃 %d 条重复 ts", path.name, dup)
        df = df.drop_duplicates(subset=["ts"], keep="first")
    return df.set_index("ts").sort_index()


def build_base_dws() -> pd.DataFrame:
    parts: List[pd.DataFrame] = []
    for meta in SOURCES:
        block = _load_one_homepage(meta)
        if block is None or block.empty:
            continue
        parts.append(block)
    if not parts:
        raise RuntimeError("没有任何首页源表被成功加载，无法构建 DWS")

    out = parts[0]
    for p in parts[1:]:
        out = out.join(p, how="outer")
    out = out.sort_index()
    out = out.reset_index()
    # 列顺序：按 BASE_COL_ORDER，存在的写入，其余（不应有）放后
    cols = [c for c in BASE_COL_ORDER if c in out.columns]
    extra = [c for c in out.columns if c not in cols]
    if extra:
        logger.warning("未在标准列清单中的列将被丢弃: %s", extra)
    out = out[cols]
    intraday_ffill_bfill(out, HOMEPAGE_PRICE_FILL_COLS)
    # 以负荷预测为主时间轴：去掉尾部「负荷预测缺失」的行（电价等更长表会多出尾部）
    if "load_forecast" in out.columns:
        while len(out) > 0 and pd.isna(out["load_forecast"].iloc[-1]):
            out = out.iloc[:-1]
    return out


def run_reingest(*, backup: bool = True, merge_sudun: bool = True) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = build_base_dws()
    logger.info("合并后: %d 行, %d 列, ts %s ~ %s",
                 len(df), len(df.columns),
                 df["ts"].min(), df["ts"].max())

    if DWS_PATH.is_file() and backup:
        shutil.copy2(DWS_PATH, DWS_LEGACY_BACKUP)
        logger.info("已备份旧表: %s", DWS_LEGACY_BACKUP)

    df.to_csv(DWS_PATH, index=False)
    logger.info("已写入: %s", DWS_PATH)

    sudun = SOURCE_DIR / "内蒙苏敦站.csv"
    if merge_sudun and sudun.is_file():
        from .merge_sudun_prices_into_dws import merge_into_dws

        merge_into_dws()
        merged = pd.read_csv(DWS_PATH, parse_dates=["ts"])
        if "load_forecast" in merged.columns:
            while len(merged) > 0 and pd.isna(merged["load_forecast"].iloc[-1]):
                merged = merged.iloc[:-1]
        merged.to_csv(DWS_PATH, index=False)
        logger.info("苏敦合并后尾部裁剪: %d 行 → %s", len(merged), merged["ts"].max())
    elif merge_sudun:
        logger.info("未找到 %s，跳过苏敦合并", sudun)

    return DWS_PATH


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description="内蒙 dws_15min_features 重新入库")
    ap.add_argument("--no-backup", action="store_true", help="不备份旧 dws_15min_features.csv")
    ap.add_argument("--no-sudun", action="store_true", help="不调用苏敦合并")
    args = ap.parse_args()
    run_reingest(backup=not args.no_backup, merge_sudun=not args.no_sudun)


if __name__ == "__main__":
    main()
