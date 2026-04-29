"""
V10 MILP 策略评估：用 V10 联合预测结果调用 15 分钟 MILP 求解器。
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("NM_V8_TARGET", "price_sudun_500kv1m_nodal")
os.environ.setdefault("NM_V8_EXTRA_LAG1",
                       "price_sudun_500kv1m_energy,price_sudun_500kv1m_cong")
os.environ.setdefault("NM_V8_HOURLY_AGG", "mean4")
os.environ.setdefault("NM_MIN_FEATURE_DATE", "2024-12-14")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.strategy_milp_15min import run as milp_run, plot_weekly

logger = logging.getLogger(__name__)

OUTPUT_DIR = ROOT / "output"
ACTUAL_XLSX = ROOT / "source_data" / "日清算结果查询电厂侧(1)_副本.xlsx"


def evaluate(exp_dir: str = "v10.0-joint", label: str = "V10.0-Joint MILP策略"):
    v10_dir = OUTPUT_DIR / "experiments" / exp_dir
    pred_csv = v10_dir / "test_predictions_hourly.csv"
    if not pred_csv.exists():
        logger.error("未找到预测文件: %s", pred_csv)
        return

    out_dir = v10_dir / "plots_milp_15min_carry_soc"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = v10_dir / "milp_strategy_results.csv"

    logger.info("=" * 60)
    logger.info("V10 MILP 策略评估: %s", exp_dir)
    logger.info("=" * 60)

    df = milp_run(
        pred_csv=pred_csv,
        actual_xlsx=ACTUAL_XLSX,
        out_csv=out_csv,
        label=label,
        carry_soc=True,
    )

    plot_weekly(df, out_dir)
    logger.info("绘图已保存: %s", out_dir)


if __name__ == "__main__":
    import argparse
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="v10.0-joint", help="实验目录名")
    parser.add_argument("--label", default="V10.0-Joint MILP策略")
    args = parser.parse_args()
    evaluate(exp_dir=args.exp, label=args.label)
