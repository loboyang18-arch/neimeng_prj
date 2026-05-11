#!/usr/bin/env bash
# 新约束多场景全量评估 — 后台一键运行
# 用法: nohup bash scripts/run_all_scenarios.sh > logs/scenarios_all.log 2>&1 &
set -euo pipefail

cd "$(dirname "$0")/.."
PROJ_ROOT="$(pwd)"

# 激活 conda
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate power 2>/dev/null || true

mkdir -p logs

echo "=========================================="
echo "多场景评估开始 $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

python scripts/eval_new_constraints_scenarios.py \
  --scenarios "A,B,C,D,E" \
  --start "2026-01-27" \
  --end   "2026-04-17"

echo "=========================================="
echo "全部完成 $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
