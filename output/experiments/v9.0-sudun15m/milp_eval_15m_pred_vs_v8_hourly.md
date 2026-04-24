# MILP 充放电：v9.0 15 分钟预测价 vs v8.0 小时预测价

## 实验设置

- **共同区间**：2026-01-25 ～ 2026-04-18（**84 天**）
- **实际结算电价**：`source_data/日清算结果查询电厂侧(1)_副本.xlsx`（15min 实际节点价，用于执行收益结算与 PF 基准）
- **策略 A（基线）**：`v8.0-jan25-sudun500/test_predictions_hourly.csv` — 日 24 点预测，上采样为 96 点参与 MILP（现有逻辑）
- **策略 B（本实验）**：`v9.0-sudun15m/test_predictions_15min.csv` — 日 **96 点**预测，**不**上采样，直接作 MILP 目标价
- **MILP 模式**：跨日 SOC（`--carry_soc`），与 `v8.0-jan25-sudun500/strategy_milp_15min_carry_soc_result.csv` 对齐
- **代码**：`scripts/strategy_milp_15min.py` 已支持 `test_predictions_15min.csv`（`pred_native_15m`）；周图里预测价画为**红色虚线 15min 曲线**（非小时阶梯）

## 结果汇总

| 指标 | v8 小时预测 MILP | v9 15m 预测 MILP | 差值 |
|------|------------------|------------------|------|
| 净收益（元） | 12,279,000 | 11,661,000 | -618,000（约 **-5.0%**） |
| 完全预知 PF 净（元） | 23,780,000 | 23,763,000 | ≈一致（同实际价曲线） |
| **兑现率** | **51.6%** | **49.1%** | **-2.5pp** |

- PF 两列应几乎相同；微小差异来自跨日参数里「次日预测均价」在 v8 用 24 点、v9 用 96 点取均值，对终端项有轻微影响。
- **结论（本段样本）**：在相同 MILP 与评估框架下，**v9 的 15min 直接决策未优于 v8 的小时上采样**，净收益与兑现率均略低。可能原因包括：v9 与 v8 训练目标/数据段不完全一致、15min 预测噪声更大、与 jan25 专用小时模型未公平对齐等。后续可做控制变量（同一套特征、同一测试期重训 15m 与 1h）再比。

## 产出文件

- 结果 CSV：`strategy_milp_15min_carry_soc_pred15m.csv`（列含 `pred_native_15m=true`）
- 周图目录：`plots_milp_15min_pred15m/`

## 复现

```bash
python scripts/strategy_milp_15min.py \
  --pred output/experiments/v9.0-sudun15m/test_predictions_15min.csv \
  --actual_xlsx "source_data/日清算结果查询电厂侧(1)_副本.xlsx" \
  --out output/experiments/v9.0-sudun15m/strategy_milp_15min_carry_soc_pred15m.csv \
  --plots output/experiments/v9.0-sudun15m/plots_milp_15min_pred15m \
  --start 2026-01-25 --end 2026-04-18 \
  --carry_soc \
  --label "MILP-15min (v9.0 15m预测价)"
```

生成时间：以仓库内该文件 mtime 为准。
