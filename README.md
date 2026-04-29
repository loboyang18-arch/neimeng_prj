# 内蒙独立储能电站充放电智能决策系统

> **电站**：苏敦独立储能一期（200 MW / 800 MWh）  
> **区域**：内蒙古电力市场  
> **目标**：基于电价预测 + 数学规划，自动生成 15 分钟级充放电调度方案

---

## 项目概述

本项目构建了一套 **"预测 → 优化"** 的储能电站充放电决策系统：

1. **电价预测**（V8 Conv2D 多任务网络）：利用 7 天历史电价、负荷、新能源等特征，预测次日 24 小时节点电价
2. **充放电优化**（15 分钟 MILP）：基于预测电价，求解混合整数线性规划，输出 96 个时段的充放电功率
3. **决策导向训练**（SPO+）：通过 Smart Predict then Optimize 方法，让预测模型直接优化决策收益而非预测精度

---

## 目录结构

```
neimeng_prj/
├── src/                          # 核心源码
│   ├── model_v8_multitask.py     # V8.0 Conv2D 多任务价格预测模型
│   ├── model_v10_joint.py        # V10.0 Pure Transformer 全日联合预测
│   ├── build_dws_15min_features.py  # 首页数据入库 → DWS 宽表
│   ├── fill_sudun_dws_gaps.py    # 苏敦节点电价缺测补齐
│   ├── config.py                 # 全局配置（电站参数、路径）
│   ├── train_spo_plus.py         # SPO+ 决策导向训练 (V8)
│   ├── train_v10_spo_plus.py     # SPO+ 决策导向训练 (V10)
│   ├── train_dfl.py              # DFL 决策聚焦学习（实验性）
│   ├── dfl_optimizer_layer.py    # 可微 LP 松弛层（DFL 专用）
│   ├── eval_dfl.py               # DFL 评估脚本
│   ├── eval_v10_milp.py          # V10 MILP 策略评估
│   ├── anomaly/                  # 异常日检测 + 保守降级（最新成果）
│   │   ├── labels.py             # z-shape 反向日标签构建
│   │   ├── features.py           # 日级特征（214 维）
│   │   ├── train_detector.py     # LightGBM 检测器训练
│   │   ├── rule_detector.py      # 物理规则检测器（推荐）
│   │   ├── eval_with_fallback.py # 三策略 MILP 评估
│   │   └── report.py             # 报告生成
│   └── rl/                       # 强化学习模块（实验性）
│       ├── battery_cfg.py        # 电池 MDP 参数
│       ├── battery_env.py        # Gymnasium 环境
│       ├── train_sac.py          # SAC 训练
│       ├── v8_policy.py          # V8 编码器嵌入 RL 策略网络
│       └── ...
│
├── scripts/                      # 策略脚本与工具
│   ├── strategy_milp_15min.py    # 15 分钟 MILP 充放电策略（主策略）
│   ├── strategy_milp.py          # 小时级 MILP（旧版对比）
│   ├── compute_performance_metrics.py  # 评价指标计算
│   ├── export_dashboard_data.py  # 看板数据导出
│   └── ...
│
├── docs/                         # 技术文档
│   ├── battery_dispatch_strategy.md   # 决策策略与回测分析
│   ├── dws_ingest_and_v8.md          # 数据入库与 V8 模型说明
│   └── battery_power_profile_analysis.md
│
├── output/                       # 实验输出（大部分被 .gitignore 排除）
│   ├── experiments/              # 各版本实验结果
│   │   ├── v8.0-jan25-sudun500/  # V8 MSE 基准实验
│   │   ├── v8.0-spo/            # SPO+ v1（递增 α）
│   │   ├── v8.0-spo-v2/         # SPO+ v2（固定 α + early stopping）
│   │   ├── v8.0-dfl/            # DFL 实验
│   │   └── ...
│   └── dashboard/                # 看板数据文件
│
├── source_data/                  # 原始数据（不入库）
├── run_v*.py                     # 旧版模型入口（v4-v7）
└── .gitignore
```

---

## 核心模块

### 1. 电价预测模型 — V8 Conv2D Multi-Task

- **文件**：`src/model_v8_multitask.py`
- **架构**：3 层 Conv2D + 共享特征 → 回归头（价格）+ 方向头（涨/平/跌分类）
- **输入**：`(C=22, H=12, W=7)` 张量 — 22 通道特征 × 12 个 15 分钟时间槽 × 7 天回看
- **特征**：LAG0（当日市场预测）+ LAG1（历史电价）+ LAG2（历史实际出力）
- **输出**：24 个小时级节点电价预测
- **训练数据**：2023-06 至 2026-01 的 DWS 15 分钟特征

### 2. 充放电 MILP 优化器

- **文件**：`scripts/strategy_milp_15min.py`
- **求解器**：SciPy `milp`（HiGHS 内核）
- **变量**：96 个时段的充电功率 c[t]、放电功率 d[t]、充放电状态 yc[t]/yd[t]（0-1 整数）、SOC[t]
- **约束**：功率上限、SOC 上下界、爬坡速率、最小连续运行时长（4 时段）、充放互斥、日充电量上限
- **模式**：支持日清零 / 跨日 SOC 传递两种模式

### 3. V10 全日联合预测 — Pure Transformer

- **文件**：`src/model_v10_joint.py`
- **架构**：每小时原始特征 (C×4×7) 展平 → Linear 投影 → Transformer Encoder → 24 输出
- **优势**：跨小时关系由 self-attention 建模，MAE 优于 V8（106.8 vs 119）
- **限制**：MAE 提升未直接转化为决策收益增益（V8/V10 的日内排序能力相似）

### 4. SPO+ 决策导向训练

- **文件**：`src/train_spo_plus.py` (V8) / `src/train_v10_spo_plus.py` (V10)
- **核心思想**：不追求预测准确，而是让预测引导 MILP 做出高收益决策
- **方法**：SPO+ 解析梯度 `∂L/∂ŷ = 2·[x*(2ŷ-y) − x*(ŷ)]`，每个样本调用 2 次完整 MILP
- **混合损失**：`(1-α)·MSE + α·SPO+`，α 可固定或递增
- **结论**：在测试集分布漂移情况下，决策导向训练易过拟合训练集决策模式 → 决策质量未明显超越 MSE baseline
- **对比实验**：
  - DFL（LP 松弛 + cvxpylayers）→ 因 LP-MILP gap 效果不佳
  - SPO+（完整 MILP 黑箱）→ V8/V10 上均未稳定超越 MSE baseline

### 5. 异常日检测 + 保守降级（**当前最佳方案**）

- **目录**：`src/anomaly/`
- **背景**：诊断发现 V10 vs V8 的全部差距来自 11 个反向日（V10 -44 万 vs V8 +2 万）
- **方法**：基于物理直觉的两条规则识别反向日，对其采用 no-op 保守策略
  ```
  wind_forecast_morning_evening_diff < -2000  (傍晚风远超早晨)
  AND reserve_neg_capacity_min > 3000        (持续高负备用)
  ```
- **测试集表现**：P=0.80, R=0.73, F1=0.76
- **关键洞察**：LightGBM 训练集→测试集分布漂移导致完全失败（11 个反向日全漏检），物理规则反而稳健
- **风险提示**：阈值仅在 2026-02 ~ 04 测试期调优，跨季节使用前需重新调参

---

## 实验结果

回测区间：2026-01-27 ~ 2026-04-17（81 天），跨日 SOC 模式

| 策略 | 净收益（万元） | 兑现率 | 全年外推（亿元） |
|------|-------------|--------|--------------|
| 实际现货运营 | 807 | — | 0.364 |
| 小时级 MILP（旧版） | 987 | 45.1% | 0.445 |
| MSE Baseline（V8 预测 → MILP） | 1,158 | 51.2% | 0.522 |
| MSE Baseline（V10 预测 → MILP） | 1,148 | 50.7% | 0.517 |
| SPO+ V8 | 1,179 | 52.1% | 0.531 |
| **V10 + 异常日检测 + no-op** | **1,211** | **53.5%** | **0.546** |
| Oracle 上限（已知反向日 + no-op） | 1,208 | 53.3% | 0.544 |
| 完全预知 PF（理论上限） | 2,264 | 100% | 1.020 |

详细对比报告：[output/experiments/anomaly-fallback/REPORT.md](output/experiments/anomaly-fallback/REPORT.md)

---

## 环境配置

```bash
# 使用 conda power 环境（含 GPU 版 PyTorch）
conda activate power

# 主要依赖
# Python 3.x, PyTorch 2.5+ (CUDA), pandas, numpy, scipy, matplotlib, openpyxl
```

---

## 快速运行

```bash
conda activate power

# === 价格预测模型 ===
# V8 Conv2D 多任务模型
python -m src.model_v8_multitask

# V10 全日联合预测（Pure Transformer，200 epoch）
python -m src.model_v10_joint
python -m src.eval_v10_milp                # MILP 策略评估

# === 决策导向训练（实验性） ===
python -m src.train_spo_plus               # SPO+ on V8
python -m src.train_v10_spo_plus           # SPO+ on V10

# === 异常日检测 + 保守降级（推荐生产方案） ===
python -m src.anomaly.labels               # 1. 构建反向日标签（基于训练集 mean shape）
python -m src.anomaly.features             # 2. 构建日级特征（214 维）
python -m src.anomaly.train_detector       # 3. (可选) 训练 LightGBM（仅诊断用）
python -m src.anomaly.rule_detector        # 4. 应用物理规则检测器
python -m src.anomaly.eval_with_fallback --mode rule    # 5. MILP 三策略评估
python -m src.anomaly.report               # 6. 生成对比报告

# === MILP 策略回测（独立使用） ===
python -m scripts.strategy_milp_15min \
  --pred output/experiments/v10.0-joint/test_predictions_hourly.csv \
  --actual_xlsx "source_data/日清算结果查询电厂侧(1)_副本.xlsx" \
  --out output/experiments/v10.0-joint/strategy_result.csv \
  --plots output/experiments/v10.0-joint/plots \
  --start 2026-01-27 --end 2026-04-17 \
  --carry_soc

# === 看板数据导出 ===
python scripts/export_dashboard_data.py
```

---

## 文档

- [充放电决策策略与回测分析](docs/battery_dispatch_strategy.md)
- [DWS 数据入库与 V8 模型说明](docs/dws_ingest_and_v8.md)
- [看板数据文件说明](output/dashboard/README.md)
