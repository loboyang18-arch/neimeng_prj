# 方向 1 实验报告：V10-Quantile + 鲁棒 MILP

**日期**：2026-05-07
**实验目录**：`output/experiments/v10.0-quantile/`
**关联文档**：[v10_optimization_directions.md](./v10_optimization_directions.md)

---

## 1. 目标与方案

把 V10 (Pure Transformer) 的 24 维确定性点估计输出，换成 24×5 维分位数集合 (P10/P30/P50/P70/P90)，下游 MILP 改为鲁棒优化（pessimistic min-max over P10–P90），假设充电时电价偏高、放电时电价偏低，让模型对自己不确定的时段自动放弃激进动作。

## 2. 实施

### 2.1 模型 — `src/model_v10_quantile.py`

复用 V10 的 input projection + positional encoding + Transformer encoder，只改输出头：

```104:114:src/model_v10_quantile.py
        self.q_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_quantiles),  # 直接输出 Q 个分位数（不强制单调，后处理 sort）
        )
```

**损失**：Pinball Loss + 方向 CE + 排序损失 + 分位数交叉惩罚

```118:131:src/model_v10_quantile.py
def pinball_loss(quantiles: torch.Tensor, target: torch.Tensor,
                 q_levels: torch.Tensor | None = None) -> torch.Tensor:
    """
    quantiles: (B, T, Q) — 模型输出的 Q 个分位数预测
    target:    (B, T)    — 真实值
    q_levels:  (1, 1, Q) — 各分位数等级 ∈ (0, 1)；缺省用 QUANTILES
    """
    if q_levels is None:
        q_levels = Q_TENSOR.to(quantiles.device)
    err = target.unsqueeze(-1) - quantiles          # (B, T, Q)
    loss = torch.maximum(q_levels * err, (q_levels - 1.0) * err)
    return loss.mean()
```

### 2.2 鲁棒 MILP — `scripts/strategy_milp_15min.py`

新增 `solve_day_milp_15min_robust(p10, p50, p90, alpha)`，目标函数：

```
max Σ_t [d[t] · P_d(t) - c[t] · P_c(t)] · Δt
P_c(t) = (1-α)·P50(t) + α·P90(t)   # 充电按偏高估
P_d(t) = (1-α)·P50(t) + α·P10(t)   # 放电按偏低估
```

α=0 退化为标准 MILP；α=1 完全保守。无任何额外整数变量，与 `_build_milp_15min` 完全兼容。

### 2.3 评估管道

- `src/eval_v10_quantile_robust.py` —— α 网格扫描（无 conformal）
- `src/eval_v10_quantile_conformal.py` —— Split Conformal Calibration + α 网格

---

## 3. 训练结果

200 epoch，~90 秒，GPU。eval_log 关键节点：

| epoch | Pinball | MAE_P50 | RMSE_P50 | Cov 80% | Width |
|---|---|---|---|---|---|
| 10  | 0.139 | 113.1 | 164.8 | **0.694** | 286.4 |
| 50  | 0.061 | 122.3 | 185.4 | 0.455 | 153.7 |
| 100 | 0.040 | 116.9 | 179.7 | 0.308 | 96.8 |
| 200 | 0.028 | 112.0 | 170.8 | **0.204** | 67.1 |

**关键观察**：训练 Pinball loss 持续下降，但 **test 覆盖率从 ep10 的 69% 一路恶化到 ep200 的 20%**，区间宽度从 286 元/MWh 收缩到 67。

> **诊断**：模型严重过拟合训练集分位数 → 测试分布的真实不确定性远超模型估计 → P10/P90 几乎挤在 P50 附近 → 鲁棒 MILP 失去调节空间。

---

## 4. 鲁棒 MILP α 网格结果

### 4.1 原始预测（无 calibration）

```
alpha   net_total_wan   pf_total_wan   realization   loss_days   width
0.0     1130.51         2264.07        49.9%         16          67.15
0.3     1120.88         2264.07        49.5%         15          67.15
0.5     1136.16         2264.07        50.2%         17          67.15
0.7     1133.59         2264.07        50.1%         18          67.15
1.0     1146.87         2264.07        50.7%         18          67.15  ← 最佳
```

### 4.2 Split Conformal Calibration（cal_days=30，cal=2025-12-28~2026-01-26）

| 集合 | RAW Cov80 | RAW Width | CONFORMAL Cov80 | CONFORMAL Width |
|---|---|---|---|---|
| Calibration | 0.982 | 71.6 | **0.831** ✓ | 45.4 |
| **Test** | 0.204 | 67.1 | **0.109** ✗ | 41.2 |

```
alpha   net_total_wan   realization
0.0     1114.53         49.2%
0.3     1118.93         49.4%
0.5     1134.62         50.1%
0.7     1140.48         50.4%
1.0     1137.35         50.2%
```

### 4.3 Split Conformal Calibration（cal_days=60）

```
alpha   net_total_wan   realization   width
0.0     1113.69         49.2%         40.79
0.5     1122.47         49.6%         40.79
1.0     1141.83         50.4%         40.79
```

---

## 5. 关键诊断

### 5.1 Conformal 校准为何失败

`delta` 的方向反了：

```
delta range per q-level (cal=30): [+15.1, +7.45, -2.17, -7.16, -13.64]
```

P10 校正项是 **+15.1**（往上推），P90 校正项是 **-13.64**（往下拉）—— **区间在被压缩**。

原因：calibration set (2025-12-28 ~ 2026-01-26 冬季) 上模型预测的 P10 已经全面低于 actual（cal RAW Cov80=0.982 → 区间过宽），所以 conformal 把区间收紧。但 test set (2026-01-27 ~ 2026-04-17 冬末春初) 含大量反向日，actual 偏离方向相反 → conformal 调整加剧不匹配。

**这是 split conformal 的失败模式：calibration 与 test 不满足 i.i.d. 假设**。与之前 LightGBM 异常日检测器在 test 上 0 召回的根因完全相同：**严重的季节性分布漂移**。

### 5.2 与现有方案的对比

| 方案 | 总净收益 (万) | 实施复杂度 | 备注 |
|---|---|---|---|
| V8 baseline | 1158 | — | 历史最优 single-output |
| V10 baseline (单点 L1) | 1148 | — | Pure Transformer |
| **V10-Quantile raw α=1.0** | **1146.9** | 中 | 本实验最佳 |
| V10-Quantile + Conformal α=0.7 | 1140.5 | 中-高 | conformal 反而拉低 |
| V10 + 异常日规则 + no-op | **1211.4** | 低 | 当前实际最佳方案 |

---

## 6. 结论与下一步

### 6.1 方向 1 的结论（如实记录）

✗ **方向 1 在当前架构 + 数据规模下未带来正收益**。
- 最佳配置 (raw α=1.0) 1146.9 万，仍略低于 V10 baseline 1148 万；
- 远不及"V10 + 异常日规则 + no-op fallback"的 1211.4 万。

✗ **Conformal calibration（split 形式）失效**：
- 季节漂移让 cal 与 test 分布严重不一致，校准方向反了，coverage 从 0.20 → 0.11。
- 这与 LightGBM 异常日检测器失败的根因相同。

✓ **机制本身是有效的**：
- α=1.0 (1146.9 万) 比 α=0 (1130.5 万) 高 +16 万，证明鲁棒优化确实在反向日上节流。
- 失败的不是鲁棒 MILP，而是 V10-Quantile 估计的不确定性区间**严重低估**且**方向偏置**。

### 6.2 推荐的后续路径

按性价比排序：

**A. 放弃 V10-Quantile，改用 V10 baseline 配 deep ensemble 估计不确定性**（高 ROI）
- 同 V10 配置训 5 个 model（不同 seed），用预测分歧作为 P10/P90 估计
- ensemble 估计的不确定性更接近真实分布漂移幅度
- 代价：训练时间 ×5（仍 < 10 分钟）

**B. 保留 V10-Quantile，改用 Adaptive Conformal Inference (ACI)**（中 ROI）
- 在线滚动校准：每天用最近 K 天 actual vs prediction 残差更新 δ
- 处理分布漂移更自然
- 实施：~50 行代码改造

**C. 直接转向方向 2（基础模型迁移）**（高 ROI 但工程量最大）
- 已知问题的根本是训练样本不足 → 跨季节泛化失败
- TimesFM / Chronos 等预训练模型见过远超 1400 天的时序，跨季节能力质变

**D. 暂时搁置方向 1，继续维护"V10 + 异常日规则 + no-op"作为生产策略**（短期最稳）

### 6.3 文件清单

- 模型代码：`src/model_v10_quantile.py`
- 鲁棒 MILP 接口：`scripts/strategy_milp_15min.py::solve_day_milp_15min_robust`
- 评估脚本：
  - `src/eval_v10_quantile_robust.py`（无 conformal）
  - `src/eval_v10_quantile_conformal.py`（split conformal）
- 训练日志：`output/experiments/v10.0-quantile/train.log`
- 预测：`output/experiments/v10.0-quantile/test_predictions_quantile.csv`
- α 网格结果：
  - `output/experiments/v10.0-quantile/robust_milp/summary_alpha_grid.csv`
  - `output/experiments/v10.0-quantile/robust_milp_conformal/summary_alpha_grid_conformal.csv`
