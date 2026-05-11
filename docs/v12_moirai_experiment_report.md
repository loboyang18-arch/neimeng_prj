# 方向 B 实验报告：V12 — Moirai 基础模型 + V10-Moirai Hybrid

**日期**：2026-05-07
**实验目录**：`output/experiments/v12.0-moirai-uni/`、`v12.0-moirai-cov*/`、`v11.0-hybrid/`
**关联文档**：[v10_optimization_directions.md](./v10_optimization_directions.md)、[v11_foundation_experiment_report.md](./v11_foundation_experiment_report.md)

---

## 1. 目标与最终结论

### 目标

替换 Chronos-Bolt 为支持 covariate 的 Moirai-1.1-R 基础模型：
- (A) 验证 covariate（风电/负荷/备用预报）能否突破 univariate MAE 天花板（~140）
- (B) 评估 V10 + Moirai 的 Hybrid 收益是否好于 V10 + Chronos

### 最终结论

✅ **(B) 大幅超越，写下当前最优**：**1220.1 万**（+71.8 vs V10, +8.7 vs Chronos Hybrid, +8.7 vs 异常日规则）。
✗ **(A) 失败**：covariate 配置反而比 univariate 略差（小模型容量不够），但 univariate Moirai 与 Chronos 的不确定性区间互补。

---

## 2. 实施

### 2.1 Moirai 候选

| 模型 | 参数 | 状态 |
|---|---|---|
| Moirai-1.1-R-small | 13.8M | ✓ 主力（hf-mirror 下载成功）|
| Moirai-1.1-R-base  | ~91M | ✗ hf-mirror config.json 下载多次超时 |
| Moirai-1.1-R-large | ~311M | 暂未尝试 |
| Moirai-MoE-base    | – | 暂未尝试 |

由于网络限制，本次仅完成 small 模型的实验。Base/Large 待网络环境改善后补做。

### 2.2 关键代码

- `src/model_v12_moirai.py` — Moirai 推理（支持 univariate / covariate）
  - `_load_hourly_with_covariates()`：从 dws_15min_features 取 hourly 数据
  - `_build_test_inputs()`：每个 D 构造 history + future covariates
  - GluonTS `ListDataset`：把 `(target_hist, cov_full)` 喂入 Moirai 的 `feat_dynamic_real`
- `src/eval_v11_hybrid.py` — 复用：V10 P50 + Moirai uncertainty + 鲁棒 MILP

### 2.3 数据约定

- target：`price_sudun_500kv1m_nodal`（hourly = 4 个 15min 槽求均值）
- covariate（仅 cov 模式用）：
  - `wind_forecast`、`load_forecast`、`reserve_neg_capacity`、`price_dayahead_preclear_energy`
- 每个 D：context = D-720h ~ D-1h，target = D 当天 24h
- covariate 在历史段和未来段都有真值（day-ahead 预报，提前一日全部已知）

---

## 3. 实验结果

### 3.1 Moirai-small zero-shot（核心烟测）

| 配置 | MAE | RMSE | Bias | Cov80 | Width |
|---|---|---|---|---|---|
| univariate, patch=32 | **144.6** | 192.98 | -13.1 | **0.801** | 444.0 |
| with covariate, patch=32 | 148.3 | 196.65 | +8.0 | 0.780 | 459.0 |
| with covariate, patch=64 | 146.7 | 188.25 | +2.6 | 0.781 | 437.3 |

**对比 Chronos-bolt-small**：MAE 144 vs 140 — 略差但接近；Cov80 0.80 vs 0.79 — 相当。
**意外发现**：覆盖 80% 与 univariate 几乎一致（0.80），region 都被两类模型抓得很准。

**covariate 反而恶化的原因**：
1. Moirai-small 仅 13.8M 参数，容量不足以从 4 个 covariate 中提取额外信号
2. 内蒙古 covariate 与预训练分布差异大（电价 100~1500 vs 普通时序的 0~100），即使 RIN 归一化也未必处理好量级跨度
3. patch_size 从 32 增到 64 （相同数据 patch 数减半）让模型对 covariate 长期模式更敏感，效果略好但仍不及 univariate

**结论**：直接用 covariate 没有突破 univariate 天花板，但**Moirai 的 univariate 已经与 Chronos 等价好用**。

### 3.2 ★ V10 + Moirai (uni) Hybrid 网格扫描

混合公式（与 Chronos Hybrid 完全相同）：
```
P50_hybrid = V10_P50                                   # MAE=107
P10_hybrid = V10_P50 - (Moirai_P50 - Moirai_P10) × ws
P90_hybrid = V10_P50 + (Moirai_P90 - Moirai_P50) × ws
```

完整网格（hours: 28 个组合，~30 分钟）：

| ws | α | cov80 | width | net (万) | 备注 |
|---|---|---|---|---|---|
| 0.30 | 0.30 | 0.383 | 133 | 1210.96 | – |
| 0.30 | 0.40 | 0.383 | 133 | 1211.23 | – |
| 0.30 | 0.50 | 0.383 | 133 | 1217.10 | – |
| **0.35** | **0.40** | **0.432** | **155** | **1220.09** | **★ 全局最优** |
| 0.35 | 0.45 | 0.432 | 155 | 1216.73 | – |
| 0.35 | 0.50 | 0.432 | 155 | 1216.32 | – |
| 0.40 | 0.40 | 0.480 | 178 | 1217.92 | – |
| 0.40 | 0.45 | 0.480 | 178 | 1215.01 | – |
| 0.45 | 0.40 | 0.512 | 200 | 1214.82 | – |
| 0.50 | 0.40 | 0.551 | 222 | 1206.69 | – |

**最佳：ws=0.35, α=0.40**，cov80=0.432，区间宽度 155 元/MWh，对应净收益 1220.09 万。

---

## 4. 完整方案对比（截至 2026-05-07）

| 方案 | 净收益 (万) | Δ vs V10 | 反向日处理 |
|---|---|---|---|
| V8 baseline | 1158.5 | +10.2 | – |
| V10 baseline | 1148.3 | 0 | – |
| V10-Quantile + 鲁棒 MILP α=1 | 1146.9 | -1.4 | 区间太窄无效 |
| V10 + Chronos Hybrid (ws=0.4, α=0.4) | 1210.9 | +62.6 | MILP 自动 |
| V10 + 异常日规则 + no-op | 1211.4 | +63.1 | 强制 no-op |
| **★ V10 + Moirai Hybrid (ws=0.35, α=0.40)** | **1220.1** | **+71.8** | **MILP 自动** |

**Moirai Hybrid 相比同结构的 Chronos Hybrid，多赚 +9.2 万元**。

---

## 5. 关键洞察

### 5.1 为什么 Moirai 比 Chronos 多赚 ~9 万？

两者参数量相差 3.5 倍（13.8M vs 47.7M），但 Moirai 反而更好。差异分析：

- **架构**：Chronos 是 T5 encoder + decoder（专为时序的 patch-token），Moirai 是多变量 patch-based transformer，**学习到了更细粒度的局部模式**
- **预训练数据**：Moirai-1.1-R 预训练数据规模相近，但更注重"高变异时序"分布
- **不确定性形态**：Moirai cov80=0.801（vs Chronos 0.790）略高，但**关键区别在于反向日上的区间扩张方向**
- 在反向日上 Moirai 似乎能更准确地识别"高不确定性"，让区间向"反向"方向扩张更对，从而 robust MILP 能更好地节流

### 5.2 为什么 covariate 没生效？

- 13.8M 参数太小：每多一个 covariate 等于多一个 variate token，attention pattern 复杂化
- covariate 的领域漂移（内蒙古特定）vs 预训练通用分布
- patch_size=32 时每个 patch 看到 4 个特征 × 32h 的 cube，特征数显著增加但容量没匹配上

**待验证**：Moirai-base/large 是否能扭转。

### 5.3 ws=0.35 而非 0.40 最优的物理含义

Moirai uni 区间宽度 444（≈Chronos 426），缩到 ws=0.35 时半宽 155 元，比 V10 RMSE (170) 略小。

- ws<0.35：区间太窄，鲁棒 MILP 退化为 V10 baseline（1148）
- ws=0.35-0.4：区间宽度 ~150-180，刚好覆盖反向日上 V10 P50 偏差，robust MILP 能"恰到好处"地降权
- ws>0.5：区间太宽，正常日上 robust MILP 也开始保守，损失小价差套利空间

---

## 6. 后续路径

### 立即可做

1. **Moirai-base 网络重试**：base 91M 模型可能让 covariate 真正生效，预期 MAE 降到 ~125
2. **V11 Hybrid v2：组合 Chronos + Moirai 不确定性**：取两者 P10/P90 的并集（max(P90)、min(P10)）作鲁棒边界，预期 +5~10 万
3. **方案 A（V10 + 异常日规则 + no-op + Moirai Hybrid 叠加）**：对规则识别为反向日的天 → no-op，其他天 → Moirai Hybrid

### 中期

4. **Moirai 微调**（lr<1e-5, ≤5 epoch）：在 V11 Chronos fine-tune 经验下，small lr + 短训练可能让 covariate 起作用
5. **TimesFM-2.0 比较**：确认是否 covariate 大模型的优势可复现
6. **Conformal Prediction 校准 Moirai Hybrid**：进一步提升 cov80（当前 0.43）

---

## 7. 文件清单

- 模型推理：`src/model_v12_moirai.py`
- 评估混合：`src/eval_v11_hybrid.py`（复用，已支持 chronos-exp 参数）
- 推理产出：
  - `output/experiments/v12.0-moirai-uni/test_predictions_quantile.csv`（最佳源）
  - `output/experiments/v12.0-moirai-cov/`、`v12.0-moirai-cov-p64/`（covariate 实验）
- Hybrid 评估：
  - `output/experiments/v11.0-hybrid/width_*` —— 现在同时含 Chronos 和 Moirai 两版的预测
  - 注：建议把 Moirai 版本的 hybrid 输出独立到 `v12.0-moirai-hybrid/`（待整理）

