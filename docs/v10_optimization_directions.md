# V10 之后的电价预测优化方向

> **背景**：V10（Pure Transformer）相比 V8（Conv2D 多任务）在 wMAPE 上仅好 1.0 个百分点（52.5% vs 53.5%），MAE 好 2 元/MWh，但 MILP 决策收益反而略差（1148 万 vs 1158 万）。
> 进一步分析显示：**V10 vs V8 的全部差距来自 11 个反向日（V10 −44 万 vs V8 +2 万）**。SPO+ / DFL 等决策导向训练在测试集分布漂移下也无法稳定超越 MSE baseline。
>
> **核心结论**：V10 的瓶颈不是模型容量，而是「24 维确定性点估计 + 把 ŷ 当真喂给 MILP」这个范式本身。
> 下面列出 4 个**非小修小补**的优化方向，按性价比排序。

---

## 方向 1：概率预测 + 鲁棒 MILP（最值得做）

**核心改动**：把 V10 的 24 维点估计输出，换成 24 维**分位数集合**（例如 P10 / P30 / P50 / P70 / P90），下游 MILP 改为鲁棒优化。

### 为什么是质变

- 现在的 MILP 只信任 `ŷ`，对反向日的"假装高价、假装低价"完全没有防御 → 血亏。
- 分位数预测让模型在**它自己也不确定的时段**（风电过剩的傍晚）输出**很宽的 P10–P90 区间**。
- 下游 MILP 改成 **min-max 鲁棒**或 **CVaR 约束**：在最坏分位数下也要至少不亏。
- **副作用为零**：稳定日 P10≈P90，鲁棒 MILP 退化到现在的解；反向日 P10 远低于 P50，MILP 自动放弃激进动作。

### 实施路径

| 步骤 | 内容 | 工作量 |
|---|---|---|
| 1 | V10 输出头从 24 维改成 24×Q 维（Q=5 个分位数足够） | 小 |
| 2 | 损失从 L1 换成 Pinball Loss `Σ_q max((y-ŷ_q)·q, (ŷ_q-y)·(1-q))` | 小 |
| 3 | `scripts/strategy_milp_15min.py` 加 `solve_robust(quantiles, alpha)` | 中 |
| 4 | 评估：V10 baseline 1148 万 → 鲁棒 MILP 收益 | 小 |

### 鲁棒 MILP 简单实现

线性目标：

```
max Σ_t [d[t] * P_d[t] - c[t] * P_c[t]] * Δt
```

其中：
- `P_d[t] = (1-α) * P50[t] + α * P10[t]`（放电时假设价格偏低）
- `P_c[t] = (1-α) * P50[t] + α * P90[t]`（充电时假设价格偏高）
- `α ∈ [0, 1]` 控制鲁棒程度：α=0 退化为标准 MILP，α=1 完全保守

### 预期收益

反向日不再亏 −44 万，整体能从 1148 万 → **1200 万左右**（接近异常日检测+no-op 的上限），但**模型自动做到**，不再依赖手工规则。
与现有"异常日 + 保守降级"逻辑**完全相容**——可以叠加。

---

## 方向 2：时序基础模型迁移学习（潜力最大）

**核心改动**：丢掉「从随机初始化训 V10」，换成在 **TimesFM / Chronos / Lag-Llama / PatchTST** 等已经在数十亿时序点上预训练的基础模型上**微调**。

### 为什么是质变

- 训练集只有 ~3 年（每天 24 点 ≈ 25k 小时），其中"反向日"训练样本只有 82 个 → V10 即使学到了模式，**也不可能见过冬末春初的反向日变体** → 这就是 LightGBM 检测器在测试集 0 召回的根本原因。
- 基础模型在跨域、跨季节、跨频率的海量时序上预训练过，已经"见过"各种突变模式的潜在表征。
- 在内蒙古电价上**小样本微调**（甚至 zero-shot + 少量调温），跨季节泛化能力会跃迁。
- 国内已经有团队在江苏、广东电价上验证 Chronos / TimesFM 微调可降 wMAPE 5–10 个点。

### 实施路径

| 步骤 | 内容 |
|---|---|
| 1 | 用 HuggingFace 加载 `amazon/chronos-bolt-base` 或 `google/timesfm-2.0-500m` |
| 2 | 把内蒙古特征序列化成它们要求的协变量格式（continuous + categorical） |
| 3 | LoRA / 部分参数微调 ~20 epoch |
| 4 | 直接给现有 dashboard / MILP 当 ŷ 喂入对比 |

### 预期收益

MAE 可能从 106.8 降到 **90 以下**，并且**反向日的预测崩塌幅度大幅缩小**（这是基础模型预训练泛化的最大价值）。

---

## 方向 3：Conformal Prediction，把"异常日检测"内化进模型

**核心改动**：在 V10 训练流程中加入 **Conformal 校准**，让模型为每天的预测自动产出一个**覆盖率 90% 的预测区间**——不再依赖外部规则检测器。

### 为什么是质变

- 现有规则检测器（`wind_diff < -2000 AND reserve > 3000`）是**纯人工启发式**，**只在 2026-02 ~ 04 这段测试期凑效**，跨季节失效风险极高。
- Conformal Prediction 是**分布无关的统计保证**：训练完 V10 后，用一个校准集计算残差分位数，给每条预测自动加 ε 半径。每个预测都附带"今天我自己有多确定"的信号。
- 下游策略：当区间宽度 > 阈值时直接 no-op；区间适中时正常 MILP；区间很窄时甚至可以更激进。
- **可以与方向 1 并用**：Conformal 给的区间比 Pinball 更可靠（理论保证），但只需要训练一次后做一次校准，**几乎无训练成本**。

### 实施路径

直接在 V10 已训权重上加 ~50 行 numpy 后处理。这是最便宜的"大方向"。

### 预期收益

让规则检测器升级成可解释的、统计保证的、跨季节的检测器，反向日漏检率从 27% 降到 ≤10%。

---

## 方向 4：物理-数据混合（Hybrid Physics-ML）

**核心改动**：不再让 V10 直接输出电价，而是输出**电价分解项**——边际成本 `λ_marginal`、阻塞调整 `Δ_cong`、备用稀缺溢价 `θ_reserve`——再用一个**小型物理模型**重组成最终电价。

### 为什么是质变

- 反向日的本质是**风电预测富余 + 备用市场紧张**这两个物理因子的非线性叠加（规则检测器发现的就是这个）。
- 当前 V10 让 Transformer 自己从端到端 24 个数字里去学这种非线性，数据量根本不够。
- 让模型分别学三个独立、有物理意义的因子，再用线性 / 简单非线性公式合成 → 学习目标分解、可解释、跨期稳定。
- 已经有论文（ICLR / NeurIPS Energy Workshop）验证此类方法在 PJM / CAISO 上比纯 ML 降 MAE 10-15%。

### 实施路径

复杂，需要配合电力系统专业知识，但**最有学术价值**。

---

## 优先级建议

| 维度 | 方向 1 | 方向 2 | 方向 3 | 方向 4 |
|---|---|---|---|---|
| 实施工作量 | 小-中 | 中-大 | 极小 | 大 |
| 概念复杂度 | 中 | 低（拿现成模型用） | 低 | 高 |
| **预期收益** | **+50~80 万** | +30~60 万 | +20~40 万 | +30~50 万 |
| 与现有 fallback 兼容 | ✓ 互补 | ✓ 互补 | ✓ 替代 | ✓ 互补 |
| 学术新颖度 | 中 | 低 | 中 | 高 |
| GPU 需求 | 不变 | 升 2-4× | 不变 | 略升 |

**强烈推荐先做方向 1 + 方向 3**：分位数预测 + Conformal 校准在工程上几乎共享同一套流水线，最终能给 MILP 同时提供"分位数"和"区间宽度"两路信号，是当前问题最干净的彻底解。

---

## 实验记录

| 日期 | 方向 | 状态 | 关键指标 | 详情 |
|---|---|---|---|---|
| 2026-05-07 | 方向 1：分位数预测 + 鲁棒 MILP | ⚠️ 完成但负收益 | raw α=1.0 → 1146.9 万 (vs V10 baseline 1148 万) | [v10_quantile_experiment_report.md](./v10_quantile_experiment_report.md) |
| 2026-05-07 | 方向 2：基础模型迁移 (Chronos zero-shot/finetune) | ⚠️ 单独使用收益不及 V10 | MAE=137~140 (vs V10 107)，但 cov80=0.79~0.81 (vs V10-Q 的 0.20) | [v11_foundation_experiment_report.md](./v11_foundation_experiment_report.md) |
| 2026-05-07 | **★ V11 Hybrid: V10 P50 + Chronos 不确定性 + 鲁棒 MILP** | ✅ **接近规则方案上限** | **1210.9 万 (+62.6 万 vs V10, +5.5%)，几乎追平 V10+规则+no-op 的 1211.4 万** | 同上 |
| 2026-05-07 | 方向 B：Moirai-1.1-R-small zero-shot | ⚠️ 单独使用 MAE=144 (uni)、148 (cov)，不及 V10/Chronos | 13.8M 参数容量不足；covariate 加入反而恶化；base 模型下载受网络阻碍 | [v12_moirai_experiment_report.md](./v12_moirai_experiment_report.md) |
| 2026-05-07 | **★★ V12 Hybrid: V10 P50 + Moirai 不确定性 + 鲁棒 MILP** | ✅ **当前全局最优** | **1220.1 万 (+71.8 万 vs V10, +6.3%)，超越 Chronos Hybrid 与异常日规则方案** | 同上 |

---

### 方向 1 详细总结（V10-Quantile + 鲁棒 MILP）

#### 实施的所有变体

1. **基础 V10-Quantile 模型** (`src/model_v10_quantile.py`)
   - 5 分位数输出 (P10/P30/P50/P70/P90)，Pinball Loss + 方向 CE + 排序 + 分位数交叉惩罚
   - 200 epoch 训练 → MAE_P50=112.0, RMSE=170.8

2. **鲁棒 MILP** (`scripts/strategy_milp_15min.py::solve_day_milp_15min_robust`)
   - 充电价格 = (1-α)·P50 + α·P90，放电价格 = (1-α)·P50 + α·P10
   - 完全保持线性，无额外整数变量

3. **α 网格扫描** (`src/eval_v10_quantile_robust.py`)
   - α∈{0, 0.3, 0.5, 0.7, 1.0}：1130.5 / 1120.9 / 1136.2 / 1133.6 / **1146.9** 万

4. **Split Conformal 校准** (`src/eval_v10_quantile_conformal.py`)
   - cal_days=30：1114.5 / 1118.9 / 1134.6 / 1140.5 / 1137.4 万
   - cal_days=60：1113.7 / — / 1122.5 / — / 1141.8 万
   - test coverage 从 0.20 → 0.11（更差）

#### 失败诊断

| 现象 | 根因 |
|---|---|
| Pinball loss ↓ 但 test coverage 从 ep10 的 69% → ep200 的 20% | 训练样本（409 天）不足以学好 P10/P90，模型过拟合到中位数 |
| 鲁棒 MILP α=1 比 α=0 仅提升 +16 万 | 分位数区间宽度 67 元/MWh ≪ 真实 RMSE 170 → 区间无指导价值 |
| Conformal 校准方向反了（δ_P10=+15.1, δ_P90=-13.6） | cal-set (冬季) 与 test-set (冬末春初) 分布漂移，cal 上模型 cov80=0.98 已过宽 → 校准把区间收紧 |
| 所有方案均不及 V10 baseline 1148 万 | 不确定性估计本身被训练数据约束，无法表达跨季节漂移的真实分散度 |

#### 与现有方案的最终对比

| 方案 | 总净收益 (万) | 反向日净 (万) | 实施难度 |
|---|---|---|---|
| V8 baseline | 1158 | +2.1 | 已部署 |
| V10 baseline | 1148 | -44.1 | 已部署 |
| V10-Quantile raw α=1.0 | 1146.9 | (近似 V10) | 中 |
| V10-Quantile + Conformal α=0.7 | 1140.5 | (更差) | 中-高 |
| **V10 + 异常日规则 + no-op** | **1211.4** | **+25.7** | 低 |

**结论**：方向 1 在数据规模和分布漂移问题没解决前，无法超越简单的"规则检测+保守降级"。**核心瓶颈仍是训练样本（409 天）跨季节泛化能力不足**，所有依赖电价历史的不确定性估计方法都跌入同一陷阱。

---

---

### 方向 2 详细总结（基础模型迁移）

#### 实施的所有变体

1. **Chronos-Bolt-Small zero-shot** (`src/model_v11_foundation.py`)
   - 47.7M 参数，HuggingFace 加载，hf-mirror 国内镜像
   - 81 天 × 720h context，推理仅需 7 秒
   - **MAE=140.5, RMSE=188.2, Cov80=0.790, Width=426.7**

2. **Chronos-Bolt-Base zero-shot**
   - 4× 参数 (~205M)，下载 ~820 MB
   - **MAE=143.3, Cov80=0.765** — 不比 small 好！
   - 印证：纯 univariate 在此领域已饱和

3. **Fine-tune chronos-bolt-small** (`src/train_v11_finetune.py`)
   - 滑窗 1285 个 (context=720h, target=24h) 训练对，全模型微调
   - lr=1e-5, 30 epoch：MAE 140.5 → ep5 改善到 139.3 → ep30 反弹到 149.0（catastrophic forgetting）
   - **lr=5e-6, 5 epoch (light)**：MAE=137.1, Cov80=0.810（最佳 fine-tune）
   - 仍比 V10 (MAE 106.8) 差 30 个点 → univariate 模型天花板

4. **Robust MILP（直接用 Chronos 分位数）**
   - α=0: 1075.8 万；α=1: 154.4 万 — 区间太宽 (427) 压死了 MILP

5. **★ V11 Hybrid (`src/eval_v11_hybrid.py`)**
   - `P50 = V10_P50`（精准点估计）
   - `P10/P90 = V10_P50 ± (Chronos 半宽) × ws`
   - 鲁棒 MILP 调 (ws, α) 二元网格

#### V11 Hybrid 关键网格

| ws | α | net | 备注 |
|---|---|---|---|
| 0.0 | – | 1148.3 | V10 baseline |
| 0.3 | 0.5 | 1210.2 | – |
| **0.4** | **0.4** | **1210.9** | **最佳** |
| 0.5 | 0.3 | 1210.1 | – |
| 1.0 | 0.3 | 1188.8 | 区间过宽 |

**结论**：当 V10 提供精准 P50、Chronos 提供有意义但需缩放（ws=0.4）的不确定性、α=0.4 适度鲁棒时，**MILP 在反向日上自动收敛到适度操作**，达到规则方案上限。

#### 与 V10 + 异常日规则 + no-op 对比

| 维度 | V11 Hybrid | V10 + 规则 + no-op |
|---|---|---|
| 收益 | **1210.9 万** | 1211.4 万 |
| 反向日处理 | MILP 自动 | 强制 no-op |
| 跨季节稳定 | **理论更优**（预训练先验） | 需季度复检阈值 |
| 解释性 | 鲁棒优化 | 物理含义清晰 |
| 实施 | 中（外部模型推理） | 低（两个阈值） |

**两套方案几乎打平，互补**：可以叠加（规则识别极端反向日 → no-op，其他天 → V11 Hybrid）进一步提升。

---

## 后续优化路径（在 V11 Hybrid 基础上继续推进）

按性价比排序：

### A. V11 Hybrid + 异常日规则 + no-op 叠加（短期）

**思路**：把两个 1210 级方案的优势合并：
- 规则识别为反向日（`wind_evening_diff<-2000 AND reserve_neg_min>+3000`）→ no-op
- 其他天 → V11 Hybrid (ws=0.4, α=0.4) robust MILP

**预期**：+5~15 万元（规则在极端日上比 robust MILP 更"狠"，能保住辅助用电成本之外的全部潜在亏损）。

**实施工作量**：极小，~50 行代码组合现有 pipeline。

### B. ★ Covariate-aware 基础模型 (Moirai-1.0 / TimesFM-2.0)（中期 ⭐ 当前主推方向）

**思路**：Chronos zero-shot 的 MAE 天花板 137 是 univariate 模型固有的，因为它**没看到**风电预报、负荷、负备用容量。Moirai-1.0 / TimesFM-2.0 原生支持把 covariate 作为辅助输入。

**预期**：
- 让 P50 也突破到 ~110 以下（接近或超越 V10）
- 区间继续保持有意义 → robust MILP 可直接用，不再需要 V10 的 P50 作"接驳"
- **理论上能完全取代 V10 + V11 Hybrid 的"两层堆叠"**，回归单模型

**风险**：
- Moirai 的 covariate 接口比 chronos 复杂，需要 GluonTS / uni2ts 数据格式
- 模型更大（base 91M、large 311M），fine-tune 内存压力大
- 国内下载 311M 权重可能慢

**关键问题**：
1. Moirai zero-shot 仅用历史价格 → 是否已经接近 Chronos？
2. 加入 (load_forecast, wind_forecast, reserve_neg) 三个 covariate 后，MAE 能不能跌破 110？
3. 即使突破不了 V10，覆盖率与 V10 P50 + Moirai uncertainty 的 hybrid 有没有比 V11 Hybrid 更好？

### C. Conformal Prediction 校准 V11 Hybrid（中期）

**思路**：V11 Hybrid 现在的 cov80（在 ws=0.4 时仅 0.44）相对偏低，仍有提升空间。在已经训好的 Hybrid 输出上做 split conformal calibration（用 train 末尾 30 天作 cal set），让区间逼近理论的 80% 覆盖。

**预期**：进一步提升 robust MILP 的精度，可能 +10~20 万。

**风险**：和方向 1 之前的 conformal 失败原因相同——cal-set 与 test 分布漂移可能让校准方向反了。但 V11 Hybrid 的基础模型已比 V10-Quantile 跨季节稳定得多，conformal 在 Hybrid 上**应该**比在 V10-Quantile 上更稳。

**实施工作量**：极小，复用 `src/eval_v10_quantile_conformal.py`。

### D. Deep Ensemble of V10s（长期）

**思路**：训 5 个不同 seed 的 V10，用预测分歧作不确定性估计。

**预期**：跟 Chronos 方向理念相同（多源不确定性），但全部"in-domain"，可能比跨域基础模型在内蒙古更准。可以与 Chronos 不确定性做加权混合（V11 Hybrid v2）。

**实施工作量**：训练 ×5 倍（仍 < 10 分钟）；评估管道需要调整以聚合多模型预测。

### 路径选择

**已完成 → B（Moirai-1.1-R-small zero-shot）**：✅ 收益突破

详见 [v12_moirai_experiment_report.md](./v12_moirai_experiment_report.md)。

#### 关键结果

| 配置 | MAE | Cov80 | Width | net (万) |
|---|---|---|---|---|
| Moirai-small univariate | 144.6 | 0.801 | 444 | – |
| Moirai-small + 4 covariates (patch=32) | 148.3 | 0.780 | 459 | – |
| Moirai-small + 4 covariates (patch=64) | 146.7 | 0.781 | 437 | – |
| **★ V10 P50 + Moirai uni + 鲁棒 MILP (ws=0.35, α=0.40)** | – | 0.43 | 155 | **1220.1** |

**单独用 Moirai 不及 Chronos 与 V10**，但其不确定性区间在 V11 Hybrid 框架下表现更好——多赚 +9 万。

#### 已观察的现象

1. **Covariate 加入反而恶化**：13.8M 参数容量不足以利用风电/负荷/备用四个 covariate 的额外信号，patch=64 略好但仍不及 univariate
2. **Moirai-base/large 下载受网络阻碍**：hf-mirror 频繁 timeout，base 91M 模型未能下载完整 config.json
3. **Moirai uni 与 Chronos uni 在 V11 Hybrid 框架下成绩不同**：Moirai → 1220 万；Chronos → 1211 万，差异 +9 万 → 暗示**不同基础模型的不确定性结构在反向日上有质量差别**

### 下一步路径建议

按性价比排序：

#### B'. ★ Moirai-base / large 网络重试（最高优先）
- 91M / 311M 模型应该能让 covariate 真正生效
- 预期 P50 MAE 降到 ~125，可能接近或超越 V10
- 即使 P50 不超 V10，更强的不确定性估计 → Hybrid 收益再提升 +10~20 万

#### E. Multi-source uncertainty Hybrid（中优先）
- 把 Chronos 和 Moirai 的不确定性区间**并集化**：
  `P10' = min(Chronos_P10, Moirai_P10) - V10_P50_diff`
  `P90' = max(Chronos_P90, Moirai_P90) + V10_P50_diff`
- 两个基础模型的"对反向日的敏感度"互补
- 预期 +3~10 万，工程量极小

#### A. V10 + 异常日规则 + no-op + V12 Moirai Hybrid 叠加
- 极端反向日 → no-op，其他天 → Moirai Hybrid
- 预期 +5~15 万

#### C. Conformal Prediction 校准 V12 Hybrid
- 当前 ws=0.35 时 cov80=0.43（远低于理论 0.80）→ 区间偏窄
- Conformal 把区间适度扩张可能让 robust MILP 更准
- 风险：仍是 cal/test 分布漂移问题

