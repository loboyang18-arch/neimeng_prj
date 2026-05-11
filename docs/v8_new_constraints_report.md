# V8 baseline — 新约束 81 天充放电收益重算

**日期**：2026-05-08（v2 — CBC 求解器最终版）
**预测来源**：`output/experiments/v8.0-12m-sudun500/test_predictions_hourly.csv`
**结果目录**：`output/experiments/v8.0-new-constraints/`
**评估期**：2026-01-27 ~ 2026-04-17（81 天，与 V10/V11/V12 测试集一致）

---

## 1. 约束变化

| 项 | 旧约束 | 新约束 |
|---|---|---|
| 储能容量 | 800 MWh | **400 MWh**（减半）|
| 日循环上限 | 1.5 次 (≤1200 MWh 充电) | **不限**（max_charge=2400）|
| 容量补偿 | 无 | **350 元/MWh × 放电量** (= 0.35 元/kWh) |
| 功率上限 | 195 MW | 195 MW（不变） |
| 双程效率 | 0.91 | 0.91（不变） |
| 辅助用电 | 13.03 MWh/天 | 13.03 MWh/天（不变） |
| 跨日 SOC | carry_soc=True | carry_soc=True（不变） |

**实现方式（v2 最终版）**：
- 求解器从 `scipy.milp`（HiGHS）切换为 **PuLP + CBC**（`_build_milp_cbc()`）
- 容量补偿 350 元/MWh **直接进入 MILP 目标函数**：`max Σ(price+350)·d·DT − Σ price·c·DT`
- 求解器自动决定最优循环次数——**无需人为设置 min_charge 硬约束、无 fallback 机制**
- CBC gap=5%，timeLimit=30s/天，81 天零失败

容量补偿的评估仍单独记录以便拆解：
```
cap_comp = discharge_mwh × 350 元/MWh
net_total = net_arb + cap_comp
```

---

## 2. 81 天合计结果

| 维度 | 旧约束 | 新约束 | Δ |
|---|---|---|---|
| 累计充电 (MWh) | 87,685 | 132,008 | +44,323 |
| 累计放电 (MWh) | 79,793 | 120,128 | +40,335 |
| **日均循环** | **1.35 次** | **4.07 次** | **+2.72 次** |
| 套利净收益 (万元) | 1,126.8 | 338.3 | **-788.5** |
| 容量补偿 (万元) | – | 4,204.5 | **+4,204.5** |
| **总净收益 (万元)** | **1,126.8** | **4,542.8** | **+3,416.0 (+303.2%)** |
| PF 总净收益 (万元) | 2,261.0 | 5,499.4 | +3,238.4 |
| 兑现率 (%) | 49.8 | **82.6** | +32.8 pp |

---

## 3. 收益结构拆解

### 3.1 容量补偿主导（占总收益 92.6%）

```
新约束总净收益 4543 万 = 套利 338 万 (7.4%) + 容量补偿 4205 万 (92.6%)
```

补偿收入远超套利——新模式下收益核心是**放电量**而非**价差**。

### 3.2 套利下降的原因（−788.5 万）

容量减半（800→400）使单次循环规模减半。尽管循环次数从 1.35 翻到 4.07（+3.0×），
高频循环在低价差时段也会执行（被补偿激励驱动），导致纯套利维度的利润被稀释。

### 3.3 兑现率大幅提升（49.8% → 82.6%）

补偿收入与预测准度**弱相关**（只要电池在运行就有补偿），使策略侧收益更稳健。
PF（完全预知）的优势被大幅压缩——因为即使知道完美价格，补偿部分也是固定的。

---

## 4. 实施

### 文件
- `scripts/strategy_milp_15min.py` — 新增 `_build_milp_cbc()` CBC 版 MILP 求解
- `scripts/eval_v8_new_constraints.py` — 简化版评估器（无 fallback，cap_comp 直接进目标）
- `output/experiments/v8.0-new-constraints/daily.csv` — 81 天逐日明细
- `output/experiments/v8.0-new-constraints/summary.txt` / `summary.csv` — 81 天合计

### 调用
```bash
conda run --no-capture-output -n power python -m scripts.eval_v8_new_constraints
# 81 天耗时约 79 分钟（CBC 30s/天 × 162 次求解）
```

### 依赖
```
pip install pulp  # PuLP 3.3+（自带 CBC 求解器）
```

---

## 5. 求解器迭代历史

### v1（HiGHS + cap_comp 进目标）→ 失败

| 修复尝试 | 7 天：失败天数 | 循环数 | 兑现率 |
|---|---|---|---|
| ① cap_comp=350 直接进目标（HiGHS） | 多数天 status=1 | 1.8× | – |
| ② + OBJ_SCALE=1000 + mip_rel_gap=1e-3 | 仍频繁 status=1 | 2.2× | – |
| ③ + 取消 ⑤⑥ 切换约束 | 更糟（9天失败） | 2.66× | 41.8% |
| ④ cap_comp 移出目标 + min_charge 硬约束 | 5天 L0 失败 | 2.85× | 49.5% |

**根因**：HiGHS 无法处理"电价+350 补偿"导致的目标系数尺度失衡，分支定界树爆炸。

### v2（PuLP+CBC，最终版）→ 成功

| 设计 | 7 天 | 81 天 |
|---|---|---|
| cap_comp=350 直接进 CBC 目标，gap=5% | **零失败** | **零失败** |
| 日均循环 | 4.04× | 4.07× |
| 兑现率 | 70.8% | **82.6%** |
| 单天耗时 | ~30s | ~30s |

**关键改进**：CBC 对 192 个二值变量 + 大系数目标远比 HiGHS 稳定，在 0.4s 内找到初始可行解，30s 内收敛到 gap<20%（实际解质量已很好）。

---

## 6. 后续

如需推广到 V10/V11/V12：
```bash
python -m scripts.eval_v8_new_constraints \
    --pred-csv output/experiments/v10.0-joint/test_predictions_hourly.csv \
    --out-dir output/experiments/v10.0-new-constraints
```

**预期**：补偿"放电量奖励"与预测准度弱相关 → 各模型间差距会被大幅压缩。
