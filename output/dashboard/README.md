# 看板数据文件说明

> **目录**：`output/dashboard/`  
> **更新方式**：运行 `conda run -n power python scripts/export_dashboard_data.py`  
> **数据来源**：苏敦独立储能一期（200 MW / 800 MWh），回测区间 2026-01-27 ～ 2026-04-17（排除 1月25-26日调试日及4月18日不完整日）  
> **决策策略**：15分钟 MILP（跨日SOC模式），详见 `docs/battery_dispatch_strategy.md`

---

## 文件列表

| 文件名 | 行数 | 粒度 | 说明 |
|--------|------|------|------|
| `15min_timeseries.csv` | 7,776 | 15 分钟 / 时段 | 核心时序表，充放电功率 + 电价 + 收益 |
| `daily_summary.csv` | 81 | 日 | 逐日充放电量、收益、PF 对照 |
| `weekly_summary.csv` | 12 | 周 | 逐周收益汇总，含实际现货对比 |
| `model_predictions.csv` | 2,040 | 小时 | 预测模型输出与实际电价对比 |
| `strategy_comparison.csv` | 5 | 策略 | 各方案横向对比（含理论最优） |
| `metrics_strategy_daily.csv` | 81 | 日 | **策略** 逐日评价指标（7.1 + 7.2 共 12 项） |
| `metrics_actual_daily.csv` | 81 | 日 | **实际运营** 逐日评价指标（7.1 + 7.2 共 12 项） |
| `metrics_summary.csv` | — | 全期 | 策略 vs 实际 汇总对比（含 1.4 集中度、1.5 循环 P50/P90） |

---

## 字段说明

### 1. `15min_timeseries.csv` — 15 分钟时序主表

每行代表一个 15 分钟时段的调度状态与收益，共 81 天 × 96 时段 = 7,776 行。

| 字段 | 类型 | 单位 | 说明 |
|------|------|------|------|
| `datetime` | datetime | — | 时段起始时刻，格式 `YYYY-MM-DD HH:MM:00` |
| `date` | string | — | 日期，格式 `YYYY-MM-DD` |
| `slot` | int | — | 当日时段序号，0 = 00:00，95 = 23:45 |
| `hour` | int | — | 整点小时（0 ～ 23） |
| `minute` | int | — | 分钟（0 / 15 / 30 / 45） |
| `state` | string | — | 运行状态：`充电` / `放电` / `待机` |
| `charge_mw` | float | MW | 本时段充电功率（电网侧输入） |
| `discharge_mw` | float | MW | 本时段放电功率（电网侧输出） |
| `net_power_mw` | float | MW | 净功率 = `discharge_mw - charge_mw`（正=向电网送电，负=从电网取电） |
| `soc_mwh` | float | MWh | 本时段末电池荷电状态（0 ～ 800） |
| `charge_energy_mwh` | float | MWh | 本时段充电电量 = `charge_mw × 0.25` |
| `discharge_energy_mwh` | float | MWh | 本时段放电电量 = `discharge_mw × 0.25` |
| `pred_price` | float | 元/MWh | 预测节点电价（小时级，每4时段相同） |
| `actual_price` | float | 元/MWh | 实际15分钟节点电价（来自日清算文件） |
| `slot_revenue` | float | 元 | 本时段净收益 = `(discharge_energy - charge_energy) × actual_price`（正=收益，负=成本） |

> **注**：`slot_revenue` 为时段级毛收益，未扣除日固定辅助用电成本（13.03 MWh/天）。

---

### 2. `daily_summary.csv` — 逐日汇总

每行为一个交易日的调度汇总，共 81 行。

| 字段 | 类型 | 单位 | 说明 |
|------|------|------|------|
| `date` | string | — | 日期 |
| `week` | string | — | ISO 周次，格式 `YYYY-WNN` |
| `charge_window` | string | — | 充电时间段，如 `04:00–22:00` |
| `discharge_window` | string | — | 放电时间段，如 `06:00–23:45` |
| `charge_mwh` | float | MWh | 当日总充电量（电网侧） |
| `discharge_mwh` | float | MWh | 当日总放电量（电网侧） |
| `soc_end_mwh` | float | MWh | 日末荷电状态（跨日保留量；日清零模式为 0） |
| `avg_charge_price` | float | 元/MWh | 充电加权平均实际电价 |
| `avg_discharge_price` | float | 元/MWh | 放电加权平均实际电价 |
| `gross_yuan` | float | 元 | 毛收益 = 放电收入 − 充电成本 |
| `aux_cost_yuan` | float | 元 | 辅助用电成本 = 日均实际电价 × 13.03 MWh |
| `net_yuan` | float | 元 | 净收益 = 毛收益 − 辅助用电成本 |
| `pf_net_yuan` | float | 元 | 完全预知（PF）净收益（用实际电价求解同一 MILP 的理论上限） |
| `realization_pct` | float | % | 兑现率 = `net_yuan / pf_net_yuan × 100` |

---

### 3. `weekly_summary.csv` — 逐周对比

每行为一个 ISO 周的汇总，共 13 周。

| 字段 | 类型 | 单位 | 说明 |
|------|------|------|------|
| `week` | string | — | ISO 周次，格式 `YYYY-WNN` |
| `days` | int | — | 本周有效交易天数 |
| `charge_mwh` | float | MWh | 周充电总量 |
| `discharge_mwh` | float | MWh | 周放电总量 |
| `net_wan` | float | 万元 | 策略净收益（万） |
| `pf_net_wan` | float | 万元 | PF 理论最优净收益（万） |
| `actual_spot_wan` | float | 万元 | 实际现货运营净收益（万），来源：日清算结算文件 |
| `realization_pct` | float | % | 兑现率 = `net_wan / pf_net_wan × 100` |
| `vs_actual_wan` | float | 万元 | 策略超出实际现货的收益差额（正=策略更优） |
| `net_yuan` | float | 元 | 策略净收益（元，与 `net_wan × 10000` 等价） |
| `pf_net_yuan` | float | 元 | PF 净收益（元） |
| `actual_spot_yuan` | float | 元 | 实际现货净收益（元） |

> **注**：W04（1/25-1/26）为调试期，`actual_spot_wan` 为空。

---

### 4. `model_predictions.csv` — 预测模型输出

V8 模型（Conv2D 多任务网络）对苏敦 500kV.1M 节点电价的小时级预测结果，共 2,040 行（85 天 × 24 小时）。

| 字段 | 类型 | 单位 | 说明 |
|------|------|------|------|
| `datetime` | datetime | — | 小时起始时刻，格式 `YYYY-MM-DD HH:00:00` |
| `date` | string | — | 日期 |
| `week` | string | — | ISO 周次 |
| `hour` | int | — | 小时（0 ～ 23） |
| `actual_price` | float | 元/MWh | 实际小时均价（15分钟价格的算术平均） |
| `pred_price` | float | 元/MWh | 模型预测小时均价 |
| `error` | float | 元/MWh | 预测误差 = `pred_price − actual_price`（正=高估，负=低估） |
| `abs_error` | float | 元/MWh | 绝对误差 = `|error|` |

**模型精度（测试集）**：MAE = 108.57 元/MWh，RMSE = 153.05 元/MWh，相关系数 = 0.744

---

### 5. `strategy_comparison.csv` — 各策略横向对比

每行为一种策略方案，共 5 行。

| 字段 | 类型 | 单位 | 说明 |
|------|------|------|------|
| `策略` | string | — | 策略名称 |
| `回测天数` | int | 天 | 有效回测天数 |
| `回测区间` | string | — | 起止日期 |
| `充电总量MWh` | float | MWh | 回测期充电总量 |
| `放电总量MWh` | float | MWh | 回测期放电总量 |
| `毛收益万` | float | 万元 | 放电收入 − 充电成本 |
| `辅助用电成本万` | float | 万元 | 日维护用电成本合计 |
| `净收益万` | float | 万元 | 毛收益 − 辅助用电成本 |
| `理论最优PF万` | float | 万元 | 完全预知（Perfect Foresight）净收益上限；实际现货无此项 |
| `兑现率%` | float | % | `净收益 / 理论最优PF × 100`；实际现货无此项 |
| `全年外推亿` | float | 亿元 | 按日均净收益线性外推至 365 天 |
| `全年PF外推亿` | float | 亿元 | PF 的全年外推；实际现货无此项 |
| `数据来源` | string | — | 原始 CSV 文件名 |

**策略说明**：

| 策略名 | 决策粒度 | SOC模式 | PF基准粒度 |
|--------|---------|---------|-----------|
| 15min MILP（跨日SOC） | 15 分钟 | 日末可结转 | 15 分钟实际价 |
| 15min MILP（日清零） | 15 分钟 | 强制日末清零 | 15 分钟实际价 |
| 小时级 MILP | 1 小时 | 强制日末清零 | 小时实际价 |
| 启发式4h窗口（方案B） | 1 小时（固定窗口） | 强制日末清零 | 固定4h窗口 |
| 实际现货运营 | — | — | — |

---

---

### 6. `metrics_strategy_daily.csv` / `metrics_actual_daily.csv` — 逐日评价指标

两文件结构相同，分别对应策略回测与实际运营，统计区间 2026-01-27 ～ 2026-04-17（各 81 行）。

**基础字段**（与 `daily_summary.csv` 含义一致）：

| 字段 | 类型 | 单位 | 说明 |
|------|------|------|------|
| `date` / `week` | string | — | 日期 / ISO 周次 |
| `charge_mwh` | float | MWh | 当日充电量（电网侧） |
| `discharge_mwh` | float | MWh | 当日放电量（电网侧） |
| `gross_yuan` | float | 元 | 毛收益 |
| `aux_cost_yuan` | float | 元 | 辅助用电成本 |
| `net_yuan` | float | 元 | 净收益 |
| `avg_charge_price` | float | 元/MWh | 充电加权均价 |
| `avg_discharge_price` | float | 元/MWh | 放电加权均价 |
| `opt_charge_4h_price` | float | 元/MWh | 当日最优连续 4h 充电窗口均价（基准） |
| `opt_discharge_4h_price` | float | 元/MWh | 当日最优连续 4h 放电窗口均价（基准） |

**评价指标字段**：

| 字段 | 指标编号 | 单位 | 说明 |
|------|---------|------|------|
| `m11_unit_power_rev_yuan_mw` | 1.1 | 元/MW | 单位功率收益 = 毛收益 / 195 MW |
| `m12_energy_rev_yuan_mwh` | 1.2 | 元/MWh | 度电收益 = 毛收益 / 吞吐电量 |
| `m13_net_energy_rev_yuan_mwh` | 1.3 | 元/MWh | 度电净收益 = 净收益 / 吞吐电量 |
| `m21_spread_utilization_pct` | 2.1 | % | 放电价差利用率 = 实际价差 / 最优连续4h价差 |
| `m22_discharge_capture_pct` | 2.2 | % | 放电价格百分位：放电均价在当日96个时段中的百分位排名（越高越好，100%=放在最高价，50%=放在日中位数）|
| `m23_charge_capture_pct` | 2.3 | % | 充电价格百分位：充电均价在当日96个时段中的百分位排名（越低越好，0%=充在最低价，50%=充在日中位数）|
| `m24_avg_spread_yuan_mwh` | 2.4 | 元/MWh | 平均充放电价差 = 放电均价 − 充电均价 |
| `m25_rte_breakeven_yuan_mwh` | 2.5 | 元/MWh | 往返效率损耗盈亏门槛 = (1−0.91) × 放电均价，价差低于此值则亏损 |
| `m26_top_dis_capture_pct` | 2.6 | % | Top-16槽放电高价抓取率：放电量在最贵16个15min时段中的占比 |
| `m27_bot_chg_capture_pct` | 2.7 | % | Bottom-16槽充电低价抓取率：充电量在最便宜16个15min时段中的占比 |

> **指标说明**：
> - 1.4 Top-N 收益集中度 和 1.5 单次循环盈亏分布为全期汇总指标，见 `metrics_summary.csv`
> - 2.2/2.3 均为百分位指标：2.2 越高越好（放电集中在高价时段），2.3 越低越好（充电集中在低价时段）
> - 最优4h窗口为连续16个时段的滑窗最优；当最优充电窗口均价 ≤ 0（负电价日）时该日 2.1/2.2/2.3 置空

---

### 7. `metrics_summary.csv` — 全期汇总对比

竖排指标对比表，每行一个指标，列为「策略值」与「实际值」。

除逐日指标的均值外，还包含：

| 指标 | 编号 | 说明 |
|------|------|------|
| Top10%/20% 收益集中度 % | 1.4 | 前 N% 高收益日净收益占总净收益的比例 |
| 单循环 P50/P90/均值（万元） | 1.5 | 每等效满充满放循环的净盈亏（加权分位数/加权均值） |
| 等效循环总次数 | 1.5 | 全期等效满充满放循环总次数 |

> **等效循环定义**：等效满充满放循环次数 = 放电总量 / 额定容量（800 MWh）。单循环盈亏以每天为观测单位（该天净收益 / 当日等效循环数），P50/P90/均值均按当日等效循环数加权，保证 `等效循环总次数 × 单循环均值 ≈ 总净收益`。

---

## 公共约定

- **节点电价**：苏敦 500kV.1M 省内实时节点电价，来源：`日清算结果查询电厂侧(1)_副本.xlsx`
- **辅助用电**：每日固定 13.03 MWh，按当日平均实际节点电价折算成本
- **电池效率**：双程 91%，单程 √0.91 ≈ 95.39%；充电侧损耗体现在 SOC 动态，放电侧损耗体现在 SOC 消耗
- **PF（完全预知）**：用当日真实 15 分钟电价（或小时均价）作为已知条件，求解同一 MILP 模型所得理论上限，仅用于评估策略兑现率，不代表可实际达到的收益
- **全年外推**：`日均净收益 × 365`，未考虑季节性差异，仅供量级参考
