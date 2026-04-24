# 内蒙 DWS 入库与 V8.0-12m 设计说明

本文档供后续 agent 快速了解：**首页数据如何入库成 `dws_15min_features.csv`**、**苏敦如何合并**、以及 **`model_v8_multitask`（v8.0-12m）如何用这些列**。实现以仓库内代码为准。

---

## 1. 时间轴与 `ts` 约定

- 首页各 CSV 的「查询日期 + 时点」在源侧表示 **当日 96 点：00:15 … 24:00**（其中 `24:00` 会先解析为**次日 00:00** 再参与计算）。
- 入库时通过 `neimeng_ts.parse_homepage_query_clock` 与 `shift_source_ts_to_dws_grid`：**整体平移 −15 分钟**，使 DWS 中该「查询日」的数据落在 **当日 00:00 … 23:45** 的 96 个 15 分钟格上，与 v4–v8 的 `date_range(当日 0 点, periods=96, freq="15min")` 一致。
- 苏敦 `内蒙苏敦站.csv` 的 `日期`+`时间` 解析后同样经 `shift_source_ts_to_dws_grid` 与 DWS 对齐。

相关代码：`src/neimeng_ts.py`，`src/build_dws_15min_features.py`，`src/merge_sudun_prices_into_dws.py`。

---

## 2. 首页入库（`src/build_dws_15min_features.py`）

### 2.1 输入与合并

- 数据源目录：`source_data/首页/` 下多个 CSV（负荷、新能源、东送、备用、电价曲线等）。
- 各表按 `ts` **外连接**合并；缺测为 **NaN**。
- 标准输出列顺序由 **`BASE_COL_ORDER`** 定义；不在清单中的源列会告警并丢弃（例如 `东送计划实测` 若未写入 `SOURCES` 则不会进宽表）。

### 2.2 电价相关列（首页）

当前从 `电价曲线.csv` 映射进 DWS 的列包括：

| DWS 列名 | 源列名 |
|----------|--------|
| `price_unified` | 全网统一出清电价 |
| `price_hbd` | 呼包东统一出清电价 |
| `price_hbx` | 呼包西统一出清电价 |
| `price_dayahead_preclear_energy` | 日前预出清电能价格 |

源侧电价多为**按小时（或稀疏时点）**一条记录；对齐到 96 格后，**未观测的 15 分钟格为 NaN**。

### 2.3 首页电价日内补齐（仅这四列）

- 模块：`src/dws_intraday_price_fill.py` → `intraday_ffill_bfill(df, col_names)`  
- 逻辑：按 **`ts` 的自然日**分组，对列依次 **`ffill` 再 `bfill`**（就地修改）。  
- 目的：在「小时价在小时内可视为常数」的约定下，把**当日已有观测**铺到该日 DWS **已有行**的 96 个格点上，便于作为稠密特征进模型。  
- **不**为没有 DWS 行的日期发明新行。

当前参与补齐的列（常量 `HOMEPAGE_PRICE_FILL_COLS`）：

- `price_hbd`, `price_hbx`, `price_dayahead_preclear_energy`, `price_unified`

调用位置：`build_base_dws()` 中在列裁剪为 `BASE_COL_ORDER` 之后、按 `load_forecast` 做尾部裁剪之前。

### 2.4 尾部裁剪

- 以 **`load_forecast`** 为主时间轴：从表尾删除 `load_forecast` 为 NaN 的行，直到最后一行非空。

### 2.5 输出

- 路径：`output/dws_15min_features.csv`  
- 入口：`python3 -m src.build_dws_15min_features`（可选 `--no-backup`、`--no-sudun`）。

---

## 3. 苏敦合并（`src/merge_sudun_prices_into_dws.py`）

### 3.1 纳入列与节点

- 仅节点 **`内蒙.苏敦站/500kV.1M`**（与源 CSV `节点名称` 完全一致）。
- 写入 DWS 的 3 列：

  - `price_sudun_500kv1m_nodal`（节点电价）
  - `price_sudun_500kv1m_energy`（电能价格）
  - `price_sudun_500kv1m_cong`（阻塞价格）

- 同一 `ts` 多行取 `first()`；再与 DWS **`merge(..., how="left", on="ts")`**。

### 3.2 明确不做的事

- **不对苏敦三列做日内 `ffill/bfill` 补齐**。缺测保持 NaN，避免在入库阶段伪造未披露时刻的价格。
- 合并前会删除旧版 6 列名（`*_n1` / `*_mean`），避免与新版并存。

### 3.3 调用关系

- 由 `build_dws_15min_features.run_reingest` 在写出基础宽表后自动调用（若存在 `source_data/内蒙苏敦站.csv` 且未 `--no-sudun`）。
- 苏敦合并后仍会按 `load_forecast` 再裁一次尾部（读回 CSV 后循环删尾）。

---

## 4. 入库清单 vs 模型特征（重要）

- **`BASE_COL_ORDER`（入库）**：决定 **CSV 里有哪些列**。  
- **`model_v8_multitask.py` 中的 `LAG0_COLS` / `LAG1_COLS` / `LAG2_COLS` 与 `TARGET_COL`**：决定 **V8 读哪些列进张量**。  
- 二者 **不会自动同步**；新增 DWS 列后，若要让 V8 使用，必须改 V8 的列配置或通过环境变量追加。

---

## 5. V8.0-12m（`src/model_v8_multitask.py`）

### 5.1 默认目标（回归标签）

- **`TARGET_COL`**：环境变量 `NM_V8_TARGET`，默认 **`price_unified`**。  
- 日级 `day_targets`：取当日 **24 个整点**对应的 `TARGET_COL` 值（由 96 点中整点槽位索引取出）；仅当 **24 小时均有限值** 时该日进入有效样本集。

### 5.2 辅助标签（方向）

- 对每个小时样本：用 **同一日标量序列** 相邻小时差分得到涨/跌/平三分类，与回归头联合训练（`LAMBDA_DIR` 等见源码）。

### 5.3 输入通道（当前默认配置）

- **`LAG0_COLS`**（当日类型 + 时间编码拼在 `day_lag0` 里）：  
  `load_forecast`, `renewable_forecast`, `wind_forecast`, `solar_forecast`, `east_send_forecast`, `reserve_pos_capacity`, `reserve_neg_capacity`, **`price_dayahead_preclear_energy`**  
  另加 **4 维** 时间编码：`sin/cos(日内槽位)`、`sin/cos(星期)`。  
  因此 **`C_LAG0 = len(LAG0_COLS) + 4`**。

- **`LAG1_COLS`**：由 `TARGET_COL` 打头，再并入 **`_DEFAULT_LAG1`**（默认 `price_unified`, `price_hbd`, `price_hbx`，去重），最后追加 **`NM_V8_EXTRA_LAG1`**（逗号分隔列名）。含义：**D-1 … D-7 的日滞后电价通道**（含目标自身滞后）。

- **`LAG2_COLS`**：`load_actual`, `renewable_actual`, `wind_actual`, `solar_actual` — **D-2 … D-8** 窗口（见 `HourlyMultiTaskDataset` 中 `dates2` 偏移）。

- **张量形状**：`(C_TOTAL, H_SLOTS, 7)`，其中 `H_SLOTS` 由 `NM_CTX_BEFORE` / `NM_CTX_AFTER`（默认各 1 小时）× 每小时 4 槽 × … 决定；`C_TOTAL = C_LAG0 + C_LAG1 + C_LAG2`。

- **NaN 处理**：组 batch 前对特征网格 **`np.nan_to_num(..., nan=0.0)`**；首页电价经入库补齐后，理论上稀疏 NaN 应显著减少，但其它列仍可能有 NaN。

### 5.4 与「v8.0-12m」实验名的关系

- 实验输出目录由 **`NM_V8_TAG`** 控制（默认 `v8.0`）；历史目录名 **`v8.0-12m`** 表示当时运行设置了例如 **12 个月训练窗**（`NM_TRAIN_MONTHS=12`）等，而非代码里写死的字符串。以运行时的环境变量与日志为准。

### 5.5 重训注意

- 修改 `LAG0_COLS`/`LAG1_COLS` 后 **`C_TOTAL` 变化**，旧 checkpoint **不能**直接加载到新结构。

---

## 6. 常用命令

```bash
# 重建 DWS（含苏敦，除非加 --no-sudun）
python3 -m src.build_dws_15min_features

# 仅合并苏敦到已有 DWS（一般不需要单独跑，入库会调）
python3 -m src.merge_sudun_prices_into_dws
```

---

## 7. 变更历史摘要（供续查）

| 主题 | 说明 |
|------|------|
| DWS 时间平移 | 首页源 96 点 → DWS 当日 00:00–23:45（−15min） |
| 日前预出清 | 入库列 `price_dayahead_preclear_energy`；V8 放入 **LAG0** |
| 首页四价补齐 | `intraday_ffill_bfill` 仅对上述四列电价 |
| 苏敦 | 仅 500kV.1M 三列；**不入库补齐** |
| V8 与入库清单 | 解耦；改 DWS 列需同步改 `model_v8_multitask.py` 或 env |

---

*文档生成自当前仓库状态；若代码后续有改动，请以对应 `src/*.py` 为准并更新本节。*
