# v2 / v3 / Naive 对比（滚动测试期，小时聚合评估集）

| 模型 | MAE | composite | vs Naive MAE |
|------|-----|-----------|---------------|
| v2 LGB 逐点 | 138.60 | 0.6119 | 0.738× |
| v3 Level+Shape | 163.45 | 0.6887 | 0.870× |
| Naive lag24h | 187.86 | — | 1.000× |

说明：composite 来自 price_forecast_eval（越低越好）。
