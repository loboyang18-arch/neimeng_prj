# 附录标准评估对比

数据源：`/root/workspace/neimeng_prj/output/evaluation/evaluation_summary_appendix_v1.csv`

| model | valid_point_count | mae | rmse | profile_corr | direction_acc | composite_score | neg_corr_day_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v25.0 | 2688 | 140.318544 | 221.547195 | 0.6854 | 0.6126 | 0.630362 | 0.035714 |
| v8.0 | 2688 | 126.953803 | 206.214634 | 0.7205 | 0.6009 | 0.609936 | 0.044643 |


**说明**：`composite_score` 相对 Naive lag24h 基线，**越低越好**；点数不同表示覆盖测试时段长度不同。

更新汇总前请先运行：`python run_evaluate_all_models.py`
