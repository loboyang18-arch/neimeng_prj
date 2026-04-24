# v2 / v3 / v4 / Naive 对比

## Fold 16 单折对比 (test: 2025-12-15 ~ 2025-12-21)

| 模型 | test_MAE | test_RMSE | vs Naive |
|------|----------|-----------|----------|
| Naive lag24h | 187.86 | — | 1.000× |
| v3 Level+Shape | 117.64 | 147.82 | 0.626× |
| **v4 Transformer** | **86.95** | **105.16** | **0.463×** |
| v2 LGB 逐点 | 80.73 | 98.42 | 0.430× |

## 训练配置

| 参数 | 值 |
|------|-----|
| 模型 | TriAxisTransformer (88,408 params) |
| d_model | 64, nhead=4, nlayers=2 |
| epochs / patience | 200 / 30 (early stop at ~ep78) |
| 训练集 | 700 天 (2024-01-01 ~ 2025-11-30) |
| 验证集 | 14 天 |
| 测试集 | 7 天 |
| 设备 | MPS (Apple GPU) |

## 训练曲线 (Fold 16)

```
ep  0  loss=0.693  train=260.5  val=218.3  test=207.4
ep  5  loss=0.644  train=241.6  val=185.7  test=152.7
ep 10  loss=0.597  train=215.9  val=183.5  test= 99.3
ep 20  loss=0.529  train=209.9  val=176.9  test= 93.1
ep 30  loss=0.507  train=228.6  val=165.5  test= 89.9
ep 50  loss=0.473  train=322.8  val=159.7  test= 86.5  ← best_val
ep 75  loss=0.453  train=371.2  val=162.2  test= 87.2  ← early stop
```

## 全量 16 折平均 (v4 仅单折)

| 模型 | avg MAE | avg RMSE |
|------|---------|----------|
| Naive lag24h | 187.86 | — |
| v3 Shape (16折) | 163.45 | 212.59 |
| v2 LGB (16折) | 139.81 | 189.08 |
| v4 Transformer (仅fold16) | 86.95 | 105.16 |

## 结论

- v4 Transformer 在 fold 16 上 MAE=86.95，**优于 v3 (26.1%)，优于 naive (53.7%)**
- 与 v2 LGB (80.73) 相比，v4 高 7.7%，但差距不大
- v4 使用 3 轴序列 (Lag0 预报 + Lag1 价格 + Lag2 实际) 的 Transformer，训练开销较大
- 需全量 16 折运行后才能公平对比整体性能
