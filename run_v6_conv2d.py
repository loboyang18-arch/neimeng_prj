#!/usr/bin/env python3
"""V6.0 Conv2D 入口脚本。

环境变量：
  NM_TF_EPOCHS      训练轮数  (默认 200)
  NM_TF_FOLDS       最大折数  (默认 0 = 全部)
  NM_TF_FOLD_START   起始折号  (默认 1)
"""
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

from src.model_v6_conv2d import run_v6_conv2d

if __name__ == "__main__":
    run_v6_conv2d()
