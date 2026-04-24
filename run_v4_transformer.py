#!/usr/bin/env python3
"""V4.0 Tri-Axis Transformer 入口脚本。

环境变量：
  NM_TF_EPOCHS   训练轮数  (默认 30)
  NM_TF_PATIENCE  早停耐心  (默认 10)
  NM_TF_FOLDS    最大折数  (默认 0 = 全部)
"""
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

from src.model_v4_transformer import run_v4_transformer

if __name__ == "__main__":
    run_v4_transformer()
