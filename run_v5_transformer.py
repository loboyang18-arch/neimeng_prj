#!/usr/bin/env python3
"""V5.0 Vanilla Transformer 入口脚本。

环境变量：
  NM_TF_EPOCHS      训练轮数  (默认 30)
  NM_TF_PATIENCE     早停耐心  (默认 10)
  NM_TF_FOLDS       最大折数  (默认 0 = 全部)
  NM_TF_FOLD_START   起始折号  (默认 1)
"""
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

from src.model_v5_transformer import run_v5_transformer

if __name__ == "__main__":
    run_v5_transformer()
