#!/usr/bin/env python3
"""V7.0 Conv2D + Shape-Aware Loss 入口脚本。"""
import logging, sys, os
sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

from src.model_v7_shapeloss import run_v7_shapeloss
run_v7_shapeloss()
