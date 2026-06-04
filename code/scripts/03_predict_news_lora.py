#!/usr/bin/env python
"""仅推理新闻模态（不重训），用于增量 predict 或 eval 前刷新 news/seed_*.parquet。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse

import torch

from src.config import AppConfig
from src.news_predict import predict_news_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = AppConfig.load(ROOT / args.config if args.config else None)
    print("[config]", cfg.describe(), flush=True)
    torch.manual_seed(args.seed)
    predict_news_seed(cfg, args.seed)


if __name__ == "__main__":
    main()
