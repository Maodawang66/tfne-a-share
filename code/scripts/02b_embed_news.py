#!/usr/bin/env python
"""新闻链指聚合为 panel_news.parquet。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse

from src.config import AppConfig
from src.news_link import aggregate_news_daily, load_basic_name_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = AppConfig.load(ROOT / args.config if args.config else None)
    print("[config]", cfg.describe())
    data_cfg = cfg.raw["data"]
    name_map = load_basic_name_map(cfg.data_dir)
    start = data_cfg.get("news_start", "20190101")
    end = data_cfg.get("end_date")
    n_jobs = int(cfg.raw.get("training", {}).get("news_n_jobs", 4))
    print(
        f"Scanning news: {start} .. {end or 'latest'} "
        f"(CPU 字符串链指，非 GPU；n_jobs={n_jobs})",
        flush=True,
    )
    news = aggregate_news_daily(
        cfg.data_dir, start=start, end=end, name_map=name_map, n_jobs=n_jobs
    )
    out = cfg.path("news", "panel_news.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    news.to_parquet(out, index=False)
    print(f"Saved news -> {out}, rows={len(news)}")


if __name__ == "__main__":
    main()
