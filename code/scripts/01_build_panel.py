#!/usr/bin/env python
"""构建 panel_daily.parquet。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse

from src.config import AppConfig
from src.panel import build_panel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="如 config.tier_a.yaml / config.smoke_c.yaml")
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()

    cfg_path = ROOT / args.config if args.config else None
    cfg = AppConfig.load(cfg_path)
    print("[config]", cfg.describe())
    data_cfg = cfg.raw["data"]
    out = cfg.path("panel_daily.parquet")

    build_panel(
        data_dir=cfg.data_dir,
        output_path=out,
        start_date=data_cfg.get("start_date", "20190101"),
        end_date=data_cfg.get("end_date"),
        max_stocks=data_cfg.get("max_stocks"),
        liquidity_top_k=data_cfg.get("liquidity_top_k"),
        n_jobs=args.n_jobs,
    )
    print(f"Saved panel -> {out}")


if __name__ == "__main__":
    main()
