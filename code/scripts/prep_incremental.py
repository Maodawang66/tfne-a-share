#!/usr/bin/env python
"""增量 prep：仅追加新交易日（panel → features → news），供 nightly 快速更新。"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import AppConfig
from src.incremental_prep import (
    append_features_days,
    append_news_days,
    append_panel_days,
    discover_feature_lag_days,
    discover_new_news_dates,
    discover_new_panel_dates,
    features_need_rebuild,
    rebuild_features_from_panel,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="增量 prep：只追加 panel/features/news 新交易日")
    parser.add_argument("--config", default="config.server_c2_72h.yaml")
    parser.add_argument("--n-jobs", type=int, default=None, help="并行度，默认读配置 training.panel_n_jobs")
    parser.add_argument("--skip-news", action="store_true", help="跳过 02b 新闻链指")
    args = parser.parse_args()

    cfg = AppConfig.load(ROOT / args.config)
    print("[config]", cfg.describe(), flush=True)

    data_cfg = cfg.raw.get("data", {})
    train_cfg = cfg.raw.get("training", {})
    feat_cfg = cfg.raw.get("features", {})
    n_jobs = args.n_jobs if args.n_jobs is not None else int(train_cfg.get("panel_n_jobs", 4))
    news_n_jobs = int(train_cfg.get("news_n_jobs", n_jobs))

    panel_path = cfg.path("panel_daily.parquet")
    features_path = cfg.path("features", "panel_features.parquet")
    meta_path = cfg.path("features", "feature_meta.json")
    news_path = cfg.path("news", "panel_news.parquet")

    t0 = time.time()
    new_days = discover_new_panel_dates(
        cfg.data_dir,
        panel_path,
        start_date=data_cfg.get("start_date", "20190101"),
        end_date=data_cfg.get("end_date"),
    )

    if not new_days:
        print(f"[incremental] panel 已是最新（max={pd_max(panel_path)}），跳过 01", flush=True)
    else:
        print(f"[incremental] 新交易日 {new_days[0]} .. {new_days[-1]}（共 {len(new_days)} 天）", flush=True)
        t1 = time.time()
        append_panel_days(
            cfg.data_dir,
            panel_path,
            new_days,
            liquidity_top_k=data_cfg.get("liquidity_top_k"),
            max_stocks=data_cfg.get("max_stocks"),
            n_jobs=n_jobs,
        )
        print(f"  01 panel +{len(new_days)}d -> {panel_path}  ({time.time() - t1:.1f}s)", flush=True)

    if features_need_rebuild(features_path):
        print(f"[incremental] features 缺失/损坏，从 panel 全量重建 02 ...", flush=True)
        t2 = time.time()
        rebuild_features_from_panel(
            panel_path,
            features_path,
            meta_path,
            seq_len=cfg.seq_len,
            winsor_q=feat_cfg.get("winsor_q", 0.01),
        )
        print(f"  02 features rebuilt -> {features_path}  ({time.time() - t2:.1f}s)", flush=True)
    else:
        feat_lag = discover_feature_lag_days(panel_path, features_path)
        if not feat_lag:
            print(f"[incremental] features 已与 panel 对齐（max={pd_max(features_path)}），跳过 02", flush=True)
        else:
            t2 = time.time()
            append_features_days(
                panel_path,
                features_path,
                meta_path,
                feat_lag,
                seq_len=cfg.seq_len,
                winsor_q=feat_cfg.get("winsor_q", 0.01),
            )
            print(f"  02 features +{len(feat_lag)}d -> {features_path}  ({time.time() - t2:.1f}s)", flush=True)

    if not args.skip_news:
        news_start = data_cfg.get("news_start", "20190101")
        new_news_files = discover_new_news_dates(cfg.data_dir, news_path, news_start)
        if not new_news_files:
            print(f"[incremental] news 已是最新（max={pd_max(news_path)}），跳过 02b", flush=True)
        else:
            t3 = time.time()
            df = append_news_days(cfg.data_dir, news_path, new_news_files, n_jobs=news_n_jobs)
            print(
                f"  02b news +{len(new_news_files)} 文件 -> {news_path} "
                f"rows={len(df)}  ({time.time() - t3:.1f}s)",
                flush=True,
            )

    print(f"[incremental] done in {time.time() - t0:.1f}s", flush=True)


def pd_max(path: Path) -> str:
    if not path.exists() or path.stat().st_size < 64:
        return "N/A"
    import pandas as pd

    return str(pd.read_parquet(path, columns=["trade_date"])["trade_date"].max())


if __name__ == "__main__":
    main()
