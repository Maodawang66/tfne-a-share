#!/usr/bin/env python
"""特征工程 + GKX 截面变换。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse

import pandas as pd

from src.config import AppConfig
from src.features import apply_feature_transform, build_stock_features, get_seq_feature_names, get_tab_feature_names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = AppConfig.load(ROOT / args.config if args.config else None)
    print("[config]", cfg.describe())
    panel_path = cfg.path("panel_daily.parquet")
    if not panel_path.exists():
        raise FileNotFoundError("请先运行 01_build_panel.py")

    df = pd.read_parquet(panel_path)
    df = build_stock_features(df)
    feat_cfg = cfg.raw.get("features", {})
    # 保留实现波动率原值，供 risk_adjust 使用（不送入 NN 特征）
    for w in [10, 20, 60]:
        col = f"vol_{w}d"
        if col in df.columns:
            df[f"{col}_raw"] = df[col].astype(float)
    df = apply_feature_transform(df, winsor_q=feat_cfg.get("winsor_q", 0.01))

    out = cfg.path("features", "panel_features.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    meta = {
        "seq_cols": get_seq_feature_names(df),
        "tab_cols": get_tab_feature_names(df),
        "seq_len": cfg.seq_len,
        "industry_vocab": int(df["industry_id"].max() + 2),
    }
    import json

    with open(cfg.path("features", "feature_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"Saved features -> {out}")


if __name__ == "__main__":
    main()
