#!/usr/bin/env python
"""训练时序模型：档位 C=TFT-lite，档位 A=GRU 基线。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import json

import pandas as pd

from src.config import AppConfig
from src.dataset import split_by_date
from src.models.factory import build_seq_model, modal_names
from src.train.train_args import training_kwargs
from src.train.trainer import forward_seq, train_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", default=None, help="如 config.tier_a.yaml / config.smoke_c.yaml")
    args = parser.parse_args()

    cfg = AppConfig.load(ROOT / args.config if args.config else None)
    print("[config]", cfg.describe())

    df = pd.read_parquet(cfg.path("features", "panel_features.parquet"))
    with open(cfg.path("features", "feature_meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    seq_cols = meta["seq_cols"]
    tab_cols = meta.get("tab_cols", [])

    splits = cfg.splits
    train, valid, _ = split_by_date(df, splits["train_end"], splits["valid_end"])
    train = train.dropna(subset=["excess_ret_1d"] + seq_cols, how="any")
    valid = valid.dropna(subset=["excess_ret_1d"] + seq_cols, how="any")

    ind_vocab = int(df["industry_id"].max() + 2)
    model = build_seq_model(cfg, len(seq_cols), ind_vocab)
    tk = training_kwargs(cfg)

    seq_modal, _, _ = modal_names(cfg)
    ds_kw = dict(seq_cols=seq_cols, tab_cols=tab_cols or seq_cols[:1], seq_len=cfg.seq_len)
    ckpt = cfg.checkpoint_path(seq_modal, args.seed)

    train_model(model, train, valid, ds_kw, forward_seq, ckpt, ind_vocab=ind_vocab, **tk)
    print(f"[{cfg.tier_tag()}] seq model ({seq_modal}) seed={args.seed} -> {ckpt}")


if __name__ == "__main__":
    main()
