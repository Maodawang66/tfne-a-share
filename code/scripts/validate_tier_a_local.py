#!/usr/bin/env python
"""档位 A 本机预检（分钟级）。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import AppConfig
from src.models.factory import build_news_encoder, build_seq_model, build_tab_model, modal_names


def _feature_dims(cfg: AppConfig) -> tuple[int, int, int]:
    meta_path = cfg.path("features", "feature_meta.json")
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        return len(meta["seq_cols"]), len(meta["tab_cols"]), int(meta.get("industry_vocab", 64))
    return 20, 20, 64


def main():
    import torch

    cfg = AppConfig.load(ROOT / "config.tier_a_local.yaml")
    print("[config]", cfg.describe())

    if not torch.cuda.is_available():
        print("[WARN] CUDA 不可用，将回退 CPU（较慢）")
    else:
        print(f"  CUDA OK: {torch.cuda.get_device_name(0)}")

    seq_m, tab_m, news_m = modal_names(cfg)
    print(f"  modals: {seq_m}, {tab_m}, {news_m}")

    n_seq, n_tab, ind_vocab = _feature_dims(cfg)
    sl = cfg.seq_len
    seq_model = build_seq_model(cfg, n_seq, ind_vocab)
    tab_model = build_tab_model(cfg, n_tab, ind_vocab)
    news_model = build_news_encoder(cfg)
    print(f"  models OK (seq_len={sl}, n_seq={n_seq}, n_tab={n_tab}, bert={cfg.model_cfg().get('use_news_bert')})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seq_model = seq_model.to(device)
    tab_model = tab_model.to(device)
    news_model = news_model.to(device)

    x_seq = torch.randn(4, sl, n_seq, device=device)
    mask = torch.ones(4, sl, device=device)
    y_seq = seq_model(x_seq, mask=mask)
    assert torch.isfinite(y_seq).all()
    print("  GRU forward OK")

    x_tab = torch.randn(4, n_tab, device=device)
    y_tab = tab_model(x_tab)
    assert torch.isfinite(y_tab).all()
    print("  MLP forward OK")

    y_news = news_model(["news text"] * 4, device)
    assert torch.isfinite(y_news).all()
    print("  news_hash forward OK")

    data = cfg.data_dir
    if not data.exists():
        print(f"[FAIL] data_dir 不存在: {data}")
        sys.exit(1)
    print(f"  data_dir OK: {data}")
    print("\n预检通过。运行: python scripts/run_tier_a_local.py")


if __name__ == "__main__":
    main()
