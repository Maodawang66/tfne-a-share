#!/usr/bin/env python
"""融合预测：按档位写入 artifacts/predictions/{tier_tag}/。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import json

import pandas as pd
import torch

from src.checkpoint_utils import (
    PATCH_ID,
    load_seq_model_from_checkpoint,
    load_tab_model_from_checkpoint,
    resolve_ind_vocab,
    verify_ind_vocab_patch_deployed,
)
from src.config import AppConfig
from src.ensemble import ensemble_rank_fusion, estimate_fusion_weights, fuse_seed_mean
from src.models.factory import modal_names
from src.news_predict import news_parquet_stale, predict_news_seed
from src.risk_score import apply_risk_adjustment
from src.train.train_args import predict_kwargs
from src.train.trainer import forward_seq, predict_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = AppConfig.load(ROOT / args.config if args.config else None)
    verify_ind_vocab_patch_deployed(ROOT)
    print(f"[04_ensemble_predict] patch={PATCH_ID}", flush=True)
    print("[config]", cfg.describe())

    df = pd.read_parquet(cfg.path("features", "panel_features.parquet"))
    with open(cfg.path("features", "feature_meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    seq_cols, tab_cols = meta["seq_cols"], meta["tab_cols"]
    ds_kw = dict(seq_cols=seq_cols, tab_cols=tab_cols, seq_len=cfg.seq_len)
    mcfg = cfg.model_cfg()
    pk = predict_kwargs(cfg)
    seeds = cfg.seeds
    seq_modal, tab_modal, news_modal = modal_names(cfg)
    ckpt_paths = [
        cfg.checkpoint_path(seq_modal, s)
        for s in seeds
        if cfg.checkpoint_path(seq_modal, s).exists()
    ] + [
        cfg.checkpoint_path(tab_modal, s)
        for s in seeds
        if cfg.checkpoint_path(tab_modal, s).exists()
    ]
    panel_max_ind = int(df["industry_id"].max())
    panel_vocab = panel_max_ind + 2
    ind_vocab = resolve_ind_vocab(
        panel_max_ind,
        ckpt_paths,
        meta_industry_vocab=meta.get("industry_vocab"),
    )
    print(
        f"[ind_vocab] resolved={ind_vocab} panel_vocab={panel_vocab} "
        f"(max industry_id={panel_max_ind})",
        flush=True,
    )

    preds_modal = {}

    # 时序
    seq_frames = []
    for s in seeds:
        ckpt = cfg.checkpoint_path(seq_modal, s)
        if not ckpt.exists():
            continue
        model = load_seq_model_from_checkpoint(
            cfg, len(seq_cols), ckpt, panel_vocab, fallback_vocab=ind_vocab
        )
        pred = predict_model(model, df, ds_kw, forward_seq, **pk)
        out_p = cfg.prediction_dir(seq_modal) / f"seed_{s}.parquet"
        out_p.parent.mkdir(parents=True, exist_ok=True)
        pred.to_parquet(out_p, index=False)
        seq_frames.append(pred)
    if seq_frames:
        preds_modal["A"] = fuse_seed_mean(pd.concat(seq_frames), "z")

    # 截面
    tab_frames = []
    forward_tab = None
    for s in seeds:
        ckpt = cfg.checkpoint_path(tab_modal, s)
        if not ckpt.exists():
            continue
        model, forward_tab = load_tab_model_from_checkpoint(
            cfg, len(tab_cols), ckpt, panel_vocab, fallback_vocab=ind_vocab
        )
        pred = predict_model(model, df, ds_kw, forward_tab, **pk)
        out_p = cfg.prediction_dir(tab_modal) / f"seed_{s}.parquet"
        out_p.parent.mkdir(parents=True, exist_ok=True)
        pred.to_parquet(out_p, index=False)
        tab_frames.append(pred)
    if tab_frames:
        preds_modal["B"] = fuse_seed_mean(pd.concat(tab_frames), "z")

    # 新闻：优先 checkpoint 推理（C2）；否则读已有 parquet
    ens_cfg = cfg.raw.get("ensemble", {})
    panel_max = str(df["trade_date"].max())
    force_news_ckpt = bool(ens_cfg.get("predict_news_from_checkpoint", False))
    stale = news_parquet_stale(cfg, panel_max, seeds)
    news_dir = cfg.prediction_dir(news_modal)
    if force_news_ckpt or stale:
        if stale and not force_news_ckpt:
            print(f"[news] parquet 落后于 panel ({panel_max})，从 checkpoint 刷新", flush=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for s in seeds:
            ckpt = cfg.checkpoint_path(news_modal, s)
            if not ckpt.exists():
                print(f"[news] 跳过 seed={s}，无 checkpoint", flush=True)
                continue
            predict_news_seed(cfg, s, device=device)
    if news_dir.exists():
        nframes = [pd.read_parquet(p) for p in sorted(news_dir.glob("seed_*.parquet"))]
        if nframes:
            preds_modal["C"] = fuse_seed_mean(pd.concat(nframes), "z")

    if not preds_modal:
        raise RuntimeError(f"无预测结果 [{cfg.tier_tag()}]，请先训练各模态")

    splits = cfg.splits
    weights = ens_cfg.get("modal_weights")
    if weights is None:
        weights = estimate_fusion_weights(
            preds_modal,
            df,
            valid_start=cfg.valid_start(),
            valid_end=splits["valid_end"],
        )
    print("Fusion weights:", weights)

    fused = ensemble_rank_fusion(
        preds_modal,
        df,
        weights={m: weights.get(m, 1.0 / len(preds_modal)) for m in preds_modal},
        winsor_q=cfg.raw.get("ensemble", {}).get("winsor_q", 0.01),
    )
    risk_cfg = ens_cfg.get("risk_adjust", {})
    if risk_cfg.get("enabled", False):
        daily_path = cfg.path("panel_daily.parquet")
        fused = apply_risk_adjustment(
            fused,
            df,
            risk_cfg=risk_cfg,
            modal_weights=weights,
            daily_panel_path=daily_path,
        )
        print(
            f"[risk_adjust] method={risk_cfg.get('method', 'sharpe')} "
            f"vol_col={risk_cfg.get('vol_col', 'vol_20d_raw')}",
            flush=True,
        )
    else:
        fused["score_raw"] = fused["score"]

    out = cfg.ensemble_scores_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    fused.to_parquet(out, index=False)
    meta = {
        "tier": cfg.tier_tag(),
        "weights": weights,
        "risk_adjust": risk_cfg if risk_cfg.get("enabled") else None,
    }
    with open(out.parent / "fusion_weights.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
