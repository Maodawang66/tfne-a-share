"""新闻模态全面板推理（训练后或增量 predict 共用）。"""
from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.config import AppConfig
from src.models.factory import build_news_encoder, modal_names


def build_news_merged_panel(cfg: AppConfig) -> pd.DataFrame:
    panel = pd.read_parquet(cfg.path("features", "panel_features.parquet"))
    news_path = cfg.path("news", "panel_news.parquet")
    if not news_path.exists():
        raise FileNotFoundError(f"未找到 {news_path}，请先运行 02b_embed_news.py")
    news = pd.read_parquet(news_path)
    agg = news.groupby(["trade_date", "ts_code"], as_index=False).agg(
        {"text": lambda x: " ".join(x.astype(str).tolist())[:8000], "has_news": "max"}
    )
    m = panel.merge(agg, on=["trade_date", "ts_code"], how="left")
    m["has_news"] = m["has_news"].fillna(0).astype(int)
    return m[m["has_news"] == 1].copy()


def _predict_batched(model, texts: list[str], device: torch.device, batch_size: int, use_amp: bool):
    outs = []
    model.eval()
    amp_ctx = torch.amp.autocast("cuda") if use_amp and device.type == "cuda" else nullcontext()
    with torch.no_grad(), amp_ctx:
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            outs.append(model(chunk, device).float().cpu().numpy())
    return np.concatenate(outs) if outs else np.array([])


def predict_news_seed(
    cfg: AppConfig,
    seed: int,
    *,
    device: torch.device | None = None,
    write_parquet: bool = True,
) -> pd.DataFrame:
    """从 checkpoint 对全量有新闻样本推理，可选写入 predictions/{tier}/{news}/seed_*.parquet。"""
    m = build_news_merged_panel(cfg)
    if m.empty:
        raise RuntimeError("无新闻样本可预测")

    mcfg = cfg.model_cfg()
    predict_bs = int(mcfg.get("news_predict_batch_size", 128))
    _, _, news_modal = modal_names(cfg)
    ckpt = cfg.checkpoint_path(news_modal, seed)
    if not ckpt.exists():
        raise FileNotFoundError(f"缺少 news checkpoint: {ckpt}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and bool(cfg.raw.get("training", {}).get("amp", True))

    model = build_news_encoder(cfg).to(device)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()

    rows = []
    print(f"[news predict] seed={seed} rows={len(m)} batch={predict_bs} device={device}", flush=True)
    for _, g in tqdm(m.groupby("trade_date"), desc=f"news predict s{seed}", leave=False):
        texts = g["text"].tolist()
        z = _predict_batched(model, texts, device, predict_bs, use_amp)
        for i, (_, r) in enumerate(g.iterrows()):
            rows.append({"trade_date": r["trade_date"], "ts_code": r["ts_code"], "z": float(z[i])})

    out_df = pd.DataFrame(rows)
    if write_parquet:
        out_p = cfg.prediction_dir(news_modal) / f"seed_{seed}.parquet"
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_parquet(out_p, index=False)
        print(f"[{cfg.tier_tag()}] news predict seed={seed} -> {out_p}", flush=True)
    return out_df


def news_parquet_stale(cfg: AppConfig, panel_max_date: str, seeds: list[int]) -> bool:
    """预测 parquet 缺失或最大日期落后于 panel 时视为过期。"""
    _, _, news_modal = modal_names(cfg)
    news_dir = cfg.prediction_dir(news_modal)
    if not news_dir.exists():
        return True
    for s in seeds:
        p = news_dir / f"seed_{s}.parquet"
        if not p.exists():
            return True
        df = pd.read_parquet(p, columns=["trade_date"])
        if df["trade_date"].max() < panel_max_date:
            return True
    return False
