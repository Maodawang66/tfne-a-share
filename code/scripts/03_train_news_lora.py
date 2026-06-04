#!/usr/bin/env python
"""训练新闻模态：档位 C=NewsLoRA，档位 A=哈希向量+线性（冻结语义基线）。"""
from __future__ import annotations

import sys
from contextlib import nullcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.config import AppConfig
from src.news_predict import predict_news_seed
from src.dataset import split_by_date
from src.metrics import daily_ic, ic_summary
from src.models.factory import build_news_encoder, modal_names


class NewsDayDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        return {"text": r["text"], "y": torch.tensor(float(r["excess_ret_1d"]), dtype=torch.float32)}


def collate_news(batch):
    return {"texts": [b["text"] for b in batch], "y": torch.stack([b["y"] for b in batch])}


def _maybe_subsample(df: pd.DataFrame, max_samples: int | None, seed: int) -> pd.DataFrame:
    if max_samples is None or len(df) <= max_samples:
        return df
    return df.sample(n=max_samples, random_state=seed).reset_index(drop=True)


def _predict_batched(model, texts: list[str], device: torch.device, batch_size: int, use_amp: bool):
    outs = []
    model.eval()
    amp_ctx = torch.amp.autocast("cuda") if use_amp and device.type == "cuda" else nullcontext()
    with torch.no_grad(), amp_ctx:
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            outs.append(model(chunk, device).float().cpu().numpy())
    return np.concatenate(outs) if outs else np.array([])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = AppConfig.load(ROOT / args.config if args.config else None)
    print("[config]", cfg.describe())

    panel = pd.read_parquet(cfg.path("features", "panel_features.parquet"))
    news_path = cfg.path("news", "panel_news.parquet")
    if not news_path.exists():
        print("未找到 panel_news.parquet，请先运行 02b_embed_news.py")
        return

    news = pd.read_parquet(news_path)
    agg = news.groupby(["trade_date", "ts_code"], as_index=False).agg(
        {"text": lambda x: " ".join(x.astype(str).tolist())[:8000], "has_news": "max"}
    )
    m = panel.merge(agg, on=["trade_date", "ts_code"], how="left")
    m["has_news"] = m["has_news"].fillna(0).astype(int)
    m = m[m["has_news"] == 1].dropna(subset=["excess_ret_1d"])
    if m.empty:
        print("无新闻样本，跳过")
        return

    splits = cfg.splits
    train, valid, _ = split_by_date(m, splits["train_end"], splits["valid_end"])
    if len(train) < 50:
        print("新闻训练样本过少，跳过")
        return

    mcfg = cfg.model_cfg()
    max_train = mcfg.get("news_train_max_samples")
    if max_train is not None:
        max_train = int(max_train)
    train = _maybe_subsample(train, max_train, args.seed)
    news_epochs = int(mcfg.get("news_max_epochs", mcfg.get("max_epochs", 5)))
    news_patience = int(mcfg.get("news_patience", 2))
    batch_size = int(mcfg.get("news_batch_size", 48))
    predict_bs = int(mcfg.get("news_predict_batch_size", 128))
    lr = float(mcfg.get("news_lr", 2e-4 if cfg.tier == "C" else 1e-3))

    print(
        f"[news] train={len(train)}, valid={len(valid)}, epochs={news_epochs}, "
        f"batch={batch_size}, predict_batch={predict_bs}, max_train={max_train or 'all'}",
        flush=True,
    )

    model = build_news_encoder(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and bool(cfg.raw.get("training", {}).get("amp", True))
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    train_loader = DataLoader(
        NewsDayDataset(train), batch_size=batch_size, shuffle=True, collate_fn=collate_news
    )
    valid_loader = DataLoader(
        NewsDayDataset(valid), batch_size=batch_size, shuffle=False, collate_fn=collate_news
    )

    _, _, news_modal = modal_names(cfg)
    ckpt = cfg.checkpoint_path(news_modal, args.seed)
    ckpt.parent.mkdir(parents=True, exist_ok=True)

    best_ic = -1e9
    wait = 0
    amp_ctx = lambda: torch.amp.autocast("cuda") if use_amp else nullcontext()

    for epoch in range(news_epochs):
        model.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"news epoch {epoch + 1}/{news_epochs}", leave=False):
            opt.zero_grad(set_to_none=True)
            with amp_ctx():
                pred = model(batch["texts"], device)
                loss = torch.nn.functional.huber_loss(pred, batch["y"].to(device))
            if not torch.isfinite(loss):
                continue
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))

        model.eval()
        preds, labels = [], []
        with torch.no_grad(), amp_ctx():
            for batch in valid_loader:
                pred = model(batch["texts"], device)
                preds.extend(pred.float().cpu().numpy().tolist())
                labels.extend(batch["y"].cpu().numpy().tolist())
        n = min(len(preds), len(valid))
        ic = ic_summary(
            daily_ic(
                pd.DataFrame(
                    {
                        "score": preds[:n],
                        "excess_ret_1d": labels[:n],
                        "trade_date": valid["trade_date"].values[:n],
                    }
                )
            )
        )
        mean_ic = float(ic["mean_ic"])
        print(
            f"news epoch {epoch + 1}/{news_epochs}: loss={np.mean(losses) if losses else float('nan'):.4f} "
            f"val_ic={mean_ic:.4f} best={best_ic:.4f}",
            flush=True,
        )
        if mean_ic > best_ic:
            best_ic = mean_ic
            wait = 0
            torch.save({"model": model.state_dict()}, ckpt)
        else:
            wait += 1
            if wait >= news_patience:
                print(f"[news] early stop patience={news_patience}", flush=True)
                break

    if not ckpt.exists():
        print("未生成 news checkpoint")
        return

    predict_news_seed(cfg, args.seed, device=device)


if __name__ == "__main__":
    main()
