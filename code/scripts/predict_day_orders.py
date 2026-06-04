#!/usr/bin/env python
"""对指定 signal 日（默认 panel 最新日）推理并生成调仓单；复用 fusion_weights.json。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch

from src.checkpoint_utils import load_seq_model_from_checkpoint, load_tab_model_from_checkpoint, resolve_ind_vocab
from src.config import AppConfig
from src.ensemble import ensemble_rank_fusion, fuse_seed_mean
from src.models.factory import build_news_encoder, modal_names
from src.risk_score import apply_risk_adjustment
from src.strategy import StrategyParams, compute_mkt_vol_threshold, target_weights, tradable_universe
from src.train.train_args import predict_kwargs
from src.dataset import DailyStockDataset
from src.train.trainer import forward_seq, predict_model


def _recent_panel(df: pd.DataFrame, target: str, buf_days: int = 100) -> pd.DataFrame:
    dates = sorted(df["trade_date"].unique())
    if target not in dates:
        raise ValueError(f"panel 无 {target}，最新={dates[-1]}")
    i = dates.index(target)
    keep = set(dates[max(0, i - buf_days) : i + 1])
    return df[df["trade_date"].isin(keep)].copy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--trade-date", default=None, help="信号日 YYYYMMDD，默认 panel 最大交易日")
    p.add_argument("--buf-days", type=int, default=100, help="序列回看窗口内保留的交易日数")
    args = p.parse_args()

    cfg = AppConfig.load(ROOT / args.config)
    print("[config]", cfg.describe(), flush=True)

    df_full = pd.read_parquet(cfg.path("features", "panel_features.parquet"))
    target = args.trade_date or str(df_full["trade_date"].max())
    buf_days = max(args.buf_days, cfg.seq_len + 5)
    df = _recent_panel(df_full, target, buf_days)

    with open(cfg.path("features", "feature_meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    seq_cols, tab_cols = meta["seq_cols"], meta["tab_cols"]
    ds_kw = dict(seq_cols=seq_cols, tab_cols=tab_cols, seq_len=cfg.seq_len)
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
    panel_max_ind = int(df_full["industry_id"].max())
    ind_vocab = resolve_ind_vocab(
        panel_max_ind, ckpt_paths, meta_industry_vocab=meta.get("industry_vocab")
    )
    panel_vocab = panel_max_ind + 2

    infer_ds = DailyStockDataset(
        df, **ds_kw, inference_dates=[target]
    )
    n_infer = len(infer_ds)
    print(
        f"[predict] target={target} panel_rows={len(df)} infer_rows={n_infer} buf_days={buf_days}",
        flush=True,
    )

    preds_modal: dict[str, pd.DataFrame] = {}

    seq_frames = []
    for s in seeds:
        ckpt = cfg.checkpoint_path(seq_modal, s)
        if not ckpt.exists():
            continue
        model = load_seq_model_from_checkpoint(
            cfg, len(seq_cols), ckpt, panel_vocab, fallback_vocab=ind_vocab
        )
        pred = predict_model(model, df, ds_kw, forward_seq, dataset=infer_ds, **pk)
        seq_frames.append(pred)
    if seq_frames:
        preds_modal["A"] = fuse_seed_mean(pd.concat(seq_frames), "z")

    tab_frames = []
    for s in seeds:
        ckpt = cfg.checkpoint_path(tab_modal, s)
        if not ckpt.exists():
            continue
        model, forward_tab = load_tab_model_from_checkpoint(
            cfg, len(tab_cols), ckpt, panel_vocab, fallback_vocab=ind_vocab
        )
        pred = predict_model(model, df, ds_kw, forward_tab, dataset=infer_ds, **pk)
        tab_frames.append(pred)
    if tab_frames:
        preds_modal["B"] = fuse_seed_mean(pd.concat(tab_frames), "z")

    ens_cfg = cfg.raw.get("ensemble", {})
    news_dir = cfg.prediction_dir(news_modal)
    need_news = ens_cfg.get("predict_news_from_checkpoint", False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _news_for_day(seed: int) -> pd.DataFrame | None:
        fp = news_dir / f"seed_{seed}.parquet"
        if fp.exists():
            part = pd.read_parquet(fp)
            hit = part[part["trade_date"] == target]
            if len(hit):
                return hit
        if not need_news or not cfg.checkpoint_path(news_modal, seed).exists():
            return None
        from src.news_predict import build_news_merged_panel, _predict_batched

        m = build_news_merged_panel(cfg)
        m = m[m["trade_date"] == target]
        if m.empty:
            return None
        mcfg = cfg.model_cfg()
        predict_bs = int(mcfg.get("news_predict_batch_size", 128))
        use_amp = device.type == "cuda" and bool(cfg.raw.get("training", {}).get("amp", True))
        model = build_news_encoder(cfg).to(device)
        state = torch.load(cfg.checkpoint_path(news_modal, seed), map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        model.eval()
        texts = m["text"].tolist()
        z = _predict_batched(model, texts, device, predict_bs, use_amp)
        rows = [
            {"trade_date": r["trade_date"], "ts_code": r["ts_code"], "z": float(z[i])}
            for i, (_, r) in enumerate(m.iterrows())
        ]
        day_df = pd.DataFrame(rows)
        if fp.exists():
            old = pd.read_parquet(fp)
            old = old[old["trade_date"] != target]
            pd.concat([old, day_df], ignore_index=True).to_parquet(fp, index=False)
        return day_df

    nframes = []
    for s in seeds:
        part = _news_for_day(s)
        if part is not None and len(part):
            nframes.append(part)
    if nframes:
        preds_modal["C"] = fuse_seed_mean(pd.concat(nframes), "z")

    if not preds_modal:
        raise RuntimeError("无模态预测结果")

    fw_path = cfg.ensemble_scores_path().parent / "fusion_weights.json"
    if fw_path.exists():
        weights = json.load(open(fw_path, encoding="utf-8"))["weights"]
    else:
        weights = {m: 1.0 / len(preds_modal) for m in preds_modal}
    print("Fusion weights (frozen):", weights, flush=True)

    panel_day = df_full[df_full["trade_date"] == target]
    fused = ensemble_rank_fusion(
        preds_modal,
        panel_day,
        weights={m: weights.get(m, 0.0) for m in preds_modal},
        winsor_q=ens_cfg.get("winsor_q", 0.01),
    )
    risk_cfg = ens_cfg.get("risk_adjust", {})
    if risk_cfg.get("enabled", False):
        fused = apply_risk_adjustment(
            fused,
            df_full,
            risk_cfg=risk_cfg,
            modal_weights=weights,
            daily_panel_path=cfg.path("panel_daily.parquet"),
        )
    else:
        fused["score_raw"] = fused["score"]

    out_scores = cfg.ensemble_scores_path()
    if out_scores.exists():
        old = pd.read_parquet(out_scores)
        old = old[old["trade_date"] != target]
        fused_all = pd.concat([old, fused], ignore_index=True)
    else:
        fused_all = fused
    fused_all.to_parquet(out_scores, index=False)

    scfg = cfg.raw.get("strategy", {})
    q80, _ = compute_mkt_vol_threshold(df_full, cfg.splits["train_end"])
    sp = StrategyParams(
        n_max=scfg.get("n_max", 50),
        q_min=scfg.get("q_min", 0.90),
        tau=scfg.get("tau", 0.5),
        mkt_vol_q80=q80,
        w_max=scfg.get("w_max", 0.05),
        w_ind_max=scfg.get("w_ind_max", 0.25),
        c_min=scfg.get("c_min", 0.03),
        gamma_vol=scfg.get("gamma_vol", 0.05),
    )
    u = tradable_universe(panel_day, cfg.raw.get("data", {}).get("min_amount_ma", 5000))
    scores = fused.set_index("ts_code")["score"].reindex(u["ts_code"]).dropna()
    w = target_weights(
        scores,
        u.set_index("ts_code")["industry"],
        u.set_index("ts_code")["log_mv"],
        sp,
        float(panel_day["mkt_vol"].iloc[0]) if "mkt_vol" in panel_day.columns else 0.0,
    )
    orders = pd.DataFrame({"ts_code": w.index, "target_weight": w.values, "trade_date": target})
    order_path = cfg.path("orders", cfg.tier_tag(), f"orders_{target}.csv")
    order_path.parent.mkdir(parents=True, exist_ok=True)
    orders.to_csv(order_path, index=False)

    print(f"[{cfg.tier_tag()}] signal_date={target} -> 次日开盘调仓", flush=True)
    print(f"  scores -> {out_scores}", flush=True)
    print(f"  orders -> {order_path}  (n={len(orders)}, weight_sum={orders['target_weight'].sum():.4f})", flush=True)
    top = orders.sort_values("target_weight", ascending=False).head(10)
    print(top.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
