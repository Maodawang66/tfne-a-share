#!/usr/bin/env python
"""评估 IC / ICIR（按档位读取 ensemble_scores）。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import json

import pandas as pd

from src.config import AppConfig
from src.metrics import (
    daily_hit_rate_topk,
    daily_ic,
    daily_ic_strategy_topk,
    daily_ic_top_quantile,
    daily_long_short_spread,
    daily_signal_portfolio_return,
    daily_within_topk_spread,
    hit_rate_summary,
    ic_summary,
    signal_portfolio_summary,
    spread_summary,
)
from src.strategy import StrategyParams, compute_mkt_vol_threshold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = AppConfig.load(ROOT / args.config if args.config else None)
    pred_path = cfg.ensemble_scores_path()
    if not pred_path.exists():
        raise FileNotFoundError(f"请先运行 04_ensemble_predict.py [{cfg.tier_tag()}]")

    pred = pd.read_parquet(pred_path)
    splits = cfg.splits
    results = {"tier_tag": cfg.tier_tag(), "tier": cfg.tier, "smoke": cfg.smoke}

    scfg = cfg.raw.get("strategy", {})
    top_q = float(scfg.get("q_min", 0.9))
    n_max = int(scfg.get("n_max", 50))

    panel_path = cfg.path("features", "panel_features.parquet")
    eval_df = pred
    sp = None
    if panel_path.exists():
        panel = pd.read_parquet(
            panel_path,
            columns=["trade_date", "ts_code", "industry", "log_mv", "mkt_vol"],
        )
        eval_df = pred.merge(panel, on=["trade_date", "ts_code"], how="left")
        q80, _ = compute_mkt_vol_threshold(panel, splits["train_end"])
        sp = StrategyParams(
            n_max=n_max,
            q_min=top_q,
            tau=float(scfg.get("tau", 0.5)),
            w_max=float(scfg.get("w_max", 0.05)),
            w_ind_max=float(scfg.get("w_ind_max", 0.25)),
            c_min=float(scfg.get("c_min", 0.03)),
            gamma_vol=float(scfg.get("gamma_vol", 0.05)),
            mkt_vol_q80=q80,
        )

    split_defs = [
        ("train", pred["trade_date"] <= splits["train_end"]),
        ("valid", (pred["trade_date"] > splits["train_end"]) & (pred["trade_date"] <= splits["valid_end"])),
    ]
    if splits.get("test_end"):
        split_defs.append(("test", pred["trade_date"] > splits["valid_end"]))
    for name, cond in split_defs:
        sub = eval_df.loc[cond].dropna(subset=["score", "excess_ret_1d"])
        ic_k = daily_ic_strategy_topk(sub, n_max=n_max, q_min=top_q)
        block = {
            "full_universe": ic_summary(daily_ic(sub)),
            "top_quantile": ic_summary(daily_ic_top_quantile(sub, top_q=top_q)),
            "strategy_topk": {
                "n_max": n_max,
                "q_min": top_q,
                **ic_summary(ic_k),
            },
            "hit_rate_topk": hit_rate_summary(daily_hit_rate_topk(sub, n_max=n_max, q_min=top_q)),
            "within_topk_spread": spread_summary(daily_within_topk_spread(sub, n_max=n_max, q_min=top_q)),
            "long_short_spread": spread_summary(daily_long_short_spread(sub, top_q=top_q)),
        }
        if sp is not None:
            block["signal_portfolio"] = signal_portfolio_summary(
                daily_signal_portfolio_return(sub, strategy_params=sp)
            )
        # 兼容旧脚本/报告：顶层保留全市场 IC 字段
        block.update(block["full_universe"])
        results[name] = block

    out = cfg.metrics_dir() / "ic_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 策略 Top-K 指标单独落盘，便于 A/C2 对比与报告引用
    strategy_keys = (
        "strategy_topk",
        "hit_rate_topk",
        "within_topk_spread",
        "signal_portfolio",
    )
    strategy_out = {
        "tier_tag": cfg.tier_tag(),
        "config": cfg.raw.get("__config_file__", args.config),
        "n_max": n_max,
        "q_min": top_q,
        "splits": dict(splits),
    }
    for split in results:
        strategy_out[split] = {k: results[split].get(k) for k in strategy_keys if k in results[split]}
    sk_path = cfg.metrics_dir() / "strategy_topk.json"
    with open(sk_path, "w", encoding="utf-8") as f:
        json.dump(strategy_out, f, indent=2, ensure_ascii=False)

    valid_sk = results.get("valid", {}).get("strategy_topk", {})
    print(
        f"\n[{cfg.tier_tag()}] valid IC@K (n_max={n_max}): "
        f"mean_ic={valid_sk.get('mean_ic', 0):.4f} icir={valid_sk.get('icir', 0):.3f} "
        f"n_days={valid_sk.get('n_days', 0)}",
        flush=True,
    )
    print(f"Saved -> {out}", flush=True)
    print(f"Saved -> {sk_path}", flush=True)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
