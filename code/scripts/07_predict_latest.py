#!/usr/bin/env python
"""最新交易日调仓建议（按档位）。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse

import pandas as pd

from src.config import AppConfig
from src.strategy import StrategyParams, compute_mkt_vol_threshold, target_weights, tradable_universe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = AppConfig.load(ROOT / args.config if args.config else None)
    panel = pd.read_parquet(cfg.path("features", "panel_features.parquet"))
    pred = pd.read_parquet(cfg.ensemble_scores_path())
    last_date = pred["trade_date"].max()
    day = panel[panel["trade_date"] == last_date]
    scores = pred[pred["trade_date"] == last_date].set_index("ts_code")["score"]

    scfg = cfg.raw.get("strategy", {})
    q80, _ = compute_mkt_vol_threshold(panel, cfg.splits["train_end"])
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
    u = tradable_universe(day, cfg.raw.get("data", {}).get("min_amount_ma", 5000))
    scores = scores.reindex(u["ts_code"]).dropna()
    w = target_weights(
        scores,
        u.set_index("ts_code")["industry"],
        u.set_index("ts_code")["log_mv"],
        sp,
        float(day["mkt_vol"].iloc[0]) if "mkt_vol" in day.columns else 0.0,
    )
    out = pd.DataFrame({"ts_code": w.index, "target_weight": w.values, "trade_date": last_date})
    path = cfg.path("orders", cfg.tier_tag(), f"orders_{last_date}.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print(f"[{cfg.tier_tag()}] Saved -> {path}")


if __name__ == "__main__":
    main()
