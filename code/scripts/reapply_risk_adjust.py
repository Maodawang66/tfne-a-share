#!/usr/bin/env python
"""对已有 ensemble_scores 施加/刷新风险调整打分（无需重跑 GPU 推理）。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.config import AppConfig
from src.metrics import daily_ic, ic_summary
from src.risk_score import apply_risk_adjustment


def main():
    import argparse

    p = argparse.ArgumentParser(description="刷新 risk_adjust 后的 score（保留 score_raw）")
    p.add_argument("--config", required=True)
    args = p.parse_args()

    cfg = AppConfig.load(ROOT / args.config)
    ens_cfg = cfg.raw.get("ensemble", {})
    risk_cfg = ens_cfg.get("risk_adjust", {})
    if not risk_cfg.get("enabled", False):
        print(f"[{cfg.tier_tag()}] risk_adjust.enabled=false，跳过")
        return

    pred_path = cfg.ensemble_scores_path()
    if not pred_path.exists():
        raise FileNotFoundError(f"未找到 {pred_path}，请先运行 04")

    fused = pd.read_parquet(pred_path)
    if "score_raw" not in fused.columns and "z_A" in fused.columns:
        fused["score_raw"] = fused["score"]

    panel = pd.read_parquet(cfg.path("features", "panel_features.parquet"))
    fw_path = pred_path.parent / "fusion_weights.json"
    weights = {}
    if fw_path.exists():
        with open(fw_path, encoding="utf-8") as f:
            meta = json.load(f)
        weights = meta.get("weights", meta)

    fused = apply_risk_adjustment(
        fused,
        panel,
        risk_cfg=risk_cfg,
        modal_weights=weights,
        daily_panel_path=cfg.path("panel_daily.parquet"),
    )
    fused.to_parquet(pred_path, index=False)

    splits = cfg.splits
    valid = fused[
        (fused["trade_date"] > splits["train_end"]) & (fused["trade_date"] <= splits["valid_end"])
    ].dropna(subset=["excess_ret_1d"])
    ic_raw = ic_summary(daily_ic(valid, score_col="score_raw"))
    ic_adj = ic_summary(daily_ic(valid, score_col="score"))
    print(
        f"[{cfg.tier_tag()}] risk_adjust={risk_cfg.get('method')} "
        f"valid IC raw={ic_raw['mean_ic']:.4f} adj={ic_adj['mean_ic']:.4f} "
        f"-> {pred_path}"
    )


if __name__ == "__main__":
    main()
