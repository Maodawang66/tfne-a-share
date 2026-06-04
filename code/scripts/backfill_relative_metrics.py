#!/usr/bin/env python
"""从已有 equity_curve.csv 补算相对基准指标（无需重跑回测）。"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.backtest import compute_relative_metrics, enrich_equity_with_benchmark, load_benchmark_equity
from src.config import AppConfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--period", default="valid", choices=["valid", "test"])
    args = p.parse_args()

    cfg = AppConfig.load(ROOT / args.config)
    bcfg = cfg.raw.get("backtest", {})
    benchmark_code = str(bcfg.get("benchmark", "000300.SH"))
    scfg = cfg.raw.get("strategy", {})
    initial = float(scfg.get("initial_capital", 1_000_000))

    eq_path = cfg.backtest_dir(args.period) / "equity_curve.csv"
    if not eq_path.exists():
        raise FileNotFoundError(f"未找到 {eq_path}")

    equity = pd.read_csv(eq_path, dtype={"trade_date": str})
    dates = equity["trade_date"].astype(str).tolist()
    bench = load_benchmark_equity(cfg.data_dir, benchmark_code, dates, initial)
    rel = compute_relative_metrics(equity, bench, initial, index_code=benchmark_code)

    metrics_path = cfg.backtest_dir(args.period) / "metrics.json"
    metrics = {}
    if metrics_path.exists():
        with open(metrics_path, encoding="utf-8") as f:
            metrics = json.load(f)
    metrics.update(rel)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    enriched = enrich_equity_with_benchmark(equity, bench)
    enriched.to_csv(eq_path, index=False)
    print(json.dumps(rel, indent=2, ensure_ascii=False))
    print(f"Updated -> {metrics_path}")


if __name__ == "__main__":
    main()
