#!/usr/bin/env python
"""计算指数基准的标准回测指标（与 src/backtest.compute_metrics 一致）。"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.backtest import compute_metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index", default="000300.SH")
    p.add_argument("--start", default="20240701")
    p.add_argument("--end", default="20241231")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    data_dir = ROOT.parent / "data"
    idx = pd.read_csv(data_dir / "market" / f"{args.index}.csv", dtype=str)
    idx["trade_date"] = idx["trade_date"].astype(str)
    sub = idx[(idx["trade_date"] >= args.start) & (idx["trade_date"] <= args.end)].copy()
    sub = sub.sort_values("trade_date")
    close = pd.to_numeric(sub["close"], errors="coerce")
    sub["equity"] = close / float(close.iloc[0]) * 1_000_000

    metrics = compute_metrics(sub[["trade_date", "equity"]], initial=1_000_000)
    metrics["index"] = args.index
    metrics["start"] = args.start
    metrics["end"] = args.end

    out = Path(args.out) if args.out else ROOT / "artifacts" / "metrics" / "benchmark_valid.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
