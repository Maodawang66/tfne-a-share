#!/usr/bin/env python
"""打印档位 A 与 C2 的策略 Top-K 指标对比（读 metrics/*/strategy_topk.json）。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load(tag: str) -> dict | None:
    p = ROOT / "artifacts" / "metrics" / tag / "strategy_topk.json"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _fmt_block(name: str, block: dict) -> None:
    sk = block.get("strategy_topk") or {}
    hr = block.get("hit_rate_topk") or {}
    ws = block.get("within_topk_spread") or {}
    sp = block.get("signal_portfolio") or {}
    print(
        f"  {name:5s} | IC@K={sk.get('mean_ic', 0):+.4f} ICIR@K={sk.get('icir', 0):+.3f} "
        f"| hit={hr.get('mean_hit_rate', 0)*100:.1f}% "
        f"| within_spread_ann={ws.get('ann_spread', 0)*100:.1f}% "
        f"| signal_ann={sp.get('ann_return', 0)*100:.1f}% "
        f"| n={sk.get('n_days', 0)}"
    )


def main():
    configs = [
        ("tier_a", "config.tier_a_local.yaml"),
        ("tier_c2", "config.server_c2_72h.yaml"),
    ]
    print("策略 Top-K 指标（与 target_weights 选股范围一致：Q(q_min) + Top n_max）\n")
    for tag, cfg_name in configs:
        data = _load(tag)
        if data is None:
            print(f"[{tag}] 未找到 strategy_topk.json，请先运行：")
            print(f"  conda activate ai25")
            print(f"  python scripts/05_evaluate_ic.py --config {cfg_name}\n")
            continue
        print(
            f"[{tag}] n_max={data.get('n_max')} q_min={data.get('q_min')} "
            f"valid={data.get('splits', {}).get('valid_start', '?')}–{data.get('splits', {}).get('valid_end', '?')}"
        )
        for split in ("train", "valid", "test"):
            if split in data:
                _fmt_block(split, data[split])
        print()


if __name__ == "__main__":
    main()
