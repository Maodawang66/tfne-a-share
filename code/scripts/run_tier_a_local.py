#!/usr/bin/env python
"""档位 A 本机基线一键全流程（config.tier_a_local.yaml）。"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG = "config.tier_a_local.yaml"
PY = sys.executable


def main():
    p = argparse.ArgumentParser(description="档位 A 本机基线（GRU+MLP+新闻哈希）")
    p.add_argument("--prep-only", action="store_true", help="只跑 01–02b")
    p.add_argument("--train-eval-only", action="store_true", help="跳过 prep，从 03 训练到 07")
    args = p.parse_args()

    cfg_flag = ["--config", CFG]
    steps: list[list[str]] = []

    if not args.train_eval_only:
        nj = "4"
        steps.extend(
            [
                [PY, str(ROOT / "scripts" / "01_build_panel.py"), "--n-jobs", nj, *cfg_flag],
                [PY, str(ROOT / "scripts" / "02_make_features.py"), *cfg_flag],
                [PY, str(ROOT / "scripts" / "02b_embed_news.py"), *cfg_flag],
            ]
        )
    if args.prep_only:
        pass
    elif not args.prep_only:
        for script, extra in [
            ("03_train_tft.py", ["--seed", "42"]),
            ("03_train_ftt.py", ["--seed", "42"]),
            ("03_train_news_lora.py", ["--seed", "42"]),
        ]:
            steps.append([PY, str(ROOT / "scripts" / script), *extra, *cfg_flag])
        for script in (
            "04_ensemble_predict.py",
            "05_evaluate_ic.py",
            "06_backtest.py",
            "07_predict_latest.py",
        ):
            steps.append([PY, str(ROOT / "scripts" / script), *cfg_flag])

    if not steps:
        print("无步骤", flush=True)
        return

    print(f"[tier_a local] config={CFG} steps={len(steps)}", flush=True)
    for cmd in steps:
        print("\n>>>", " ".join(cmd), flush=True)
        subprocess.check_call(cmd, cwd=str(ROOT))


if __name__ == "__main__":
    main()
