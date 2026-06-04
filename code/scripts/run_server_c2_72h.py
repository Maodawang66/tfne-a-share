#!/usr/bin/env python
"""
TFNE-C2 · 第二代档位 C 入口（config.server_c2_72h.yaml）。

  prep   : 01 + 02 + 02b（CPU，Top3000，必须单独 sbatch）
  train  : TFT + FTT + NewsLoRA × 3 seeds
  eval   : 04 融合 + 05 IC + 06 回测 + 07 订单

  train-tft | train-ftt | train-news | predict-news  可拆分

示例：
  python scripts/run_server_c2_72h.py prep
  python scripts/run_server_c2_72h.py train
  python scripts/run_server_c2_72h.py predict-news --seeds 42 123 456
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.checkpoint_utils import verify_ind_vocab_patch_deployed
from src.config import AppConfig
from src.preflight import print_preflight, run_eval_preflight, run_train_preflight

CONFIG_NAME = "config.server_c2_72h.yaml"


def _py(script: str, *cli_args: str) -> list[str]:
    return [sys.executable, str(ROOT / "scripts" / script), *cli_args]


def run_cmd(cmd: list[str]) -> None:
    t0 = time.time()
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(ROOT))
    print(f"<<< done in {(time.time() - t0) / 3600:.2f} h", flush=True)


def _config_flag() -> list[str]:
    return ["--config", CONFIG_NAME]


def preflight(for_train: bool = False, for_eval: bool = False, seeds: list[int] | None = None) -> None:
    cfg_path = ROOT / CONFIG_NAME
    if not cfg_path.exists():
        raise FileNotFoundError(f"缺少配置 {cfg_path}")
    cfg = AppConfig.load(cfg_path)
    print("[preflight]", cfg.describe(), flush=True)

    if for_eval:
        result = run_eval_preflight(cfg, seeds=seeds, require_cuda=False)
    elif for_train:
        result = run_train_preflight(cfg, require_cuda=True, config_name=CONFIG_NAME)
    else:
        return

    print_preflight(result, title="preflight")
    result.exit_if_fatal()


def stage_prep_incremental() -> None:
    run_cmd(_py("prep_incremental.py") + _config_flag())


def stage_prep() -> None:
    nj = int(AppConfig.load(ROOT / CONFIG_NAME).raw.get("training", {}).get("panel_n_jobs", 16))
    flag = _config_flag()
    run_cmd(_py("01_build_panel.py") + ["--n-jobs", str(nj)] + flag)
    run_cmd(_py("02_make_features.py") + flag)
    run_cmd(_py("02b_embed_news.py") + flag)


def stage_train(seeds: list[int] | None, modals: list[str]) -> None:
    cfg = AppConfig.load(ROOT / CONFIG_NAME)
    seeds = seeds or cfg.seeds
    preflight(for_train=True)
    flag = _config_flag()
    modal_scripts = {
        "tft": "03_train_tft.py",
        "ftt": "03_train_ftt.py",
        "news": "03_train_news_lora.py",
    }
    for modal in modals:
        script = modal_scripts[modal]
        for s in seeds:
            run_cmd(_py(script, "--seed", str(s)) + flag)


def stage_predict_news(seeds: list[int] | None) -> None:
    cfg = AppConfig.load(ROOT / CONFIG_NAME)
    seeds = seeds or cfg.seeds
    flag = _config_flag()
    for s in seeds:
        run_cmd(_py("03_predict_news_lora.py", "--seed", str(s)) + flag)


def stage_eval(seeds: list[int] | None = None) -> None:
    verify_ind_vocab_patch_deployed(ROOT)
    cfg = AppConfig.load(ROOT / CONFIG_NAME)
    seeds = seeds or cfg.seeds
    preflight(for_eval=True, seeds=seeds)
    flag = _config_flag()
    run_cmd(_py("04_ensemble_predict.py") + flag)
    run_cmd(_py("05_evaluate_ic.py") + flag)
    run_cmd(_py("06_backtest.py") + flag)
    run_cmd(_py("07_predict_latest.py") + flag)


def main() -> None:
    parser = argparse.ArgumentParser(description="TFNE-C2 · 72h 满预算流水线")
    parser.add_argument(
        "stage",
        choices=[
            "prep",
            "prep-inc",
            "train",
            "train-tft",
            "train-ftt",
            "train-news",
            "predict-news",
            "eval",
            "all",
        ],
    )
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    args = parser.parse_args()

    if args.stage == "prep":
        stage_prep()
    elif args.stage == "prep-inc":
        stage_prep_incremental()
    elif args.stage == "train":
        stage_train(args.seeds, ["tft", "ftt", "news"])
    elif args.stage == "train-tft":
        stage_train(args.seeds, ["tft"])
    elif args.stage == "train-ftt":
        stage_train(args.seeds, ["ftt"])
    elif args.stage == "train-news":
        stage_train(args.seeds, ["news"])
    elif args.stage == "predict-news":
        stage_predict_news(args.seeds)
    elif args.stage == "eval":
        stage_eval(args.seeds)
    else:
        stage_prep()
        stage_train(args.seeds, ["tft", "ftt", "news"])
        stage_eval(args.seeds)


if __name__ == "__main__":
    main()
