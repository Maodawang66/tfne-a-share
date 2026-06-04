#!/usr/bin/env python
"""档位 A 基线全流程（GRU + MLP + 新闻哈希，对照实验 E1）。"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG = "config.tier_a_local.yaml"  # 本机默认；通用配置可用 config.tier_a.yaml

if __name__ == "__main__":
    cmd = [sys.executable, str(ROOT / "scripts" / "run_tier_a_local.py")]
    print(">>>", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(ROOT))
