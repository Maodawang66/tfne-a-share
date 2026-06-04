#!/usr/bin/env python
"""检查 panel_features：仅 inf/缺列/极端值/样本过少会 fail；NaN 只警告。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import AppConfig
from src.preflight import check_panel_health, print_preflight


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.server_c_tfne_72h.yaml")
    parser.add_argument("--strict-nan", action="store_true", help="调试：有任何 NaN 即 exit 1")
    args = parser.parse_args()

    cfg = AppConfig.load(ROOT / args.config)
    result = check_panel_health(cfg, strict_nan=args.strict_nan)
    print_preflight(result, title="health")
    result.exit_if_fatal()


if __name__ == "__main__":
    main()
