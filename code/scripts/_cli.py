"""脚本公共参数。"""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def base_parser(description: str = "") -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="配置文件路径，默认 code/config.yaml；可用 config.tier_a.yaml / config.smoke_c.yaml",
    )
    return p


def load_cfg(args) -> "AppConfig":
    from src.config import AppConfig

    if args.config:
        path = Path(args.config)
        if not path.is_absolute():
            path = ROOT / path
        return AppConfig.load(path)
    return AppConfig.load()
