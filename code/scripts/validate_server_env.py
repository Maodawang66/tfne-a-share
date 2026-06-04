#!/usr/bin/env python
"""服务器 ai25 全项目环境预检（登录节点可跑；CUDA 在登录节点为 False 也正常）。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    ok = True

    def check(name: str, fn):
        nonlocal ok
        try:
            fn()
            print(f"  [OK] {name}")
        except Exception as e:
            ok = False
            print(f"  [FAIL] {name}: {e}")

    print("[validate_server_env] 全项目依赖检查")
    check("pandas", lambda: __import__("pandas"))
    check("pyarrow", lambda: __import__("pyarrow"))
    check("torch", lambda: __import__("torch"))
    check("transformers", lambda: __import__("transformers"))
    check("peft", lambda: __import__("peft"))

    import torch

    print(f"  torch {torch.__version__}, cuda built={torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print("  cuda available=False（登录节点正常；GPU 作业内应变为 True）")

    code = ROOT / "scripts" / "run_server_c_tfne_72h.py"
    if not code.exists():
        ok = False
        print(f"  [FAIL] missing {code}")
    else:
        print(f"  [OK] {code}")

    legacy = ROOT / "scripts" / "run_server_c.py"
    if legacy.exists():
        print(f"  [OK] {legacy}")

    if ok:
        print("\n预检通过。可提交 prep / train 作业。")
    else:
        print("\n预检失败。运行: bash scripts/setup_server_ai25.sh")
        sys.exit(1)


if __name__ == "__main__":
    main()
