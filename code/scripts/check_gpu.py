#!/usr/bin/env python
"""检查本机 PyTorch 是否可用 CUDA，并给出训练时如何观察 GPU 占用。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from src.train.gpu_setup import configure_cuda, log_cuda_memory, print_gpu_status


def main():
    print("=" * 60)
    print_gpu_status("check")
    if not torch.cuda.is_available():
        print("\n建议：安装带 CUDA 的 PyTorch，例如 cu118 轮子。")
        return

    configure_cuda(tf32=True)
    x = torch.randn(4096, 128, 20, device="cuda")
    w = torch.randn(128, 64, device="cuda")
    for _ in range(50):
        y = torch.matmul(x.mean(2), w)
        y = torch.relu(y)
    torch.cuda.synchronize()
    log_cuda_memory("stress")
    print("\n若上面显示 GPU 名称且 mem>0，说明 Python 正在使用显卡。")
    print("任务管理器 → 性能 → GPU → 选「CUDA」或「Compute_0」，不要只看「3D」。")
    print("或在另一终端运行: nvidia-smi -l 1")


if __name__ == "__main__":
    main()
