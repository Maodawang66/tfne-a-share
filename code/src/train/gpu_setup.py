"""CUDA 检测与训练前 GPU 环境配置。"""

from __future__ import annotations

import torch


def print_gpu_status(prefix: str = "") -> bool:
    """打印 GPU 信息；返回是否将使用 CUDA。"""
    tag = f"[{prefix}] " if prefix else ""
    if not torch.cuda.is_available():
        print(f"{tag}CUDA 不可用，将使用 CPU（任务管理器里不会出现 Python GPU 占用）", flush=True)
        return False
    i = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(i)
    print(
        f"{tag}GPU: {props.name} | "
        f"显存 {props.total_memory / 1024**3:.1f} GB | "
        f"CUDA {torch.version.cuda} | "
        f"PyTorch {torch.__version__}",
        flush=True,
    )
    return True


def configure_cuda(tf32: bool = True) -> None:
    if not torch.cuda.is_available():
        return
    torch.backends.cudnn.benchmark = True
    if tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def log_cuda_memory(prefix: str = "") -> None:
    if not torch.cuda.is_available():
        return
    alloc = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    tag = f"[{prefix}] " if prefix else ""
    print(f"{tag}CUDA mem: allocated={alloc:.0f} MB, reserved={reserved:.0f} MB", flush=True)


def maybe_compile(model: torch.nn.Module, enabled: bool) -> torch.nn.Module:
    if not enabled or not torch.cuda.is_available():
        return model
    if not hasattr(torch, "compile"):
        print("[gpu] torch.compile 不可用，跳过", flush=True)
        return model
    try:
        print("[gpu] 启用 torch.compile（首轮可能较慢）...", flush=True)
        return torch.compile(model)
    except Exception as e:
        print(f"[gpu] torch.compile 失败，使用 eager: {e}", flush=True)
        return model
