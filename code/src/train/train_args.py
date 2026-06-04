"""从 AppConfig 解析训练超参。"""

from __future__ import annotations

from typing import Any, Dict

import torch

from src.config import AppConfig


def _resolve_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        print("[training] 配置为 cuda 但未检测到 GPU，回退 cpu", flush=True)
        return "cpu"
    return name


def training_kwargs(cfg: AppConfig) -> Dict[str, Any]:
    t = dict(cfg.raw.get("training", {}))
    m = cfg.model_cfg()
    device = _resolve_device(str(t.get("device", "cuda")))
    return {
        "batch_size": int(m.get("batch_stocks", 2048)),
        "device_name": device,
        "amp": bool(t.get("amp", device == "cuda")),
        "grad_accum_steps": max(1, int(t.get("grad_accum_steps", 1))),
        "grad_clip": float(t.get("grad_clip", 1.0)),
        "compile_model": bool(t.get("compile", False)),
        "tf32": bool(t.get("tf32", True)),
        "lr": float(m.get("lr", 1e-3)),
        "max_epochs": int(m.get("max_epochs", 30)),
        "patience": int(m.get("patience", 8)),
        "huber_delta": float(m.get("huber_delta", 1.0)),
    }


def predict_kwargs(cfg: AppConfig) -> Dict[str, Any]:
    t = dict(cfg.raw.get("training", {}))
    m = cfg.model_cfg()
    device = _resolve_device(str(t.get("device", "cuda")))
    return {
        "batch_size": int(m.get("batch_stocks", 2048)),
        "device_name": device,
        "amp": bool(t.get("amp", device == "cuda")),
    }
