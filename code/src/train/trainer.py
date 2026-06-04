"""训练循环：预计算数据 + GPU batch + AMP + 梯度累积。"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import DailyStockDataset, collate_daily, split_by_date
from src.metrics import daily_ic, ic_summary
from src.train.gpu_setup import configure_cuda, log_cuda_memory, maybe_compile, print_gpu_status


def get_device(name: str = "cuda") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        print("[train] 请求 cuda 但不可用，回退 cpu", flush=True)
        return torch.device("cpu")
    return torch.device(name)


def _autocast_ctx(enabled: bool, device: torch.device):
    if enabled and device.type == "cuda":
        return torch.amp.autocast("cuda")
    return nullcontext()


def _grad_scaler(enabled: bool, device: torch.device):
    if enabled and device.type == "cuda":
        return torch.amp.GradScaler("cuda")
    return torch.amp.GradScaler("cpu", enabled=False)


def _move_batch(batch: dict, device: torch.device) -> dict:
    nb = device.type == "cuda"
    out = {
        "seq": batch["seq"].to(device, non_blocking=nb),
        "mask": batch["mask"].to(device, non_blocking=nb),
        "tab": batch["tab"].to(device, non_blocking=nb),
        "industry_id": batch["industry_id"].to(device, non_blocking=nb),
        "log_mv": batch["log_mv"].to(device, non_blocking=nb),
        "listing_age": batch["listing_age"].to(device, non_blocking=nb),
        "y": batch["y"].to(device, non_blocking=nb),
        "trade_date": batch["trade_date"],
        "ts_code": batch["ts_code"],
    }
    return out


def huber_loss(pred: torch.Tensor, target: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    return nn.functional.huber_loss(pred, target, delta=delta)


def _make_loader(ds: DailyStockDataset, batch_size: int, shuffle: bool, pin_memory: bool) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_daily,
        num_workers=0,
        pin_memory=pin_memory,
        drop_last=False,
    )


@torch.no_grad()
def eval_ic(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    forward_fn: Callable,
    use_amp: bool,
) -> float:
    model.eval()
    preds, labels, dates = [], [], []
    for batch in loader:
        bg = _move_batch(batch, device)
        with _autocast_ctx(use_amp, device):
            pred = forward_fn(model, bg, device, use_amp)
        preds.extend(pred.detach().float().cpu().numpy().tolist())
        labels.extend(bg["y"].cpu().numpy().tolist())
        dates.extend(bg["trade_date"])
    df = pd.DataFrame({"score": preds, "excess_ret_1d": labels, "trade_date": dates})
    return ic_summary(daily_ic(df))["mean_ic"]


def train_model(
    model: nn.Module,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    dataset_kwargs: dict,
    forward_fn: Callable,
    ckpt_path: Path,
    lr: float = 1e-3,
    max_epochs: int = 30,
    patience: int = 8,
    batch_size: int = 2048,
    device_name: str = "cuda",
    huber_delta: float = 1.0,
    amp: bool = True,
    grad_accum_steps: int = 1,
    grad_clip: float = 1.0,
    compile_model: bool = False,
    tf32: bool = True,
    ind_vocab: Optional[int] = None,
) -> dict:
    print_gpu_status("train")
    configure_cuda(tf32=tf32)
    device = get_device(device_name)
    use_amp = bool(amp and device.type == "cuda")
    pin_memory = device.type == "cuda"
    grad_accum_steps = max(1, grad_accum_steps)

    model = model.to(device)
    model = maybe_compile(model, compile_model and device.type == "cuda")
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = _grad_scaler(use_amp, device)

    print("Building datasets (precompute sequences once) ...", flush=True)
    train_ds = DailyStockDataset(train_df, **dataset_kwargs)
    valid_ds = DailyStockDataset(valid_df, **dataset_kwargs)
    eff_batch = batch_size * grad_accum_steps
    print(
        f"train={len(train_ds)}, valid={len(valid_ds)}, device={device}, amp={use_amp}, "
        f"micro_batch={batch_size}, accum={grad_accum_steps}, effective_batch≈{eff_batch}, "
        f"epochs={max_epochs}",
        flush=True,
    )

    train_loader = _make_loader(train_ds, batch_size, shuffle=True, pin_memory=pin_memory)
    valid_loader = _make_loader(valid_ds, batch_size, shuffle=False, pin_memory=pin_memory)

    best_ic = -1e9
    wait = 0
    history = []
    opt.zero_grad(set_to_none=True)

    for epoch in range(max_epochs):
        model.train()
        losses = []
        nan_batches = 0
        n_batches = 0
        micro = 0
        for batch in tqdm(train_loader, desc=f"epoch {epoch + 1}/{max_epochs}", leave=True):
            n_batches += 1
            bg = _move_batch(batch, device)
            with _autocast_ctx(use_amp, device):
                pred = forward_fn(model, bg, device, use_amp)
                loss = huber_loss(pred, bg["y"], huber_delta) / grad_accum_steps
            if not torch.isfinite(loss):
                nan_batches += 1
                continue
            scaler.scale(loss).backward()
            micro += 1
            losses.append(loss.item() * grad_accum_steps)

            if micro % grad_accum_steps == 0:
                if grad_clip > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

        if micro % grad_accum_steps != 0:
            if grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

        if device.type == "cuda":
            log_cuda_memory(f"epoch {epoch + 1}")

        if n_batches > 0 and nan_batches == n_batches:
            print(
                f"epoch {epoch + 1}: 全部 {n_batches} 个 batch loss 非有限数，"
                f"本 epoch 未更新权重（请检查 AMP/特征 NaN/学习率）",
                flush=True,
            )
        elif nan_batches > 0:
            print(
                f"epoch {epoch + 1}: {nan_batches}/{n_batches} batch loss 非有限数，已跳过",
                flush=True,
            )

        print(f"epoch {epoch + 1}: validating IC ...", flush=True)
        val_ic = eval_ic(model, valid_loader, device, forward_fn, use_amp)
        epoch_loss = float(np.mean(losses)) if losses else float("nan")
        history.append({"epoch": epoch, "loss": epoch_loss, "val_ic": val_ic, "nan_batches": nan_batches})
        print(
            f"epoch {epoch + 1}: loss={epoch_loss:.4f} val_ic={val_ic:.4f} best={best_ic:.4f}",
            flush=True,
        )
        if not losses and epoch >= 1:
            print("[train] 连续 epoch 无有效 loss，提前停止", flush=True)
            break
        if val_ic > best_ic:
            best_ic = val_ic
            wait = 0
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            to_save = model._orig_mod if hasattr(model, "_orig_mod") else model
            payload = {"model": to_save.state_dict(), "epoch": epoch}
            if ind_vocab is not None:
                payload["ind_vocab"] = int(ind_vocab)
            torch.save(payload, ckpt_path)
        else:
            wait += 1
            if wait >= patience:
                break

    if ckpt_path.exists():
        try:
            state = torch.load(ckpt_path, map_location=device, weights_only=False)
        except TypeError:
            state = torch.load(ckpt_path, map_location=device)
        target = model._orig_mod if hasattr(model, "_orig_mod") else model
        target.load_state_dict(state["model"])

    return {"best_val_ic": best_ic, "history": history}


@torch.no_grad()
def predict_model(
    model: nn.Module,
    df: pd.DataFrame,
    dataset_kwargs: dict,
    forward_fn: Callable,
    device_name: str = "cuda",
    batch_size: int = 2048,
    amp: bool = False,
    dataset: DailyStockDataset | None = None,
) -> pd.DataFrame:
    print_gpu_status("predict")
    device = get_device(device_name)
    use_amp = bool(amp and device.type == "cuda")
    pin_memory = device.type == "cuda"
    model = model.to(device)
    model.eval()
    ds = dataset if dataset is not None else DailyStockDataset(df, **dataset_kwargs)
    loader = _make_loader(ds, batch_size, shuffle=False, pin_memory=pin_memory)

    rows = []
    for batch in tqdm(loader, desc="predict", leave=False):
        bg = _move_batch(batch, device)
        with _autocast_ctx(use_amp, device):
            pred = forward_fn(model, bg, device, use_amp)
        pred_np = pred.detach().float().cpu().numpy()
        for i in range(len(pred_np)):
            rows.append(
                {
                    "ts_code": bg["ts_code"][i],
                    "trade_date": bg["trade_date"][i],
                    "z": float(pred_np[i]),
                }
            )
    return pd.DataFrame(rows)


def forward_seq(model, batch, device, use_amp: bool = False):
    return model(
        batch["seq"],
        batch["industry_id"],
        batch["log_mv"],
        batch["listing_age"],
        batch["mask"],
    )


def forward_tab(model, batch, device, use_amp: bool = False):
    return model(batch["tab"], batch["industry_id"])
