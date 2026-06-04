"""Checkpoint 元数据：ind_vocab 等与 panel 不一致时仍能 load。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from src.config import AppConfig
from src.models.factory import build_seq_model, build_tab_model, get_tab_forward_fn

PATCH_ID = "ind_vocab_resolve_v2"


def infer_ind_vocab_from_state_dict(state_dict: Dict[str, Any]) -> Optional[int]:
    """从 state_dict 推断 industry_vocab（SeqTFT / TabMLP / rtdl cat_embeddings）。"""
    for key in ("ind_emb.weight", "model.ind_emb.weight"):
        w = state_dict.get(key)
        if w is not None:
            return int(w.shape[0])
    for key, w in state_dict.items():
        if "cat_embeddings" in key and key.endswith(".weight"):
            return int(w.shape[0])
    return None


def load_checkpoint_bundle(ckpt_path: Path) -> Dict[str, Any]:
    try:
        return torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(ckpt_path, map_location="cpu")


def ind_vocab_from_checkpoint(ckpt_path: Path) -> Optional[int]:
    bundle = load_checkpoint_bundle(ckpt_path)
    if "ind_vocab" in bundle:
        return int(bundle["ind_vocab"])
    return infer_ind_vocab_from_state_dict(bundle.get("model", {}))


def resolve_ind_vocab(
    panel_max_industry_id: int,
    ckpt_paths: list[Path],
    meta_industry_vocab: Optional[int] = None,
) -> int:
    """
    构建模型用的 industry_vocab：取 panel 与所有 checkpoint 中的最大值。
    forward 里会对 industry_id clamp；panel 行业编号若与训练时不一致，应重跑 C2 prep。
    """
    panel_vocab = int(panel_max_industry_id + 2)
    vocabs = [panel_vocab]
    if meta_industry_vocab is not None:
        vocabs.append(int(meta_industry_vocab))
    for p in ckpt_paths:
        if p.exists():
            v = ind_vocab_from_checkpoint(p)
            if v is not None:
                vocabs.append(v)
    resolved = max(vocabs)
    if resolved > panel_vocab:
        print(
            f"[warn] ind_vocab panel={panel_vocab} < checkpoint={resolved}；"
            "将用 checkpoint 大小加载（panel 可能被 Top500 prep 覆盖，建议服务器重跑 C2 prep）",
            flush=True,
        )
    return resolved


def _ind_vocab_for_load(
    bundle: Dict[str, Any],
    ckpt_path: Path,
    panel_vocab: int,
    fallback: Optional[int] = None,
) -> int:
    ckpt_vocab = bundle.get("ind_vocab")
    if ckpt_vocab is not None:
        ckpt_vocab = int(ckpt_vocab)
    else:
        ckpt_vocab = infer_ind_vocab_from_state_dict(bundle.get("model", {}))
    if ckpt_vocab is None:
        ckpt_vocab = fallback if fallback is not None else panel_vocab
        if ckpt_vocab == panel_vocab:
            print(f"[warn] {ckpt_path}: 无法从 checkpoint 推断 ind_vocab，回退 panel={panel_vocab}", flush=True)
    return max(int(panel_vocab), int(ckpt_vocab))


def _load_state(model: nn.Module, state_dict: Dict[str, Any], ckpt_path: Path) -> None:
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        raise RuntimeError(
            f"加载 checkpoint 失败: {ckpt_path}\n{e}\n"
            "请确认已上传 src/checkpoint_utils.py 与 scripts/04_ensemble_predict.py，"
            f"且日志含 [04_ensemble_predict] patch={PATCH_ID}"
        ) from e


def load_seq_model_from_checkpoint(
    cfg: AppConfig,
    n_features: int,
    ckpt_path: Path,
    panel_vocab: int,
    fallback_vocab: Optional[int] = None,
) -> nn.Module:
    bundle = load_checkpoint_bundle(ckpt_path)
    ind_vocab = _ind_vocab_for_load(bundle, ckpt_path, panel_vocab, fallback_vocab)
    model = build_seq_model(cfg, n_features, ind_vocab)
    _load_state(model, bundle["model"], ckpt_path)
    return model


def load_tab_model_from_checkpoint(
    cfg: AppConfig,
    n_features: int,
    ckpt_path: Path,
    panel_vocab: int,
    fallback_vocab: Optional[int] = None,
) -> tuple[nn.Module, Any]:
    bundle = load_checkpoint_bundle(ckpt_path)
    ind_vocab = _ind_vocab_for_load(bundle, ckpt_path, panel_vocab, fallback_vocab)
    model = build_tab_model(cfg, n_features, ind_vocab)
    _load_state(model, bundle["model"], ckpt_path)
    return model, get_tab_forward_fn(model)


def verify_ind_vocab_patch_deployed(code_root: Path) -> None:
    """eval 前检查：防止服务器仍跑旧版 04。"""
    script = code_root / "scripts" / "04_ensemble_predict.py"
    utils = code_root / "src" / "checkpoint_utils.py"
    if not utils.exists():
        raise FileNotFoundError(f"缺少 {utils}，请先上传 ind_vocab 修复文件")
    text = script.read_text(encoding="utf-8")
    if "resolve_ind_vocab" not in text or "load_seq_model_from_checkpoint" not in text:
        raise RuntimeError(
            f"{script} 仍是旧版（无 ind_vocab 修复）。"
            f"请上传最新 04_ensemble_predict.py（patch={PATCH_ID}）"
        )
