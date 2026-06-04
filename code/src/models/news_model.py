"""新闻编码：RoBERTa+LoRA 或轻量哈希回退。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

try:
    from transformers import AutoModel, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    HAS_BERT = True
except ImportError:
    HAS_BERT = False

from sklearn.feature_extraction.text import HashingVectorizer


def _resolve_model_path(model_name: str) -> str:
    """支持本地目录或 HF 镜像（环境变量 HF_ENDPOINT，如 https://hf-mirror.com）。"""
    p = Path(model_name)
    if p.is_dir():
        return str(p.resolve())
    local = os.environ.get("NEWS_MODEL_LOCAL")
    if local and Path(local).is_dir():
        return str(Path(local).resolve())
    return model_name


class NewsEncoder(nn.Module):
    def __init__(
        self,
        use_bert: bool = True,
        model_name: str = "hfl/chinese-roberta-wwm-ext",
        lora_r: int = 8,
        lora_alpha: int = 16,
        out_dim: int = 64,
        hash_dim: int = 256,
        max_length: int = 256,
    ):
        super().__init__()
        self.use_bert = use_bert and HAS_BERT
        self.max_length = max_length
        self.out_dim = out_dim

        if self.use_bert:
            name_or_path = _resolve_model_path(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(name_or_path)
            base = AutoModel.from_pretrained(name_or_path)
            cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, target_modules=["query", "value"])
            self.encoder = get_peft_model(base, cfg)
            hidden = base.config.hidden_size
            self.pool_query = nn.Parameter(torch.randn(1, 1, hidden) * 0.02)
            self.proj = nn.Linear(hidden, out_dim)
        else:
            self.hash = HashingVectorizer(n_features=hash_dim, alternate_sign=False)
            self.proj = nn.Sequential(
                nn.Linear(hash_dim, out_dim),
                nn.ReLU(),
            )
            self._hash_fitted = False

        self.head = nn.Linear(out_dim, 1)

    def encode_texts(self, texts: List[str], device: torch.device) -> torch.Tensor:
        if self.use_bert:
            enc = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = self.encoder(**enc).last_hidden_state
            q = self.pool_query.expand(out.size(0), -1, -1)
            attn = torch.softmax((out * q).sum(-1), dim=-1).unsqueeze(-1)
            pooled = (out * attn).sum(1)
            return self.proj(pooled)
        mat = self.hash.transform(texts)
        if hasattr(mat, "toarray"):
            mat = mat.toarray()
        x = torch.tensor(mat, dtype=torch.float32, device=device)
        return self.proj(x)

    def forward(self, texts: List[str], device: torch.device) -> torch.Tensor:
        emb = self.encode_texts(texts, device)
        return self.head(emb).squeeze(-1)
