"""按档位 A / 档位 C 构建模型与 forward 函数。"""

from __future__ import annotations

from typing import Callable, Tuple

import torch.nn as nn

from src.config import AppConfig
from src.models.news_model import NewsEncoder
from src.models.seq_model import SeqTFTLite
from src.models.tab_model import TabFTTransformer
from src.models.tier_a import SeqGRU, TabMLP


def build_seq_model(cfg: AppConfig, n_seq_features: int, industry_vocab: int) -> nn.Module:
    m = cfg.model_cfg()
    if cfg.tier == "A":
        return SeqGRU(
            n_features=n_seq_features,
            hidden=int(m.get("gru_hidden", 64)),
            dropout=float(m.get("dropout", 0.2)),
        )
    return SeqTFTLite(
        n_seq_features=n_seq_features,
        hidden_size=int(m.get("hidden_size", 128)),
        industry_vocab=industry_vocab,
        dropout=float(m.get("dropout", 0.2)),
    )


def build_tab_model(cfg: AppConfig, n_tab_features: int, industry_vocab: int) -> nn.Module:
    m = cfg.model_cfg()
    if cfg.tier == "A":
        return TabMLP(
            n_features=n_tab_features,
            hidden=int(m.get("mlp_hidden", 128)),
            dropout=float(m.get("dropout", 0.2)),
        )
    return TabFTTransformer(
        n_num_features=n_tab_features,
        cat_cardinalities=[industry_vocab],
        d_token=int(m.get("ftt_d_token", 128)),
        n_blocks=int(m.get("ftt_n_blocks", 4)),
        dropout=float(m.get("dropout", 0.2)),
        industry_vocab=industry_vocab,
    )


def build_news_encoder(cfg: AppConfig) -> NewsEncoder:
    m = cfg.model_cfg()
    if cfg.tier == "A":
        # 档位 A：冻结语义，仅轻量哈希 + 线性（等价于冻结 BERT 的工程简化）
        return NewsEncoder(
            use_bert=False,
            hash_dim=int(m.get("news_hash_dim", 256)),
            out_dim=int(m.get("news_out_dim", 32)),
        )
    use_bert = bool(m.get("use_news_bert", True))
    return NewsEncoder(
        use_bert=use_bert,
        model_name=str(m.get("news_model_name", "hfl/chinese-roberta-wwm-ext")),
        lora_r=int(m.get("lora_r", 8)),
        lora_alpha=int(m.get("lora_alpha", 16)),
        max_length=int(m.get("news_max_length", 256)),
        out_dim=int(m.get("news_out_dim", 64)),
    )


def get_tab_forward_fn(model: nn.Module) -> Callable:
    if getattr(model, "use_rtdl", False):

        def forward_tab_rtdl(model, batch, device, use_amp: bool = False):
            x_cat = batch["industry_id"].unsqueeze(-1)
            return model(batch["tab"], batch["industry_id"], x_cat)

        return forward_tab_rtdl

    from src.train.trainer import forward_tab

    return forward_tab


def modal_names(cfg: AppConfig) -> Tuple[str, str, str]:
    """返回 checkpoint / prediction 子目录名（时序、截面、新闻）。"""
    if cfg.tier == "A":
        return "gru", "mlp", "news_hash"
    return "tft", "ftt", "news"
