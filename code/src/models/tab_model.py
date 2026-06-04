"""FT-Transformer：优先 rtdl，否则 MLP 回退。"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    from rtdl import FTTransformer

    HAS_RTDL = True
except ImportError:
    HAS_RTDL = False


class TabFTTransformer(nn.Module):
    def __init__(
        self,
        n_num_features: int,
        cat_cardinalities: list[int] | None = None,
        d_token: int = 128,
        n_blocks: int = 4,
        dropout: float = 0.2,
        industry_vocab: int = 64,
    ):
        super().__init__()
        self.use_rtdl = HAS_RTDL and cat_cardinalities is not None
        if self.use_rtdl:
            self.model = FTTransformer(
                n_num_features=n_num_features,
                cat_cardinalities=cat_cardinalities,
                d_token=d_token,
                n_blocks=n_blocks,
                attention_dropout=dropout,
                ffn_d_hidden=d_token * 2,
                ffn_dropout=dropout,
                d_out=1,
            )
        else:
            self.ind_emb = nn.Embedding(industry_vocab, d_token // 4)
            in_dim = n_num_features + d_token // 4
            layers = []
            dim = in_dim
            for _ in range(n_blocks):
                layers += [nn.Linear(dim, d_token), nn.ReLU(), nn.Dropout(dropout)]
                dim = d_token
            layers.append(nn.Linear(dim, 1))
            self.mlp = nn.Sequential(*layers)
            self.n_num = n_num_features

    def forward(
        self,
        x_num: torch.Tensor,
        industry_id: torch.Tensor | None = None,
        x_cat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.use_rtdl:
            return self.model(x_num, x_cat).squeeze(-1)
        ind_e = self.ind_emb(industry_id.clamp(0, self.ind_emb.num_embeddings - 1))
        x = torch.cat([x_num, ind_e], dim=-1)
        return self.mlp(x).squeeze(-1)
