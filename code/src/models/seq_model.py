"""TFT-lite: LSTM + 静态变量 embedding + 时序注意力。"""

from __future__ import annotations

import torch
import torch.nn as nn


class SeqTFTLite(nn.Module):
    def __init__(
        self,
        n_seq_features: int,
        hidden_size: int = 128,
        n_static: int = 3,
        static_vocab: int = 128,
        industry_vocab: int = 64,
        dropout: float = 0.2,
        n_layers: int = 2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.ind_emb = nn.Embedding(industry_vocab, hidden_size // 4)
        self.static_proj = nn.Linear(hidden_size // 4 + 2, hidden_size)

        self.lstm = nn.LSTM(
            n_seq_features,
            hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.attn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        x_seq: torch.Tensor,
        industry_id: torch.Tensor,
        log_mv: torch.Tensor,
        listing_age: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # x_seq: (B, T, F)
        h, _ = self.lstm(x_seq)
        if mask is not None:
            m = mask.unsqueeze(-1)
            h = h.masked_fill(~m.bool(), 0.0)
            attn_logits = self.attn(h).squeeze(-1)
            mask_fill = torch.tensor(-1e4, device=attn_logits.device, dtype=attn_logits.dtype)
            attn_logits = attn_logits.masked_fill(~mask.bool(), mask_fill)
            # 全零 mask 行 softmax 会 0/0→nan，回退为均匀权重
            row_valid = mask.sum(dim=-1).clamp(min=1.0)
            attn_w = torch.softmax(attn_logits, dim=-1)
            bad = ~torch.isfinite(attn_w).all(dim=-1)
            if bad.any():
                uniform = mask / row_valid.unsqueeze(-1)
                attn_w = torch.where(bad.unsqueeze(-1), uniform, attn_w)
        else:
            attn_logits = self.attn(h).squeeze(-1)
            attn_w = torch.softmax(attn_logits, dim=-1)
        attn_w = attn_w.unsqueeze(-1)
        ctx = (h * attn_w).sum(dim=1)

        ind_e = self.ind_emb(industry_id.clamp(0, self.ind_emb.num_embeddings - 1))
        static_in = torch.cat([ind_e, log_mv.unsqueeze(-1), listing_age.unsqueeze(-1)], dim=-1)
        static_h = self.static_proj(static_in)

        fused = torch.cat([ctx, static_h], dim=-1)
        return self.head(fused).squeeze(-1)
