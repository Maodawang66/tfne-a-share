"""档位 A：小 GRU + MLP（本地 smoke test，见项目方案第十五节）。"""

from __future__ import annotations

import torch
import torch.nn as nn


class SeqGRU(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x_seq, industry_id=None, log_mv=None, listing_age=None, mask=None):
        h, _ = self.gru(x_seq)
        if mask is not None:
            idx = mask.sum(1).long().clamp(min=1) - 1
            last = h[torch.arange(h.size(0)), idx]
        else:
            last = h[:, -1]
        return self.head(last).squeeze(-1)


class TabMLP(nn.Module):
    def __init__(self, n_features: int, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x_num, industry_id=None, x_cat=None):
        return self.net(x_num).squeeze(-1)
