"""按交易日采样的 PyTorch 数据集（初始化时预计算序列，避免 GPU 空等 CPU）。"""

from __future__ import annotations

from collections import deque
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


def _safe_f32(v, default: float = 0.0) -> float:
    """NaN/Inf/缺失 → default（Python 里 float(nan) or 0 仍为 nan，需显式处理）。"""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    return default if not np.isfinite(x) else x


class DailyStockDataset(Dataset):
    """每个样本为 (ts_code, trade_date)；序列在 __init__ 中一次性预计算。"""

    def __init__(
        self,
        df: pd.DataFrame,
        seq_cols: List[str],
        tab_cols: List[str],
        seq_len: int,
        label_col: str = "excess_ret_1d",
        precompute: bool = True,
        inference_dates: Optional[Iterable[str]] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.seq_cols = seq_cols
        self.tab_cols = tab_cols
        self.seq_len = seq_len
        self.label_col = label_col
        self.n_seq = len(seq_cols)
        self.n_tab = len(tab_cols)
        self._sample_idx: np.ndarray | None = None

        if precompute:
            self._precompute()
        else:
            self.panel = self.df.sort_values(["ts_code", "trade_date"])
            self._seq = self._mask = self._tab = None

        if inference_dates is not None:
            dates = {str(d) for d in inference_dates}
            self._sample_idx = np.flatnonzero(
                self.df["trade_date"].astype(str).isin(dates).to_numpy()
            )
            if len(self._sample_idx) == 0:
                raise ValueError(f"inference_dates 在 panel 中无样本: {sorted(dates)}")

    def _precompute(self) -> None:
        n = len(self.df)
        self._seq = np.zeros((n, self.seq_len, self.n_seq), dtype=np.float32)
        self._mask = np.zeros((n, self.seq_len), dtype=np.float32)
        self._tab = np.zeros((n, self.n_tab), dtype=np.float32)
        self._y = np.zeros(n, dtype=np.float32)
        self._industry_id = np.zeros(n, dtype=np.int64)
        self._log_mv = np.zeros(n, dtype=np.float32)
        self._listing_age = np.zeros(n, dtype=np.float32)

        order = self.df.sort_values(["ts_code", "trade_date"]).index.to_numpy()
        buffers: Dict[str, deque] = {}

        for idx in tqdm(order, desc="precompute_seq", leave=False):
            row = self.df.loc[idx]
            code = row["ts_code"]
            feat = np.nan_to_num(
                row[self.seq_cols].to_numpy(dtype=np.float32),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            if code not in buffers:
                buffers[code] = deque(maxlen=self.seq_len)
            buffers[code].append(feat)
            window = list(buffers[code])
            t = len(window)
            self._seq[idx, -t:] = window
            self._mask[idx, -t:] = 1.0
            self._tab[idx] = np.nan_to_num(
                row[self.tab_cols].to_numpy(dtype=np.float32),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            y = row[self.label_col]
            self._y[idx] = _safe_f32(y, 0.0)
            self._industry_id[idx] = int(row.get("industry_id", 0) or 0)
            self._log_mv[idx] = _safe_f32(row.get("log_mv", 0), 0.0)
            self._listing_age[idx] = _safe_f32(row.get("listing_age_days", 0), 0.0) / 3650.0

    def __len__(self) -> int:
        if self._sample_idx is not None:
            return len(self._sample_idx)
        return len(self.df)

    def _get_seq_legacy(self, ts_code: str, end_date: str) -> Tuple[np.ndarray, np.ndarray]:
        sub = self.panel[self.panel["ts_code"] == ts_code]
        sub = sub[sub["trade_date"] <= end_date].tail(self.seq_len)
        t = len(sub)
        seq = np.nan_to_num(
            sub[self.seq_cols].to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
        )
        mask = np.ones(t, dtype=np.float32)
        if t < self.seq_len:
            pad = np.zeros((self.seq_len - t, seq.shape[1]), dtype=np.float32)
            seq = np.vstack([pad, seq])
            mask = np.concatenate([np.zeros(self.seq_len - t), mask])
        return seq, mask

    def __getitem__(self, idx: int) -> dict:
        if self._sample_idx is not None:
            idx = int(self._sample_idx[idx])
        if self._seq is not None:
            return {
                "seq": torch.from_numpy(self._seq[idx]),
                "mask": torch.from_numpy(self._mask[idx]),
                "tab": torch.from_numpy(self._tab[idx]),
                "industry_id": torch.tensor(self._industry_id[idx], dtype=torch.long),
                "log_mv": torch.tensor(self._log_mv[idx], dtype=torch.float32),
                "listing_age": torch.tensor(self._listing_age[idx], dtype=torch.float32),
                "y": torch.tensor(self._y[idx], dtype=torch.float32),
                "trade_date": self.df.iloc[idx]["trade_date"],
                "ts_code": self.df.iloc[idx]["ts_code"],
            }
        row = self.df.iloc[idx]
        seq, mask = self._get_seq_legacy(row["ts_code"], row["trade_date"])
        tab = np.nan_to_num(
            row[self.tab_cols].to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
        )
        y = row[self.label_col]
        y = _safe_f32(y, 0.0)
        return {
            "seq": torch.tensor(seq),
            "mask": torch.tensor(mask),
            "tab": torch.tensor(tab),
            "industry_id": torch.tensor(int(row.get("industry_id", 0) or 0), dtype=torch.long),
            "log_mv": torch.tensor(_safe_f32(row.get("log_mv", 0), 0.0), dtype=torch.float32),
            "listing_age": torch.tensor(
                _safe_f32(row.get("listing_age_days", 0), 0.0) / 3650.0, dtype=torch.float32
            ),
            "y": torch.tensor(y, dtype=torch.float32),
            "trade_date": row["trade_date"],
            "ts_code": row["ts_code"],
        }


def collate_daily(batch: List[dict]) -> dict:
    return {
        "seq": torch.stack([b["seq"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "tab": torch.stack([b["tab"] for b in batch]),
        "industry_id": torch.stack([b["industry_id"] for b in batch]),
        "log_mv": torch.stack([b["log_mv"] for b in batch]),
        "listing_age": torch.stack([b["listing_age"] for b in batch]),
        "y": torch.stack([b["y"] for b in batch]),
        "trade_date": [b["trade_date"] for b in batch],
        "ts_code": [b["ts_code"] for b in batch],
    }


def split_by_date(df: pd.DataFrame, train_end: str, valid_end: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[df["trade_date"] <= train_end]
    valid = df[(df["trade_date"] > train_end) & (df["trade_date"] <= valid_end)]
    test = df[df["trade_date"] > valid_end]
    return train, valid, test
