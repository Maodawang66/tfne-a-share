"""GKX 式截面 winsorize + 秩变换（逐日，无全样本泄露）。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def winsorize_series(s: pd.Series, q: float = 0.01) -> pd.Series:
    if s.notna().sum() < 5:
        return s
    lo, hi = s.quantile(q), s.quantile(1 - q)
    return s.clip(lo, hi)


def rank_to_unit(s: pd.Series) -> pd.Series:
    """截面秩线性映射到 [-1, 1]；NA 保持 NA。"""
    valid = s.notna()
    if valid.sum() < 2:
        return s
    r = s[valid].rank(method="average")
    n = len(r)
    if n == 1:
        out = pd.Series(0.0, index=s.index)
    else:
        mapped = 2.0 * (r - 1) / (n - 1) - 1.0
        out = pd.Series(np.nan, index=s.index)
        out.loc[valid] = mapped.values
    return out


def cross_section_transform(
    df: pd.DataFrame,
    cols: list[str],
    date_col: str = "trade_date",
    winsor_q: float = 0.01,
    to_rank: bool = True,
) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns and not pd.api.types.is_float_dtype(out[c]):
            out[c] = pd.to_numeric(out[c], errors="coerce").astype(np.float64)
    for d, g in out.groupby(date_col, sort=False):
        idx = g.index
        for c in cols:
            if c not in out.columns:
                continue
            s = winsorize_series(g[c], winsor_q)
            if to_rank:
                s = rank_to_unit(s)
            out.loc[idx, c] = pd.to_numeric(s, errors="coerce").astype(np.float64).values
    return out


def cross_section_winsorize_predictions(
    pred: pd.Series,
    date_index: pd.Series,
    q: float = 0.01,
) -> pd.Series:
    """对预测值按日 winsorize。"""
    df = pd.DataFrame({"pred": pred, "trade_date": date_index})
    out = pred.copy()
    for _, g in df.groupby("trade_date"):
        out.loc[g.index] = winsorize_series(g["pred"], q).values
    return out


def cross_section_rank(values: pd.Series, groups: pd.Series) -> pd.Series:
    """按 groups（trade_date）截面秩，归一化到 (0,1]。"""
    ranks = pd.Series(np.nan, index=values.index)
    for _, g in pd.DataFrame({"v": values, "d": groups}).groupby("d"):
        r = g["v"].rank(method="average", pct=True)
        ranks.loc[g.index] = r.values
    return ranks
