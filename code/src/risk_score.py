"""
风险调整打分（Risk-Adjusted Alpha Scoring）

业界常见做法（与本模块方法对应）：
- sharpe：均值-方差筛选，score ∝ μ̂/σ（Markowitz 1952；因子投资中广泛使用）
- certainty_equivalent：μ - (γ/2)σ²（Merton 1969；JPMorgan QIS 等确定性等价组合）
- rank_penalty：rank(μ) - λ·rank(σ)（Barra/MSCI 分离 alpha 与 risk 的简化版）

说明：
- 本流水线 NN 仅预测 E[r]（excess_ret_1d）；σ 用过去 20 日实现波动（vol_20d_raw）估计，
  与 Barra 用独立风险模型、GKX(2020) 在组合层处理风险的思路一致。
- 无需重训模型：在 04 秩融合之后对 score 做后处理即可。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.preprocess import cross_section_rank


def _compute_z_blend(fused: pd.DataFrame, weights: Optional[Dict[str, float]] = None) -> pd.Series:
    """融合模态 winsorized z（z_A/z_B/z_C）为单一 alpha 预测 μ̂。"""
    modal_cols = {"A": "z_A", "B": "z_B", "C": "z_C"}
    if weights is None:
        weights = {m: 1.0 for m in modal_cols if modal_cols[m] in fused.columns}
    z = pd.Series(0.0, index=fused.index, dtype=np.float64)
    wsum = 0.0
    for m, col in modal_cols.items():
        if col not in fused.columns:
            continue
        w = float(weights.get(m, 0.0))
        if w <= 0:
            continue
        v = fused[col].to_numpy(dtype=np.float64)
        valid = np.isfinite(v)
        z.iloc[valid] += v[valid] * w
        wsum += w
    if wsum > 0:
        z = z / wsum
    return z


def _compute_raw_vol_from_daily(daily: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    df = daily.sort_values(["ts_code", "trade_date"]).copy()
    df["ret_1d"] = df.groupby("ts_code")["close"].pct_change()
    mp = max(3, window // 2)
    df["vol_20d_raw"] = df.groupby("ts_code")["ret_1d"].transform(
        lambda s: s.rolling(window, min_periods=mp).std()
    )
    return df[["trade_date", "ts_code", "vol_20d_raw"]]


def attach_realized_vol(
    fused: pd.DataFrame,
    panel: pd.DataFrame,
    vol_col: str = "vol_20d_raw",
    daily_panel_path: Optional[Path] = None,
    vol_floor: float = 0.005,
) -> pd.Series:
    """为 fused 行附加实现波动率（优先 panel 列，否则从 panel_daily 重算）。"""
    keys = fused[["trade_date", "ts_code"]].copy()
    if vol_col in panel.columns:
        vol_df = panel[["trade_date", "ts_code", vol_col]].drop_duplicates()
    elif daily_panel_path is not None and daily_panel_path.exists():
        daily = pd.read_parquet(daily_panel_path, columns=["trade_date", "ts_code", "close"])
        vol_df = _compute_raw_vol_from_daily(daily)
        vol_col = "vol_20d_raw"
    else:
        raise KeyError(
            f"未找到波动率列 {vol_col}；请重新运行 02_make_features.py 生成 *_raw 列，"
            "或保留 panel_daily.parquet 供回退计算。"
        )
    merged = keys.merge(vol_df, on=["trade_date", "ts_code"], how="left")
    vol = pd.to_numeric(merged[vol_col], errors="coerce").astype(float)
    return vol.clip(lower=vol_floor).reset_index(drop=True)


def apply_risk_adjustment(
    fused: pd.DataFrame,
    panel: pd.DataFrame,
    risk_cfg: Optional[dict] = None,
    modal_weights: Optional[Dict[str, float]] = None,
    daily_panel_path=None,
) -> pd.DataFrame:
    """
    在秩融合 score 上施加风险调整，写回 fused['score']，并保留 score_raw。

    risk_cfg 字段：
      enabled, method, vol_col, vol_floor, risk_aversion, gamma
    """
    ra = risk_cfg or {}
    out = fused.copy()
    out["score_raw"] = out["score"]

    if not ra.get("enabled", False):
        return out

    method = str(ra.get("method", "sharpe")).lower()
    vol_col = str(ra.get("vol_col", "vol_20d_raw"))
    vol_floor = float(ra.get("vol_floor", 0.005))
    lam = float(ra.get("risk_aversion", 0.5))
    gamma = float(ra.get("gamma", 1.0))

    vol = attach_realized_vol(out, panel, vol_col=vol_col, daily_panel_path=daily_panel_path, vol_floor=vol_floor)
    out["vol_used"] = vol.values
    z_blend = _compute_z_blend(out, modal_weights)
    out["z_blend"] = z_blend

    if method == "sharpe":
        mu = z_blend.copy()
        if not mu.notna().any():
            mu = out["score_raw"].astype(float)
        ratio = mu / vol
        out["score"] = cross_section_rank(ratio, out["trade_date"])
    elif method in ("certainty_equivalent", "ce", "merton"):
        mu = z_blend.copy()
        if not mu.notna().any():
            mu = out["score_raw"].astype(float)
        ce = mu - gamma * (vol ** 2)
        out["score"] = cross_section_rank(ce, out["trade_date"])
    elif method in ("rank_penalty", "penalty"):
        vol_r = cross_section_rank(vol, out["trade_date"])
        out["score"] = out["score_raw"] - lam * vol_r
    else:
        raise ValueError(f"未知 risk_adjust.method: {method}")

    return out
