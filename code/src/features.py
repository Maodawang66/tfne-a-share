"""时序/截面特征工程。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.preprocess import cross_section_transform

SEQ_BASE_COLS = [
    "ret_1d", "ret_5d", "ret_10d", "ret_20d", "ret_60d",
    "vol_10d", "vol_20d", "vol_60d",
    "amp", "vwap_dev", "log_amount",
    "net_mf_ratio", "elg_imb", "lg_imb",
    "turnover_rate_f",
]

TAB_COLS = [
    "turnover_rate_f", "volume_ratio", "pb", "pe_ttm", "pe_missing", "log_mv",
    "net_mf_ratio", "elg_imb", "lg_imb",
]


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(n, min_periods=n).mean()
    loss = (-delta.clip(upper=0)).rolling(n, min_periods=n).mean()
    rs = gain / (loss + 1e-9)
    return 100 - 100 / (1 + rs)


def build_stock_features(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.sort_values(["ts_code", "trade_date"]).copy()
    gb = df.groupby("ts_code", group_keys=False)

    df["ret_1d"] = gb["close"].pct_change()
    for w in [5, 10, 20, 60]:
        df[f"ret_{w}d"] = gb["close"].pct_change(w)
    for w in [10, 20, 60]:
        mp = max(3, w // 2)
        # transform 保证索引与 df 对齐（pandas 2.x 下 groupby().rolling() 直接赋值会错位）
        df[f"vol_{w}d"] = gb["ret_1d"].transform(
            lambda s, window=w, min_p=mp: s.rolling(window, min_periods=min_p).std()
        )
    df["amp"] = (df["high"] - df["low"]) / (df["close"] + 1e-9)
    df["vwap_dev"] = (df["close"] - df["vwap"]) / (df["vwap"] + 1e-9)
    df["log_amount"] = np.log1p(df["amount_yuan"])
    if "turnover_rate_f" not in df.columns and "turnover_rate" in df.columns:
        df["turnover_rate_f"] = df["turnover_rate"]
    df["rsi_14"] = gb["close"].transform(_rsi)

    # 行业编码
    industries = df["industry"].fillna("未知").astype("category")
    df["industry_id"] = industries.cat.codes.astype(int)

    return df


def apply_feature_transform(
    df: pd.DataFrame,
    winsor_q: float = 0.01,
) -> pd.DataFrame:
    cols = [c for c in SEQ_BASE_COLS + TAB_COLS if c in df.columns]
    cols = list(dict.fromkeys(cols))
    return cross_section_transform(df, cols, winsor_q=winsor_q, to_rank=True)


def get_seq_feature_names(df: pd.DataFrame) -> list[str]:
    names = [c for c in SEQ_BASE_COLS if c in df.columns]
    if "rsi_14" in df.columns:
        names.append("rsi_14")
    return names


def get_tab_feature_names(df: pd.DataFrame) -> list[str]:
    return [c for c in TAB_COLS if c in df.columns]
