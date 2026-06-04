"""IC / ICIR 等指标。"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def select_strategy_topk(
    g: pd.DataFrame,
    score_col: str = "score",
    n_max: int = 50,
    q_min: float = 0.90,
) -> pd.DataFrame:
    """
    与 strategy.target_weights 第一步一致：score ≥ Q(q_min) 后取 Top n_max。
    返回策略当日实际候选池（尚未做 softmax 赋权）。
    """
    s = g.dropna(subset=[score_col])
    if s.empty:
        return s
    hi = s[score_col].quantile(q_min)
    return s[s[score_col] >= hi].sort_values(score_col, ascending=False).head(n_max)


def daily_ic(
    df: pd.DataFrame,
    score_col: str = "score",
    label_col: str = "excess_ret_1d",
    date_col: str = "trade_date",
    method: str = "spearman",
) -> pd.Series:
    ics = []
    dates = []
    for d, g in df.groupby(date_col):
        s = g[[score_col, label_col]].dropna()
        if len(s) < 30:
            continue
        if method == "spearman":
            ic, _ = spearmanr(s[score_col], s[label_col])
        else:
            ic = s[score_col].corr(s[label_col])
        if np.isfinite(ic):
            ics.append(ic)
            dates.append(d)
    return pd.Series(ics, index=dates, name="ic")


def ic_summary(ic_series: pd.Series) -> Dict[str, float]:
    if ic_series.empty:
        return {"mean_ic": 0.0, "std_ic": 0.0, "icir": 0.0, "n_days": 0}
    m = float(ic_series.mean())
    s = float(ic_series.std(ddof=1)) if len(ic_series) > 1 else 0.0
    return {
        "mean_ic": m,
        "std_ic": s,
        "icir": m / s if s > 1e-9 else 0.0,
        "n_days": int(len(ic_series)),
    }


def daily_long_short_spread(
    df: pd.DataFrame,
    score_col: str = "score",
    label_col: str = "excess_ret_1d",
    date_col: str = "trade_date",
    top_q: float = 0.9,
    bottom_q: float = 0.1,
) -> pd.Series:
    """每日 Top 组减 Bottom 组的平均超额收益（多头 spread，非对冲组合）。"""
    spreads = []
    dates = []
    for d, g in df.groupby(date_col):
        s = g[[score_col, label_col]].dropna()
        if len(s) < 30:
            continue
        hi = s[score_col].quantile(top_q)
        lo = s[score_col].quantile(bottom_q)
        top = s.loc[s[score_col] >= hi, label_col]
        bot = s.loc[s[score_col] <= lo, label_col]
        if len(top) < 5 or len(bot) < 5:
            continue
        spreads.append(float(top.mean() - bot.mean()))
        dates.append(d)
    return pd.Series(spreads, index=dates, name="long_short_spread")


def daily_ic_top_quantile(
    df: pd.DataFrame,
    score_col: str = "score",
    label_col: str = "excess_ret_1d",
    date_col: str = "trade_date",
    top_q: float = 0.9,
    method: str = "spearman",
) -> pd.Series:
    """仅在 Top 分位股票子集上计算 IC（与策略 q_min 对齐）。"""
    ics = []
    dates = []
    for d, g in df.groupby(date_col):
        s = g[[score_col, label_col]].dropna()
        if len(s) < 30:
            continue
        hi = s[score_col].quantile(top_q)
        sub = s[s[score_col] >= hi]
        if len(sub) < 10:
            continue
        if method == "spearman":
            ic, _ = spearmanr(sub[score_col], sub[label_col])
        else:
            ic = sub[score_col].corr(sub[label_col])
        if np.isfinite(ic):
            ics.append(float(ic))
            dates.append(d)
    return pd.Series(ics, index=dates, name="ic_top")


def daily_ic_strategy_topk(
    df: pd.DataFrame,
    score_col: str = "score",
    label_col: str = "excess_ret_1d",
    date_col: str = "trade_date",
    n_max: int = 50,
    q_min: float = 0.90,
    min_names: int = 10,
    method: str = "spearman",
) -> pd.Series:
    """
    策略 Top-K IC（IC@K）：仅在「Q(q_min) 过滤 + Top n_max」候选池内算 Spearman IC。

    与全市场 IC 的区别：K 通常 30~50 只，衡量的是「即将买入的组合内能否再排序」，
    与 target_weights 的选股范围严格对齐，比 top_quantile（整个 10% 分位，~300 只）更贴近策略。
    """
    ics = []
    dates = []
    for d, g in df.groupby(date_col):
        sub = select_strategy_topk(g, score_col=score_col, n_max=n_max, q_min=q_min)
        sub = sub[[score_col, label_col]].dropna()
        if len(sub) < min_names:
            continue
        if method == "spearman":
            ic, _ = spearmanr(sub[score_col], sub[label_col])
        else:
            ic = sub[score_col].corr(sub[label_col])
        if np.isfinite(ic):
            ics.append(float(ic))
            dates.append(d)
    return pd.Series(ics, index=dates, name="ic_strategy_topk")


def daily_hit_rate_topk(
    df: pd.DataFrame,
    score_col: str = "score",
    label_col: str = "excess_ret_1d",
    date_col: str = "trade_date",
    n_max: int = 50,
    q_min: float = 0.90,
    min_names: int = 5,
) -> pd.Series:
    """Top-K 命中率：候选池内次日超额收益 > 0 的占比。"""
    hits = []
    dates = []
    for d, g in df.groupby(date_col):
        sub = select_strategy_topk(g, score_col=score_col, n_max=n_max, q_min=q_min)
        sub = sub[[label_col]].dropna()
        if len(sub) < min_names:
            continue
        hits.append(float((sub[label_col] > 0).mean()))
        dates.append(d)
    return pd.Series(hits, index=dates, name="hit_rate_topk")


def daily_within_topk_spread(
    df: pd.DataFrame,
    score_col: str = "score",
    label_col: str = "excess_ret_1d",
    date_col: str = "trade_date",
    n_max: int = 50,
    q_min: float = 0.90,
    min_names: int = 10,
) -> pd.Series:
    """
    Top-K 内部分 spread：按 score 排序后，前半 vs 后半的平均超额收益差。
    直接反映「在将要持有的 K 只里，高分是否比低分赚更多」。
    """
    spreads = []
    dates = []
    for d, g in df.groupby(date_col):
        sub = select_strategy_topk(g, score_col=score_col, n_max=n_max, q_min=q_min)
        sub = sub[[score_col, label_col]].dropna().sort_values(score_col, ascending=False)
        if len(sub) < min_names:
            continue
        mid = len(sub) // 2
        if mid < 3:
            continue
        top = sub.iloc[:mid][label_col]
        bot = sub.iloc[mid:][label_col]
        spreads.append(float(top.mean() - bot.mean()))
        dates.append(d)
    return pd.Series(spreads, index=dates, name="within_topk_spread")


def daily_signal_portfolio_return(
    df: pd.DataFrame,
    score_col: str = "score",
    label_col: str = "excess_ret_1d",
    date_col: str = "trade_date",
    industry_col: str = "industry",
    log_mv_col: str = "log_mv",
    mkt_vol_col: str = "mkt_vol",
    strategy_params=None,
) -> pd.Series:
    """
    信号层组合收益：对当日全截面 score 调用 target_weights，计算 Σ w_i · excess_ret_1d。
    不含交易成本，衡量「若按目标权重完美成交，次日超额收益多少」。
    """
    from src.strategy import StrategyParams, target_weights

    params = strategy_params or StrategyParams()
    need_cols = [score_col, label_col, industry_col, log_mv_col]
    rets = []
    dates = []
    for d, g in df.groupby(date_col):
        sub = g.dropna(subset=need_cols)
        if len(sub) < 30:
            continue
        idx = sub["ts_code"] if "ts_code" in sub.columns else sub.index
        scores = pd.Series(sub[score_col].values, index=idx)
        industries = pd.Series(sub[industry_col].values, index=idx)
        log_mv = pd.Series(sub[log_mv_col].values, index=idx)
        mkt_vol = (
            float(sub[mkt_vol_col].iloc[0])
            if mkt_vol_col in sub.columns and pd.notna(sub[mkt_vol_col].iloc[0])
            else 0.0
        )
        w = target_weights(scores, industries, log_mv, params, mkt_vol)
        if w.empty:
            continue
        y = pd.Series(sub[label_col].values, index=idx).reindex(w.index)
        rets.append(float((w * y).sum()))
        dates.append(d)
    return pd.Series(rets, index=dates, name="signal_portfolio_ret")


def hit_rate_summary(hit_series: pd.Series) -> Dict[str, float]:
    if hit_series.empty:
        return {"mean_hit_rate": 0.0, "std_hit_rate": 0.0, "n_days": 0}
    return {
        "mean_hit_rate": float(hit_series.mean()),
        "std_hit_rate": float(hit_series.std(ddof=1)) if len(hit_series) > 1 else 0.0,
        "n_days": int(len(hit_series)),
    }


def signal_portfolio_summary(
    ret_series: pd.Series,
    trading_days_per_year: int = 252,
) -> Dict[str, float]:
    if ret_series.empty:
        return {
            "mean_daily_return": 0.0,
            "std_daily_return": 0.0,
            "ann_return": 0.0,
            "ann_vol": 0.0,
            "sharpe": 0.0,
            "n_days": 0,
        }
    m = float(ret_series.mean())
    s = float(ret_series.std(ddof=1)) if len(ret_series) > 1 else 0.0
    ann = m * trading_days_per_year
    vol = s * np.sqrt(trading_days_per_year) if s > 0 else 0.0
    return {
        "mean_daily_return": m,
        "std_daily_return": s,
        "ann_return": ann,
        "ann_vol": vol,
        "sharpe": ann / vol if vol > 1e-12 else 0.0,
        "n_days": int(len(ret_series)),
    }


def spread_summary(spread_series: pd.Series, trading_days_per_year: int = 252) -> Dict[str, float]:
    if spread_series.empty:
        return {"mean_spread": 0.0, "std_spread": 0.0, "ann_spread": 0.0, "n_days": 0}
    m = float(spread_series.mean())
    return {
        "mean_spread": m,
        "std_spread": float(spread_series.std(ddof=1)) if len(spread_series) > 1 else 0.0,
        "ann_spread": m * trading_days_per_year,
        "n_days": int(len(spread_series)),
    }


def modal_icir_weights(
    ic_by_modal: Dict[str, pd.Series],
) -> Dict[str, float]:
    """按各模态 ICIR（非负）归一化权重。"""
    scores = {}
    for m, ic in ic_by_modal.items():
        s = ic_summary(ic)
        scores[m] = max(s["icir"], 0.0)
    total = sum(scores.values())
    if total < 1e-9:
        n = len(scores)
        return {m: 1.0 / n for m in scores}
    return {m: v / total for m, v in scores.items()}
