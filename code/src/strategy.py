"""目标权重组合 + T+1 调仓。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class StrategyParams:
    n_max: int = 50
    q_min: float = 0.90
    tau: float = 0.5
    w_max: float = 0.05
    w_ind_max: float = 0.25
    c_min: float = 0.03
    gamma_vol: float = 0.05
    delta: float = 0.01
    omega_max: float = 0.25
    mkt_vol_q80: float = 0.0
    downweight_microcap: bool = True


def compute_mkt_vol_threshold(panel: pd.DataFrame, train_end: str) -> Tuple[float, float]:
    sub = panel[panel["trade_date"] <= train_end]
    vol = sub.groupby("trade_date")["mkt_vol"].first().dropna()
    if vol.empty:
        return 0.0, 0.0
    return float(vol.quantile(0.8)), float(vol.median())


def tradable_universe(
    day_df: pd.DataFrame,
    min_amount_ma: float,
    min_listing_days: int = 60,
) -> pd.DataFrame:
    u = day_df.copy()
    u = u[(u.get("is_st", 0) == 0) & (u.get("is_bj", 0) == 0)]
    u = u[u.get("vol", 0) > 0]
    u = u[u.get("listing_age_days", 0) >= min_listing_days]
    u = u[u.get("amount_ma20", 0) >= min_amount_ma * 10000]  # min_amount_ma 万元
    return u


def target_weights(
    scores: pd.Series,
    industries: pd.Series,
    log_mv: pd.Series,
    params: StrategyParams,
    mkt_vol: float = 0.0,
) -> pd.Series:
    df = pd.DataFrame({"score": scores, "industry": industries, "log_mv": log_mv}).dropna(subset=["score"])
    if df.empty:
        return pd.Series(dtype=float)

    q = df["score"].quantile(params.q_min)
    df = df[df["score"] >= q].sort_values("score", ascending=False).head(params.n_max)
    if df.empty:
        return pd.Series(dtype=float)

    r = df["score"].rank(method="average", pct=True)
    logits = np.exp((r.values - 0.5) / max(params.tau, 1e-6))
    w = logits / logits.sum()
    df["w"] = w

    # 单票 cap
    for _ in range(5):
        excess = df["w"].clip(upper=params.w_max)
        rem = 1.0 - excess.sum()
        free = df["w"] > params.w_max
        if rem <= 0 or not free.any():
            df["w"] = excess
            break
        df.loc[~free, "w"] = df.loc[~free, "w"]
        df.loc[free, "w"] = params.w_max
        if rem > 0:
            df.loc[~free, "w"] += rem * (df.loc[~free, "w"] / df.loc[~free, "w"].sum())

    # 行业 cap
    for _ in range(5):
        ind_sum = df.groupby("industry")["w"].sum()
        over = ind_sum[ind_sum > params.w_ind_max]
        if over.empty:
            break
        for ind, s in over.items():
            scale = params.w_ind_max / s
            df.loc[df["industry"] == ind, "w"] *= scale
        df["w"] /= df["w"].sum()

    # 微盘降权
    if params.downweight_microcap and len(df) > 10:
        q10 = df["log_mv"].quantile(0.1)
        df.loc[df["log_mv"] <= q10, "w"] *= 0.5
        df["w"] /= df["w"].sum()

    c_t = params.c_min
    if mkt_vol > params.mkt_vol_q80:
        c_t += params.gamma_vol
    df["w"] *= (1.0 - c_t)
    return df["w"]


@dataclass
class ExecutionConstraints:
    """回测执行层现实约束（A/C2 共用，由 config.backtest 注入）。"""

    lot_size: int = 100
    limit_up_pct: float = 9.5
    limit_down_pct: float = -9.5
    block_limit_up_buy: bool = True
    block_limit_down_sell: bool = True
    block_suspended: bool = True


def _pct_chg_on(panel: pd.DataFrame, code: str) -> float:
    s = panel.set_index("ts_code").get("pct_chg", pd.Series())
    v = s.get(code, 0)
    return float(v) if pd.notna(v) else 0.0


def _vol_on(panel: pd.DataFrame, code: str) -> float:
    s = panel.set_index("ts_code").get("vol", pd.Series())
    v = s.get(code, 0)
    return float(v) if pd.notna(v) else 0.0


def can_buy(code: str, exec_panel: pd.DataFrame, exe: ExecutionConstraints) -> bool:
    if exe.block_suspended and _vol_on(exec_panel, code) <= 0:
        return False
    if exe.block_limit_up_buy and _pct_chg_on(exec_panel, code) >= exe.limit_up_pct:
        return False
    return True


def can_sell(code: str, exec_panel: pd.DataFrame, exe: ExecutionConstraints) -> bool:
    if exe.block_suspended and _vol_on(exec_panel, code) <= 0:
        return False
    if exe.block_limit_down_sell and _pct_chg_on(exec_panel, code) <= exe.limit_down_pct:
        return False
    return True


@dataclass
class Holding:
    ts_code: str
    shares: int
    buy_date: str
    sellable: bool


def rebalance_day(
    W: float,
    cash: float,
    holdings: Dict[str, Holding],
    day_panel: pd.DataFrame,
    scores: pd.Series,
    prices: Dict[str, float],
    trade_date: str,
    params: StrategyParams,
    mkt_vol: float,
    exec_panel: Optional[pd.DataFrame] = None,
    exe: Optional[ExecutionConstraints] = None,
) -> Tuple[float, Dict[str, Holding], List[dict]]:
    """返回新 cash, holdings, orders。exec_panel 为 T+1 执行日截面（涨跌停/停牌判断）。"""
    orders = []
    exe = exe or ExecutionConstraints()
    exec_panel = exec_panel if exec_panel is not None else day_panel
    lot = max(int(exe.lot_size), 1)
    ind = day_panel.set_index("ts_code")["industry"]
    lmv = day_panel.set_index("ts_code")["log_mv"]
    w_tgt = target_weights(scores, ind.reindex(scores.index), lmv.reindex(scores.index), params, mkt_vol)

    V_cur = {}
    for code, h in holdings.items():
        p = prices.get(code, np.nan)
        if np.isfinite(p):
            V_cur[code] = h.shares * p

    all_codes = set(w_tgt.index) | set(V_cur.keys())
    deltas = {}
    for code in all_codes:
        vt = W * w_tgt.get(code, 0.0)
        vc = V_cur.get(code, 0.0)
        dv = vt - vc
        if abs(dv) / W < params.delta:
            continue
        if dv < 0 and code in holdings and not holdings[code].sellable:
            continue
        deltas[code] = dv

    # 换手上限
    to = sum(abs(v) for v in deltas.values()) / (2 * W + 1e-9)
    if to > params.omega_max:
        scale = params.omega_max / to
        deltas = {k: v * scale for k, v in deltas.items()}

    # 先卖
    for code, dv in sorted(deltas.items(), key=lambda x: x[1]):
        if dv >= 0:
            continue
        p = prices.get(code)
        if not p or code not in holdings:
            continue
        if not can_sell(code, exec_panel, exe):
            continue
        shares_sell = min(
            holdings[code].shares,
            int(abs(dv) / (lot * p)) * lot,
        )
        if shares_sell <= 0:
            continue
        cash += shares_sell * p
        holdings[code].shares -= shares_sell
        orders.append({"ts_code": code, "side": "sell", "shares": shares_sell, "price": p})
        if holdings[code].shares <= 0:
            del holdings[code]

    # 后买
    buy_list = [(c, dv) for c, dv in deltas.items() if dv > 0]
    buy_list.sort(key=lambda x: -x[1])
    for code, dv in buy_list:
        p = prices.get(code)
        if not p or p <= 0:
            continue
        if not can_buy(code, exec_panel, exe):
            continue
        need = int(dv / (lot * p)) * lot
        cost = need * p
        if need <= 0 or cost > cash:
            continue
        cash -= cost
        if code in holdings:
            holdings[code].shares += need
        else:
            holdings[code] = Holding(code, need, trade_date, False)
        orders.append({"ts_code": code, "side": "buy", "shares": need, "price": p})

    return cash, holdings, orders
