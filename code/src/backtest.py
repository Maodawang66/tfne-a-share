"""组合回测引擎。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.strategy import (
    ExecutionConstraints,
    Holding,
    StrategyParams,
    compute_mkt_vol_threshold,
    rebalance_day,
    tradable_universe,
)


@dataclass
class BacktestConfig:
    initial_capital: float = 1_000_000
    commission: float = 0.00015
    stamp_tax: float = 0.0005
    slippage: float = 0.0005
    lot_size: int = 100
    min_commission: float = 5.0
    benchmark: str = "000300.SH"
    limit_up_pct: float = 9.5
    limit_down_pct: float = -9.5
    block_limit_up_buy: bool = True
    block_limit_down_sell: bool = True
    block_suspended: bool = True


def execution_constraints_from_backtest(bt_cfg: BacktestConfig) -> ExecutionConstraints:
    return ExecutionConstraints(
        lot_size=bt_cfg.lot_size,
        limit_up_pct=bt_cfg.limit_up_pct,
        limit_down_pct=bt_cfg.limit_down_pct,
        block_limit_up_buy=bt_cfg.block_limit_up_buy,
        block_limit_down_sell=bt_cfg.block_limit_down_sell,
        block_suspended=bt_cfg.block_suspended,
    )


def load_benchmark_equity(
    data_dir: Path,
    index_code: str,
    trade_dates: List[str],
    initial_capital: float = 1_000_000,
) -> pd.DataFrame:
    """按策略净值日期对齐指数买入持有净值（同起点资金）。"""
    path = data_dir / "market" / f"{index_code}.csv"
    if not path.exists():
        raise FileNotFoundError(f"基准指数文件不存在: {path}")
    idx = pd.read_csv(path, dtype=str)
    idx["trade_date"] = idx["trade_date"].astype(str)
    idx["close"] = pd.to_numeric(idx["close"], errors="coerce")
    idx = idx.dropna(subset=["close"]).sort_values("trade_date")
    sub = idx[idx["trade_date"].isin(trade_dates)].copy()
    if sub.empty:
        return pd.DataFrame(columns=["trade_date", "bench_equity"])
    sub["bench_equity"] = sub["close"] / float(sub["close"].iloc[0]) * initial_capital
    return sub[["trade_date", "bench_equity"]].reset_index(drop=True)


def compute_relative_metrics(
    equity: pd.DataFrame,
    bench: pd.DataFrame,
    initial: float,
    index_code: str = "000300.SH",
    risk_free_rate: float = 0.0,
    trading_days_per_year: int = 252,
) -> dict:
    """
    相对基准（默认沪深 300）的组合评价指标。

    - excess.total_return / excess.cagr：相对净值曲线（组合/基准）的超额
    - information_ratio：年化超额收益 / 跟踪误差（组合层 IR）
    - alpha_ann / beta：日收益对基准回归的 Jensen alpha 与 beta
    - relative_max_drawdown：相对净值曲线的最大回撤
    - win_rate / up_capture / down_capture：相对基准的日度胜负与涨跌捕获
    """
    merged = equity[["trade_date", "equity"]].merge(bench, on="trade_date", how="inner")
    if len(merged) < 2:
        return {"benchmark_index": index_code, "n_aligned_days": int(len(merged))}

    port = merged["equity"].astype(float)
    b = merged["bench_equity"].astype(float)
    n = len(merged) - 1
    if n < 1:
        return {"benchmark_index": index_code, "n_aligned_days": int(len(merged))}

    r_p = port.pct_change().dropna()
    r_b = b.pct_change().dropna()
    r_x = r_p - r_b

    bench_metrics = compute_metrics(
        merged[["trade_date", "bench_equity"]].rename(columns={"bench_equity": "equity"}),
        initial,
        risk_free_rate=risk_free_rate,
        trading_days_per_year=trading_days_per_year,
    )

    rel = port / port.iloc[0] / (b / b.iloc[0])
    excess_total = float(rel.iloc[-1] - 1.0)
    excess_cagr = float(rel.iloc[-1] ** (trading_days_per_year / n) - 1.0) if n > 0 else 0.0
    ann_excess = float(r_x.mean() * trading_days_per_year)

    te = float(r_x.std(ddof=1) * np.sqrt(trading_days_per_year)) if n > 1 else 0.0
    ir = float((ann_excess - risk_free_rate) / te) if te > 1e-12 else 0.0

    var_b = float(r_b.var(ddof=1)) if n > 1 else 0.0
    beta = float(np.cov(r_p, r_b, ddof=1)[0, 1] / var_b) if var_b > 1e-12 else 0.0
    alpha_ann = float((r_p.mean() - beta * r_b.mean()) * trading_days_per_year)

    rel_peak = np.maximum.accumulate(rel.values)
    rel_dd = rel.values / rel_peak - 1.0
    relative_mdd = float(rel_dd.min())

    win_rate = float((r_p > r_b).mean())
    up_mask = r_b > 0
    down_mask = r_b < 0
    up_capture = (
        float(r_p[up_mask].mean() / r_b[up_mask].mean())
        if up_mask.any() and abs(float(r_b[up_mask].mean())) > 1e-12
        else None
    )
    down_capture = (
        float(r_p[down_mask].mean() / r_b[down_mask].mean())
        if down_mask.any() and abs(float(r_b[down_mask].mean())) > 1e-12
        else None
    )

    return {
        "benchmark_index": index_code,
        "n_aligned_days": int(len(merged)),
        "benchmark": {
            "total_return": bench_metrics.get("total_return"),
            "cagr": bench_metrics.get("cagr"),
            "ann_return": bench_metrics.get("ann_return"),
            "ann_vol": bench_metrics.get("ann_vol"),
            "sharpe": bench_metrics.get("sharpe"),
            "max_drawdown": bench_metrics.get("max_drawdown"),
        },
        "excess": {
            "total_return": excess_total,
            "cagr": excess_cagr,
            "ann_return": ann_excess,
            "tracking_error": te,
            "information_ratio": ir,
            "alpha_ann": alpha_ann,
            "beta": beta,
            "relative_max_drawdown": relative_mdd,
            "win_rate": win_rate,
            "up_capture": up_capture,
            "down_capture": down_capture,
        },
    }


def enrich_equity_with_benchmark(equity: pd.DataFrame, bench: pd.DataFrame) -> pd.DataFrame:
    """在净值曲线中追加基准与相对净值列。"""
    out = equity.merge(bench, on="trade_date", how="left")
    if "bench_equity" in out.columns and out["bench_equity"].notna().any():
        p0 = float(out["equity"].iloc[0])
        b0 = float(out["bench_equity"].iloc[0])
        out["relative_equity"] = (out["equity"] / p0) / (out["bench_equity"] / b0)
    return out


def run_backtest(
    panel: pd.DataFrame,
    predictions: pd.DataFrame,
    strategy_params: StrategyParams,
    bt_cfg: BacktestConfig,
    start_date: str,
    end_date: str,
    min_amount_ma: float = 5000,
) -> tuple[pd.DataFrame, dict]:
    pred = predictions.copy()
    pred = pred[(pred["trade_date"] >= start_date) & (pred["trade_date"] <= end_date)]
    dates = sorted(pred["trade_date"].unique())
    panel_idx = panel.set_index(["trade_date", "ts_code"])

    cash = bt_cfg.initial_capital
    holdings: Dict[str, Holding] = {}
    equity_rows = []

    for i, d in enumerate(dates):
        day_p = panel[panel["trade_date"] == d].copy()
        if day_p.empty:
            continue
        prices_close = day_p.set_index("ts_code")["close"].to_dict()

        # 净值
        mv = sum(holdings[c].shares * prices_close.get(c, 0) for c in holdings)
        W = cash + mv

        if i + 1 >= len(dates):
            equity_rows.append({"trade_date": d, "equity": W, "cash": cash, "n_hold": len(holdings)})
            break

        d_next = dates[i + 1]
        day_scores = pred[pred["trade_date"] == d].set_index("ts_code")["score"]
        u = tradable_universe(day_p, min_amount_ma)
        day_scores = day_scores.reindex(u["ts_code"]).dropna()

        mkt_vol = float(day_p["mkt_vol"].iloc[0]) if "mkt_vol" in day_p.columns else 0.0

        # 执行价用次日开盘（若无 open 则用 close）
        day_next = panel[panel["trade_date"] == d_next]
        prices_exec = (
            day_next.set_index("ts_code")["open"]
            .fillna(day_next.set_index("ts_code")["close"])
            .to_dict()
        )
        exe = execution_constraints_from_backtest(bt_cfg)

        cash, holdings, orders = rebalance_day(
            W,
            cash,
            holdings,
            day_p,
            day_scores,
            prices_exec,
            d_next,
            strategy_params,
            mkt_vol,
            exec_panel=day_next,
            exe=exe,
        )

        # 费用（含最低佣金）
        for o in orders:
            val = o["shares"] * o["price"]
            cash -= max(val * bt_cfg.commission, bt_cfg.min_commission)
            if o["side"] == "sell":
                cash -= val * bt_cfg.stamp_tax
            cash -= val * bt_cfg.slippage

        # T+1: 前一日买入今日可卖
        for h in holdings.values():
            if h.buy_date < d_next:
                h.sellable = True

        mv = sum(holdings[c].shares * prices_close.get(c, 0) for c in holdings)
        W = cash + mv
        equity_rows.append({"trade_date": d, "equity": W, "cash": cash, "n_hold": len(holdings)})

    equity = pd.DataFrame(equity_rows)
    metrics = compute_metrics(equity, bt_cfg.initial_capital)
    metrics["execution_constraints"] = {
        "lot_size": bt_cfg.lot_size,
        "commission": bt_cfg.commission,
        "stamp_tax": bt_cfg.stamp_tax,
        "slippage": bt_cfg.slippage,
        "min_commission": bt_cfg.min_commission,
        "limit_up_pct": bt_cfg.limit_up_pct,
        "limit_down_pct": bt_cfg.limit_down_pct,
        "block_limit_up_buy": bt_cfg.block_limit_up_buy,
        "block_limit_down_sell": bt_cfg.block_limit_down_sell,
        "block_suspended": bt_cfg.block_suspended,
        "t_plus_1": True,
    }
    return equity, metrics


def compute_metrics(
    equity: pd.DataFrame,
    initial: float,
    risk_free_rate: float = 0.0,
    trading_days_per_year: int = 252,
) -> dict:
    """
    标准组合回测指标（日频净值序列）。

    - total_return: 区间总收益率
    - cagr: 复利年化收益率 (P_T/P_0)^(252/n)-1
    - ann_return: 算术年化收益率 mean(r_d)*252（Sharpe 分子用）
    - ann_vol: 年化波动率 std(r_d)*sqrt(252)
    - sharpe: (ann_return - rf) / ann_vol
    - max_drawdown: 最大回撤 min_t (P_t - peak_t)/peak_t
    """
    if equity.empty or len(equity) < 2:
        return {}
    eq = equity["equity"].values.astype(float)
    r = pd.Series(eq).pct_change().dropna()
    n = int(len(r))
    if n < 1:
        return {}

    total_return = float(eq[-1] / initial - 1)
    cagr = float((eq[-1] / initial) ** (trading_days_per_year / n) - 1)
    ann_return = float(r.mean() * trading_days_per_year)
    ann_vol = float(r.std(ddof=1) * np.sqrt(trading_days_per_year)) if n > 1 else 0.0
    sharpe = float((ann_return - risk_free_rate) / ann_vol) if ann_vol > 1e-12 else 0.0

    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    max_dd = float(dd.min())

    return {
        "total_return": total_return,
        "cagr": cagr,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "n_trading_days": n,
        "final_equity": float(eq[-1]),
        # 兼容旧字段名
        "ann_return_cagr": cagr,
    }


def save_backtest_outputs(
    equity: pd.DataFrame,
    metrics: dict,
    out_dir: Path,
    bench: Optional[pd.DataFrame] = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    eq_out = enrich_equity_with_benchmark(equity, bench) if bench is not None and not bench.empty else equity
    eq_out.to_csv(out_dir / "equity_curve.csv", index=False)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
