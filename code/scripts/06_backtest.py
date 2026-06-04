#!/usr/bin/env python
"""策略回测（按档位读取预测与输出指标）。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import json

import pandas as pd

from src.backtest import (
    BacktestConfig,
    compute_relative_metrics,
    load_benchmark_equity,
    run_backtest,
    save_backtest_outputs,
)
from src.config import AppConfig
from src.strategy import StrategyParams, compute_mkt_vol_threshold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = AppConfig.load(ROOT / args.config if args.config else None)
    print("[config]", cfg.describe())

    panel = pd.read_parquet(cfg.path("features", "panel_features.parquet"))
    pred_path = cfg.ensemble_scores_path()
    if not pred_path.exists():
        raise FileNotFoundError(f"请先运行 04 [{cfg.tier_tag()}]")

    pred = pd.read_parquet(pred_path)
    scfg = cfg.raw.get("strategy", {})
    splits = cfg.splits
    q80, _ = compute_mkt_vol_threshold(panel, splits["train_end"])
    sp = StrategyParams(
        n_max=scfg.get("n_max", 50),
        q_min=scfg.get("q_min", 0.90),
        tau=scfg.get("tau", 0.5),
        w_max=scfg.get("w_max", 0.05),
        w_ind_max=scfg.get("w_ind_max", 0.25),
        c_min=scfg.get("c_min", 0.03),
        gamma_vol=scfg.get("gamma_vol", 0.05),
        delta=scfg.get("delta", 0.01),
        omega_max=scfg.get("omega_max", 0.25),
        mkt_vol_q80=q80,
    )
    bcfg = cfg.raw.get("backtest", {})
    benchmark_code = str(bcfg.get("benchmark", "000300.SH"))
    btcfg = BacktestConfig(
        initial_capital=scfg.get("initial_capital", 1_000_000),
        commission=scfg.get("commission", 0.00015),
        stamp_tax=scfg.get("stamp_tax", 0.0005),
        slippage=scfg.get("slippage", 0.0005),
        lot_size=int(bcfg.get("lot_size", scfg.get("lot_size", 100))),
        min_commission=float(bcfg.get("min_commission", 5.0)),
        benchmark=benchmark_code,
        limit_up_pct=float(bcfg.get("limit_up_pct", 9.5)),
        limit_down_pct=float(bcfg.get("limit_down_pct", -9.5)),
        block_limit_up_buy=bool(bcfg.get("block_limit_up_buy", True)),
        block_limit_down_sell=bool(bcfg.get("block_limit_down_sell", True)),
        block_suspended=bool(bcfg.get("block_suspended", True)),
    )
    min_amt = cfg.raw.get("data", {}).get("min_amount_ma", 5000)

    valid_start = cfg.valid_start()
    valid_end = splits["valid_end"]
    periods = [("valid", valid_start, valid_end)]
    # test_end=null：报告只报 valid；FDL2026 模拟赛（如 6.1--6.10）为样本外，不在此回测
    test_end_cfg = splits.get("test_end")
    if test_end_cfg:
        test_start = splits.get("test_start")
        if not test_start:
            from datetime import datetime, timedelta

            d = datetime.strptime(str(valid_end), "%Y%m%d") + timedelta(days=1)
            test_start = d.strftime("%Y%m%d")
        test_end = str(test_end_cfg)
        if test_start <= test_end:
            periods.append(("test", test_start, test_end))

    for period, start, end in periods:
        equity, metrics = run_backtest(panel, pred, sp, btcfg, start, end, min_amount_ma=min_amt)
        dates = equity["trade_date"].astype(str).tolist()
        bench = load_benchmark_equity(cfg.data_dir, benchmark_code, dates, btcfg.initial_capital)
        if not bench.empty:
            rel = compute_relative_metrics(
                equity,
                bench,
                btcfg.initial_capital,
                index_code=benchmark_code,
            )
            metrics.update(rel)
        out_dir = cfg.backtest_dir(period)
        save_backtest_outputs(equity, metrics, out_dir, bench=bench if not bench.empty else None)
        print(f"[{cfg.tier_tag()}] {period}", metrics)


if __name__ == "__main__":
    main()
