"""从 data/ 构建日频面板 panel_daily.parquet。"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Set

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.calendar_util import load_trade_calendar, open_dates, next_trade_date


def _read_one_day(args: tuple) -> Optional[pd.DataFrame]:
    data_dir, day, st_set = args
    daily_path = data_dir / "daily" / f"{day}.csv"
    if not daily_path.exists():
        return None
    daily = pd.read_csv(daily_path, dtype={"ts_code": str, "trade_date": str})
    if daily.empty:
        return None

    metric_path = data_dir / "metric" / f"{day}.csv"
    if metric_path.exists():
        metric = pd.read_csv(metric_path, dtype={"ts_code": str, "trade_date": str})
        daily = daily.merge(metric, on=["ts_code", "trade_date"], how="left", suffixes=("", "_m"))
        if "close_m" in daily.columns:
            daily.drop(columns=["close_m"], inplace=True, errors="ignore")

    mf_path = data_dir / "moneyflow" / f"{day}.csv"
    if mf_path.exists():
        mf = pd.read_csv(mf_path, dtype={"ts_code": str, "trade_date": str})
        daily = daily.merge(mf, on=["ts_code", "trade_date"], how="left")

    daily["is_st"] = daily["ts_code"].isin(st_set).astype(int)
    daily["is_bj"] = daily["ts_code"].str.endswith(".BJ").astype(int)
    return daily


def load_st_sets(data_dir: Path, days: List[str]) -> dict[str, Set[str]]:
    st_dir = data_dir / "stock_st"
    out = {}
    for d in days:
        p = st_dir / f"{d}.csv"
        if p.exists():
            df = pd.read_csv(p, dtype=str)
            out[d] = set(df["ts_code"].tolist())
        else:
            out[d] = set()
    return out


def load_market(data_dir: Path, code: str = "000300.SH") -> pd.DataFrame:
    mkt = pd.read_csv(data_dir / "market" / f"{code}.csv", dtype=str)
    mkt["trade_date"] = mkt["trade_date"].astype(str)
    mkt["mkt_ret"] = pd.to_numeric(mkt["pct_chg"], errors="coerce") / 100.0
    mkt = mkt.sort_values("trade_date")
    mkt["mkt_vol"] = mkt["mkt_ret"].rolling(20, min_periods=10).std()
    return mkt[["trade_date", "mkt_ret", "mkt_vol"]]


def build_panel(
    data_dir: Path,
    output_path: Path,
    start_date: str = "20190101",
    end_date: Optional[str] = None,
    max_stocks: Optional[int] = None,
    liquidity_top_k: Optional[int] = None,
    n_jobs: int = 4,
) -> pd.DataFrame:
    days = open_dates(data_dir, start=start_date, end=end_date)
    if not days:
        raise FileNotFoundError("无可用交易日")

    basic = pd.read_csv(data_dir / "basic.csv", dtype=str)
    basic = basic[~basic["ts_code"].str.endswith(".BJ")]

    st_map = load_st_sets(data_dir, days)
    mkt = load_market(data_dir)

    tasks = [(data_dir, d, st_map.get(d, set())) for d in days]
    chunks: List[pd.DataFrame] = []

    if n_jobs <= 1:
        for t in tqdm(tasks, desc="build_panel"):
            df = _read_one_day(t)
            if df is not None:
                chunks.append(df)
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futs = {ex.submit(_read_one_day, t): t[1] for t in tasks}
            for fut in tqdm(as_completed(futs), total=len(futs), desc="build_panel"):
                df = fut.result()
                if df is not None:
                    chunks.append(df)

    panel = pd.concat(chunks, ignore_index=True)
    panel = panel.merge(basic[["ts_code", "industry", "market", "list_date", "name"]], on="ts_code", how="left")
    panel = panel.merge(mkt, on="trade_date", how="left")

    # 资金流衍生
    if "net_mf_amount" in panel.columns and "amount" in panel.columns:
        amt_yuan = panel["amount"] * 1000.0
        panel["net_mf_ratio"] = panel["net_mf_amount"] / (amt_yuan / 10000.0 + 1e-6)
    if all(c in panel.columns for c in ["buy_elg_amount", "sell_elg_amount"]):
        panel["elg_imb"] = (panel["buy_elg_amount"] - panel["sell_elg_amount"]) / (
            panel["buy_elg_amount"] + panel["sell_elg_amount"] + 1e-6
        )
    if all(c in panel.columns for c in ["buy_lg_amount", "sell_lg_amount"]):
        panel["lg_imb"] = (panel["buy_lg_amount"] - panel["sell_lg_amount"]) / (
            panel["buy_lg_amount"] + panel["sell_lg_amount"] + 1e-6
        )

    panel["log_mv"] = np.log1p(pd.to_numeric(panel.get("total_mv", np.nan), errors="coerce"))
    panel["pe_missing"] = panel["pe_ttm"].isna().astype(int) if "pe_ttm" in panel.columns else 0

    panel["list_date"] = panel["list_date"].astype(str)
    panel["trade_date"] = panel["trade_date"].astype(str)
    panel["list_ts"] = pd.to_datetime(panel["list_date"], format="%Y%m%d", errors="coerce")
    panel["trade_ts"] = pd.to_datetime(panel["trade_date"], format="%Y%m%d", errors="coerce")
    panel["listing_age_days"] = (panel["trade_ts"] - panel["list_ts"]).dt.days

    panel.sort_values(["ts_code", "trade_date"], inplace=True)
    panel["amount_yuan"] = panel["amount"] * 1000.0
    panel["amount_ma20"] = (
        panel.groupby("ts_code")["amount_yuan"].transform(lambda x: x.rolling(20, min_periods=5).mean())
    )

    # 标签（因果：t 特征预测 t->t+1）
    panel["fwd_ret_1d"] = panel.groupby("ts_code")["close"].pct_change().shift(-1)
    panel["mkt_ret_fwd"] = panel.groupby("trade_date")["mkt_ret"].shift(-1)
    panel["excess_ret_1d"] = panel["fwd_ret_1d"] - panel["mkt_ret_fwd"]

    # 过滤 ST / 北交所
    panel = panel[(panel["is_st"] == 0) & (panel["is_bj"] == 0)].copy()

    if liquidity_top_k:
        panel["liq_rank"] = panel.groupby("trade_date")["amount_ma20"].rank(ascending=False, method="first")
        panel = panel[panel["liq_rank"] <= liquidity_top_k]

    if max_stocks:
        # 按全样本平均流动性取 top
        liq = panel.groupby("ts_code")["amount_ma20"].mean().sort_values(ascending=False)
        keep = set(liq.head(max_stocks).index)
        panel = panel[panel["ts_code"].isin(keep)]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(output_path, index=False)
    return panel
