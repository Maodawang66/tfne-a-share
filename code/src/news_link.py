"""新闻链指与按股聚合（CPU；I/O + 字符串匹配，非神经网络）。"""

from __future__ import annotations

import bisect
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
from tqdm import tqdm

from src.calendar_util import open_dates


def load_basic_name_map(data_dir: Path) -> Dict[str, Set[str]]:
    basic = pd.read_csv(data_dir / "basic.csv", dtype=str)
    mp: Dict[str, Set[str]] = {}
    for _, r in basic.iterrows():
        code = r["ts_code"]
        names = {r["name"], r.get("symbol", "")}
        names = {n for n in names if n and isinstance(n, str) and len(n) >= 2}
        for n in names:
            mp.setdefault(n, set()).add(code)
    return mp


class TradeCalendarCache:
    """缓存交易日历，避免每条新闻重复读 trade_cal.csv。"""

    def __init__(self, data_dir: Path):
        self.open_days = open_dates(data_dir)
        self.open_set = set(self.open_days)

    def map_trade_date(self, cal_date: str, dt_str: str) -> str:
        try:
            t = pd.to_datetime(dt_str)
        except Exception:
            return self._nearest_open(cal_date)
        if t.hour > 15 or (t.hour == 15 and t.minute > 0):
            nxt = self._next_open(cal_date)
            return nxt if nxt else self._nearest_open(cal_date)
        if cal_date in self.open_set:
            return cal_date
        return self._nearest_open(cal_date)

    def _next_open(self, date: str) -> Optional[str]:
        i = bisect.bisect_right(self.open_days, date)
        return self.open_days[i] if i < len(self.open_days) else None

    def _nearest_open(self, date: str) -> str:
        i = bisect.bisect_right(self.open_days, date) - 1
        if i >= 0:
            return self.open_days[i]
        return self.open_days[0] if self.open_days else date


def _match_codes(text: str, name_items: List[Tuple[str, Set[str]]]) -> Set[str]:
    matched: Set[str] = set()
    for name, codes in name_items:
        if name in text:
            matched |= codes
    return matched


def _process_one_day(
    args: Tuple[str, str, List[Tuple[str, Set[str]]], List[str]],
) -> List[dict]:
    """单日新闻 CSV → 链指记录（供多进程调用）。"""
    p_str, data_dir_str, name_items, open_days = args
    cal = Path(p_str).stem
    data_dir = Path(data_dir_str)
    cal_cache = TradeCalendarCache.__new__(TradeCalendarCache)
    cal_cache.open_days = open_days
    cal_cache.open_set = set(open_days)

    try:
        news = pd.read_csv(p_str, dtype=str)
    except Exception:
        return []
    if news.empty:
        return []

    rows = []
    for _, r in news.iterrows():
        text = f"{r.get('title', '')} {r.get('content', '')}"
        trade_d = cal_cache.map_trade_date(cal, str(r.get("datetime", cal)))
        matched = _match_codes(text, name_items)
        if not matched:
            rows.append(
                {"trade_date": trade_d, "ts_code": "__MARKET__", "text": text[:2000], "has_news": 1}
            )
        else:
            for code in matched:
                rows.append(
                    {"trade_date": trade_d, "ts_code": code, "text": text[:2000], "has_news": 1}
                )
    return rows


def aggregate_news_daily(
    data_dir: Path,
    start: str,
    end: Optional[str],
    name_map: Dict[str, Set[str]],
    n_jobs: int = 1,
) -> pd.DataFrame:
    news_dir = data_dir / "news"
    files = sorted(news_dir.glob("*.csv"))
    files = [p for p in files if p.stem >= start and (not end or p.stem <= end)]
    if not files:
        return pd.DataFrame(columns=["trade_date", "ts_code", "text", "has_news"])

    # 长公司名优先，略减误匹配
    name_items = sorted(name_map.items(), key=lambda x: len(x[0]), reverse=True)
    open_days = open_dates(data_dir)
    rows: List[dict] = []

    if n_jobs <= 1:
        cal_cache = TradeCalendarCache(data_dir)
        for p in tqdm(files, desc="news_csv", unit="day"):
            for _, r in pd.read_csv(p, dtype=str).iterrows():
                text = f"{r.get('title', '')} {r.get('content', '')}"
                trade_d = cal_cache.map_trade_date(p.stem, str(r.get("datetime", p.stem)))
                matched = _match_codes(text, name_items)
                if not matched:
                    rows.append(
                        {
                            "trade_date": trade_d,
                            "ts_code": "__MARKET__",
                            "text": text[:2000],
                            "has_news": 1,
                        }
                    )
                else:
                    for code in matched:
                        rows.append(
                            {
                                "trade_date": trade_d,
                                "ts_code": code,
                                "text": text[:2000],
                                "has_news": 1,
                            }
                        )
    else:
        task_args = [(str(p), str(data_dir), name_items, open_days) for p in files]
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futs = {ex.submit(_process_one_day, a): a[0] for a in task_args}
            for fut in tqdm(as_completed(futs), total=len(futs), desc="news_csv", unit="day"):
                rows.extend(fut.result())

    if not rows:
        return pd.DataFrame(columns=["trade_date", "ts_code", "text", "has_news"])
    return pd.DataFrame(rows)
