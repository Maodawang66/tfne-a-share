"""交易日历工具。"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd


def load_trade_calendar(data_dir: Path, exchange: str = "SSE") -> pd.DataFrame:
    cal = pd.read_csv(data_dir / "trade_cal.csv", dtype=str)
    cal = cal[cal["exchange"] == exchange].copy()
    cal["cal_date"] = cal["cal_date"].astype(str)
    cal["is_open"] = cal["is_open"].astype(int)
    return cal.sort_values("cal_date").reset_index(drop=True)


def open_dates(
    data_dir: Path,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> List[str]:
    cal = load_trade_calendar(data_dir)
    days = cal.loc[cal["is_open"] == 1, "cal_date"].tolist()
    if start:
        days = [d for d in days if d >= start]
    if end:
        days = [d for d in days if d <= end]
    return days


def next_trade_date(data_dir: Path, date: str) -> Optional[str]:
    days = open_dates(data_dir, start=date)
    for d in days:
        if d > date:
            return d
    return None
