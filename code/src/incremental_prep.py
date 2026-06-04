"""增量 prep：仅追加新交易日到 panel / features / news（不重读全历史 CSV）。"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.calendar_util import open_dates
from src.features import (
    apply_feature_transform,
    build_stock_features,
    get_seq_feature_names,
    get_tab_feature_names,
)
from src.news_link import aggregate_news_daily, load_basic_name_map
from src.panel import _read_one_day, load_market, load_st_sets

PANEL_MA_LOOKBACK = 25
FEATURE_LOOKBACK = 65


def _align_df_to_schema(df: pd.DataFrame, schema) -> pd.DataFrame:
    import pyarrow as pa

    out = df.copy()
    for field in schema:
        name = field.name
        if name not in out.columns:
            out[name] = pd.NA
    cols = [f.name for f in schema]
    out = out[cols]
    for field in schema:
        name = field.name
        if pa.types.is_integer(field.type):
            out[name] = pd.to_numeric(out[name], errors="coerce").astype("Int64")
        elif pa.types.is_floating(field.type):
            out[name] = pd.to_numeric(out[name], errors="coerce").astype("float64")
        elif pa.types.is_string(field.type) or pa.types.is_large_string(field.type):
            out[name] = out[name].astype(str)
    return out


def _append_parquet_df(path: Path, new_df: pd.DataFrame, *, replace_dates: Optional[List[str]] = None) -> None:
    """追加 parquet 行（流式复制旧文件，避免全量 pandas concat）。"""
    import pyarrow as pa
    import pyarrow.parquet as pq

    if new_df.empty:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    new_df = new_df.copy()
    new_df["trade_date"] = new_df["trade_date"].astype(str)

    if not path.exists() or path.stat().st_size < 64:
        pq.write_table(pa.Table.from_pandas(new_df, preserve_index=False), path)
        return

    if replace_dates:
        replace_set = {str(d) for d in replace_dates}
        tmp = path.with_suffix(".parquet.tmp")
        pf = pq.ParquetFile(path)
        schema = pf.schema_arrow
        try:
            with pq.ParquetWriter(tmp, schema) as writer:
                for rg in range(pf.num_row_groups):
                    chunk = pf.read_row_group(rg).to_pandas()
                    chunk["trade_date"] = chunk["trade_date"].astype(str)
                    chunk = chunk[~chunk["trade_date"].isin(replace_set)]
                    if not chunk.empty:
                        writer.write_table(
                            pa.Table.from_pandas(_align_df_to_schema(chunk, schema), schema=schema)
                        )
                append_df = _align_df_to_schema(new_df, schema)
                writer.write_table(pa.Table.from_pandas(append_df, schema=schema))
            tmp.replace(path)
        except Exception:
            if tmp.exists():
                tmp.unlink()
            raise
        return

    existing_max = str(pd.read_parquet(path, columns=["trade_date"])["trade_date"].max())
    new_df = new_df[new_df["trade_date"] > existing_max]
    if new_df.empty:
        return

    tmp = path.with_suffix(".parquet.tmp")
    pf = pq.ParquetFile(path)
    schema = pf.schema_arrow
    append_df = _align_df_to_schema(new_df, schema)
    try:
        with pq.ParquetWriter(tmp, schema) as writer:
            for rg in range(pf.num_row_groups):
                writer.write_table(pf.read_row_group(rg))
            writer.write_table(pa.Table.from_pandas(append_df, schema=schema))
        tmp.replace(path)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def _daily_csv_dates(data_dir: Path, start: str, end: Optional[str] = None) -> List[str]:
    days = open_dates(data_dir, start=start, end=end)
    daily_dir = data_dir / "daily"
    return [d for d in days if (daily_dir / f"{d}.csv").exists()]


def _tail_dates(sorted_dates: List[str], before: str, n: int) -> List[str]:
    prior = [d for d in sorted_dates if d < before]
    return prior[-n:]


def discover_new_panel_dates(
    data_dir: Path,
    panel_path: Path,
    start_date: str,
    end_date: Optional[str] = None,
) -> List[str]:
    available = _daily_csv_dates(data_dir, start_date, end_date)
    if not panel_path.exists():
        return available
    existing = pd.read_parquet(panel_path, columns=["trade_date"])
    max_d = str(existing["trade_date"].max())
    return [d for d in available if d > max_d]


def _read_days_parallel(
    data_dir: Path,
    days: List[str],
    n_jobs: int,
) -> pd.DataFrame:
    st_map = load_st_sets(data_dir, days)
    tasks = [(data_dir, d, st_map.get(d, set())) for d in days]
    chunks: List[pd.DataFrame] = []

    if n_jobs <= 1:
        for t in tasks:
            df = _read_one_day(t)
            if df is not None:
                chunks.append(df)
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futs = {ex.submit(_read_one_day, t): t[1] for t in tasks}
            for fut in as_completed(futs):
                df = fut.result()
                if df is not None:
                    chunks.append(df)

    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


def _enrich_raw_panel(raw: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    basic = pd.read_csv(data_dir / "basic.csv", dtype=str)
    basic = basic[~basic["ts_code"].str.endswith(".BJ")]
    mkt = load_market(data_dir)

    panel = raw.merge(basic[["ts_code", "industry", "market", "list_date", "name"]], on="ts_code", how="left")
    panel = panel.merge(mkt, on="trade_date", how="left")

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
    panel = panel[(panel["is_st"] == 0) & (panel["is_bj"] == 0)].copy()
    panel["amount_yuan"] = panel["amount"] * 1000.0
    return panel


def _recompute_panel_labels(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.sort_values(["ts_code", "trade_date"]).copy()
    panel["fwd_ret_1d"] = panel.groupby("ts_code")["close"].pct_change().shift(-1)
    panel["mkt_ret_fwd"] = panel.groupby("trade_date")["mkt_ret"].shift(-1)
    panel["excess_ret_1d"] = panel["fwd_ret_1d"] - panel["mkt_ret_fwd"]
    return panel


def append_panel_days(
    data_dir: Path,
    panel_path: Path,
    new_days: List[str],
    *,
    liquidity_top_k: Optional[int] = None,
    max_stocks: Optional[int] = None,
    n_jobs: int = 4,
) -> pd.DataFrame:
    if not new_days:
        return pd.read_parquet(panel_path)

    if not panel_path.exists():
        raise FileNotFoundError(f"缺少 {panel_path}，请先全量运行 01_build_panel.py")

    existing = pd.read_parquet(panel_path)
    existing["trade_date"] = existing["trade_date"].astype(str)
    all_dates = sorted(existing["trade_date"].unique())

    raw_new = _read_days_parallel(data_dir, new_days, n_jobs)
    if raw_new.empty:
        raise FileNotFoundError(f"新交易日 {new_days} 无 daily CSV 数据")

    raw_new = _enrich_raw_panel(raw_new, data_dir)
    tail = _tail_dates(all_dates, new_days[0], PANEL_MA_LOOKBACK)
    hist_cols = [c for c in existing.columns if c in raw_new.columns or c in ("amount_yuan", "amount_ma20")]
    hist = existing[existing["trade_date"].isin(tail)][hist_cols].copy()

    work = pd.concat([hist, raw_new], ignore_index=True)
    work.sort_values(["ts_code", "trade_date"], inplace=True)
    work["amount_ma20"] = work.groupby("ts_code")["amount_yuan"].transform(
        lambda x: x.rolling(20, min_periods=5).mean()
    )

    parts: List[pd.DataFrame] = []
    for d in new_days:
        day = work[work["trade_date"] == d].copy()
        if day.empty:
            continue
        if liquidity_top_k:
            day["liq_rank"] = day.groupby("trade_date")["amount_ma20"].rank(
                ascending=False, method="first"
            )
            day = day[day["liq_rank"] <= liquidity_top_k]
        parts.append(day)

    if not parts:
        raise RuntimeError(f"新交易日 {new_days} 经流动性过滤后无样本")

    new_part = pd.concat(parts, ignore_index=True)
    if max_stocks:
        liq = existing.groupby("ts_code")["amount_ma20"].mean().sort_values(ascending=False)
        keep = set(liq.head(max_stocks).index)
        new_part = new_part[new_part["ts_code"].isin(keep)]

    merged = pd.concat([existing, new_part], ignore_index=True)
    merged = merged.drop_duplicates(["ts_code", "trade_date"], keep="last")
    merged = _recompute_panel_labels(merged)

    panel_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(panel_path, index=False)
    return merged


def rebuild_features_from_panel(
    panel_path: Path,
    features_path: Path,
    meta_path: Path,
    *,
    seq_len: int,
    winsor_q: float = 0.01,
) -> pd.DataFrame:
    """从 panel_daily 全量重建 features（不重新读 daily CSV）。"""
    if not panel_path.exists():
        raise FileNotFoundError(f"缺少 {panel_path}，请先运行 01_build_panel.py")

    df = pd.read_parquet(panel_path)
    df = build_stock_features(df)
    for w in [10, 20, 60]:
        col = f"vol_{w}d"
        if col in df.columns:
            df[f"{col}_raw"] = df[col].astype(float)
    df = apply_feature_transform(df, winsor_q=winsor_q)

    features_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = features_path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(features_path)

    import json

    meta = {
        "seq_cols": get_seq_feature_names(df),
        "tab_cols": get_tab_feature_names(df),
        "seq_len": seq_len,
        "industry_vocab": int(df["industry_id"].max() + 2),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[incremental] rebuilt features -> {features_path} rows={len(df)}", flush=True)
    return df


def features_need_rebuild(features_path: Path) -> bool:
    return (not features_path.exists()) or features_path.stat().st_size < 64


def append_features_days(
    panel_path: Path,
    features_path: Path,
    meta_path: Path,
    new_days: List[str],
    *,
    seq_len: int,
    winsor_q: float = 0.01,
) -> pd.DataFrame:
    if not new_days:
        if features_need_rebuild(features_path):
            raise FileNotFoundError(f"缺少 {features_path}，请先 rebuild_features_from_panel")
        return pd.read_parquet(features_path, columns=["trade_date"])

    if features_need_rebuild(features_path):
        raise FileNotFoundError(f"缺少 {features_path}，请先 rebuild_features_from_panel")

    panel = pd.read_parquet(panel_path)
    panel["trade_date"] = panel["trade_date"].astype(str)

    all_dates = sorted(panel["trade_date"].unique())
    tail = _tail_dates(all_dates, new_days[0], max(FEATURE_LOOKBACK, seq_len + 5))
    hist_panel = panel[panel["trade_date"].isin(tail)].copy()
    new_panel = panel[panel["trade_date"].isin(new_days)].copy()
    work = pd.concat([hist_panel, new_panel], ignore_index=True)
    work = work.drop_duplicates(["ts_code", "trade_date"], keep="last")
    work.sort_values(["ts_code", "trade_date"], inplace=True)

    work = build_stock_features(work)
    for w in [10, 20, 60]:
        col = f"vol_{w}d"
        if col in work.columns:
            work[f"{col}_raw"] = work[col].astype(float)

    new_feat = work[work["trade_date"].isin(new_days)].copy()
    new_feat = apply_feature_transform(new_feat, winsor_q=winsor_q)

    _append_parquet_df(features_path, new_feat)

    import json

    meta = {}
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
    meta.setdefault("seq_cols", get_seq_feature_names(new_feat))
    meta.setdefault("tab_cols", get_tab_feature_names(new_feat))
    meta["seq_len"] = seq_len
    meta["industry_vocab"] = max(
        int(meta.get("industry_vocab", 0)),
        int(new_feat["industry_id"].max() + 2),
    )
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return pd.read_parquet(features_path, columns=["trade_date"])


def discover_feature_lag_days(panel_path: Path, features_path: Path) -> List[str]:
    panel_dates = sorted(
        pd.read_parquet(panel_path, columns=["trade_date"])["trade_date"].astype(str).unique()
    )
    if features_need_rebuild(features_path):
        return []
    feat_max = str(pd.read_parquet(features_path, columns=["trade_date"])["trade_date"].max())
    return [d for d in panel_dates if d > feat_max]


def discover_new_news_dates(data_dir: Path, news_path: Path, news_start: str) -> List[str]:
    news_dir = data_dir / "news"
    files = sorted(p.stem for p in news_dir.glob("*.csv") if p.stem >= news_start)
    if not news_path.exists():
        return files
    existing = pd.read_parquet(news_path, columns=["trade_date"])
    max_d = str(existing["trade_date"].max())
    return [d for d in files if d > max_d]


def append_news_days(
    data_dir: Path,
    news_path: Path,
    new_file_dates: List[str],
    *,
    n_jobs: int = 4,
) -> pd.DataFrame:
    if not new_file_dates:
        if news_path.exists():
            return pd.read_parquet(news_path)
        return pd.DataFrame(columns=["trade_date", "ts_code", "text", "has_news"])

    name_map = load_basic_name_map(data_dir)
    start = min(new_file_dates)
    end = max(new_file_dates)
    new_part = aggregate_news_daily(data_dir, start=start, end=end, name_map=name_map, n_jobs=n_jobs)
    if new_part.empty:
        print(f"[incremental] news {start}..{end} 无链指记录", flush=True)
        if news_path.exists():
            return pd.read_parquet(news_path)
        return new_part

    new_part["trade_date"] = new_part["trade_date"].astype(str)
    file_set = set(new_file_dates)
    new_part = new_part[new_part["trade_date"].isin(file_set)]
    if new_part.empty:
        if news_path.exists():
            return pd.read_parquet(news_path)
        return new_part

    existing_max = None
    if news_path.exists() and news_path.stat().st_size >= 64:
        existing_max = str(pd.read_parquet(news_path, columns=["trade_date"])["trade_date"].max())
    if existing_max:
        new_part = new_part[new_part["trade_date"] > existing_max]
    if new_part.empty:
        return pd.read_parquet(news_path) if news_path.exists() else new_part
    _append_parquet_df(news_path, new_part)
    return pd.read_parquet(news_path)
