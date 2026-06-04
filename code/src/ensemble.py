"""多模态、多 seed 秩融合。"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.metrics import daily_ic, ic_summary, modal_icir_weights
from src.preprocess import cross_section_rank, cross_section_winsorize_predictions


def load_modal_predictions(
    artifacts_dir: Path,
    modal: str,
    seeds: List[int],
) -> pd.DataFrame:
    frames = []
    for s in seeds:
        p = artifacts_dir / "predictions" / modal / f"seed_{s}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        df["seed"] = s
        df["modal"] = modal
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"无预测文件: {modal}")
    return pd.concat(frames, ignore_index=True)


def fuse_seed_mean(df: pd.DataFrame, z_col: str = "z") -> pd.DataFrame:
    g = df.groupby(["trade_date", "ts_code"], as_index=False)[z_col].mean()
    return g.rename(columns={z_col: "z_mean"})


def ensemble_rank_fusion(
    preds: Dict[str, pd.DataFrame],
    panel: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None,
    winsor_q: float = 0.01,
) -> pd.DataFrame:
    """
    preds: modal -> DataFrame[trade_date, ts_code, z_mean]
    返回含 z_A,z_B,z_C,score
    """
    base = panel[["trade_date", "ts_code", "excess_ret_1d"]].drop_duplicates()
    out = base.copy()

    rank_cols = []
    modal_to_col = {"A": "z_A", "B": "z_B", "C": "z_C"}
    for modal, pdf in preds.items():
        col = modal_to_col.get(modal, f"z_{modal}")
        z = pdf.rename(columns={"z_mean": col})
        out = out.merge(z, on=["trade_date", "ts_code"], how="left")
        out[col] = cross_section_winsorize_predictions(
            out[col], out["trade_date"], q=winsor_q
        )
        rcol = f"rank_{modal}"
        out[rcol] = cross_section_rank(out[col], out["trade_date"])
        rank_cols.append(rcol)

    if weights is None:
        weights = {m: 1.0 / len(rank_cols) for m in preds}

    # 仅累加有效秩；避免 0 * NaN 污染 score
    score = np.zeros(len(out), dtype=np.float64)
    for m in preds:
        w = float(weights.get(m, 0.0))
        if w <= 0:
            continue
        r = out[f"rank_{m}"].to_numpy(dtype=np.float64)
        valid = np.isfinite(r)
        score[valid] += r[valid] * w
    out["score"] = score

    return out


def estimate_fusion_weights(
    preds: Dict[str, pd.DataFrame],
    panel: pd.DataFrame,
    valid_start: str,
    valid_end: str,
) -> Dict[str, float]:
    ic_by_modal = {}
    for modal, pdf in preds.items():
        m = pdf.merge(
            panel[["trade_date", "ts_code", "excess_ret_1d"]],
            on=["trade_date", "ts_code"],
            how="inner",
        )
        m = m[(m["trade_date"] >= valid_start) & (m["trade_date"] <= valid_end)]
        ic = daily_ic(m, score_col="z_mean", label_col="excess_ret_1d")
        ic_by_modal[modal] = ic
    return modal_icir_weights(ic_by_modal)
