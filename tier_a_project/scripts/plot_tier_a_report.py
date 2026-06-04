#!/usr/bin/env python
"""从档位 A 产物生成实验报告用图（写入 tier_a_project/figures/）。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "code"  # .../pro/code
PRO = ROOT.parent
FIG = PRO / "tier_a_project" / "figures"
ART = ROOT / "artifacts"

sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def plot_equity():
    eq = pd.read_csv(ART / "backtest/tier_a/valid/equity_curve.csv")
    eq["trade_date"] = pd.to_datetime(eq["trade_date"], format="%Y%m%d")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(eq["trade_date"], eq["equity"] / 1e6, color="#2563eb", lw=1.5)
    ax.set_title("档位 A 验证集净值曲线（2024H2）")
    ax.set_xlabel("日期")
    ax.set_ylabel("净值（百万元）")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_valid_equity.png", dpi=150)
    plt.close(fig)


def plot_fusion_weights():
    w = json.load(open(ART / "predictions/tier_a/fusion_weights.json", encoding="utf-8"))["weights"]
    labels = ["GRU\n(时序)", "MLP\n(截面)", "NewsHash\n(新闻)"]
    vals = [w["A"], w["B"], w["C"]]
    colors = ["#2563eb", "#16a34a", "#ca8a04"]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(vals, labels=labels, autopct="%1.1f%%", colors=colors, startangle=90)
    ax.set_title("三模态 ICIR 融合权重")
    fig.tight_layout()
    fig.savefig(FIG / "fig3_fusion_weights.png", dpi=150)
    plt.close(fig)


def plot_orders():
    o = pd.read_csv(ART / "orders/tier_a/orders_20241231.csv")
    o = o.sort_values("target_weight", ascending=True).tail(15)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(o["ts_code"], o["target_weight"] * 100, color="#6366f1")
    ax.set_xlabel("目标权重（%）")
    ax.set_title("最新调仓 Top15 权重（2024-12-31）")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_latest_orders.png", dpi=150)
    plt.close(fig)


def plot_ic_bars():
    ic = json.load(open(ART / "metrics/tier_a/ic_summary.json", encoding="utf-8"))
    splits = ["train", "valid"]
    means = [ic[s]["mean_ic"] for s in splits]
    icirs = [ic[s]["icir"] for s in splits]
    x = range(len(splits))
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.bar([i - 0.2 for i in x], means, width=0.4, label="mean IC", color="#2563eb")
    ax2 = ax1.twinx()
    ax2.bar([i + 0.2 for i in x], icirs, width=0.4, label="ICIR", color="#f97316", alpha=0.85)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(["训练集", "验证集"])
    ax1.set_ylabel("mean IC")
    ax2.set_ylabel("ICIR")
    ax1.set_title("IC / ICIR 对比")
    lines1, lab1 = ax1.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lab1 + lab2, loc="upper left")
    ax1.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_ic_summary.png", dpi=150)
    plt.close(fig)


def plot_ic_timeseries():
    from scipy.stats import spearmanr

    pred = pd.read_parquet(ART / "predictions/tier_a/ensemble_scores.parquet")
    pred = pred.dropna(subset=["score", "excess_ret_1d"])
    ics, dates = [], []
    for d, g in pred.groupby("trade_date"):
        if len(g) < 30:
            continue
        ic, _ = spearmanr(g["score"], g["excess_ret_1d"])
        if pd.notna(ic):
            ics.append(ic)
            dates.append(d)
    s = pd.Series(ics, index=pd.to_datetime(dates, format="%Y%m%d"))
    roll = s.rolling(20, min_periods=10).mean()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(s.index, s.values, width=2, color="#93c5fd", alpha=0.6, label="日度 IC")
    ax.plot(roll.index, roll.values, color="#1d4ed8", lw=1.8, label="20 日滚动均值")
    ax.axhline(0, color="#64748b", lw=0.8)
    ax.axvline(pd.Timestamp("2024-07-01"), color="#ef4444", ls="--", lw=1, label="验证集起点")
    ax.set_title("验证集及之前样本：日度 Spearman IC")
    ax.set_xlabel("日期")
    ax.set_ylabel("IC")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "fig5_ic_timeseries.png", dpi=150)
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    plot_equity()
    plot_ic_bars()
    plot_fusion_weights()
    plot_orders()
    plot_ic_timeseries()
    print(f"[OK] 图表已写入 {FIG}")


if __name__ == "__main__":
    main()
