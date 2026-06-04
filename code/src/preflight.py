"""训练前统一预检：区分 fatal / warn / info，避免 NaN、单 seed、配置对照等误阻断 train。"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.config import AppConfig


@dataclass
class PreflightResult:
    infos: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    fatals: list[str] = field(default_factory=list)

    def merge(self, other: PreflightResult) -> None:
        self.infos.extend(other.infos)
        self.warnings.extend(other.warnings)
        self.fatals.extend(other.fatals)

    @property
    def ok(self) -> bool:
        return not self.fatals

    def exit_if_fatal(self) -> None:
        import sys

        if self.fatals:
            sys.exit(1)


def print_preflight(result: PreflightResult, title: str = "preflight") -> None:
    for line in result.infos:
        print(f"[{title}] [INFO] {line}", flush=True)
    for line in result.warnings:
        print(f"[{title}] [WARN] {line}", flush=True)
    if result.fatals:
        print(f"[{title}] [FATAL]", flush=True)
        for line in result.fatals:
            print(f"  {line}", flush=True)
    elif not result.warnings and not result.infos:
        print(f"[{title}] OK", flush=True)
    else:
        print(f"[{title}] OK（无致命问题）", flush=True)


def is_c2_72h_balanced(cfg: AppConfig) -> bool:
    profile = (cfg.raw.get("report_notes") or {}).get("profile", "")
    return profile == "tfne_c2_72h_balanced"


def is_5090_72h_max(cfg: AppConfig) -> bool:
    profile = (cfg.raw.get("report_notes") or {}).get("profile", "")
    return profile == "tier_c_5090_72h_max"


def is_72h_guaranteed(cfg: AppConfig) -> bool:
    profile = (cfg.raw.get("report_notes") or {}).get("profile", "")
    if profile == "tfne_c_72h_guaranteed":
        return True
    if cfg.smoke or cfg.tier != "C":
        return False
    data = cfg.raw.get("data", {})
    models = cfg.model_cfg()
    training = cfg.raw.get("training", {})
    return (
        data.get("liquidity_top_k") == 2000
        and list(models.get("seeds", [])) == [42]
        and training.get("amp") is False
        and int(models.get("max_epochs", 30)) <= 8
    )


def check_config_notes(cfg: AppConfig, config_name: str = "") -> PreflightResult:
    """配置说明：72h 版打印有意取舍；smoke 版与 server_c 差异仅 warn，不 fail。"""
    out = PreflightResult()
    name = config_name or "当前配置"

    if is_c2_72h_balanced(cfg):
        notes = cfg.raw.get("report_notes") or {}
        data = cfg.raw.get("data", {})
        models = cfg.model_cfg()
        training = cfg.raw.get("training", {})
        est = notes.get("estimated_gpu_hours", "28–58")
        baseline = notes.get("measured_baseline_gpu_hours", {})
        out.infos.append(
            f"{name} 为 TFNE-C2（第二代）：tag={cfg.tier_tag()}、Top{data.get('liquidity_top_k')}、"
            f"seeds={models.get('seeds')}、train≤{cfg.splits.get('train_end')}、"
            f"valid={cfg.valid_start()}–{cfg.splits.get('valid_end')}、"
            f"max_epochs={models.get('max_epochs')}、amp={training.get('amp')}"
        )
        if baseline:
            out.infos.append(
                f"  7707 实测基线: {baseline.get('total')}h GPU "
                f"(TFT {baseline.get('tft')}h + FTT {baseline.get('ftt')}h + News {baseline.get('news')}h "
                f"@ Top{baseline.get('top_k')}, {baseline.get('seeds')} seed)"
            )
        out.infos.append(f"  C2 GPU 粗算 {est} h（预算上限 72h）")
        for line in notes.get("improvements_vs_7707", []):
            out.infos.append(f"  相对 7707: {line}")
        return out

    if is_5090_72h_max(cfg):
        notes = cfg.raw.get("report_notes") or {}
        data = cfg.raw.get("data", {})
        models = cfg.model_cfg()
        training = cfg.raw.get("training", {})
        est = notes.get("estimated_gpu_hours", "32–55")
        out.infos.append(
            f"{name} 为 5090·72h 尽量满血 C：Top{data.get('liquidity_top_k')}、"
            f"seeds={models.get('seeds')}、max_epochs={models.get('max_epochs')}、"
            f"amp={training.get('amp')}；GPU 粗算 {est} h"
        )
        for line in notes.get("sacrifices_vs_server_c", []):
            out.infos.append(f"  相对 server_c 缩减: {line}")
        return out

    if is_72h_guaranteed(cfg):
        data = cfg.raw.get("data", {})
        models = cfg.model_cfg()
        training = cfg.raw.get("training", {})
        out.infos.append(
            f"{name} 为 72h 保证版：Top{data.get('liquidity_top_k')}、"
            f"seeds={models.get('seeds')}、amp={training.get('amp')}、"
            f"max_epochs={models.get('max_epochs')}（相对满血 server_c 的有意缩减）"
        )
        return out

    try:
        import yaml

        code_root = cfg.code_root
        if "smoke" not in config_name.lower():
            out.infos.append(f"{name} 非 smoke 场景，跳过与 server_c 的逐项比对")
            return out

        server_path = code_root / "config.server_c.yaml"
        smoke_path = code_root / config_name
        if not server_path.exists() or not smoke_path.exists():
            out.warnings.append("找不到 smoke 或 server_c 配置文件，跳过对照")
            return out

        with open(server_path, encoding="utf-8") as f:
            server = yaml.safe_load(f)
        with open(smoke_path, encoding="utf-8") as f:
            smoke = yaml.safe_load(f)
        st = smoke.get("smoke_test", {})
        checks = [
            ("features.seq_len", smoke.get("features", {}).get("seq_len"), server.get("features", {}).get("seq_len")),
            ("models.hidden_size", st.get("hidden_size"), server.get("models", {}).get("hidden_size")),
            ("training.amp", smoke.get("training", {}).get("amp"), server.get("training", {}).get("amp")),
        ]
        for key, a, b in checks:
            if a != b:
                out.warnings.append(f"smoke vs server_c 不一致: {key} smoke={a} server={b}（仅提醒，不阻断）")
    except Exception as e:
        out.warnings.append(f"配置对照跳过: {e}")

    return out


def check_train_artifacts(cfg: AppConfig) -> PreflightResult:
    out = PreflightResult()
    feat = cfg.path("features", "panel_features.parquet")
    news = cfg.path("news", "panel_news.parquet")
    meta = cfg.path("features", "feature_meta.json")
    if not feat.exists():
        out.fatals.append(f"缺少 {feat}，请先 prep")
    if not news.exists():
        out.fatals.append(f"缺少 {news}，请先 prep")
    if not meta.exists():
        out.fatals.append(f"缺少 {meta}，请先 prep")
    return out


def check_panel_health(cfg: AppConfig, *, strict_nan: bool = False) -> PreflightResult:
    out = PreflightResult()
    feat_path = cfg.path("features", "panel_features.parquet")
    meta_path = cfg.path("features", "feature_meta.json")
    if not feat_path.exists() or not meta_path.exists():
        out.fatals.append("panel 特征产物缺失，请先 prep")
        return out

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    seq_cols = meta["seq_cols"]
    tab_cols = meta.get("tab_cols", [])
    cols = list(dict.fromkeys(seq_cols + tab_cols + ["excess_ret_1d", "log_mv", "listing_age_days"]))

    df = pd.read_parquet(feat_path, columns=cols + ["trade_date"])
    out.infos.append(
        f"panel_features rows={len(df):,}, dates={df['trade_date'].min()}–{df['trade_date'].max()}"
    )

    nan_cols: list[str] = []
    for c in cols:
        if c not in df.columns:
            out.fatals.append(f"缺失列: {c}")
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        n_nan = int(s.isna().sum())
        arr = s.to_numpy(dtype=np.float64)
        n_inf = int(np.isinf(arr).sum())
        if n_inf:
            out.fatals.append(f"{c}: inf={n_inf}")
        if n_nan:
            pct = 100.0 * n_nan / len(df)
            nan_cols.append(f"{c}: nan={n_nan} ({pct:.2f}%)")
        if len(s.dropna()) > 0:
            q = s.quantile([0.001, 0.999])
            if abs(q.iloc[0]) > 1e6 or abs(q.iloc[1]) > 1e6:
                out.fatals.append(f"{c}: 极端分位 [{q.iloc[0]:.4g}, {q.iloc[1]:.4g}]")

    train_end = cfg.splits.get("train_end", "20240630")
    train = df[df["trade_date"] <= train_end]
    label_cols = ["excess_ret_1d"] + seq_cols
    train_ready = train.dropna(subset=[c for c in label_cols if c in train.columns], how="any")
    n_ready = len(train_ready)
    batch = int(cfg.model_cfg().get("batch_stocks", 4096))
    out.infos.append(f"train rows={len(train):,}, dropna 后可用于 TFT≈{n_ready:,}, est batches/epoch≈{max(n_ready // batch, 1)}")
    if n_ready < 100_000:
        out.fatals.append(f"dropna 后训练样本过少: {n_ready}")

    if nan_cols:
        out.warnings.append(
            "部分列含 NaN（A 股常见：亏损无 PE、新股 rolling 不足）；训练时会 dropna/nan_to_num"
        )
        for w in nan_cols[:10]:
            out.warnings.append(f"  {w}")
        if len(nan_cols) > 10:
            out.warnings.append(f"  ... 共 {len(nan_cols)} 列含 NaN")
        if strict_nan:
            out.fatals.append("strict-nan：存在 NaN")

    return out


def check_runtime(cfg: AppConfig, *, require_cuda: bool = False) -> PreflightResult:
    out = PreflightResult()
    import torch

    if not torch.cuda.is_available():
        msg = "CUDA 不可用"
        if require_cuda:
            out.fatals.append(f"{msg}（require_cuda=true：请在 GPU 计算节点 / sbatch 作业内运行）")
        else:
            out.warnings.append(f"{msg}（登录节点正常；GPU 作业内再验证）")
    else:
        out.infos.append(f"CUDA OK: {torch.cuda.get_device_name(0)}")

    m = cfg.model_cfg()
    if m.get("use_news_bert"):
        try:
            from transformers import AutoModel  # noqa: F401
            from peft import LoraConfig  # noqa: F401

            out.infos.append("transformers + peft 可导入")
        except ImportError as e:
            out.fatals.append(f"use_news_bert=true 但 NLP 依赖缺失: {e}")

    feat = cfg.path("features", "panel_features.parquet")
    if feat.exists():
        df = pd.read_parquet(feat, columns=["trade_date"])
        out.infos.append(f"panel_features dates {df['trade_date'].min()}–{df['trade_date'].max()}")
        valid_end = cfg.splits.get("valid_end", "20241231")
        if df["trade_date"].max() <= valid_end:
            out.warnings.append("特征最大日期 ≤ valid_end，test 段可能为空")
    else:
        out.fatals.append("无 panel_features.parquet")

    meta_path = cfg.path("features", "feature_meta.json")
    if not meta_path.exists():
        out.fatals.append("无 feature_meta.json")
        return out

    with open(meta_path, encoding="utf-8") as f:
        meta_j = json.load(f)
    sl = cfg.seq_len
    use_amp = bool(cfg.raw.get("training", {}).get("amp", False))

    try:
        from src.models.factory import build_seq_model, build_tab_model, build_news_encoder

        ind_vocab = 64
        seq_m = build_seq_model(cfg, len(meta_j["seq_cols"]), ind_vocab)
        build_tab_model(cfg, len(meta_j["tab_cols"]), ind_vocab)
        out.infos.append(f"时序/截面模型构建 OK (seq_len={sl})")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        seq_m = seq_m.to(device)
        b, t, f = 4, sl, len(meta_j["seq_cols"])
        x = torch.randn(b, t, f, device=device)
        mask = torch.ones(b, t, device=device)
        ind = torch.zeros(b, dtype=torch.long, device=device)
        mv = torch.randn(b, device=device)
        age = torch.randn(b, device=device)
        amp_on = use_amp and device.type == "cuda"
        with torch.amp.autocast("cuda", enabled=amp_on):
            y = seq_m(x, ind, mv, age, mask)
        if not torch.isfinite(y).all():
            out.fatals.append("TFT 单 batch 前向含 NaN/Inf")
        else:
            out.infos.append(f"TFT 单 batch 前向 OK (amp={amp_on})")
        if m.get("use_news_bert"):
            try:
                build_news_encoder(cfg)
                out.infos.append("NewsEncoder(BERT) 加载 OK")
            except Exception as e:
                out.fatals.append(
                    f"NewsLoRA 无法加载: {e}；可设 HF_ENDPOINT=https://hf-mirror.com"
                )
    except Exception as e:
        out.fatals.append(f"模型前向失败: {e}")

    if len(cfg.seeds) < 2 and not is_72h_guaranteed(cfg) and not is_c2_72h_balanced(cfg):
        out.warnings.append("仅 1 个 seed，未覆盖多 seed 融合（72h 保证版可忽略）")

    return out


def check_eval_checkpoints(cfg: AppConfig, seeds: Iterable[int] | None = None) -> PreflightResult:
    out = PreflightResult()
    seeds = list(seeds or cfg.seeds)
    for modal in ("tft", "ftt", "news"):
        for s in seeds:
            ckpt = cfg.checkpoint_path(modal, s)
            if not ckpt.exists():
                out.fatals.append(f"缺少 checkpoint: {ckpt}（请先 train 或指定已有 seed）")
    return out


def run_train_preflight(
    cfg: AppConfig,
    *,
    require_cuda: bool = False,
    strict_nan: bool = False,
    config_name: str = "",
) -> PreflightResult:
    result = PreflightResult()
    result.merge(check_config_notes(cfg, config_name))
    result.merge(check_train_artifacts(cfg))
    result.merge(check_panel_health(cfg, strict_nan=strict_nan))
    result.merge(check_runtime(cfg, require_cuda=require_cuda))
    return result


def run_eval_preflight(
    cfg: AppConfig,
    *,
    seeds: Iterable[int] | None = None,
    require_cuda: bool = False,
) -> PreflightResult:
    result = PreflightResult()
    result.merge(check_train_artifacts(cfg))
    result.merge(check_eval_checkpoints(cfg, seeds=seeds))
    if require_cuda:
        result.merge(check_runtime(cfg, require_cuda=True))
    return result
