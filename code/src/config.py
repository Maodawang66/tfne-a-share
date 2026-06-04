"""加载 config.yaml；支持档位 A 基线、档位 C 主实验、档位 C 本地 smoke。"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


CODE_ROOT = Path(__file__).resolve().parents[1]


def _resolve(path_str: str, base: Path) -> Path:
    p = Path(path_str)
    if not p.is_absolute():
        p = (base / p).resolve()
    return p


class AppConfig:
    def __init__(
        self,
        raw: dict,
        code_root: Path,
        data_dir: Path,
        artifacts_dir: Path,
        tier: str,
        smoke: bool,
    ):
        self.raw = raw
        self.code_root = code_root
        self.data_dir = data_dir
        self.artifacts_dir = artifacts_dir
        self.tier = tier.upper()
        self.smoke = smoke

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "AppConfig":
        config_path = config_path or CODE_ROOT / "config.yaml"
        with open(config_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        code_root = config_path.parent.resolve()
        paths = raw.get("paths", {})
        data_dir = _resolve(paths.get("data_dir", "../data"), code_root)
        artifacts_dir = _resolve(paths.get("artifacts_dir", "./artifacts"), code_root)
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        tier = str(raw.get("tier", "C")).upper()
        smoke = bool(raw.get("smoke_test", {}).get("enabled", False))

        # 档位 A：基线超参（可与 smoke 叠加以缩小数据量）
        if tier == "A":
            cls._apply_tier_a(raw)

        # 档位 C + smoke：本地小规模调试验证（架构仍为 TFT/FTT/NewsLoRA）
        if tier == "C" and smoke:
            cls._apply_smoke_c(raw)

        return cls(
            raw=raw,
            code_root=code_root,
            data_dir=data_dir,
            artifacts_dir=artifacts_dir,
            tier=tier,
            smoke=smoke,
        )

    @staticmethod
    def _apply_tier_a(raw: dict) -> None:
        ta = raw.get("tier_a", {})
        raw.setdefault("features", {})
        raw["features"]["seq_len"] = ta.get("seq_len", 20)
        raw.setdefault("models", {})
        m = raw["models"]
        m.update(
            {
                "seeds": ta.get("seeds", [42]),
                "hidden_size": ta.get("gru_hidden", 64),
                "gru_hidden": ta.get("gru_hidden", 64),
                "mlp_hidden": ta.get("mlp_hidden", 128),
                "max_epochs": ta.get("max_epochs", 10),
                "patience": ta.get("patience", 5),
                "batch_stocks": ta.get("batch_stocks", 512),
                "use_news_bert": False,
                "news_hash_dim": ta.get("news_hash_dim", 256),
                "news_out_dim": ta.get("news_out_dim", 32),
            }
        )
        raw.setdefault("data", {})
        if ta.get("max_stocks"):
            raw["data"]["max_stocks"] = ta["max_stocks"]
        if ta.get("start_date"):
            raw["data"]["start_date"] = ta["start_date"]
            raw["data"]["news_start"] = ta.get("news_start", ta["start_date"])
        if ta.get("end_date"):
            raw["data"]["end_date"] = ta["end_date"]

    @staticmethod
    def _apply_smoke_c(raw: dict) -> None:
        st = raw.get("smoke_test", {})
        raw.setdefault("data", {})
        if st.get("max_stocks") is not None:
            raw["data"]["max_stocks"] = st["max_stocks"]
        if st.get("liquidity_top_k") is not None:
            raw["data"]["liquidity_top_k"] = st["liquidity_top_k"]
            raw["data"].pop("max_stocks", None)
        if st.get("start_date"):
            raw["data"]["start_date"] = st["start_date"]
            raw["data"]["news_start"] = st.get("news_start", st["start_date"])
        if st.get("end_date"):
            raw["data"]["end_date"] = st["end_date"]
        raw.setdefault("models", {})
        m = raw["models"]
        m["seeds"] = st.get("seeds", [42])
        m["max_epochs"] = st.get("max_epochs", 3)
        m["hidden_size"] = st.get("hidden_size", m.get("hidden_size_smoke", 64))
        m["batch_stocks"] = st.get("batch_stocks", m.get("batch_stocks_smoke", 4096))
        if st.get("hidden_size"):
            m["hidden_size"] = st["hidden_size"]
        if st.get("ftt_n_blocks") is not None:
            m["ftt_n_blocks"] = st["ftt_n_blocks"]
        if st.get("ftt_d_token") is not None:
            m["ftt_d_token"] = st["ftt_d_token"]
        if st.get("news_max_length") is not None:
            m["news_max_length"] = st["news_max_length"]
        raw.setdefault("features", {})
        raw["features"]["seq_len"] = st.get("seq_len", raw["features"].get("seq_len_smoke", 20))
        if "use_news_bert" in st:
            m["use_news_bert"] = bool(st["use_news_bert"])
        if st.get("patience") is not None:
            m["patience"] = st["patience"]
        for key in (
            "news_max_epochs",
            "news_batch_size",
            "news_train_max_samples",
            "news_predict_batch_size",
            "news_patience",
            "news_lr",
        ):
            if st.get(key) is not None:
                m[key] = st[key]

    def get(self, *keys: str, default: Any = None) -> Any:
        d = self.raw
        for k in keys:
            if not isinstance(d, dict):
                return default
            d = d.get(k)
            if d is None:
                return default
        return d

    def model_cfg(self) -> dict:
        return dict(self.raw.get("models", {}))

    @property
    def seq_len(self) -> int:
        return int(self.get("features", "seq_len", default=60))

    @property
    def seeds(self) -> List[int]:
        return list(self.get("models", "seeds", default=[42]))

    @property
    def splits(self) -> dict:
        return self.get("splits", default={})

    def path(self, *parts: str) -> Path:
        return self.artifacts_dir.joinpath(*parts)

    def tier_tag(self) -> str:
        """artifacts 内分目录：tier_a / tier_c_smoke / tier_c / 自定义 artifacts_tag。"""
        override = self.raw.get("artifacts_tag")
        if override:
            return str(override)
        if self.tier == "A":
            return "tier_a"
        if self.smoke:
            return "tier_c_smoke"
        return "tier_c"

    def valid_start(self) -> str:
        """融合定权 / 新闻验证的起始日：显式 valid_start 或 train_end 次日。"""
        splits = self.splits
        if splits.get("valid_start"):
            return str(splits["valid_start"])
        train_end = str(splits.get("train_end", "20240630"))
        from datetime import datetime, timedelta

        d = datetime.strptime(train_end, "%Y%m%d") + timedelta(days=1)
        return d.strftime("%Y%m%d")

    def checkpoint_path(self, modal: str, seed: int) -> Path:
        return self.path("checkpoints", self.tier_tag(), modal, f"seed_{seed}", "best.pt")

    def prediction_dir(self, modal: str) -> Path:
        return self.path("predictions", self.tier_tag(), modal)

    def ensemble_scores_path(self) -> Path:
        return self.path("predictions", self.tier_tag(), "ensemble_scores.parquet")

    def metrics_dir(self) -> Path:
        return self.path("metrics", self.tier_tag())

    def backtest_dir(self, period: str = "test") -> Path:
        return self.path("backtest", self.tier_tag(), period)

    def describe(self) -> str:
        return f"tier={self.tier}, smoke={self.smoke}, tag={self.tier_tag()}, seq_len={self.seq_len}, seeds={self.seeds}"
