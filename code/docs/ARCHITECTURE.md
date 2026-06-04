# 项目架构与文件说明

本文档说明 `code/` 目录的**整体架构**、**数据流**、**档位 A / C2 差异**，以及**现存每个文件的作用**。快速上手见根目录 [`README.md`](../README.md)。

---

## 1. 项目定位

本项目实现 A 股多模态深度学习选股流水线，对应实验报告 `report.tex` 中的两档方案：

| 档位 | 代号 | 配置 | 模型 | 股票池 |
|------|------|------|------|--------|
| **A** | RGNE-Lite | `config.tier_a_local.yaml` | SeqGRU + TabMLP + 新闻哈希 | Top 500 |
| **C2** | TFNE-C2 | `config.server_c2_72h.yaml` | SeqTFTLite + TabFTTransformer + NewsLoRA | Top 3000 |

两档共用同一套 `src/` 核心库与 `01`–`07` 流水线脚本，通过 YAML 配置切换模型架构、数据规模与产物子目录（`tier_tag`）。

**评价口径（与报告一致）：** `test_end=null`，只在 **valid**（2025-04-01 ~ 2026-05-20）上报告 IC 与回测；FDL2026 模拟赛为样本外，仅由 `07_predict_latest.py` 出单。

---

## 2. 整体架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         ../data/  （原始 CSV）                           │
│  daily/ metric/ moneyflow/ news/ basic.csv trade_cal.csv market/ ...    │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │
                    ┌───────────────▼───────────────┐
                    │   Prep（CPU，01–02–02b）       │
                    │   src/panel, features, news   │
                    └───────────────┬───────────────┘
                                    │
          ┌─────────────────────────┼─────────────────────────┐
          ▼                         ▼                         ▼
   panel_daily.parquet      panel_features.parquet      panel_news.parquet
   （共享路径）              + feature_meta.json          （新闻文本）
          │                         │                         │
          └─────────────────────────┼─────────────────────────┘
                                    │
                    ┌───────────────▼───────────────┐
                    │   DailyStockDataset           │
                    │   src/dataset.py              │
                    └───────────────┬───────────────┘
                                    │
          ┌─────────────────────────┼─────────────────────────┐
          ▼                         ▼                         ▼
    03_train_tft              03_train_ftt            03_train_news_lora
    （时序模态 A/B）             （截面模态 A/B）           （新闻模态 A/C）
          │                         │                         │
          └─────────────────────────┼─────────────────────────┘
                                    │
                    checkpoints/{tier_tag}/{modal}/seed_*/best.pt
                                    │
                    ┌───────────────▼───────────────┐
                    │   04_ensemble_predict         │
                    │   src/ensemble.py             │
                    │   多 seed 均值 + ICIR 定权     │
                    └───────────────┬───────────────┘
                                    │
                         ensemble_scores.parquet
                         fusion_weights.json
                                    │
          ┌─────────────────────────┼─────────────────────────┐
          ▼                         ▼                         ▼
    05_evaluate_ic            06_backtest              07_predict_latest
    src/metrics.py            src/backtest.py          src/strategy.py
    IC/ICIR/IC@K              净值曲线/Sharpe           orders_*.csv
```

### 2.1 模块分层

| 层级 | 职责 | 主要代码 |
|------|------|----------|
| **配置** | 读 YAML、档位超参、`tier_tag` 路径 | `src/config.py`、三个 `config.*.yaml` |
| **数据** | 面板构建、GKX 变换、新闻链指、Dataset | `panel.py`、`features.py`、`preprocess.py`、`news_link.py`、`dataset.py` |
| **模型** | 按档位构建 GRU/TFT、MLP/FTT、Hash/LoRA | `src/models/*`、`factory.py` |
| **训练** | 循环、AMP、早停、保存 checkpoint | `src/train/trainer.py` |
| **融合** | 截面秩变换、多模态 ICIR 加权 | `src/ensemble.py` |
| **策略** | 目标权重、可交易池、行业/单票约束 | `src/strategy.py`、`risk_score.py` |
| **回测** | T+1、涨跌停、软换仓、成本 | `src/backtest.py` |
| **脚本** | CLI 入口、Slurm、一键编排 | `scripts/*.py`、`scripts/*.sh` |

### 2.2 档位 A 与 C2 在代码中的分叉

```
config.yaml ──► AppConfig.load()
                    │
        tier == "A" ──► _apply_tier_a() ──► factory: SeqGRU / TabMLP / NewsHash
                    │                         tier_tag → tier_a
                    │
        tier == "C" ──► artifacts_tag=tier_c2 ──► SeqTFTLite / TabFTTransformer / NewsLoRA
                                              tier_tag → tier_c2
```

- **A**：`03_train_tft.py` 实际训练 GRU（由 `factory.build_seq_model` 按 tier 选择）。
- **C2**：同一脚本训练 TFT-lite；需 `requirements-nlp.txt` 与 GPU；checkpoint 含 `ind_vocab`（行业 embedding 词表大小）。
- **共享 prep 路径**：`artifacts/panel_daily.parquet` 等无 tier 前缀，后跑 prep 会覆盖；A/C2 的 checkpoint 与 metrics 在各自 `tier_a/`、`tier_c2/` 下隔离。

---

## 3. 目录树与文件说明

以下按路径列出 **code/ 内现存的全部 86 个条目**（含 artifacts 中已有产物样例）。

### 3.1 根目录

| 文件 | 作用 |
|------|------|
| [`README.md`](../README.md) | 快速入门：两档配置、流水线、常用命令 |
| [`requirements.txt`](../requirements.txt) | 核心 Python 依赖（numpy、pandas、torch、pyyaml 等） |
| [`requirements-nlp.txt`](../requirements-nlp.txt) | C2 新闻 LoRA 额外依赖（transformers、peft 等） |
| [`.gitignore`](../.gitignore) | Git 忽略规则（artifacts 大文件、缓存等） |
| [`config.tier_a_local.yaml`](../config.tier_a_local.yaml) | **档位 A 本机主配置**：Top500、GRU/MLP/Hash、valid-only、`tier_a` |
| [`config.tier_a.yaml`](../config.tier_a.yaml) | 档位 A 备用配置（与 local 切分对齐，参数略异） |
| [`config.server_c2_72h.yaml`](../config.server_c2_72h.yaml) | **档位 C2 主配置**：Top3000、3 seed、TFT/FTT/LoRA、`tier_c2`、amp=false |

### 3.2 `src/` — 核心库

#### 包根

| 文件 | 作用 |
|------|------|
| [`src/__init__.py`](../src/__init__.py) | 包标识 |
| [`src/config.py`](../src/config.py) | 加载 YAML；档位 A 超参覆盖；`tier_tag()`、`checkpoint_path()`、`ensemble_scores_path()` 等路径助手 |
| [`src/calendar_util.py`](../src/calendar_util.py) | 交易日历：开闭市日、下一交易日 |
| [`src/panel.py`](../src/panel.py) | 从 CSV 构建长表面板：量价/基本面/资金流合并、ST 过滤、标签 `excess_ret_1d` |
| [`src/features.py`](../src/features.py) | 滚动时序特征（ret/vol/RSI 等）与截面特征列定义 |
| [`src/preprocess.py`](../src/preprocess.py) | GKX 截面 winsorize + rank 变换（防未来信息泄露） |
| [`src/news_link.py`](../src/news_link.py) | 新闻标题/正文链指到 `ts_code`；盘后新闻映射到下一交易日 |
| [`src/dataset.py`](../src/dataset.py) | `DailyStockDataset`：$L$ 日序列 + 截面 + 行业 id + 标签；`split_by_date` |
| [`src/ensemble.py`](../src/ensemble.py) | 多模态/多 seed 预测秩融合；验证集 ICIR 定权；输出 `ensemble_scores.parquet` |
| [`src/metrics.py`](../src/metrics.py) | 日 IC、ICIR、spread；`strategy_topk`（IC@K、命中率、signal_portfolio 等） |
| [`src/risk_score.py`](../src/risk_score.py) | 档位 A 可选 `risk_adjust`：对 score 做风险调整 |
| [`src/strategy.py`](../src/strategy.py) | `tradable_universe`、`target_weights`：Top 分位、行业/单票上限、现金缓冲 |
| [`src/backtest.py`](../src/backtest.py) | `run_backtest`：T+1 开盘价成交、涨跌停、软换仓 `delta`/`omega_max` |
| [`src/news_predict.py`](../src/news_predict.py) | C2 从 News checkpoint 全量推理新闻分数 |
| [`src/checkpoint_utils.py`](../src/checkpoint_utils.py) | 从 checkpoint 推断 `ind_vocab` 并安全加载 Seq 模型（修复行业数不匹配） |
| [`src/preflight.py`](../src/preflight.py) | train/eval 前检查：数据、GPU、checkpoint、patch 是否部署 |

#### `src/models/`

| 文件 | 作用 |
|------|------|
| [`src/models/__init__.py`](../src/models/__init__.py) | 导出主要模型类 |
| [`src/models/factory.py`](../src/models/factory.py) | **模型工厂**：按 `tier` 构建 GRU/TFT、MLP/FTT、Hash/LoRA；模态名 A/B/C |
| [`src/models/tier_a.py`](../src/models/tier_a.py) | 档位 A：`SeqGRU`、`TabMLP` |
| [`src/models/seq_model.py`](../src/models/seq_model.py) | 档位 C：`SeqTFTLite`（时序 + 静态行业/市值嵌入） |
| [`src/models/tab_model.py`](../src/models/tab_model.py) | 档位 C：`TabFTTransformer`（无 rtdl 时回退 MLP） |
| [`src/models/news_model.py`](../src/models/news_model.py) | 新闻编码：`NewsEncoder`（BERT+LoRA 或 HashingVectorizer） |

#### `src/train/`

| 文件 | 作用 |
|------|------|
| [`src/train/__init__.py`](../src/train/__init__.py) | 导出 `train_model`、`predict_model` |
| [`src/train/trainer.py`](../src/train/trainer.py) | 训练/验证循环、早停、保存 `best.pt`（含 `ind_vocab` 元数据） |
| [`src/train/train_args.py`](../src/train/train_args.py) | 从配置解析 lr、batch、grad_clip 等训练参数 |
| [`src/train/gpu_setup.py`](../src/train/gpu_setup.py) | CUDA/TF32/compile 设置与显存日志 |

### 3.3 `scripts/` — 命令行入口

#### 主流水线（01–07）

| 文件 | 作用 |
|------|------|
| [`scripts/01_build_panel.py`](../scripts/01_build_panel.py) | 调用 `build_panel` → `artifacts/panel_daily.parquet` |
| [`scripts/02_make_features.py`](../scripts/02_make_features.py) | 特征工程 + GKX → `features/panel_features.parquet`、`feature_meta.json` |
| [`scripts/02b_embed_news.py`](../scripts/02b_embed_news.py) | 新闻链指 → `news/panel_news.parquet` |
| [`scripts/03_train_tft.py`](../scripts/03_train_tft.py) | 训练时序模态（A=GRU，C2=TFT）；`--seed` |
| [`scripts/03_train_ftt.py`](../scripts/03_train_ftt.py) | 训练截面模态（A=MLP，C2=FTT）；`--seed` |
| [`scripts/03_train_news_lora.py`](../scripts/03_train_news_lora.py) | 训练新闻模态（A=Hash 线性，C2=LoRA）；`--seed` |
| [`scripts/03_predict_news_lora.py`](../scripts/03_predict_news_lora.py) | C2 增量：仅从 News checkpoint 重跑全量新闻预测 |
| [`scripts/04_ensemble_predict.py`](../scripts/04_ensemble_predict.py) | 加载各模态 checkpoint → 融合 → `ensemble_scores.parquet`、`fusion_weights.json` |
| [`scripts/05_evaluate_ic.py`](../scripts/05_evaluate_ic.py) | 计算 IC/ICIR/spread/strategy_topk → `metrics/{tier_tag}/` |
| [`scripts/06_backtest.py`](../scripts/06_backtest.py) | 验证集回测 → `backtest/{tier_tag}/valid/` |
| [`scripts/07_predict_latest.py`](../scripts/07_predict_latest.py) | 最后一个交易日目标权重 → `orders/{tier_tag}/orders_YYYYMMDD.csv` |

#### 一键编排

| 文件 | 作用 |
|------|------|
| [`scripts/run_tier_a_local.py`](../scripts/run_tier_a_local.py) | **档位 A 本机一键**：prep → 03×3 → 04–07；`config.tier_a_local.yaml` |
| [`scripts/run_tier_a.py`](../scripts/run_tier_a.py) | 转调 `run_tier_a_local.py` 的薄封装 |
| [`scripts/run_server_c2_72h.py`](../scripts/run_server_c2_72h.py) | **档位 C2 分阶段入口**：`prep` / `train` / `eval` / 单模态子命令 |

#### 评估与诊断

| 文件 | 作用 |
|------|------|
| [`scripts/print_strategy_topk_metrics.py`](../scripts/print_strategy_topk_metrics.py) | 打印 A/C2 的 IC@K、signal_ann 等（读 `strategy_topk.json`） |
| [`scripts/validate_tier_a_local.py`](../scripts/validate_tier_a_local.py) | A 本机环境/数据/配置预检 |
| [`scripts/validate_server_env.py`](../scripts/validate_server_env.py) | 服务器 conda/GPU/路径检查 |
| [`scripts/check_gpu.py`](../scripts/check_gpu.py) | GPU 可用性与简单算力测试 |
| [`scripts/check_panel_features_health.py`](../scripts/check_panel_features_health.py) | 检查 panel/features 行数、日期范围、缺失列 |
| [`scripts/compute_benchmark_metrics.py`](../scripts/compute_benchmark_metrics.py) | 计算相对基准（如沪深 300）指标 |
| [`scripts/backfill_relative_metrics.py`](../scripts/backfill_relative_metrics.py) | 向已有 metrics 补写相对大盘字段 |
| [`scripts/reapply_risk_adjust.py`](../scripts/reapply_risk_adjust.py) | 对 A 的 ensemble score 重新应用 risk_adjust |

#### 服务器 / Slurm

| 文件 | 作用 |
|------|------|
| [`scripts/web_c2_prep_command.sh`](../scripts/web_c2_prep_command.sh) | SCOW 提交 C2 CPU prep 作业 |
| [`scripts/web_c2_gpu_command.sh`](../scripts/web_c2_gpu_command.sh) | SCOW 提交 C2 GPU 训练作业 |
| [`scripts/web_c2_eval_command.sh`](../scripts/web_c2_eval_command.sh) | SCOW 提交 C2 eval 作业（含 patch 检查） |
| [`scripts/slurm_prep_cpu_c2_72h.sbatch`](../scripts/slurm_prep_cpu_c2_72h.sbatch) | Slurm：C2 prep（16 CPU） |
| [`scripts/slurm_train_gpu_c2_72h.sbatch`](../scripts/slurm_train_gpu_c2_72h.sbatch) | Slurm：C2 train + eval |
| [`scripts/slurm_tier_a_gpu.sbatch`](../scripts/slurm_tier_a_gpu.sbatch) | Slurm：档位 A GPU 训练（可选） |
| [`scripts/setup_server_ai25.sh`](../scripts/setup_server_ai25.sh) | 服务器 conda `ai25` 环境初始化 |
| [`scripts/fix_crlf.sh`](../scripts/fix_crlf.sh) | Windows 上传后修复 shell 脚本换行符 |

#### 脚本工具（内部）

| 文件 | 作用 |
|------|------|
| [`scripts/__init__.py`](../scripts/__init__.py) | 包标识 |
| [`scripts/_bootstrap.py`](../scripts/_bootstrap.py) | 将 `code/` 加入 `sys.path` |
| [`scripts/_cli.py`](../scripts/_cli.py) | 公共 argparse 助手（`--config`）；当前主流水线未统一引用 |

### 3.4 `docs/` — 补充文档

| 文件 | 作用 |
|------|------|
| [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) | **本文档**：架构 + 全文件说明 |
| [`docs/TIER_A_LOCAL.md`](TIER_A_LOCAL.md) | 档位 A 本机运行细节（3060、耗时、常见问题） |
| [`docs/USTC_SLURM_PREP.md`](USTC_SLURM_PREP.md) | USTC 算力平台 Slurm 提交说明 |
| [`docs/USTC_SUBMIT_SIMPLE.md`](USTC_SUBMIT_SIMPLE.md) | 简化版服务器提交备忘 |

### 3.5 `artifacts/` — 运行产物（非源码，按 tier 隔离）

产物由流水线生成；以下为当前仓库中**已有样例**的路径模式。

#### 共享 prep（无 tier 前缀，A/C2 共用，易被后跑 prep 覆盖）

| 路径 | 作用 |
|------|------|
| `artifacts/panel_daily.parquet` | 01 输出的长表面板（若存在，体积大，可能未纳入 git） |
| `artifacts/features/panel_features.parquet` | 02 特征表 |
| `artifacts/features/feature_meta.json` | 特征列名、行业词表 `industry_vocab` 等元数据 |
| `artifacts/news/panel_news.parquet` | 02b 新闻链指结果 |

#### 档位 A（`tier_a/`）

| 路径 | 作用 |
|------|------|
| `artifacts/checkpoints/tier_a/{gru,mlp,news_hash}/seed_42/best.pt` | A 三模态 checkpoint |
| `artifacts/predictions/tier_a/ensemble_scores.parquet` | 融合后每日 score |
| `artifacts/predictions/tier_a/fusion_weights.json` | 验证集 ICIR 融合权重 |
| `artifacts/metrics/tier_a/ic_summary.json` | IC/ICIR/spread 汇总 |
| `artifacts/metrics/tier_a/strategy_topk.json` | IC@K、命中率、signal_portfolio 等 |
| `artifacts/backtest/tier_a/valid/metrics.json` | 验证集回测指标（报告引用） |
| `artifacts/backtest/tier_a/valid/equity_curve.csv` | 验证集净值曲线 |
| `artifacts/backtest/tier_a/test/*` | 历史 test 段产物（`test_end=null` 后报告不用，保留作参考） |
| `artifacts/orders/tier_a/orders_*.csv` | 调仓单 |

#### 档位 C2（`tier_c2/`）

| 路径 | 作用 |
|------|------|
| `artifacts/checkpoints/tier_c2/{tft,ftt,news}/seed_{42,123,456}/best.pt` | C2 九组 checkpoint |
| `artifacts/predictions/tier_c2/ensemble_scores.parquet` | 融合 score |
| `artifacts/predictions/tier_c2/fusion_weights.json` | TFT/FTT/News 融合权重 |
| `artifacts/metrics/tier_c2/ic_summary.json` | IC 汇总 |
| `artifacts/metrics/tier_c2/strategy_topk.json` | 策略 Top-K 指标 |
| `artifacts/backtest/tier_c2/valid/metrics.json` | 验证集回测 |
| `artifacts/backtest/tier_c2/valid/equity_curve.csv` | 净值曲线 |
| `artifacts/orders/tier_c2/orders_20260520.csv` | 最新调仓单 |

#### 其他 metrics

| 路径 | 作用 |
|------|------|
| `artifacts/metrics/benchmark_valid.json` | `compute_benchmark_metrics.py` 输出的基准对照 |

---

## 4. 典型执行路径

### 4.1 档位 A（本机）

```bash
cd code
python scripts/validate_tier_a_local.py          # 可选预检
python scripts/run_tier_a_local.py               # 全流程
python scripts/print_strategy_topk_metrics.py    # A/C2 策略指标对照
```

### 4.2 档位 C2（服务器）

```bash
cd code
python scripts/run_server_c2_72h.py prep         # CPU：01–02–02b
python scripts/run_server_c2_72h.py train        # GPU：03×3 模态 × 3 seed
python scripts/run_server_c2_72h.py eval         # 04–07 + preflight
```

或使用 `scripts/web_c2_*.sh` / `slurm_*_c2_72h.sbatch` 提交作业。

### 4.3 仅重跑评估（已有 checkpoint）

```bash
python scripts/04_ensemble_predict.py --config config.server_c2_72h.yaml
python scripts/05_evaluate_ic.py --config config.server_c2_72h.yaml
python scripts/06_backtest.py --config config.server_c2_72h.yaml
python scripts/07_predict_latest.py --config config.server_c2_72h.yaml
```

### 4.4 报告图表（仓库上级目录）

```bash
python tier_c2_project/scripts/plot_tier_c2_report.py
# 输出 → tier_c2_project/figures/fig*.png → report.tex 引用
```

---

## 5. 与 `report.tex` 的对应关系

| 报告章节 | 代码/产物 |
|----------|-----------|
| 数据处理 | `01`–`02b`，`src/panel.py`、`features.py`、`preprocess.py` |
| 模型 | `src/models/*`，`03_train_*` |
| 融合 | `src/ensemble.py`，`04_ensemble_predict.py`，`fusion_weights.json` |
| 策略 | `src/strategy.py`，`config.*.yaml` 中 `strategy` 段 |
| 回测 | `src/backtest.py`，`06_backtest.py`，`backtest/*/valid/` |
| IC / IC@K | `src/metrics.py`，`05_evaluate_ic.py`，`print_strategy_topk_metrics.py` |
| 实验数字 | `metrics/tier_a/`、`metrics/tier_c2/`、`backtest/*/valid/metrics.json` |
| 插图 | `tier_c2_project/scripts/plot_tier_c2_report.py` |

---

## 6. 依赖关系简图（import）

```
scripts/0*.py
    └── src/config.py
    └── src/panel | features | news_link | dataset
    └── src/train/trainer.py ──► src/models/factory.py
    └── src/ensemble.py ──► src/metrics.py, src/preprocess.py
    └── src/backtest.py ──► src/strategy.py
    └── src/checkpoint_utils.py ──► factory（C2 eval 专用）

run_server_c2_72h.py ──► preflight.py, checkpoint_utils.py
03_predict_news_lora.py ──► news_predict.py
```

---

## 7. 注意事项

1. **必须显式 `--config`**：已移除通用 `config.yaml`，单步脚本须指定三个 yaml 之一。
2. **prep 共享路径**：A 与 C2 的 prep 写入同一 `panel_daily.parquet`；对比实验时注意 prep 顺序与 `liquidity_top_k`。
3. **C2 eval 行业数**：加载 TFT 时须与训练时 `ind_vocab` 一致，见 `checkpoint_utils.py`。
4. **NLP 依赖**：C2 训练/推理 NewsLoRA 需 `pip install -r requirements-nlp.txt`。
5. **valid-only**：`test_end=null` 时 05/06 不产出有意义的 test 段；报告只引用 valid。

---

*文档版本：与 code/ 清理后 86 文件清单一致（2026-05）。*
