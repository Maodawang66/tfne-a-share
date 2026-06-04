# TFNE · A 股多模态深度学习趋势预测与模拟交易

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Academic%20use-lightgrey.svg)]()

中国科学技术大学《深度学习基础》课程大作业完整工程：**利用价量、基本面、资金流与新闻文本，在 A 股流动性股票池上构建神经网络排序模型，完成 IC 评估、历史回测与同花顺 FDL2026 模拟交易出单。**

主方案 **TFNE-C2**（TFT-lite + FT-Transformer + NewsLoRA 多 seed 秩融合），并保留档位 A（RGNE-Lite）本机基线作对照。

---

## 特性

- **多模态融合**：时序深度网络 + 表格 Transformer + 中文 RoBERTa 新闻语义（LoRA 微调）
- **严格防泄露**：walk-forward 切分、逐期截面秩变换、valid-only 报告口径
- **可复现流水线**：`01` 面板 → `02` 特征 → `02b` 新闻 → `03` 训练 → `04` 融合 → `05/06` 评估 → nightly 出单
- **增量更新**：`prep-inc` 仅追加新交易日，适合每日收盘后快速刷新
- **策略层**：Top-K 目标权重组合（C2 默认 **每日最多 10 只**），输出 `orders_YYYYMMDD.csv`

---

## 仓库结构

```text
tfne-a-share/
├── code/                 # 主代码（配置、脚本、src、文档）
│   ├── config.server_c2_72h.yaml
│   ├── config.tier_a_local.yaml
│   ├── scripts/          # 流水线入口
│   ├── src/              # 数据、模型、训练、策略、回测
│   └── README.md         # 详细运行手册（数据、环境、出单）
├── data/                 # 原始 CSV（日频目录未入库，见下文）
│   └── README.md
├── report.tex            # 实验报告源文件
├── tier_a_project/       # 档位 A 图表脚本
├── tier_c2_project/      # 档位 C2 图表脚本
└── 设想.md / 项目方案.md  # 方案与设计说明
```

---

## 快速开始

### 1. 克隆与环境

```bash
git clone https://github.com/Maodawang66/tfne-a-share.git
cd tfne-a-share/code

conda create -n ai25 python=3.10 -y
conda activate ai25
pip install -r requirements.txt
pip install -r requirements-nlp.txt   # C2 NewsLoRA 必须
```

### 2. 准备数据

本仓库**不包含**约 4GB 的日频 CSV（`data/daily/`、`metric/`、`moneyflow/`、`news/` 等），需按 [`data/README.md`](data/README.md) 自行放置到项目根目录的 `data/` 下。

已入库的小文件：`basic.csv`、`trade_cal.csv`、`market/*.csv`。

### 3. 档位 C2（推荐）

```bash
# 首次：全量预处理 + GPU 训练（服务器约 28–42h）
python scripts/run_server_c2_72h.py prep
python scripts/run_server_c2_72h.py train
python scripts/run_server_c2_72h.py eval

# 每交易日：增量 prep + 10 只出单
python scripts/run_server_c2_72h.py prep-inc
python scripts/predict_day_orders.py --config config.server_c2_72h.yaml --buf-days 70
```

调仓单输出：`code/artifacts/orders/tier_c2/orders_YYYYMMDD.csv`（需本地训练后生成；checkpoint 未入库）。

### 4. 档位 A（本机 6GB GPU 调试）

```bash
python scripts/run_tier_a_local.py
```

---

## 模型档位

| 档位 | 配置 | 架构 | 股票池 |
|------|------|------|--------|
| **A** | `code/config.tier_a_local.yaml` | GRU + TabMLP + 新闻哈希 | Top 500 |
| **C2** | `code/config.server_c2_72h.yaml` | TFT-lite + FTT + NewsLoRA × 3 seeds | Top 3000 |

策略参数（C2）：`strategy.n_max: 10`（组合最多 10 只股票，改配置即可，无需重训）。

---

## 文档

| 文档 | 内容 |
|------|------|
| [**code/README.md**](code/README.md) | 数据同步、环境、流水线、十只策略出单、验证步骤 |
| [**code/docs/ARCHITECTURE.md**](code/docs/ARCHITECTURE.md) | 架构与模块说明 |
| [**code/docs/USTC_SLURM_PREP.md**](code/docs/USTC_SLURM_PREP.md) | 本科生算力平台 / conda `ai25` |
| [**设想.md**](设想.md) | C2 方案与耗时估算 |

---

## 未纳入 Git 的内容

为控制仓库体积（GitHub 单文件约 100MB 限制），以下内容在 [`.gitignore`](.gitignore) 中排除，请在本地或服务器单独备份：

- `data/daily/`、`data/metric/`、`data/moneyflow/`、`data/news/`、`data/stock_st/`（日频原始数据）
- `code/artifacts/`（`panel_features.parquet`、checkpoint `*.pt`、回测与订单产物）
- `*.tar.gz` 打包归档

克隆后首次使用需在目标机器上跑 `prep` 或从备份恢复 `artifacts/`。

---

## 评价指标

- 预测：**IC / ICIR**、策略 **IC@K**（K = `n_max`）
- 组合：valid 段成本后 **年化收益、夏普、最大回撤**，相对沪深 300 超额
- 交易：T+1 调仓、涨跌停与停牌约束、Top-K 目标权重

---

## 声明

本项目为**课程学习与研究用途**，不构成投资建议。数据来源于课程提供的 A 股公开字段；请勿将仓库用于商业实盘或未经授权的数据再分发。

---

## 作者

[Maodawang66](https://github.com/Maodawang66) · 中国科学技术大学
