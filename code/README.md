# TFNE-C A 股深度学习（档位 A + C2）

本目录为 `report.tex` 复现所需代码：从 `../data/` 读 CSV，产物写入 `./artifacts/`。**档位 A（RGNE-Lite）** 与 **档位 C2（TFNE-C2）** 使用不同配置与 `tier_tag`，checkpoint 互不覆盖。

> **架构与逐文件说明**见 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)。  
> **服务器 Slurm / conda 环境**见 [`docs/USTC_SLURM_PREP.md`](docs/USTC_SLURM_PREP.md)、[`docs/USTC_SUBMIT_SIMPLE.md`](docs/USTC_SUBMIT_SIMPLE.md)。

| 档位 | 配置 | 模型 | artifacts 子目录 |
|------|------|------|------------------|
| **A 本机基线** | `config.tier_a_local.yaml` | GRU + MLP + 新闻哈希 | `tier_a/` |
| **C2 主实验** | `config.server_c2_72h.yaml` | TFT-lite + FTT + NewsLoRA | `tier_c2/` |

> 各脚本须显式传 `--config`；无默认 `config.yaml`。  
> **C2 十只策略现状与出单**：见 **§五**（含前置检查、验证 CSV 行数、旧 50 只产物说明）。  
> **推荐 nightly**：`prep-inc` + `predict_day_orders.py`。

---

## 一、目录与路径

| 环境 | 项目根 | 数据目录 | 代码目录 |
|------|--------|----------|----------|
| 本机 Windows | `...\deep learning\pro` | `...\pro\data` | `...\pro\code` |
| USTC 服务器 | `/home/scc/pb23151824/pro` | `/home/scc/pb23151824/pro/data` | `/home/scc/pb23151824/pro/code` |

配置里 `paths.data_dir: ../data` 表示相对 **`code/`** 上一级，即与 `code/` 同级的 `data/` 文件夹。

---

## 二、拉取与准备数据

### 2.1 数据从哪来

原始 CSV **不由本仓库自动下载**，需自行准备或从课程/平台获取，按 [`../data/README.md`](../data/README.md) 的目录结构放置。

**首次全量**至少需要：

| 路径（相对 `data/`） | 用途 |
|----------------------|------|
| `basic.csv` | 股票列表、行业、上市日 |
| `trade_cal.csv` | 交易日历 |
| `daily/YYYYMMDD.csv` | 日频量价（2016 起，按日一个文件） |
| `market/*.csv` | 指数（000300.SH 等，作 benchmark） |
| `metric/YYYYMMDD.csv` | 基本面（可选但 C2 会用） |
| `moneyflow/YYYYMMDD.csv` | 资金流（可选但 C2 会用） |
| `news/YYYYMMDD.csv` | 东方财富快讯（2019 起） |
| `stock_st/YYYYMMDD.csv` | 每日 ST 名单 |

文件名中的日期为 **8 位字符串**，如 `20260520.csv`。

### 2.2 本机检查数据是否齐全

```powershell
cd "C:\Users\Lenovo\Desktop\deep learning\pro"

# 应有 thousands 个 daily 文件
(Get-ChildItem data\daily\*.csv).Count
Get-ChildItem data\daily\*.csv | Sort-Object Name | Select-Object -Last 3 Name
```

Linux / 服务器：

```bash
ls data/daily/*.csv | wc -l
ls -t data/daily/*.csv | head -3
```

### 2.3 每日增量更新（nightly 前必做）

比赛/模拟交易需要 **T 日收盘后** 把新交易日 CSV 放进 `data/`，与全量目录结构相同：

```text
data/daily/20260529.csv
data/metric/20260529.csv      # 若有
data/moneyflow/20260529.csv   # 若有
data/news/20260529.csv        # 若有
data/stock_st/20260529.csv    # 若有
```

代码不会联网拉 Tushare；**缺文件则 panel 最大日期停在上次 prep 的末尾**。

### 2.4 本机 → 服务器同步（USTC 算力平台）

**首次上传（全量 data + code）：**

```powershell
cd "C:\Users\Lenovo\Desktop\deep learning\pro"
tar -czvf data.tar.gz data
tar -czvf code.tar.gz code
```

通过平台「文件管理」上传到 `/home/scc/pb23151824/`，在登录 Shell 解压：

```bash
mkdir -p /home/scc/pb23151824/pro
cd /home/scc/pb23151824/pro
tar -xzf /home/scc/pb23151824/data.tar.gz
tar -xzf /home/scc/pb23151824/code.tar.gz
# 解压后应有 pro/data 与 pro/code
```

**只更新若干新 CSV（推荐 nightly）：** 在平台文件管理里直接上传到 `pro/data/daily/` 等子目录，或本机 scp/rsync 增量文件（路径以平台实际为准）。

**只更新代码：** 重新打包 `code.tar.gz` 上传解压，或只覆盖改动的 `code/scripts/`、`code/src/` 文件。

### 2.5 数据 → 中间产物（prep 做什么）

prep 把 CSV 转成 `artifacts/` 下的 parquet，后续训练/推理只读这些文件：

```text
../data/*.csv
  → artifacts/panel_daily.parquet           # 01 合并面板
  → artifacts/features/panel_features.parquet + feature_meta.json   # 02 特征
  → artifacts/news/panel_news.parquet       # 02b 新闻链指
```

字段说明见 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)。

---

## 三、环境与依赖

### 3.1 Conda 环境 `ai25`

**本机 Windows：**

```powershell
conda activate ai25
cd "C:\Users\Lenovo\Desktop\deep learning\pro\code"
pip install -r requirements.txt
pip install -r requirements-nlp.txt   # C2 NewsLoRA 必须
```

**USTC 服务器（须先 source，否则 conda 找不到）：**

```bash
source /home/scc/pb23151824/miniforge3/etc/profile.d/conda.sh
conda activate ai25
cd /home/scc/pb23151824/pro/code
export HF_ENDPOINT=https://hf-mirror.com
export DISABLE_SAFETENSORS_CONVERSION=true
```

首次未装环境时：

```bash
bash /home/scc/pb23151824/pro/code/scripts/setup_server_ai25.sh
```

验证：

```bash
python scripts/validate_server_env.py
python scripts/check_gpu.py          # GPU 节点 / 作业内
python scripts/check_panel_features_health.py   # prep 后检查 features
```

---

## 四、运行程序（流水线）

### 4.1 总览

```text
01_build_panel        面板 + excess_ret_1d
02_make_features      GKX 截面变换 + 时序/截面特征
02b_embed_news        新闻链指
03_train_*            A=GRU/MLP/哈希；C2=TFT/FTT/NewsLoRA
04_ensemble_predict   多 seed 秩融合 score
05_evaluate_ic        IC / ICIR / strategy_topk
06_backtest           目标权重策略回测（valid-only）
07_predict_latest     按 ensemble 最新日生成调仓 CSV
predict_day_orders    指定 signal 日推理 + 出单（nightly 推荐）
```

### 4.2 档位 A（本机调试 / 对照）

Top500、单 seed、无需 BERT 微调，适合 RTX 3060 6GB：

```bash
cd code
conda activate ai25

# 一键：prep → train → eval → 订单
python scripts/run_tier_a_local.py

# 分步
python scripts/run_tier_a_local.py --prep-only
python scripts/run_tier_a_local.py --train-eval-only

# 策略 Top-K 指标打印
python scripts/print_strategy_topk_metrics.py
```

配置：`config.tier_a_local.yaml`（`strategy.n_max: 30`）。

### 4.3 档位 C2（主实验 / 服务器）

Top3000、3 seed、需 GPU + `requirements-nlp.txt`。配置：`config.server_c2_72h.yaml`（`strategy.n_max: 10`）。

**首次或换股票池 / 换 TopK 后 — 全量 prep（CPU，约数小时）：**

```bash
cd code
conda activate ai25

python scripts/run_server_c2_72h.py prep
# 等价于依次跑 01 + 02 + 02b
```

**已有 panel，仅追加新交易日 — 增量 prep（nightly，约 2–5 分钟/天）：**

```bash
python scripts/run_server_c2_72h.py prep-inc
# 或
python scripts/prep_incremental.py --config config.server_c2_72h.yaml --n-jobs 4
```

若 `panel_features.parquet` 损坏或缺失，增量脚本会**自动从 panel 全量重建 features**。

**训练（GPU，首次约 28–42 h，3 seed）：**

```bash
python scripts/run_server_c2_72h.py train
# 拆分：train-tft | train-ftt | train-news | predict-news
```

**离线评估（valid 段融合 + IC + 回测 + 最新日订单）：**

```bash
python scripts/run_server_c2_72h.py eval
```

**Slurm 提交（服务器）：**

```bash
sbatch scripts/slurm_prep_cpu_c2_72h.sbatch
sbatch scripts/slurm_train_gpu_c2_72h.sbatch
```

图形界面整段命令见 `scripts/web_c2_prep_command.sh`、`web_c2_gpu_command.sh`、`web_c2_eval_command.sh`。

### 4.4 单步脚本（均需 `--config`）

```bash
python scripts/01_build_panel.py --config config.server_c2_72h.yaml --n-jobs 4
python scripts/02_make_features.py --config config.server_c2_72h.yaml
python scripts/02b_embed_news.py --config config.server_c2_72h.yaml
python scripts/03_train_tft.py --config config.server_c2_72h.yaml --seed 42
python scripts/04_ensemble_predict.py --config config.server_c2_72h.yaml
python scripts/05_evaluate_ic.py --config config.server_c2_72h.yaml
python scripts/06_backtest.py --config config.server_c2_72h.yaml
python scripts/07_predict_latest.py --config config.server_c2_72h.yaml
```

---

## 五、C2 十只策略：现状、改动与出单

本节说明：**策略是否已是 10 只**、**磁盘上旧文件为何还是 50 只**、以及**如何跑一遍得到正确的 10 只调仓单**。

### 5.0 现状一览

| 项目 | 当前状态 | 说明 |
|------|----------|------|
| **C2 配置** | `config.server_c2_72h.yaml` → `strategy.n_max: 10` | 已改，是唯一的功能性改动 |
| **策略代码** | `src/strategy.py` 等读 `n_max` 参数 | 未改；默认 `50` 只是缺配置时的兜底 |
| **出单脚本** | `predict_day_orders.py` / `07_predict_latest.py` | 通过 `scfg.get("n_max", 50)` 读 YAML → 实际为 **10** |
| **模型 checkpoint** | `artifacts/checkpoints/tier_c2/` | 与 `n_max` **无关**，改 10 只**不必重训** |
| **旧调仓 CSV** | 如 `orders_20260520.csv`、`orders_20260525.csv` | 多为改配置**之前**生成，**约 50 行**，不能直接用 |
| **旧评估 JSON** | `artifacts/metrics/tier_c2/strategy_topk.json` | 可能仍写 `"n_max": 50`，需重跑 `05`/`06` 才与 10 对齐 |
| **档位 A** | `config.tier_a_local.yaml` → `n_max: 30` | 与 C2 无关 |

**结论：**

- **配置上** C2 已是「最多持 10 只」；
- **磁盘上的旧 `orders_*.csv` 不会自动变**，必须按 §5.3 重新跑 `predict_day_orders.py`；
- **没有漏改的代码**导致「配置了 10 却出 50」——旧文件只是历史产物。

### 5.1 「10 只」在代码里是什么意思

选股与权重在 `src/strategy.py` 的 `target_weights()`：

1. 可交易池：剔除 ST、北交所、停牌、上市不足 60 日、流动性不足；
2. 分数过滤：保留 `score ≥ Q(q_min)`（默认 `q_min=0.90`）；
3. 取 Top **`n_max`** 只（C2 为 **10**）；
4. 秩加权 + 单票/行业/微盘等约束 → **目标权重**，权重和 ≈ 1（尽量满仓）。

**注意语义：**

- `n_max: 10` = **组合里最多持有 10 只股票**（目标权重 CSV 里 ≤ 10 行）；
- **不是**作业 PDF 那种「每天固定卖出 k 只、再买入 k 只」的规则；
- 实际买卖笔数由分数变化、`delta`（小幅不调）、`omega_max`（换手上限）等决定，**不必恰好换 10 只**。

参数在 YAML 的 `strategy:` 段；改 **`n_max`** 只影响组合构建与回测/出单，**不影响模型训练**。

### 5.2 从 50 改到 10：改了什么、在哪

| 文件 | 是否改动 | 作用 |
|------|----------|------|
| `config.server_c2_72h.yaml` | **是**（`n_max: 50` → `10`） | 唯一必改项 |
| `README.md` | 是（文档） | 说明用法 |
| `src/strategy.py` | 否 | `head(params.n_max)`，随配置变 |
| `src/metrics.py` | 否 | IC@K 的 K = 传入的 `n_max` |
| `scripts/predict_day_orders.py` | 否 | 读 `strategy.n_max` |
| `scripts/05_evaluate_ic.py` / `06_backtest.py` / `07_predict_latest.py` | 否 | 同上 |

**改动都在 `code/` 目录内**；项目根目录的 `data/`、`report.tex` 等与此无关。

`predict_day_orders.py` 末尾 `head(10)` **仅用于终端打印前 10 行**，不决定持仓数量；决定数量的是前面的 `target_weights(..., n_max=10)`。

### 5.3 跑一遍生成 10 只调仓单（逐步）

以下以 **C2 + 本机或服务器 GPU** 为例；**不必重训**，前提是 checkpoint 已存在。

#### 步骤 0：前置检查

```bash
cd code
conda activate ai25
# 服务器须先：source .../miniforge3/etc/profile.d/conda.sh

# 配置里应是 n_max: 10
grep -A2 "^strategy:" config.server_c2_72h.yaml

# C2 三个模态 checkpoint 至少各有一个 seed（通常 42/123/456）
ls artifacts/checkpoints/tier_c2/

# features 与 panel 日期（出单依赖 panel_features.parquet）
python -c "
import pandas as pd
f=pd.read_parquet('artifacts/features/panel_features.parquet', columns=['trade_date'])
print('panel_features max:', f.trade_date.max())
"
```

| 检查项 | 期望 | 不满足时 |
|--------|------|----------|
| `strategy.n_max` | `10` | 确认用的是 `config.server_c2_72h.yaml` |
| `artifacts/checkpoints/tier_c2/*` | 存在 `tft/ftt/news` 的 `.pt` | 先跑 `run_server_c2_72h.py train` |
| `panel_features.parquet` | 有目标 signal 日 | 跑 `prep-inc` 或全量 `prep` |
| `data/daily/` | 含最新交易日 CSV | 先更新 `../data/` 再 `prep-inc` |

#### 步骤 1：更新数据（若要比「已有 panel 最大日」更新的 signal 日）

把新日 CSV 放入 `../data/daily/`（及 `metric/`、`news/` 等），然后：

```bash
python scripts/run_server_c2_72h.py prep-inc
# 或
python scripts/prep_incremental.py --config config.server_c2_72h.yaml --n-jobs 4
```

若 panel 已是最新、只是重算订单，可**跳过**本步。

#### 步骤 2：推理 + 出单（核心）

```bash
# 默认 signal 日 = panel_features 最大 trade_date
python scripts/predict_day_orders.py --config config.server_c2_72h.yaml --buf-days 70

# 或指定 signal 日
python scripts/predict_day_orders.py --config config.server_c2_72h.yaml --trade-date 20260525 --buf-days 70
```

**必须带** `--config config.server_c2_72h.yaml`；不传配置时脚本可能读不到 `n_max: 10`。

成功时终端类似：

```text
[tier_c2] signal_date=20260525 -> 次日开盘调仓
  orders -> artifacts/orders/tier_c2/orders_20260525.csv  (n=10, weight_sum=0.98xx)
```

`n=` 应 **≤ 10**（通常正好 10；可交易池极窄时可能更少）。

#### 步骤 3：验证 CSV 确实是 10 只

**PowerShell（本机）：**

```powershell
$csv = Import-Csv "artifacts\orders\tier_c2\orders_20260525.csv"
$csv.Count                    # 应 <= 10
($csv.target_weight | Measure-Object -Sum).Sum   # 应接近 1
$csv | Sort-Object { [double]$_.target_weight } -Descending | Select-Object -First 5
```

**Bash：**

```bash
# 数据行数（不含表头）应 <= 10
tail -n +2 artifacts/orders/tier_c2/orders_20260525.csv | wc -l

# 权重和
python -c "
import pandas as pd
o=pd.read_csv('artifacts/orders/tier_c2/orders_20260525.csv')
print('n=',len(o),'weight_sum=',o.target_weight.sum())
print(o.sort_values('target_weight',ascending=False))
"
```

若行数仍约 **50**，说明：用了旧 CSV 没重跑、或 `--config` 指错了文件、或看错档位目录（`tier_a` vs `tier_c2`）。

#### 步骤 4（可选）：报告指标与 10 只对齐

旧 `strategy_topk.json` 里 `n_max: 50` 不影响模拟盘出单，但报告 IC@K 会不一致。需要时：

```bash
python scripts/05_evaluate_ic.py --config config.server_c2_72h.yaml
python scripts/06_backtest.py --config config.server_c2_72h.yaml
python scripts/print_strategy_topk_metrics.py
```

确认 `artifacts/metrics/tier_c2/strategy_topk.json` 顶层为 `"n_max": 10`。

#### 一条龙命令（panel 已最新、checkpoint 已有）

```bash
cd code && conda activate ai25
python scripts/predict_day_orders.py --config config.server_c2_72h.yaml --buf-days 70
```

#### 每个交易日 nightly（数据已更新后）

```bash
python scripts/run_server_c2_72h.py prep-inc
python scripts/predict_day_orders.py --config config.server_c2_72h.yaml --buf-days 70
```

粗估 **20–35 分钟/晚**（prep-inc 2–5 min + predict 15–30 min，RTX 3060 6GB）。

### 5.4 输出文件说明

| 文件 | 含义 |
|------|------|
| `artifacts/predictions/tier_c2/ensemble_scores.parquet` | 当日 score 追加/更新 |
| `artifacts/orders/tier_c2/orders_YYYYMMDD.csv` | **调仓建议（模拟盘用，≤10 行）** |

**`orders_YYYYMMDD.csv` 格式：**

```csv
ts_code,target_weight,trade_date
600519.SH,0.1023,20260525
000858.SZ,0.0987,20260525
...
```

- `trade_date`：**信号日 T**（T 日收盘信息算分）；
- **T+1 开盘**按 `target_weight` 调仓；
- 数据行数 ≤ `n_max`（C2 为 10），`target_weight` 之和 ≈ 1。

### 5.6 离线 eval 出单（与 nightly 的区别）

跑完 `run_server_c2_72h.py eval` 后，`07_predict_latest.py` 会按 **ensemble_scores 中最大 trade_date** 写订单（不是 panel 最大日，若两者不一致以 scores 为准）：

```bash
python scripts/07_predict_latest.py --config config.server_c2_72h.yaml
# → artifacts/orders/tier_c2/orders_<last_score_date>.csv
```

同样读 `strategy.n_max`，出单行数 ≤ 10。valid 段批量 scores + 回测由 `04`–`06` 完成；**对新交易日出单仍推荐 §5.3 的 `predict_day_orders`**。

### 5.7 策略评价指标（报告用）

```bash
python scripts/05_evaluate_ic.py --config config.server_c2_72h.yaml
python scripts/06_backtest.py --config config.server_c2_72h.yaml
python scripts/print_strategy_topk_metrics.py
```

| 文件 | 内容 |
|------|------|
| `artifacts/metrics/tier_c2/ic_summary.json` | 全市场 IC / ICIR |
| `artifacts/metrics/tier_c2/strategy_topk.json` | **IC@K**（K=`n_max`，应为 10） |
| `artifacts/backtest/tier_c2/valid/metrics.json` | valid 段成本后夏普、回撤等 |
| `artifacts/predictions/tier_c2/ensemble_scores.parquet` | 每日每票融合 score |
| `artifacts/predictions/tier_c2/fusion_weights.json` | 各模态融合权重 |

旧 JSON 若仍为 `"n_max": 50`，是改配置前跑的 eval；按 §5.3 步骤 4 重跑即可。

### 5.8 常见问题

| 现象 | 原因 | 处理 |
|------|------|------|
| `orders_*.csv` 约 50 行 | 旧文件未重跑 | 按 §5.3 重跑 `predict_day_orders` |
| 终端 `n=50` | 未传 `--config config.server_c2_72h.yaml` | 显式传配置 |
| `Command 'conda' not found`（服务器） | 未 `source conda.sh` | 见 §三 |
| predict 报错缺 checkpoint | 未训练 C2 | `run_server_c2_72h.py train` |
| panel 无目标日 | data 未更新或未 prep-inc | 更新 CSV 后 `prep-inc` |

---

## 六、快速命令速查

```bash
cd code
conda activate ai25

# --- 档位 A 本机 ---
python scripts/run_tier_a_local.py
python scripts/print_strategy_topk_metrics.py

# --- 档位 C2：首次 ---
python scripts/run_server_c2_72h.py prep
python scripts/run_server_c2_72h.py train
python scripts/run_server_c2_72h.py eval

# --- 档位 C2：每交易日（10 只出单）---
python scripts/run_server_c2_72h.py prep-inc
python scripts/predict_day_orders.py --config config.server_c2_72h.yaml --buf-days 70
# 验证：orders CSV 数据行应 <= 10（见 §5.3 步骤 3）
```

---

## 七、目录索引

| 路径 | 说明 |
|------|------|
| `config.tier_a_local.yaml` | 档位 A 本机（Top500，`n_max=30`） |
| `config.tier_a.yaml` | 档位 A 备用 |
| `config.server_c2_72h.yaml` | 档位 C2（Top3000 / 3 seed，`n_max=10`） |
| `src/` | 数据处理、模型、训练、融合、策略、回测 |
| `scripts/run_tier_a_local.py` | A 一键全流程 |
| `scripts/run_server_c2_72h.py` | C2 prep / prep-inc / train / eval |
| `scripts/predict_day_orders.py` | **指定日推理 + 出单** |
| `scripts/prep_incremental.py` | 增量 prep |
| `docs/ARCHITECTURE.md` | 架构 + 全文件索引 |
| `docs/TIER_A_LOCAL.md` | A 本机说明 |
| `docs/USTC_SLURM_PREP.md` | 服务器环境与 Slurm |

## 八、依赖

```bash
pip install -r requirements.txt
pip install -r requirements-nlp.txt   # C2 NewsLoRA 必须
```

报告图表（项目根目录）：`python ../tier_c2_project/scripts/plot_tier_c2_report.py`
