# USTC 本科生算力平台 · CPU prep 提交指南（档位 C）

| 环境 | 项目根目录 | 数据目录 | 代码目录 |
|------|------------|----------|----------|
| 本机 Windows | `C:\Users\Lenovo\Desktop\deep learning\pro` | `C:\Users\Lenovo\Desktop\deep learning\pro\data` | `C:\Users\Lenovo\Desktop\deep learning\pro\code` |
| 服务器 Linux | `/home/scc/pb23151824/pro` | `/home/scc/pb23151824/pro/data` | `/home/scc/pb23151824/pro/code` |

参考：[USTC107 深度学习作业文档](https://xinchengo.github.io/ustc107/guides/ai/deep-learning-homework/)

prep = `01_build_panel` + `02_make_features` + `02b_embed_news`（纯 CPU，`#SBATCH --gres=gpu:0`）。

档位 A GPU 提交见：`C:\Users\Lenovo\Desktop\deep learning\pro\code\docs\USTC_SUBMIT_SIMPLE.md`（服务器侧同路径：`/home/scc/pb23151824/pro/code/docs/USTC_SUBMIT_SIMPLE.md`）。

---

## 一、相关文件绝对路径

| 本机 Windows | 服务器 Linux |
|--------------|--------------|
| `C:\Users\Lenovo\Desktop\deep learning\pro\code\scripts\slurm_prep_cpu.sbatch` | `/home/scc/pb23151824/pro/code/scripts/slurm_prep_cpu.sbatch` |
| `C:\Users\Lenovo\Desktop\deep learning\pro\code\scripts\run_server_c.py` | `/home/scc/pb23151824/pro/code/scripts/run_server_c.py` |
| `C:\Users\Lenovo\Desktop\deep learning\pro\code\config.server_c_72h.yaml` | `/home/scc/pb23151824/pro/code/config.server_c_72h.yaml` |
| `C:\Users\Lenovo\Desktop\deep learning\pro\code\requirements.txt` | `/home/scc/pb23151824/pro/code/requirements.txt` |

---

## 二、上传文件

### 2.1 本机打包（PowerShell）

```powershell
cd "C:\Users\Lenovo\Desktop\deep learning\pro"

tar -czvf "C:\Users\Lenovo\Desktop\deep learning\pro\data.tar.gz" data
tar -czvf "C:\Users\Lenovo\Desktop\deep learning\pro\code.tar.gz" code
```

生成：

- `C:\Users\Lenovo\Desktop\deep learning\pro\data.tar.gz`
- `C:\Users\Lenovo\Desktop\deep learning\pro\code.tar.gz`

### 2.2 上传到服务器

将上述两个 `.tar.gz` 通过平台「文件管理」上传到：

- `/home/scc/pb23151824/data.tar.gz`
- `/home/scc/pb23151824/code.tar.gz`

解压、环境配置与 prep 见 **§三**（一次性）→ **§5.1**（预检 + 提交）。

---

## 三、登录节点配置 Miniforge（一次性，必做）

官方说明：[USTC107 深度学习作业 · 配置环境](https://xinchengo.github.io/ustc107/guides/ai/deep-learning-homework/)。  
中科大镜像说明：[Miniforge / mamba](https://mirrors.ustc.edu.cn/help/anaconda.html#miniforge-mamba)

### 3.1 为什么要这样做

| 概念 | 说明 |
|------|------|
| **登录节点** | 平台「Shell」连到的节点；用于传文件、**装环境**、提交作业；不要在这里跑 prep 训练 |
| **计算节点** | 如 `anode06`；sbatch 提交的作业在这里执行 |
| **home 共享** | `/home/scc/pb23151824` 在登录节点与计算节点**共用**，登录节点装好 Miniforge 后，作业里可直接 `conda activate ai25` |

你之前 prep 失败 `ERROR: 未找到 miniforge3 或 anaconda3`，就是因为 **§三 尚未完成**。

### 3.2 在哪里执行

平台 **「登录集群 → Shell」**（登录节点）。**不要**把下面命令写进 sbatch 作业里安装 Miniforge（慢且易超时）。

### 3.3 检查是否已安装

```bash
echo $HOME
ls -l /home/scc/pb23151824/miniforge3/etc/profile.d/conda.sh
ls -l /home/scc/pb23151824/anaconda3/etc/profile.d/conda.sh 2>/dev/null
which conda 2>/dev/null || true
```

若 `miniforge3/etc/profile.d/conda.sh` **已存在** → 跳到 **§3.5**。

### 3.4 安装 Miniforge（首次）

```bash
cd /home/scc/pb23151824

# 中科大镜像（推荐）
wget https://mirrors.ustc.edu.cn/github-release/conda-forge/miniforge/LatestRelease/Miniforge3-Linux-x86_64.sh -O Miniforge3-Linux-x86_64.sh

# 若 wget 失败，试官方：
# wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh

bash Miniforge3-Linux-x86_64.sh -b -p /home/scc/pb23151824/miniforge3

source /home/scc/pb23151824/miniforge3/etc/profile.d/conda.sh
conda -V
mamba -V
```

安装路径固定为：`/home/scc/pb23151824/miniforge3`（与作业脚本中的 `source` 路径一致）。

### 3.5 创建项目环境 `ai25`

官方示例用 `mamba create -n dl-homework python=3.14`；本项目与本机对齐，用 **`ai25` + Python 3.10**。

### 3.6 一次性配齐全项目（推荐）

**一条命令**安装 prep + GPU train + BERT/NLP 全部依赖（登录 Shell 执行）：

```bash
# 若 code 尚未解压，先：
mkdir -p /home/scc/pb23151824/pro && cd /home/scc/pb23151824/pro && tar -xzf /home/scc/pb23151824/code.tar.gz

bash /home/scc/pb23151824/pro/code/scripts/setup_server_ai25.sh
```

脚本 `/home/scc/pb23151824/pro/code/scripts/setup_server_ai25.sh` 会自动完成：

| 步骤 | 内容 |
|------|------|
| 1 | 安装 Miniforge（若尚未安装） |
| 2 | 创建 `ai25`（Python 3.10） |
| 3 | 解压 `code.tar.gz`（若需要） |
| 4 | **PyTorch CUDA 12.1** + `requirements.txt` + `requirements-nlp.txt` |
| 5 | `nvitop`、`git` |
| 6 | 写入 `~/.bashrc`：`HF_ENDPOINT` 镜像 |
| 7 | 运行 `validate_server_env.py` |

**覆盖范围：**

| 任务 | 是否支持 |
|------|----------|
| 档位 C prep（CPU） | ✓ |
| 档位 C GPU train + eval | ✓ |
| 档位 A 服务器基线 | ✓ |

你已装好 Miniforge 时，脚本会**跳过**安装步骤，只补全 ai25 与依赖。

### 3.7 手动分步安装（与脚本等价）

```bash
source /home/scc/pb23151824/miniforge3/etc/profile.d/conda.sh
mamba create -n ai25 python=3.10 -y
conda activate ai25

mamba install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r /home/scc/pb23151824/pro/code/requirements.txt
pip install -r /home/scc/pb23151824/pro/code/requirements-nlp.txt
mamba install -y nvitop git

export HF_ENDPOINT=https://hf-mirror.com
export DISABLE_SAFETENSORS_CONVERSION=true
```

### 3.8 验证

```bash
source /home/scc/pb23151824/miniforge3/etc/profile.d/conda.sh
conda activate ai25
python /home/scc/pb23151824/pro/code/scripts/validate_server_env.py
```

- 登录节点 `cuda available=False` **正常**（无 GPU）
- GPU 作业内运行 `python scripts/check_gpu.py` 应看到 `cuda True`

全部成功 → **§5.1 提交 prep** → prep 完成后 **§六 提交 train**（**无需再装依赖**）。

---

## 四、命令行提交 prep（可选，需已解压）

若已通过 §5.1 解压，可直接 sbatch；否则先执行 §5.1 步骤 2 中的解压与检查部分。

```bash
mkdir -p /home/scc/pb23151824/pro/code/logs
cd /home/scc/pb23151824/pro/code
sbatch /home/scc/pb23151824/pro/code/scripts/slurm_prep_cpu.sbatch
```

查看日志：

```bash
squeue -u pb23151824
tail -f /home/scc/pb23151824/pro/code/logs/prep_*.out
tail -f /home/scc/pb23151824/pro/code/logs/prep_*.err
```

取消作业：先执行 `squeue -u pb23151824` 查看 JOBID，再执行 `scancel JOBID`。

---

## 五、图形界面提交 prep

最长运行时间填 **2880 分钟**（48 小时）。命令框内容优先于下方表单。

### 5.1 命令框（整段粘贴）

**前提：已完成 §三（Miniforge + ai25）。** 未完成会先报 `ERROR: 未找到 miniforge3 或 anaconda3`。

**步骤 1 — 上传前检查（本机 PowerShell，打包上传前执行）：**

```powershell
Test-Path "C:\Users\Lenovo\Desktop\deep learning\pro\data"
Test-Path "C:\Users\Lenovo\Desktop\deep learning\pro\code\scripts\slurm_prep_cpu.sbatch"
Test-Path "C:\Users\Lenovo\Desktop\deep learning\pro\code\scripts\run_server_c.py"
Test-Path "C:\Users\Lenovo\Desktop\deep learning\pro\code\config.server_c_72h.yaml"
Test-Path "C:\Users\Lenovo\Desktop\deep learning\pro\code\requirements.txt"
```

五项均应返回 `True`。完成 §2.1 打包、§2.2 上传后，**先在登录节点 Shell 执行步骤 1.5 预检**（几分钟内可知 conda/依赖是否正常）；通过后再粘贴 **步骤 2** 提交 prep 作业。

**步骤 1.5 — 登录节点预检（上传后、提交作业前，Shell 粘贴，无 `#SBATCH`，不跑 prep）：**

```bash
set -euo pipefail

echo "========== 预检开始 $(date -Iseconds) =========="

# --- 压缩包 ---
test -f /home/scc/pb23151824/data.tar.gz || { echo "ERROR: 未找到 data.tar.gz"; exit 1; }
test -f /home/scc/pb23151824/code.tar.gz || { echo "ERROR: 未找到 code.tar.gz"; exit 1; }
echo "[CHECK] tar.gz 存在 OK"

# --- 解压（若 pro/code 已有则跳过）---
if [ ! -f /home/scc/pb23151824/pro/code/scripts/run_server_c.py ]; then
  echo "[CHECK] 开始解压 ..."
  mkdir -p /home/scc/pb23151824/pro
  cd /home/scc/pb23151824/pro
  tar -xzvf /home/scc/pb23151824/data.tar.gz
  tar -xzvf /home/scc/pb23151824/code.tar.gz
else
  echo "[CHECK] pro/code 已存在，跳过解压"
fi

test -d /home/scc/pb23151824/pro/data || { echo "ERROR: 缺少 pro/data"; exit 1; }
test -f /home/scc/pb23151824/pro/code/scripts/run_server_c.py || { echo "ERROR: 缺少 run_server_c.py"; exit 1; }
test -f /home/scc/pb23151824/pro/code/config.server_c_72h.yaml || { echo "ERROR: 缺少 config.server_c_72h.yaml"; exit 1; }
test -f /home/scc/pb23151824/pro/code/requirements.txt || { echo "ERROR: 缺少 requirements.txt"; exit 1; }
ls /home/scc/pb23151824/pro/data/*.csv | head
echo "[CHECK] 解压后目录 OK"

# --- conda（自动识别 miniforge / anaconda）---
set +u
if [ -f /home/scc/pb23151824/miniforge3/etc/profile.d/conda.sh ]; then
  source /home/scc/pb23151824/miniforge3/etc/profile.d/conda.sh
  echo "[CHECK] conda: miniforge3 OK"
elif [ -f /home/scc/pb23151824/anaconda3/etc/profile.d/conda.sh ]; then
  source /home/scc/pb23151824/anaconda3/etc/profile.d/conda.sh
  echo "[CHECK] conda: anaconda3 OK"
else
  echo "ERROR: 未找到 miniforge3 或 anaconda3，请先完成 §三"; exit 1
fi

if ! conda env list | grep -qE '^ai25[[:space:]]'; then
  echo "[CHECK] 创建 conda 环境 ai25 ..."
  conda create -n ai25 python=3.10 -y
else
  echo "[CHECK] conda 环境 ai25 已存在"
fi
conda activate ai25
set -u
echo "[CHECK] python=$(which python) $(python -V)"

pip install -r /home/scc/pb23151824/pro/code/requirements.txt -q
python -c "import pandas, pyarrow; print('[CHECK] deps OK: pandas+pyarrow')"

cd /home/scc/pb23151824/pro/code
python /home/scc/pb23151824/pro/code/scripts/run_server_c.py --help >/dev/null
echo "[CHECK] run_server_c.py 可执行 OK"

echo "========== 预检全部通过，可提交步骤 2 =========="
```

**步骤 2 — 解压 + 检查 + conda + prep（作业脚本框整段粘贴）：**

```bash
#!/bin/bash
#SBATCH --job-name=tfne-prep-cpu
#SBATCH --partition=Students
#SBATCH --qos=qos_stu_long
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=72:00:00
# 日志路径由网页表单填写 logs/prep_%j.out / logs/prep_%j.err，此处不要写 --output/--error

set -euo pipefail
echo "========== prep 作业开始 $(date -Iseconds) host=$(hostname) =========="

# --- 上传后检查：压缩包 ---
test -f /home/scc/pb23151824/data.tar.gz || { echo "ERROR: 未找到 data.tar.gz"; exit 1; }
test -f /home/scc/pb23151824/code.tar.gz || { echo "ERROR: 未找到 code.tar.gz"; exit 1; }
echo "[CHECK] tar.gz 存在 OK"

# --- 解压（若 §三 已解压可跳过；保留也无妨）---
if [ ! -f /home/scc/pb23151824/pro/code/scripts/run_server_c.py ]; then
  mkdir -p /home/scc/pb23151824/pro
  cd /home/scc/pb23151824/pro
  tar -xzf /home/scc/pb23151824/data.tar.gz
  tar -xzf /home/scc/pb23151824/code.tar.gz
else
  echo "[CHECK] pro/code 已存在，跳过解压"
fi

# --- 解压后检查 ---
test -d /home/scc/pb23151824/pro/data || { echo "ERROR: 缺少 pro/data"; exit 1; }
test -f /home/scc/pb23151824/pro/code/scripts/run_server_c.py || { echo "ERROR: 缺少 run_server_c.py"; exit 1; }
test -f /home/scc/pb23151824/pro/code/config.server_c_72h.yaml || { echo "ERROR: 缺少 config.server_c_72h.yaml"; exit 1; }
test -f /home/scc/pb23151824/pro/code/requirements.txt || { echo "ERROR: 缺少 requirements.txt"; exit 1; }
ls /home/scc/pb23151824/pro/data/*.csv | head
echo "[CHECK] 解压后目录 OK"

# --- conda（§三 已装 Miniforge 后，此处仅 activate）---
cd /home/scc/pb23151824/pro/code
mkdir -p /home/scc/pb23151824/pro/code/logs

set +u
if [ -f /home/scc/pb23151824/miniforge3/etc/profile.d/conda.sh ]; then
  source /home/scc/pb23151824/miniforge3/etc/profile.d/conda.sh
  echo "[CHECK] conda: miniforge3 OK"
elif [ -f /home/scc/pb23151824/anaconda3/etc/profile.d/conda.sh ]; then
  source /home/scc/pb23151824/anaconda3/etc/profile.d/conda.sh
  echo "[CHECK] conda: anaconda3 OK"
else
  echo "ERROR: 未找到 miniforge3 或 anaconda3，请先完成 §三"; exit 1
fi
conda activate ai25
set -u
echo "[CHECK] python=$(which python) $(python -V)"

pip install -r /home/scc/pb23151824/pro/code/requirements.txt -q
python -c "import pandas, pyarrow; print('[CHECK] deps OK: pandas+pyarrow')"

# --- prep ---
echo "[CHECK] 开始 prep ..."
python /home/scc/pb23151824/pro/code/scripts/run_server_c.py prep --config config.server_c_72h.yaml

echo "========== prep done $(date -Iseconds) =========="
ls -lh /home/scc/pb23151824/pro/code/artifacts/panel_daily.parquet
ls -lh /home/scc/pb23151824/pro/code/artifacts/features/panel_features.parquet
ls -lh /home/scc/pb23151824/pro/code/artifacts/news/panel_news.parquet
echo "[CHECK] prep 产物 OK"
```

步骤 2 提交后查看日志：`tail -f /home/scc/pb23151824/pro/code/logs/prep_*.out`，应依次出现 `[CHECK] tar.gz` → `[CHECK] 解压后目录` → `[CHECK] conda` → `[CHECK] deps` → `[CHECK] prep 产物`。

**步骤 3 — prep 完成后核对（作业结束后在 Shell 执行，不提交新作业）：**

步骤 2 跑的就是下面这条命令（已在脚本内，**无需再手动执行**）：

```bash
python /home/scc/pb23151824/pro/code/scripts/run_server_c.py prep --config config.server_c_72h.yaml
```

作业结束后，用下面命令确认 **prep 成功**（三个 parquet 均存在且体积合理）：

```bash
ls -lh /home/scc/pb23151824/pro/code/artifacts/panel_daily.parquet
ls -lh /home/scc/pb23151824/pro/code/artifacts/features/panel_features.parquet
ls -lh /home/scc/pb23151824/pro/code/artifacts/features/feature_meta.json
ls -lh /home/scc/pb23151824/pro/code/artifacts/news/panel_news.parquet
```

预期产物绝对路径：

```text
/home/scc/pb23151824/pro/code/artifacts/panel_daily.parquet
/home/scc/pb23151824/pro/code/artifacts/features/panel_features.parquet
/home/scc/pb23151824/pro/code/artifacts/features/feature_meta.json
/home/scc/pb23151824/pro/code/artifacts/news/panel_news.parquet
```

耗时粗估 **12–24 小时**。日志末尾出现 `[CHECK] prep 产物 OK` 且上述文件存在 → 进入 **§六** 提交 GPU train + eval。详见 `C:\Users\Lenovo\Desktop\deep learning\pro\code\docs\SERVER_72H_PLAN.md`。

### 5.2 表单字段

| 字段 | 填写值 |
|------|--------|
| 作业名 | `tfne-prep-cpu` |
| 用户 | `pb23151824` |
| 分区 | `Students` |
| QOS | `qos_stu_long` |
| 节点数 | `1` |
| **GPU 卡数** | **`0`**（prep 纯 CPU，必为 0） |
| CPU 核数 | `16` |
| 内存 | `64G` |
| 最长运行时间 | `4320`（72 小时，分钟）或填 `72`（若单位为小时） |
| 工作目录 | `/home/scc/pb23151824/pro/code` |
| 标准输出 | `logs/prep_%j.out`（**相对路径**，相对工作目录） |
| 错误输出 | `logs/prep_%j.err` |

**SCOW 网页提交注意：** 命令框里**不要**写 `#SBATCH --output=` / `#SBATCH --error=`，用表单填 `logs/prep_%j.out`；否则可能报 `Invalid directive found in batch script`。

保存作业提交文件可填：`tfne-prep-cpu.sh`。

---

## 六、prep 成功后 → GPU train + eval

```bash
cd /home/scc/pb23151824/pro/code
source /home/scc/pb23151824/miniforge3/etc/profile.d/conda.sh
conda activate ai25
pip install -r /home/scc/pb23151824/pro/code/requirements-nlp.txt
export HF_ENDPOINT=https://hf-mirror.com
export DISABLE_SAFETENSORS_CONVERSION=true

python /home/scc/pb23151824/pro/code/scripts/run_server_c.py train --config config.server_c_72h.yaml
python /home/scc/pb23151824/pro/code/scripts/run_server_c.py eval --config config.server_c_72h.yaml
```

GPU 作业脚本：

```bash
sbatch /home/scc/pb23151824/pro/code/scripts/slurm_train_gpu.sbatch
```

---

## 七、常见问题

| 问题 | 处理 |
|------|------|
| `ERROR: 未找到 miniforge3 或 anaconda3` | **先完成 §三** 在登录 Shell 安装 Miniforge，再重新提交 |
| `Invalid directive ... prep_%j.out` | 命令框去掉 `#SBATCH --output/error`；表单用 `logs/prep_%j.out` |
| 找不到 CSV | 检查 `/home/scc/pb23151824/pro/data` 是否已解压 |
| conda 报错 | 登录 Shell 重跑 §3.8 验证；确认 `source` 路径为 `/home/scc/pb23151824/miniforge3/...` |
| 预检通过但 prep 失败 | 看 `prep_*.err`；§5.1 步骤 1.5 只验环境，不验全量数据耗时 |
| 误占 GPU | 表单 GPU=0，命令框 `#SBATCH --gres=gpu:0` |
| 运行时间不足 | 表单最长运行时间改为 72h；或 `#SBATCH --time=72:00:00` |

---

## 八、与本机 today 预演

| | 本机 | 服务器 prep |
|--|------|-------------|
| 代码目录 | `C:\Users\Lenovo\Desktop\deep learning\pro\code` | `/home/scc/pb23151824/pro/code` |
| 配置 | `C:\Users\Lenovo\Desktop\deep learning\pro\code\config.local_preplay_today.yaml` | `/home/scc/pb23151824/pro/code/config.server_c_72h.yaml` |
| 产物目录 | `C:\Users\Lenovo\Desktop\deep learning\pro\code\artifacts\checkpoints\tier_c_smoke\` | `/home/scc/pb23151824/pro/code/artifacts/checkpoints/tier_c/` |
