# 服务器提交傻瓜版（档位 A 基线 · GPU）

| 环境 | 项目根目录 | 数据目录 | 代码目录 |
|------|------------|----------|----------|
| 本机 Windows | `C:\Users\Lenovo\Desktop\deep learning\pro` | `C:\Users\Lenovo\Desktop\deep learning\pro\data` | `C:\Users\Lenovo\Desktop\deep learning\pro\code` |
| 服务器 Linux | `/home/scc/pb23151824/pro` | `/home/scc/pb23151824/pro/data` | `/home/scc/pb23151824/pro/code` |

参考：[USTC107 深度学习作业文档](https://xinchengo.github.io/ustc107/guides/ai/deep-learning-homework/) · 服务器环境配置见同文档 **§三** 或 `USTC_SLURM_PREP.md` **§三**。

---

## 一、本机已有脚本（无需新写）

| 本机绝对路径 | 作用 |
|--------------|------|
| `C:\Users\Lenovo\Desktop\deep learning\pro\code\scripts\run_tier_a_local.py` | 档位 A 全流程 |
| `C:\Users\Lenovo\Desktop\deep learning\pro\code\scripts\slurm_tier_a_gpu.sbatch` | Slurm 作业模板 |
| `C:\Users\Lenovo\Desktop\deep learning\pro\code\config.tier_a_local.yaml` | 档位 A 配置 |
| `C:\Users\Lenovo\Desktop\deep learning\pro\code\requirements.txt` | Python 依赖 |

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

解压、检查与训练见 **§3.2** 整段脚本。

---

## 三、图形界面提交（档位 A · GPU）

最长运行时间填 **480 分钟**（8 小时）。命令框内容优先于下方表单。

### 3.1 顶部

| 字段 | 填写值 |
|------|--------|
| 集群 | 高性能计算平台 |
| 作业名 | tfne-tier-a |

### 3.2 命令 / 脚本框（整段粘贴）

**步骤 1 — 上传前检查（本机 PowerShell，打包上传前执行）：**

```powershell
Test-Path "C:\Users\Lenovo\Desktop\deep learning\pro\data"
Test-Path "C:\Users\Lenovo\Desktop\deep learning\pro\code\scripts\run_tier_a_local.py"
Test-Path "C:\Users\Lenovo\Desktop\deep learning\pro\code\requirements.txt"
```

三项均应返回 `True`。完成 §2.1 打包、§2.2 上传后，在平台 Shell 或作业脚本框粘贴下方 **步骤 2**（含解压、检查、训练）。

**步骤 2 — 解压 + 检查 + 提交训练（服务器 bash，整段粘贴）：**

```bash
#!/bin/bash
#SBATCH --job-name=tfne-tier-a
#SBATCH --partition=Student
#SBATCH --qos=qos_stu001
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/home/scc/pb23151824/pro/code/logs/tier_a_%j.out
#SBATCH --error=/home/scc/pb23151824/pro/code/logs/tier_a_%j.err

set -euo pipefail

# --- 上传后检查：压缩包 ---
test -f /home/scc/pb23151824/data.tar.gz || { echo "ERROR: 未找到 data.tar.gz"; exit 1; }
test -f /home/scc/pb23151824/code.tar.gz || { echo "ERROR: 未找到 code.tar.gz"; exit 1; }

# --- 解压 ---
mkdir -p /home/scc/pb23151824/pro
cd /home/scc/pb23151824/pro
tar -xzvf /home/scc/pb23151824/data.tar.gz
tar -xzvf /home/scc/pb23151824/code.tar.gz

# --- 解压后检查 ---
test -d /home/scc/pb23151824/pro/data || { echo "ERROR: 缺少 pro/data"; exit 1; }
test -f /home/scc/pb23151824/pro/code/scripts/run_tier_a_local.py || { echo "ERROR: 缺少 run_tier_a_local.py"; exit 1; }
test -f /home/scc/pb23151824/pro/code/requirements.txt || { echo "ERROR: 缺少 requirements.txt"; exit 1; }
ls /home/scc/pb23151824/pro/data/*.csv | head
ls /home/scc/pb23151824/pro/code/scripts/run_tier_a_local.py
ls /home/scc/pb23151824/pro/code/requirements.txt

# 预期目录结构：
# /home/scc/pb23151824/pro/data/
# /home/scc/pb23151824/pro/code/scripts/
# /home/scc/pb23151824/pro/code/config.tier_a_local.yaml
# /home/scc/pb23151824/pro/code/requirements.txt
# /home/scc/pb23151824/pro/code/src/

# --- 训练 ---
cd /home/scc/pb23151824/pro/code
mkdir -p /home/scc/pb23151824/pro/code/logs

set +u
source /home/scc/pb23151824/miniforge3/etc/profile.d/conda.sh
conda activate ai25
set -u

pip install -r /home/scc/pb23151824/pro/code/requirements.txt -q
python /home/scc/pb23151824/pro/code/scripts/run_tier_a_local.py

echo "DONE $(date -Iseconds)"
ls -lh /home/scc/pb23151824/pro/code/artifacts/checkpoints/tier_a/gru/seed_42/best.pt
```

Conda 若在 Anaconda 下，将 `source` 一行改为：

```bash
source /home/scc/pb23151824/anaconda3/etc/profile.d/conda.sh
```

### 3.3 表单字段

| 字段 | 填写值 |
|------|--------|
| 用户 | pb23151824 |
| 队列 / 分区 | Student |
| QOS | qos_stu001 |
| 节点数 | 1 |
| 单节点 GPU 卡数 | 1 |
| 单节点 CPU 核数 | 16 |
| 内存 | 32G |
| 最长运行时间 | 480 |
| 工作目录 | /home/scc/pb23151824/pro/code |
| 标准输出 | /home/scc/pb23151824/pro/code/logs/tier_a_%j.out |
| 错误输出 | /home/scc/pb23151824/pro/code/logs/tier_a_%j.err |

勾选「保存并更新提交脚本」→ 提交。

---

## 四、提交后查看日志

```bash
squeue -u pb23151824
tail -f /home/scc/pb23151824/pro/code/logs/tier_a_*.out
```

---

## 五、成功标志（服务器绝对路径）

```text
/home/scc/pb23151824/pro/code/artifacts/checkpoints/tier_a/gru/seed_42/best.pt
/home/scc/pb23151824/pro/code/artifacts/checkpoints/tier_a/mlp/seed_42/best.pt
/home/scc/pb23151824/pro/code/artifacts/checkpoints/tier_a/news_hash/seed_42/best.pt
/home/scc/pb23151824/pro/code/artifacts/predictions/tier_a/ensemble_scores.parquet
/home/scc/pb23151824/pro/code/artifacts/orders/tier_a/orders_20241231.csv
```

---

## 六、命令行提交（可选）

```bash
mkdir -p /home/scc/pb23151824/pro/code/logs
cd /home/scc/pb23151824/pro/code
source /home/scc/pb23151824/miniforge3/etc/profile.d/conda.sh
conda activate ai25
pip install -r /home/scc/pb23151824/pro/code/requirements.txt
sbatch /home/scc/pb23151824/pro/code/scripts/slurm_tier_a_gpu.sbatch
```

---

## 七、与 CPU prep 文档的区别

| 文档 | 任务 | GPU |
|------|------|-----|
| 本文 | 档位 A 基线 | 使用 1 块 GPU |
| `C:\Users\Lenovo\Desktop\deep learning\pro\code\docs\USTC_SLURM_PREP.md` | 档位 C 数据 prep | 不使用 GPU |

---

## 八、本机直接运行（对照）

```powershell
cd "C:\Users\Lenovo\Desktop\deep learning\pro\code"
conda activate ai25
python "C:\Users\Lenovo\Desktop\deep learning\pro\code\scripts\run_tier_a_local.py"
```

---

## 九、常见问题

| 问题 | 处理 |
|------|------|
| 找不到数据 | 确认 `/home/scc/pb23151824/pro/data` 与 `/home/scc/pb23151824/pro/code` 均已解压 |
| CUDA OOM | 编辑 `C:\Users\Lenovo\Desktop\deep learning\pro\code\config.tier_a_local.yaml` 中 `batch_stocks: 2048`，同步到 `/home/scc/pb23151824/pro/code/config.tier_a_local.yaml` |
| 不需要 BERT | 不要安装 `requirements-nlp.txt` |
