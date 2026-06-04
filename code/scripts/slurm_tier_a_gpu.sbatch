#!/bin/bash
# 档位 A 基线 GPU 作业
# 本机: C:\Users\Lenovo\Desktop\deep learning\pro
# 服务器代码目录: /home/scc/pb23151824/pro/code
# 文档: C:\Users\Lenovo\Desktop\deep learning\pro\code\docs\USTC_SUBMIT_SIMPLE.md

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

CODE_DIR="/home/scc/pb23151824/pro/code"
DATA_DIR="/home/scc/pb23151824/pro/data"
CONDA_SH="/home/scc/pb23151824/miniforge3/etc/profile.d/conda.sh"

cd "${CODE_DIR}"
mkdir -p /home/scc/pb23151824/pro/code/logs

echo "=== Tier A GPU job ==="
echo "host=$(hostname) start=$(date -Iseconds)"
echo "CODE_DIR=${CODE_DIR}"

set +u
source "${CONDA_SH}"
conda activate ai25
set -u

test -d "${DATA_DIR}" || { echo "ERROR: 未找到 ${DATA_DIR}"; exit 1; }

pip install -r /home/scc/pb23151824/pro/code/requirements.txt -q
python /home/scc/pb23151824/pro/code/scripts/run_tier_a_local.py

echo "=== done $(date -Iseconds) ==="
ls -lh /home/scc/pb23151824/pro/code/artifacts/checkpoints/tier_a/gru/seed_42/best.pt
ls -lh /home/scc/pb23151824/pro/code/artifacts/predictions/tier_a/ensemble_scores.parquet
