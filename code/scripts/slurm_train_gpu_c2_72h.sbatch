#!/bin/bash
# TFNE-C2 · GPU train + eval（72h）
# 文档: ../设想.md

#SBATCH --job-name=c2-gpu
#SBATCH --partition=Student
#SBATCH --qos=qos_stu001
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=/home/scc/pb23151824/pro/code/logs/train_c2_%j.out
#SBATCH --error=/home/scc/pb23151824/pro/code/logs/train_c2_%j.err

set -euo pipefail

CODE_DIR="/home/scc/pb23151824/pro/code"
CONDA_SH="/home/scc/pb23151824/miniforge3/etc/profile.d/conda.sh"
FEAT="${CODE_DIR}/artifacts/features/panel_features.parquet"
CFG="${CODE_DIR}/config.server_c2_72h.yaml"

cd "${CODE_DIR}"
mkdir -p "${CODE_DIR}/logs"

echo "=== TFNE-C2 GPU train+eval === host=$(hostname) date=$(date -Iseconds)"

set +u
source "${CONDA_SH}"
conda activate ai25
set -u

export HF_ENDPOINT=https://hf-mirror.com
export DISABLE_SAFETENSORS_CONVERSION=true
pip install -r "${CODE_DIR}/requirements-nlp.txt" -q

test -f "${FEAT}" || { echo "ERROR: 未找到 ${FEAT}，请先 sbatch slurm_prep_cpu_c2_72h.sbatch"; exit 1; }
test -f "${CFG}" || { echo "ERROR: 未找到 ${CFG}，请更新 code.tar.gz"; exit 1; }

python "${CODE_DIR}/scripts/check_gpu.py"

python "${CODE_DIR}/scripts/run_server_c2_72h.py" train
python "${CODE_DIR}/scripts/run_server_c2_72h.py" eval

echo "[CHECK] train+eval OK $(date -Iseconds)"
