#!/bin/bash
# 复制到 SCOW 网页命令框（prep 成功后）。表单日志：logs/train_c2_%j.out / .err

#SBATCH --job-name=c2-gpu
#SBATCH --partition=Students
#SBATCH --qos=qos_stu_long
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=72:00:00

set -euo pipefail
echo "========== C2 GPU START $(date -Iseconds) host=$(hostname) =========="

CODE_DIR="/home/scc/pb23151824/pro/code"
FEAT="${CODE_DIR}/artifacts/features/panel_features.parquet"

mkdir -p "${CODE_DIR}/logs"
cd "${CODE_DIR}"

set +u
source /home/scc/pb23151824/miniforge3/etc/profile.d/conda.sh
conda activate ai25
set -u

export HF_ENDPOINT=https://hf-mirror.com
export DISABLE_SAFETENSORS_CONVERSION=true
pip install -r "${CODE_DIR}/requirements-nlp.txt" -q

test -f "${FEAT}" || { echo "ERROR: 先完成 prep"; exit 1; }

python "${CODE_DIR}/scripts/check_gpu.py"
python "${CODE_DIR}/scripts/run_server_c2_72h.py" train
python "${CODE_DIR}/scripts/run_server_c2_72h.py" eval

echo "[CHECK] train+eval OK $(date -Iseconds)"
