#!/bin/bash
# TFNE-C2 · CPU prep（Top3000）
# 文档: ../设想.md

#SBATCH --job-name=c2-prep
#SBATCH --partition=Student
#SBATCH --qos=qos_stu001
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/home/scc/pb23151824/pro/code/logs/prep_c2_%j.out
#SBATCH --error=/home/scc/pb23151824/pro/code/logs/prep_c2_%j.err

set -euo pipefail

CODE_DIR="/home/scc/pb23151824/pro/code"
DATA_DIR="/home/scc/pb23151824/pro/data"
CONDA_SH="/home/scc/pb23151824/miniforge3/etc/profile.d/conda.sh"

cd "${CODE_DIR}"
mkdir -p "${CODE_DIR}/logs"

echo "=== TFNE-C2 prep === host=$(hostname) date=$(date -Iseconds)"

set +u
source "${CONDA_SH}"
conda activate ai25
set -u

test -d "${DATA_DIR}" || { echo "ERROR: 未找到 ${DATA_DIR}"; exit 1; }

python "${CODE_DIR}/scripts/run_server_c2_72h.py" prep

echo "=== prep 产物 ==="
ls -lh "${CODE_DIR}/artifacts/panel_daily.parquet"
ls -lh "${CODE_DIR}/artifacts/features/panel_features.parquet"
ls -lh "${CODE_DIR}/artifacts/news/panel_news.parquet"

python -c "
import pandas as pd
p=pd.read_parquet('${CODE_DIR}/artifacts/panel_daily.parquet', columns=['trade_date','ts_code'])
n=p[p.trade_date=='20240701'].ts_code.nunique()
print('20240701 股票数:', n, '（C2 应约 3000）')
"

echo "[CHECK] prep OK $(date -Iseconds)"
