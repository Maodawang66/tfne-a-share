#!/bin/bash
# 复制本文件「命令框部分」到 SCOW 网页「提交作业 → 命令」框。
# 表单字段见 ../设想.md §6.5；日志路径必须在表单填 logs/prep_c2_%j.out，不要写在本文件。

#SBATCH --job-name=c2-prep
#SBATCH --partition=Students
#SBATCH --qos=qos_stu_long
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=72:00:00

set -euo pipefail
echo "========== C2 prep START $(date -Iseconds) host=$(hostname) =========="

CODE_DIR="/home/scc/pb23151824/pro/code"
DATA_DIR="/home/scc/pb23151824/pro/data"

mkdir -p "${CODE_DIR}/logs"
cd "${CODE_DIR}"

set +u
if [ -f /home/scc/pb23151824/miniforge3/etc/profile.d/conda.sh ]; then
  source /home/scc/pb23151824/miniforge3/etc/profile.d/conda.sh
elif [ -f /home/scc/pb23151824/anaconda3/etc/profile.d/conda.sh ]; then
  source /home/scc/pb23151824/anaconda3/etc/profile.d/conda.sh
else
  echo "ERROR: 未找到 conda，请先安装 miniforge3"; exit 1
fi
conda activate ai25
set -u

test -d "${DATA_DIR}" || { echo "ERROR: 未找到 ${DATA_DIR}"; exit 1; }
test -f "${CODE_DIR}/config.server_c2_72h.yaml" || { echo "ERROR: 缺少 config.server_c2_72h.yaml"; exit 1; }

pip install -r "${CODE_DIR}/requirements.txt" -q

python "${CODE_DIR}/scripts/run_server_c2_72h.py" prep

ls -lh "${CODE_DIR}/artifacts/panel_daily.parquet"
ls -lh "${CODE_DIR}/artifacts/features/panel_features.parquet"
ls -lh "${CODE_DIR}/artifacts/news/panel_news.parquet"

python -c "
import pandas as pd
p=pd.read_parquet('${CODE_DIR}/artifacts/panel_daily.parquet', columns=['trade_date','ts_code'])
print('20240701 股票数:', p[p.trade_date=='20240701'].ts_code.nunique())
"

echo "[CHECK] prep OK $(date -Iseconds)"
