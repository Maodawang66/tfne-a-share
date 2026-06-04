#!/bin/bash
# =============================================================================
# TFNE-C2 · 仅 eval — SCOW 网页「命令」框用
# ⚠️ QOS：表单与下面 #SBATCH 必须一致，推荐 qos_stu_long（480 分钟 GPU 作业）
# ⚠️ 不要在命令里写 #SBATCH --output/--error（只在表单填 logs/eval_c2_%j.out）
# =============================================================================

#SBATCH --job-name=c2-eval
#SBATCH --partition=Students
#SBATCH --qos=qos_stu_long
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00

set -euo pipefail
echo "========== C2 EVAL ONLY START $(date -Iseconds) host=$(hostname) =========="

CODE_DIR="/home/scc/pb23151824/pro/code"
FEAT="${CODE_DIR}/artifacts/features/panel_features.parquet"
CKPT="${CODE_DIR}/artifacts/checkpoints/tier_c2/tft/seed_42/best.pt"
CFG="${CODE_DIR}/config.server_c2_72h.yaml"
FIX="${CODE_DIR}/src/checkpoint_utils.py"
SCRIPT04="${CODE_DIR}/scripts/04_ensemble_predict.py"
RUNNER="${CODE_DIR}/scripts/run_server_c2_72h.py"

mkdir -p "${CODE_DIR}/logs"
cd "${CODE_DIR}"

set +u
source /home/scc/pb23151824/miniforge3/etc/profile.d/conda.sh
conda activate ai25
set -u

export HF_ENDPOINT=https://hf-mirror.com
export DISABLE_SAFETENSORS_CONVERSION=true
pip install -r "${CODE_DIR}/requirements-nlp.txt" -q

# ---------- 新版代码检查 ----------
test -f "${FIX}" || { echo "ERROR: 缺少 ${FIX}"; exit 1; }
test -f "${SCRIPT04}" || { echo "ERROR: 缺少 ${SCRIPT04}"; exit 1; }
test -f "${RUNNER}" || { echo "ERROR: 缺少 ${RUNNER}"; exit 1; }

grep -q "ind_vocab_resolve_v2" "${FIX}" || {
  echo "ERROR: checkpoint_utils.py 未更新（无 ind_vocab_resolve_v2）"; exit 1; }
grep -q "load_seq_model_from_checkpoint" "${SCRIPT04}" || {
  echo "ERROR: 04_ensemble_predict.py 仍是旧版（无 load_seq_model_from_checkpoint）"; exit 1; }
grep -q "verify_ind_vocab_patch_deployed" "${SCRIPT04}" || {
  echo "ERROR: 04_ensemble_predict.py 仍是旧版（无 verify_ind_vocab_patch_deployed）"; exit 1; }
grep -q "verify_ind_vocab_patch_deployed" "${RUNNER}" || {
  echo "ERROR: run_server_c2_72h.py 未更新"; exit 1; }

echo "[CHECK] ind_vocab patch v2 代码 OK"

test -f "${FEAT}" || { echo "ERROR: 未找到 ${FEAT}，请先 prep 或确认产物路径"; exit 1; }
test -f "${CKPT}" || { echo "ERROR: 未找到 ${CKPT}，不能 eval-only"; exit 1; }
test -f "${CFG}" || { echo "ERROR: 请同步最新 code"; exit 1; }

python "${CODE_DIR}/scripts/check_gpu.py"
python "${CODE_DIR}/scripts/run_server_c2_72h.py" eval

echo "[CHECK] eval OK $(date -Iseconds)"
