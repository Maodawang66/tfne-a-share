#!/bin/bash
# 服务器登录节点一次性配置 ai25（prep + GPU train + NLP 全项目）
# 用法：bash /home/scc/pb23151824/pro/code/scripts/setup_server_ai25.sh
# 文档：/home/scc/pb23151824/pro/code/docs/USTC_SLURM_PREP.md §三

set -euo pipefail

HOME_DIR="/home/scc/pb23151824"
MINIFORGE="${HOME_DIR}/miniforge3"
CODE_DIR="${HOME_DIR}/pro/code"
ENV_NAME="ai25"

echo "========== TFNE 服务器全项目环境配置 =========="

# --- Miniforge ---
if [ ! -f "${MINIFORGE}/etc/profile.d/conda.sh" ]; then
  echo "[1/6] 安装 Miniforge ..."
  cd "${HOME_DIR}"
  wget -q https://mirrors.ustc.edu.cn/github-release/conda-forge/miniforge/LatestRelease/Miniforge3-Linux-x86_64.sh \
    -O Miniforge3-Linux-x86_64.sh
  bash Miniforge3-Linux-x86_64.sh -b -p "${MINIFORGE}"
else
  echo "[1/6] Miniforge 已存在，跳过"
fi

set +u
# shellcheck source=/dev/null
source "${MINIFORGE}/etc/profile.d/conda.sh"
set -u
conda -V

# --- conda env ai25 ---
if ! conda env list | grep -qE "^${ENV_NAME}[[:space:]]"; then
  echo "[2/6] 创建 conda 环境 ${ENV_NAME} (python=3.10) ..."
  mamba create -n "${ENV_NAME}" python=3.10 -y
else
  echo "[2/6] 环境 ${ENV_NAME} 已存在"
fi

conda activate "${ENV_NAME}"
python -V

# --- 解压 code（若需要）---
if [ ! -f "${CODE_DIR}/requirements.txt" ]; then
  echo "[3/6] 解压 code.tar.gz ..."
  mkdir -p "${HOME_DIR}/pro"
  cd "${HOME_DIR}/pro"
  tar -xzf "${HOME_DIR}/code.tar.gz"
else
  echo "[3/6] pro/code 已存在，跳过解压"
fi

test -f "${CODE_DIR}/requirements.txt" || { echo "ERROR: 未找到 ${CODE_DIR}/requirements.txt"; exit 1; }

# --- PyTorch GPU + 项目依赖 ---
echo "[4/6] 安装 PyTorch (CUDA 12.1) + requirements ..."
if python -c "import torch; exit(0 if torch.version.cuda else 1)" 2>/dev/null; then
  echo "  torch CUDA 版已存在"
else
  pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
  mamba install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia \
    || pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
fi

pip install -r "${CODE_DIR}/requirements.txt"
pip install -r "${CODE_DIR}/requirements-nlp.txt"

# --- 工具 ---
echo "[5/6] 安装 nvitop / git ..."
mamba install -y nvitop git 2>/dev/null || conda install -y nvitop git 2>/dev/null || true

# --- HF 镜像（写入 ~/.bashrc，避免重复）---
MARK="# TFNE HF mirror"
BASHRC="${HOME_DIR}/.bashrc"
if ! grep -qF "${MARK}" "${BASHRC}" 2>/dev/null; then
  cat >> "${BASHRC}" <<'EOF'

# TFNE HF mirror
export HF_ENDPOINT=https://hf-mirror.com
export DISABLE_SAFETENSORS_CONVERSION=true
EOF
  echo "  已写入 ~/.bashrc: HF_ENDPOINT"
fi
export HF_ENDPOINT=https://hf-mirror.com
export DISABLE_SAFETENSORS_CONVERSION=true

# --- 验证 ---
echo "[6/6] 验证 import ..."
python -c "
import pandas, pyarrow, torch, transformers, peft
print('pandas/pyarrow/torch/transformers/peft OK')
print('torch', torch.__version__, 'cuda built:', torch.version.cuda)
print('cuda available (login node may be False):', torch.cuda.is_available())
"
python "${CODE_DIR}/scripts/validate_server_env.py"

echo ""
echo "========== 全项目环境配置完成 =========="
echo "  activate: source ${MINIFORGE}/etc/profile.d/conda.sh && conda activate ${ENV_NAME}"
echo "  下一步: §5.1 提交 prep 作业"
echo "  GPU 训练前可在 GPU 作业/节点运行: python ${CODE_DIR}/scripts/check_gpu.py"
