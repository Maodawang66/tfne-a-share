#!/bin/bash
# 修复 Windows 上传导致的 CRLF 换行（sbatch 报错 DOS line breaks 时运行）
# 用法：bash scripts/fix_crlf.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

fix() {
  if sed -i 's/\r$//' "$1" 2>/dev/null; then
    echo "fixed: $1"
  else
    # macOS sed 备用
    sed -i '' 's/\r$//' "$1" && echo "fixed: $1"
  fi
}

for f in scripts/*.sbatch scripts/*.sh; do
  [ -f "$f" ] && fix "$f"
done

echo "Done. Retry: sbatch scripts/slurm_prep_cpu_tfne_72h.sbatch"
