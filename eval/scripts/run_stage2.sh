#!/bin/bash
# 小批量 图像注释生成 测试脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$SCRIPT_DIR"

echo "=== 小批量 图像注释生成（测试） ==="
echo "工作目录: $SCRIPT_DIR"

if command -v conda >/dev/null 2>&1; then
  source ~/.bashrc || true
  conda init || true
  echo "激活conda环境: llava"
  eval "$(conda shell.bash hook)"
  source activate llava || conda activate llava || true
  echo "✅ 环境: ${CONDA_DEFAULT_ENV:-system}"
else
  echo "⚠️ 未检测到conda，使用系统Python"
fi
cd "$REPO_ROOT"
python eval/qa/run.py --config eval/configs/config.yaml
