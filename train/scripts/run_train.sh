#!/usr/bin/env bash
# Launch the LoRA training stage via the new unified pipeline.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

if command -v conda >/dev/null 2>&1; then
  source ~/.bashrc 2>/dev/null || true
  eval "$(conda shell.bash hook)" 2>/dev/null || true
  conda activate llava 2>/dev/null || true
fi

python -m train.pipeline --config train/configs/config.yaml --run-train "$@"
