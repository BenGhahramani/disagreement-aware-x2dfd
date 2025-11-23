#!/usr/bin/env bash
# Convenience wrapper to run the full training pipeline with default config.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

# Optional: activate conda env only if caller wants it (set RUN_ALL_WITH_CONDA=1)
if [[ "${RUN_ALL_WITH_CONDA:-0}" == "1" ]] && command -v conda >/dev/null 2>&1; then
  if [[ -r "$HOME/.bashrc" ]]; then
    source "$HOME/.bashrc" 2>/dev/null || true
  fi
  eval "$(conda shell.bash hook)" 2>/dev/null || true
  conda activate llava 2>/dev/null || true
fi

python -m train.pipeline --config train/configs/config.yaml "$@"
