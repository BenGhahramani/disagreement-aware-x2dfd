#!/usr/bin/env bash
# Orchestrate two environments and run X2DFD train/test.
# 1) cd into SABRE-AT-transfer and activate 'adv' (if available)
# 2) cd into X2DFD and activate 'llava' (fallback to X2DFD), then run train.sh and test.sh

set -euo pipefail

echo "==> Step 1: Entering /data/250010183/adv/new/SABRE-AT-transfer and activating 'adv'"
ADV_DIR="/data/250010183/adv/new/SABRE-AT-transfer"
if [[ -d "$ADV_DIR" ]]; then
  cd "$ADV_DIR"
else
  echo "[WARN] Directory not found: $ADV_DIR (continuing)"
fi

if command -v conda >/dev/null 2>&1; then
  # Load bashrc and initialize conda
  source ~/.bashrc || true
  conda init || true

  echo "Activating conda env: adv"
  eval "$(conda shell.bash hook)"
  source activate adv || conda activate adv || true
  echo "Current env: ${CONDA_DEFAULT_ENV:-system}"
else
  echo "[WARN] conda not detected; using system Python"
fi

echo "==> Step 2: Entering X2DFD and activating 'llava' (fallback to 'X2DFD')"
X2DFD_DIR="/data/250010183/workspace/X2DFD"
cd "$X2DFD_DIR"

if command -v conda >/dev/null 2>&1; then
  source ~/.bashrc || true
  conda init || true

  echo "Activating conda env: llava (or fallback to X2DFD)"
  eval "$(conda shell.bash hook)"
  # Try llava first, then fallback to X2DFD
  source activate llava || conda activate llava || source activate X2DFD || conda activate X2DFD || true
  echo "Current env: ${CONDA_DEFAULT_ENV:-system}"
else
  echo "[WARN] conda not detected; using system Python"
fi

echo "==> Step 3: Running X2DFD training pipeline"
bash ./train.sh

echo "==> Step 4: Running X2DFD quick test"
bash ./test.sh

echo "All done."

