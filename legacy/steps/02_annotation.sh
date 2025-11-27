#!/usr/bin/env bash
# Step 2: Annotation (base model QA-style explanations per dataset JSON)
# Thin wrapper around train.pipeline annotate stage.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}" )/.." && pwd)"
CONFIG="${CONFIG:-${REPO_ROOT}/config/train_config.yaml}"

# Optional conda activation if requested
if [[ "${RUN_WITH_CONDA:-0}" == "1" ]] && command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV:-llava}" 2>/dev/null || true
fi

echo "[step2] annotating using config: $CONFIG"
python -m train.pipeline --config "$CONFIG" --phase annotate ${FORCE:+--force-annotate}

echo "[step2] done. Outputs under train/outputs/pipeline/runs/pipeline_*"
