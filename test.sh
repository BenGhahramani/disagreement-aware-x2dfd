#!/usr/bin/env bash
# Run LoRA inference on evaluation datasets and compute AUC automatically.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"
cd "$REPO_ROOT"

# Step 1: generate responses with fake-score turns
python -m eval.infer.runner --config eval/configs/infer_config.yaml "$@"

# Step 2: compute AUC from the latest inference run
python -m eval.tools.compute_auc
