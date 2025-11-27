#!/usr/bin/env bash
# Run LoRA inference on evaluation datasets.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"
cd "$REPO_ROOT"

# Generate responses with expert scores embedded in the prompt/turns
python -m eval.infer.runner --config eval/configs/infer_config.yaml "$@"
