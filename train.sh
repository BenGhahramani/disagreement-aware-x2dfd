#!/usr/bin/env bash
# Entry script to launch the full training pipeline.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"
cd "$REPO_ROOT"

python -m train.pipeline --config train/configs/config.yaml --run-train "$@"
