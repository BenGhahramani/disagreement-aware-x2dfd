#!/usr/bin/env bash
# Step 3: Weak Feature Supplementing (multi-expert score merge)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}" )/.." && pwd)"
INFER_CFG="${INFER_CFG:-${REPO_ROOT}/config/test_config.yaml}"
ANN_DIR="${ANN_DIR:-${REPO_ROOT}/train/outputs/pipeline/runs/pipeline_*/train/datasets}"
EXPERTS="${EXPERTS:-blending,diffusion_detector}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"
OUT_WEAK_JSON="${OUT_WEAK_JSON:-${LOG_DIR}/steps/train_weak_blending+diffdet.json}"

if [[ "$ANN_DIR" == *"*"* ]]; then
  # Resolve newest matching pipeline run if wildcard is used
  latest_run=$(ls -1dt ${REPO_ROOT}/train/outputs/pipeline/runs/pipeline_* 2>/dev/null | head -n1 || true)
  if [[ -n "$latest_run" ]]; then
    ANN_DIR="$latest_run/train/datasets"
  fi
fi

if ! compgen -G "${ANN_DIR}/*_annotations.json" >/dev/null; then
  echo "[step3][ERROR] No *_annotations.json under ${ANN_DIR}. Run step2 first or set ANN_DIR." >&2
  exit 2
fi

echo "[step3] weak-merge from: $ANN_DIR"
python -m train.pipeline \
  --infer-config "$INFER_CFG" \
  --phase weak \
  --skip-annotate \
  --reuse-datasets-dir "$ANN_DIR" \
  --experts "$EXPERTS" \
  --out-weak-json "$OUT_WEAK_JSON"

echo "[step3] wrote: $OUT_WEAK_JSON"
