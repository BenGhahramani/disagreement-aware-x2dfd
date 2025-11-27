#!/usr/bin/env bash
# Train pipeline (staged):
#   1) (optional) feature assess
#   2) explainable annotation (base model)
#   3) weak feature supply merge (multi-expert scores)
#   4) LoRA training

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"
cd "$REPO_ROOT"

# Optional conda activation when none is active
if [ -z "${CONDA_DEFAULT_ENV-}" ] && command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  TARGET_ENV="${TRAIN_CONDA_ENV:-X2DFD}"
  echo "[env] Activating conda env: ${TARGET_ENV}"
  conda activate "${TARGET_ENV}" 2>/dev/null || echo "[env][WARN] failed to activate ${TARGET_ENV}; using system Python"
fi

# Expert providers used at weak stage (comma-separated aliases/providers)
EXPERTS="${EXPERTS:-blending,diffusion_detector}"

# Auto-detect GPU list if not provided via CLI (--train-gpus) or env (GPUS)
# Priority: explicit CLI > $GPUS > $CUDA_VISIBLE_DEVICES > nvidia-smi count
NEED_TRAIN_GPUS_FLAG=true
for arg in "$@"; do
  if [[ "$arg" == "--train-gpus" ]]; then
    NEED_TRAIN_GPUS_FLAG=false
    break
  fi
done

EXTRA_ARGS=()
if $NEED_TRAIN_GPUS_FLAG; then
  if [[ -n "${GPUS:-}" ]]; then
    GPU_LIST="$GPUS"
  elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    GPU_LIST="$CUDA_VISIBLE_DEVICES"
  elif command -v nvidia-smi >/dev/null 2>&1; then
    COUNT="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
    if [[ "$COUNT" =~ ^[0-9]+$ ]] && [[ "$COUNT" -gt 0 ]]; then
      GPU_LIST="$(seq -s, 0 $((COUNT-1)))"
    else
      GPU_LIST="0"
    fi
  else
    GPU_LIST="0"
  fi
  EXTRA_ARGS+=(--train-gpus "$GPU_LIST")
fi

echo "[train] Stage 2: Explainable annotation -> Stage 3: Weak merge -> Stage 4: LoRA training"
python -m train.pipeline \
  --config train/configs/config.yaml \
  --phase train \
  --experts "$EXPERTS" \
  --train-with-scores \
  --run-train \
  "${EXTRA_ARGS[@]}" \
  "$@"

echo "[train] Done. Check train/outputs/pipeline/latest_run.json for summary."
