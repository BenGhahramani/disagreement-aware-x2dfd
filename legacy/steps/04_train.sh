#!/usr/bin/env bash
# Step 4: LoRA Training (DeepSpeed)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}" )/.." && pwd)"
INFER_CFG="${INFER_CFG:-${REPO_ROOT}/config/test_config.yaml}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"
WEAK_JSON="${WEAK_JSON:-${LOG_DIR}/steps/train_weak_blending+diffdet.json}"
BASE_MODEL="${BASE_MODEL:-weights/base/llava-v1.5-7b}"
CKPT="${CKPT:-${LOG_DIR}/steps/ckpt/llava-v1.5-7b-lora-[x2dfd]}"

if [[ ! -f "$WEAK_JSON" ]]; then
  echo "[step4][ERROR] weak dataset JSON not found: $WEAK_JSON (run step3 first or set WEAK_JSON)" >&2
  exit 2
fi

if ! command -v deepspeed >/dev/null 2>&1; then
  echo "[step4][ERROR] deepspeed not found in PATH. Please install/activate it." >&2
  exit 3
fi

# Auto-detect GPU list unless provided
TRAIN_GPUS="${TRAIN_GPUS:-}"
if [[ -z "$TRAIN_GPUS" ]]; then
  if [[ -n "${GPUS:-}" ]]; then
    TRAIN_GPUS="$GPUS"
  elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    TRAIN_GPUS="$CUDA_VISIBLE_DEVICES"
  elif command -v nvidia-smi >/dev/null 2>&1; then
    COUNT="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
    if [[ "$COUNT" =~ ^[0-9]+$ ]] && [[ "$COUNT" -gt 0 ]]; then
      TRAIN_GPUS="$(seq -s, 0 $((COUNT-1)))"
    else
      TRAIN_GPUS="0"
    fi
  else
    TRAIN_GPUS="0"
  fi
fi

if [[ -d "$CKPT" ]] && [[ "${OVERWRITE_CKPT:-0}" == "1" ]]; then
  echo "[step4][WARN] Removing existing CKPT: $CKPT"
  rm -rf "$CKPT"
fi

if [[ -d "$CKPT" ]]; then
  echo "[step4] ckpt exists: $CKPT — skipping training"
else
  echo "[step4] Training on GPU(s): $TRAIN_GPUS (quick_train.sh)"
  # Train using quick_train.sh with WEAK_JSON as dataset and CKPT as output dir
  GPUS="$TRAIN_GPUS" \
  BASE_MODEL="$BASE_MODEL" \
  VISION_TOWER="${VISION_TOWER:-weights/base/clip-vit-large-patch14-336}" \
  OUTPUT_DIR="$CKPT" \
  bash "${REPO_ROOT}/quick_train.sh" "$WEAK_JSON"
fi

if [[ ! -d "$CKPT" ]] || [[ -z $(ls -A "$CKPT" 2>/dev/null) ]]; then
  echo "[step4][ERROR] training produced no files under: $CKPT" >&2
  exit 4
fi

echo "[step4] done. CKPT: $CKPT"
