#!/usr/bin/env bash
# Small quick-run wrapper: 3 (weak) -> 4 (train) -> 5a (prompts) -> 5b (infer)
# Defaults aim for fast sanity checks; override via env when needed.
#
# Env knobs:
#   ANN_DIR               annotated JSON dir (default: datasets/annotations/five)
#   TRAIN_GPUS            GPU list for training (default: auto-detect)
#   EPOCHS/LR/TRAIN_BATCH/…  training hyperparams (see steps/_gen_train_config.py)
#   LOG_DIR               output root (default: logs)
#   MODEL_BASE            base model (default: weights/base/llava-v1.5-7b)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"
cd "$REPO_ROOT"

# Fixed defaults for small-run (edit here to change)
ANN_DIR="${REPO_ROOT}/datasets/annotations/five"
LOG_DIR="${REPO_ROOT}/logs"
MODEL_BASE="weights/base/llava-v1.5-7b"
CKPT="${LOG_DIR}/steps/ckpt/llava-v1.5-7b-lora-[x2dfd]"
# Training hyperparams (fixed)
TRAIN_GPUS="0,1"
EPOCHS="1"
TRAIN_BATCH="2"
EVAL_BATCH="2"
GRAD_ACC="1"
LR="2e-4"
LORA_R="16"
LORA_ALPHA="32"
MM_PROJECTOR_LR="2e-5"
VISION_TOWER="weights/base/clip-vit-large-patch14-336"

echo "[small] Step3: weak merge"
ANN_DIR="$ANN_DIR" LOG_DIR="$LOG_DIR" bash steps/03_weak.sh

echo "[small] Step4: train"
LOG_DIR="$LOG_DIR" BASE_MODEL="$MODEL_BASE" CKPT="$CKPT" \
  TRAIN_GPUS="$TRAIN_GPUS" \
  EPOCHS="$EPOCHS" TRAIN_BATCH="$TRAIN_BATCH" EVAL_BATCH="$EVAL_BATCH" GRAD_ACC="$GRAD_ACC" \
  LR="$LR" LORA_R="$LORA_R" LORA_ALPHA="$LORA_ALPHA" MM_PROJECTOR_LR="$MM_PROJECTOR_LR" \
  VISION_TOWER="$VISION_TOWER" \
  bash steps/04_train.sh

echo "[small] Step5a: build prompts"
python steps/05a_build_prompts.py --config config/test_config.yaml

PROMPT_RUN_DIR=$(ls -1dt logs/steps/prompts/runs/prompt_* 2>/dev/null | head -n1 || true)
if [[ -z "$PROMPT_RUN_DIR" ]]; then
  echo "[small][ERROR] No prompted datasets found." >&2; exit 3
fi
DATASETS_DIR="$PROMPT_RUN_DIR/datasets"

echo "[small] Step5b: infer with CKPT=$CKPT"
python steps/05b_infer.py \
  --inputs "$DATASETS_DIR" \
  --model-path "$CKPT" \
  --model-base "$MODEL_BASE"

echo "[small] done. CKPT: $CKPT"
