#!/usr/bin/env bash
# Unified run-all: weak -> train -> eval, using pre-annotated datasets.
# - Strict跳过 annotate：若 ANN_DIR 下无 *_annotations.json 则报错退出。
# - 训练使用 quick_train.sh；评测使用 config/test_config.yaml。
# - 支持仅评测：RUNALL_ONLY_EVAL=1

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"
cd "$REPO_ROOT"

# Optional conda activation when none is active
if [ -z "${CONDA_DEFAULT_ENV-}" ] && command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  TARGET_ENV="${RUNALL_CONDA_ENV:-llava}"
  echo "[env] Activating conda env: ${TARGET_ENV}"
  conda activate "${TARGET_ENV}" 2>/dev/null || echo "[env][WARN] failed to activate ${TARGET_ENV}; using system Python"
fi

# Defaults
ANN_DIR="${ANN_DIR:-${REPO_ROOT}/datasets/annotations/five}"
EXPERTS="${EXPERTS:-blending,diffusion_detector}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"
CKPT="${CKPT:-${LOG_DIR}/steps/ckpt/llava-v1.5-7b-lora-[x2dfd]}"
WEAK_JSON="${WEAK_JSON:-${LOG_DIR}/steps/train_weak_blending+diffdet.json}"
BASE_MODEL="${BASE_MODEL:-weights/base/llava-v1.5-7b}"
INFER_CFG="${INFER_CFG:-config/test_config.yaml}"

# Dry-run: print commands and exit
if [[ "${RUNALL_DRY:-0}" == "1" ]]; then
  set -x
  echo python -m train.pipeline --infer-config "$INFER_CFG" --phase weak --skip-annotate --reuse-datasets-dir "$ANN_DIR" --experts "$EXPERTS" --out-weak-json "$WEAK_JSON"
  echo bash "$REPO_ROOT/quick_train.sh" "$WEAK_JSON" '# OUTPUT_DIR='$CKPT
  echo python -m eval.infer.runner --config "$INFER_CFG" --experts "$EXPERTS" --model-path "$CKPT" --model-base "$BASE_MODEL"
  exit 0
fi

# Only eval short-circuit
if [[ "${RUNALL_ONLY_EVAL:-0}" == "1" ]]; then
  echo "[run_all] ONLY_EVAL=1 — skipping weak/train; running eval with CKPT=$CKPT"
  python -m eval.infer.runner \
    --config "$INFER_CFG" \
    --experts "$EXPERTS" \
    --model-path "$CKPT" \
    --model-base "$BASE_MODEL" \
    "$@"
  exit 0
fi

# Strict: require annotated JSONs under ANN_DIR
if ! compgen -G "${ANN_DIR}/*_annotations.json" >/dev/null; then
  echo "[run_all][ERROR] No *_annotations.json under ${ANN_DIR}. Provide annotated datasets and retry." >&2
  exit 2
fi

# Weak merge (multi-expert)
python -m train.pipeline \
  --infer-config "$INFER_CFG" \
  --phase weak \
  --skip-annotate \
  --reuse-datasets-dir "$ANN_DIR" \
  --experts "$EXPERTS" \
  --out-weak-json "$WEAK_JSON"

# Train via quick_train.sh (skip if CKPT exists)
if [[ -d "$CKPT" ]]; then
  echo "[run_all] ckpt exists: $CKPT — skipping training"
else
  # Auto-detect GPU list
  TRAIN_GPUS="${RUNALL_TRAIN_GPUS:-}"
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
  echo "[run_all] Training on GPU(s): $TRAIN_GPUS (quick_train.sh)"
  GPUS="$TRAIN_GPUS" \
  BASE_MODEL="$BASE_MODEL" \
  VISION_TOWER="${VISION_TOWER:-weights/base/clip-vit-large-patch14-336}" \
  OUTPUT_DIR="$CKPT" \
  bash "$REPO_ROOT/quick_train.sh" "$WEAK_JSON"
fi

# Eval
python -m eval.infer.runner \
  --config "$INFER_CFG" \
  --experts "$EXPERTS" \
  --model-path "$CKPT" \
  --model-base "$BASE_MODEL" \
  "$@"
