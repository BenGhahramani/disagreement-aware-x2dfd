#!/usr/bin/env bash
# Small end-to-end run: reuse a tiny annotated set -> weak merge (two experts)
# -> LoRA train (deepspeed) -> LoRA eval on small config.
#
# Defaults chosen to be fast and reproducible. Overridable via env vars.
#
# Key env knobs (optional):
#   RUNALL_CONDA_ENV        conda env to activate when none is active (default: llava)
#   ANN_DIR                  annotated JSON dir (default: datasets/annotations/five)
#   INFER_CFG                eval config (default: eval/configs/infer_config.quick.yaml)
#   CKPT                     LoRA output dir (default: weights/checkpoints/ckpt/FR/llava-v1.5-7b-lora-[x2dfd-small])
#   RUNALL_TRAIN_GPUS       GPU list for training (comma list). If unset, auto-detect.
#   RUNALL_OVERWRITE_CKPT   when "1", remove CKPT dir before training.
#   RUNALL_DRY              when "1", only print commands and exit.
#   RUNALL_ONLY_EVAL        when "1", skip weak/train and only run eval

set -euo pipefail

echo "==> Small Run: weak -> train -> eval (tiny config)"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"
cd "$REPO_ROOT"

# 1) Environment activation (best-effort). If a conda env is already active, keep it.
if [ -z "${CONDA_DEFAULT_ENV-}" ] && command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  TARGET_ENV="${RUNALL_CONDA_ENV:-llava}"
  echo "[env] Activating conda env: ${TARGET_ENV}"
  conda activate "${TARGET_ENV}" 2>/dev/null || echo "[env][WARN] failed to activate ${TARGET_ENV}; using system Python"
fi

# 2) Defaults (can be overridden by env)
ANN_DIR="${ANN_DIR:-${REPO_ROOT}/datasets/annotations/five}"
EXPERTS="${EXPERTS:-blending,diffusion_detector}"
CKPT="${CKPT:-/data/250010183/workspace/X2DFD/weights/checkpoints/ckpt/FR/llava-v1.5-7b-lora-[x2dfd]}"
WEAK_JSON="${WEAK_JSON:-${REPO_ROOT}/train/outputs/train_weak_small_blending+diffdet.json}"
BASE_MODEL="${BASE_MODEL:-weights/base/llava-v1.5-7b}"
INFER_CFG="${INFER_CFG:-config/test_config.yaml}"

# 3) Dry-run support
if [[ "${RUNALL_DRY:-0}" == "1" ]]; then
  set -x
  echo python -m train.pipeline --infer-config "$INFER_CFG" --phase weak --skip-annotate --reuse-datasets-dir "$ANN_DIR" --experts "$EXPERTS" --out-weak-json "$WEAK_JSON"
  echo python -m train.pipeline --infer-config "$INFER_CFG" --phase train --run-train --train-output "$CKPT" --train-gpus "<auto>" --base-model "$BASE_MODEL" --out-weak-json "$WEAK_JSON"
  echo python -m eval.infer.runner --config "$INFER_CFG" --experts "$EXPERTS" --model-path "$CKPT" --model-base "$BASE_MODEL" "$@"
  exit 0
fi

# 4) Only-eval short-circuit
if [[ "${RUNALL_ONLY_EVAL:-0}" == "1" ]]; then
  echo "[small] ONLY_EVAL=1 — skipping weak/train; running eval with CKPT=$CKPT"
  python -m eval.infer.runner \
    --config "$INFER_CFG" \
    --experts "$EXPERTS" \
    --model-path "$CKPT" \
    --model-base "$BASE_MODEL" \
    "$@"
  echo "==> Small run (only eval) completed. CKPT: $CKPT"
  exit 0
fi

# 5) Strict mode: do NOT auto-annotate. Require existing *_annotations.json under ANN_DIR.
if ! compgen -G "${ANN_DIR}/*_annotations.json" >/dev/null; then
  echo "[small][ERROR] No *_annotations.json found under ${ANN_DIR}. Please provide pre-annotated datasets and rerun." >&2
  echo "             e.g., set ANN_DIR=/abs/path/to/train/datasets (containing *_annotations.json)" >&2
  exit 2
fi

# 6) Weak merge (two experts)
python -m train.pipeline \
  --infer-config "$INFER_CFG" \
  --phase weak \
  --skip-annotate \
  --reuse-datasets-dir "$ANN_DIR" \
  --experts "$EXPERTS" \
  --out-weak-json "$WEAK_JSON"

# 7) Train (LoRA) with auto GPU detection unless provided
if [[ -d "$CKPT" ]] && [[ "${RUNALL_OVERWRITE_CKPT:-0}" == "1" ]]; then
  echo "[small][WARN] Removing existing CKPT: $CKPT"
  rm -rf "$CKPT"
fi

if [[ -d "$CKPT" ]]; then
  echo "[small] ckpt exists: $CKPT — skipping training"
else
  # Require deepspeed to be available
  if ! command -v deepspeed >/dev/null 2>&1; then
    echo "[small][ERROR] deepspeed not found in PATH. Please install/activate it before training." >&2
    exit 3
  fi
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
  echo "[small] Training on GPU(s): $TRAIN_GPUS (quick_train.sh)"
  GPUS="$TRAIN_GPUS" \
  BASE_MODEL="$BASE_MODEL" \
  VISION_TOWER="${VISION_TOWER:-weights/base/clip-vit-large-patch14-336}" \
  OUTPUT_DIR="$CKPT" \
  bash "$REPO_ROOT/quick_train.sh" "$WEAK_JSON"
  # Basic success check: CKPT dir should exist and contain some files
  if [[ ! -d "$CKPT" ]] || [[ -z $(ls -A "$CKPT" 2>/dev/null) ]]; then
    echo "[small][ERROR] training did not produce files under: $CKPT" >&2
    exit 4
  fi
fi

# 8) Eval with the same CKPT on tiny inputs (unless caller disables)
if [[ "${RUNALL_ONLY_TRAIN:-0}" != "1" ]]; then
  python -m eval.infer.runner \
    --config "$INFER_CFG" \
    --experts "$EXPERTS" \
    --model-path "$CKPT" \
    --model-base "$BASE_MODEL" \
    "$@"
else
  echo "[small] RUNALL_ONLY_TRAIN=1 set — skipping eval step."
fi

echo "==> Small run completed. CKPT: $CKPT"
