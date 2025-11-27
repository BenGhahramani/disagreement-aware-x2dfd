#!/usr/bin/env bash
# Strict LoRA training launcher (narrow training only): no annotate, no weak merge.
# Usage:
#   bash train/scripts/train_only.sh \
#     --data-json /abs/path/to/train_weak.json \
#     --output-dir /abs/path/to/ckpt_dir \
#     [--base-model weights/base/llava-v1.5-7b] \
#     [--vision-tower weights/base/clip-vit-large-patch14-336] \
#     [--gpus 0,1] [--epochs 1] [--train-bsz 2] [--lr 2e-4]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}" )/../.." && pwd)"
cd "$REPO_ROOT"

# Defaults (override by flags)
DATA_JSON=""
OUTPUT_DIR=""
BASE_MODEL="weights/base/llava-v1.5-7b"
VISION_TOWER="weights/base/clip-vit-large-patch14-336"
GPUS=""
EPOCHS=1
TRAIN_BSZ=2
EVAL_BSZ=2
LR=2e-4
MASTER_PORT=${MASTER_PORT:-29000}

# Parse flags
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-json) DATA_JSON="$2"; shift 2;;
    --output-dir) OUTPUT_DIR="$2"; shift 2;;
    --base-model) BASE_MODEL="$2"; shift 2;;
    --vision-tower) VISION_TOWER="$2"; shift 2;;
    --gpus) GPUS="$2"; shift 2;;
    --epochs) EPOCHS="$2"; shift 2;;
    --train-bsz) TRAIN_BSZ="$2"; shift 2;;
    --eval-bsz) EVAL_BSZ="$2"; shift 2;;
    --lr) LR="$2"; shift 2;;
    --) shift; break;;
    *) echo "[WARN] Unknown arg: $1"; shift;;
  esac
done

if [[ -z "$DATA_JSON" || -z "$OUTPUT_DIR" ]]; then
  echo "Usage: $0 --data-json </abs/train_weak.json> --output-dir </abs/ckpt_dir> [options...]" >&2
  exit 2
fi
if [[ ! -f "$DATA_JSON" ]]; then
  echo "[ERR] data-json not found: $DATA_JSON" >&2
  exit 3
fi
mkdir -p "$OUTPUT_DIR"

# Auto-detect GPU list if not provided
if [[ -z "$GPUS" ]]; then
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    GPUS="$CUDA_VISIBLE_DEVICES"
  elif command -v nvidia-smi >/dev/null 2>&1; then
    COUNT="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
    if [[ "$COUNT" =~ ^[0-9]+$ ]] && [[ "$COUNT" -gt 0 ]]; then
      GPUS="$(seq -s, 0 $((COUNT-1)))"
    else
      GPUS="0"
    fi
  else
    GPUS="0"
  fi
fi
INCLUDE="localhost:${GPUS}"

IMG_FOLDER=""  # conversation-style JSON already stores absolute image paths
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-False}
DDP_FIND_UNUSED_PARAMETERS=${DDP_FIND_UNUSED_PARAMETERS:-False}

ARGS=(
  deepspeed --master_port "$MASTER_PORT" --include "$INCLUDE" train/model_train.py
  --lora_enable True --lora_r 16 --lora_alpha 32 --mm_projector_lr 2e-5
  --model_name_or_path "$BASE_MODEL"
  --version v1
  --data_path "$DATA_JSON"
  --image_folder "$IMG_FOLDER"
  --vision_tower "$VISION_TOWER"
  --mm_projector_type mlp2x_gelu
  --mm_vision_select_layer -2
  --mm_use_im_start_end False
  --mm_use_im_patch_token False
  --image_aspect_ratio pad
  --group_by_modality_length True
  --output_dir "$OUTPUT_DIR"
  --bf16 True
  --gradient_checkpointing "$GRADIENT_CHECKPOINTING"
  --ddp_find_unused_parameters "$DDP_FIND_UNUSED_PARAMETERS"
  --num_train_epochs "$EPOCHS"
  --per_device_train_batch_size "$TRAIN_BSZ"
  --per_device_eval_batch_size "$EVAL_BSZ"
  --gradient_accumulation_steps 1
  --evaluation_strategy no
  --save_strategy no
  --save_total_limit 1
  --learning_rate "$LR"
  --weight_decay 0.0
  --warmup_ratio 0.03
  --lr_scheduler_type cosine
  --logging_steps 1
  --tf32 True
  --model_max_length 2048
  --dataloader_num_workers 4
  --lazy_preprocess True
  --report_to none
)

echo "[train_only] cmd: ${ARGS[*]}"
exec "${ARGS[@]}"
