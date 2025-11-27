#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"
cd "$REPO_ROOT"

JSON_PATH="${1:-/data/250010183/workspace/X2DFD/train/outputs/pipeline/runs/pipeline_20251123T140554Z/train/train_merged.json}"
if [[ ! -f "$JSON_PATH" ]]; then
  echo "Dataset JSON not found: $JSON_PATH" >&2
  exit 1
fi

# Auto-detect GPUs if not provided: prefer CUDA_VISIBLE_DEVICES, else nvidia-smi
if [[ -z "${GPUS:-}" ]]; then
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    GPUS="${CUDA_VISIBLE_DEVICES}"
  elif command -v nvidia-smi >/dev/null 2>&1; then
    COUNT="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
    if [[ "${COUNT}" =~ ^[0-9]+$ ]] && [[ "${COUNT}" -gt 0 ]]; then
      GPUS="$(seq -s, 0 $((COUNT-1)))"
    else
      GPUS="0"
    fi
  else
    GPUS="0"
  fi
fi
INCLUDE="localhost:${GPUS}"
MASTER_PORT="${MASTER_PORT:-29100}"
BASE_MODEL="${BASE_MODEL:-weights/base/llava-v1.5-7b}"
VISION_TOWER="${VISION_TOWER:-weights/base/clip-vit-large-patch14-336}"
OUTPUT_DIR="${OUTPUT_DIR:-weights/checkpoints/ckpt/FR/llava-v1.5-7b-lora-[small]}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-}"

IMG_FOLDER=""

GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-False}"
DDP_FIND_UNUSED_PARAMETERS="${DDP_FIND_UNUSED_PARAMETERS:-False}"

ARGS=(
  deepspeed --master_port "$MASTER_PORT" --include "$INCLUDE" train/model_train.py
  --lora_enable True --lora_r 16 --lora_alpha 32 --mm_projector_lr 2e-5
  --model_name_or_path "$BASE_MODEL"
  --version v1
  --data_path "$JSON_PATH"
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
  --num_train_epochs 1
  --per_device_train_batch_size 2
  --per_device_eval_batch_size 2
  --gradient_accumulation_steps 1
  --evaluation_strategy no
  --save_strategy no
  --save_total_limit 1
  --learning_rate 2e-4
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
if [[ -n "$DEEPSPEED_CONFIG" ]]; then
  ARGS+=(--deepspeed "$DEEPSPEED_CONFIG")
fi

echo "[cmd] ${ARGS[*]}"
exec "${ARGS[@]}"
