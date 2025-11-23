#!/usr/bin/env python3
"""
Thin wrapper to launch LLaVA training from this repo.

Usage (example with DeepSpeed):
  deepspeed --master_port 29000 --include localhost:0 model_train.py \
    --lora_enable True --lora_r 16 --lora_alpha 32 --mm_projector_lr 2e-5 \
    --deepspeed train/configs/zero3.json \
    --model_name_or_path weights/base/llava-v1.5-7b \
    --version v1 \
    --data_path train/outputs/pipeline/runs/<run_id>/train/train_merged.json \
    --image_folder datasets/raw/images \
    --vision_tower weights/base/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir weights/checkpoints/ckpt/FR/llava-v1.5-7b-lora-[exp] \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True

This wrapper intentionally does not parse CLI arguments. All flags
are consumed by LLaVA's internal argument parser inside train().
"""

from __future__ import annotations


def main() -> None:
    # Import inside main so that this file can be imported without side effects
    from llava.train.train import train

    # Forward all CLI flags (in sys.argv) to LLaVA's internal parser.
    # Only set the attention implementation here as per requirement.
    train(attn_implementation="flash_attention_2")


if __name__ == "__main__":
    main()
