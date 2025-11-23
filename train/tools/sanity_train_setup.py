#!/usr/bin/env python3
"""
Sanity checks for LLaVA training setup to diagnose common issues like:
  AttributeError: 'DataArguments' object has no attribute 'image_processor'

What it does:
- Loads a unified config (e.g., uniconfig_small.yaml)
- Resolves one or more training dataset JSONs
- Resolves absolute image paths using Description/image_root_prefix
- Verifies the vision tower (CLIP) image processor loads
- Runs a preprocessing pass on a few sample images
- Prints a suggested deepspeed training command with required flags

Usage:
  python sanity_train_setup.py --config train/configs/config.yaml \
    --vision-tower /path/to/clip-vit-large-patch14-336 \
    --samples 4
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

from PIL import Image


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists() or yaml is None:
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_abs(p: str, root: Optional[str]) -> str:
    if os.path.isabs(p):
        return os.path.normpath(p)
    return os.path.normpath(os.path.join(root or "", p))


def _collect_images_from_dataset_json(path: Path, fallback_root: Optional[str]) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    root = fallback_root
    if isinstance(payload, dict):
        desc = payload.get("Description") or payload.get("description")
        if isinstance(desc, str) and desc.strip():
            root = desc.strip()
    images: List[str] = []
    if isinstance(payload, dict) and isinstance(payload.get("images"), list):
        for it in payload["images"]:
            if isinstance(it, dict):
                rel = it.get("image_path") or it.get("path")
                if isinstance(rel, str) and rel:
                    images.append(_resolve_abs(rel, root))
    elif isinstance(payload, list):  # conversation schema
        for it in payload:
            if isinstance(it, dict):
                rel = it.get("image")
                if isinstance(rel, str) and rel:
                    images.append(_resolve_abs(rel, root))
    return images


def _load_image_processor(vision_tower: str):
    # Try CLIPImageProcessor then AutoImageProcessor
    try:
        from transformers import CLIPImageProcessor  # type: ignore
        return CLIPImageProcessor.from_pretrained(vision_tower)
    except Exception:
        pass
    try:
        from transformers import AutoImageProcessor  # type: ignore
        return AutoImageProcessor.from_pretrained(vision_tower)
    except Exception as e:
        raise RuntimeError(f"Failed to load image processor from '{vision_tower}': {e}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Sanity checks for training setup")
    ap.add_argument("--config", type=Path, required=True, help="Path to unified config (e.g., uniconfig_small.yaml)")
    ap.add_argument("--vision-tower", default=None, help="Path to CLIP vision tower (e.g., clip-vit-large-patch14-336)")
    ap.add_argument("--samples", type=int, default=4, help="Number of sample images to preprocess")
    args = ap.parse_args()

    cfg = _load_yaml(args.config)
    base_dir = Path(__file__).resolve().parent

    data_dir = (cfg.get("paths", {}).get("data_dir") if isinstance(cfg, dict) else None) or "data"
    if not os.path.isabs(data_dir):
        data_dir = str((base_dir / data_dir).resolve())
    image_root_prefix = (cfg.get("paths", {}).get("image_root_prefix") if isinstance(cfg, dict) else None)

    ann = (cfg.get("annotations") or {}) if isinstance(cfg, dict) else {}
    real_files = list(ann.get("real_files") or [])
    fake_files = list(ann.get("fake_files") or [])
    train_jsons = [str(Path(data_dir) / f) for f in (real_files + fake_files)]
    if not train_jsons:
        raise SystemExit("No training JSONs resolved from config.annotations.real_files/fake_files")

    tr_cfg = (cfg.get("training") or {}) if isinstance(cfg, dict) else {}
    vt = args.vision_tower or os.environ.get("VISION_TOWER") or tr_cfg.get("vision_tower")
    if not vt:
        print("[WARN] --vision-tower not provided; suggest using run_train.sh's value.")
        print("       Training will likely fail unless the training launcher sets it.")
    else:
        print(f"[OK] Using vision tower: {vt}")

    # Load a few sample images across the first training JSON
    first_json = Path(train_jsons[0])
    imgs = _collect_images_from_dataset_json(first_json, image_root_prefix)
    if not imgs:
        raise SystemExit(f"No images resolved from {first_json}")
    sample_paths = imgs[: max(1, args.samples)]
    pil_images = []
    for p in sample_paths:
        try:
            pil_images.append(Image.open(p).convert("RGB"))
        except Exception as e:
            raise SystemExit(f"Failed to open sample image {p}: {e}")
    print(f"[OK] Loaded {len(pil_images)} sample images from {first_json}")

    # Try loading the image processor and preprocess samples
    if vt:
        processor = _load_image_processor(vt)
        try:
            batch = processor(images=pil_images, return_tensors="pt")
            print("[OK] Processor output keys:", list(batch.keys()))
            for k, v in batch.items():
                try:
                    shape = tuple(v.shape)  # type: ignore[attr-defined]
                except Exception:
                    shape = None
                print(f"      {k}: {shape}")
        except Exception as e:
            raise SystemExit(f"Processor failed to preprocess samples: {e}")

    # Suggest a complete deepspeed command matching run_train.sh requirements
    base_model = (cfg.get("annotations", {}).get("model_path") if isinstance(cfg, dict) else None) or (cfg.get("model", {}).get("base") if isinstance(cfg, dict) else None)
    out_dir = (cfg.get("training", {}).get("output_dir") if isinstance(cfg, dict) else None) or "/tmp/llava_lora_ckpt"
    ds_cfg = (cfg.get("training", {}).get("deepspeed_config") if isinstance(cfg, dict) else None) or "/path/to/zero3.json"
    gpus = (cfg.get("training", {}).get("gpus") if isinstance(cfg, dict) else None) or "0"
    merged_hint = f"{Path(data_dir) / 'train_merged.json'} (pipeline will generate)"
    print("\n[Suggested deepspeed command] (align with run_train.sh):")
    print(
        " ".join([
            "deepspeed", "--master_port", "29000", "--include", f"localhost:{gpus}",
            str(Path(__file__).with_name("model_train.py")),
            "--lora_enable", "True",
            "--lora_r", "16", "--lora_alpha", "32", "--mm_projector_lr", "2e-5",
            "--deepspeed", ds_cfg,
            "--model_name_or_path", base_model or "/path/to/base",
            "--version", "v1",
            "--data_path", merged_hint,
            "--image_folder", image_root_prefix or "",
            "--vision_tower", vt or "/path/to/clip-vit-large-patch14-336",
            "--mm_projector_type", "mlp2x_gelu",
            "--mm_vision_select_layer", "-2",
            "--mm_use_im_start_end", "False",
            "--mm_use_im_patch_token", "False",
            "--image_aspect_ratio", "pad",
            "--group_by_modality_length", "True",
            "--bf16", "True",
            "--output_dir", out_dir,
            "--num_train_epochs", "1",
            "--per_device_train_batch_size", "2",
            "--per_device_eval_batch_size", "2",
            "--gradient_accumulation_steps", "1",
            "--evaluation_strategy", "no",
            "--save_strategy", "no",
            "--save_total_limit", "1",
            "--learning_rate", "2e-4",
            "--weight_decay", "0.",
            "--warmup_ratio", "0.03",
            "--lr_scheduler_type", "cosine",
            "--logging_steps", "1",
            "--tf32", "True",
            "--model_max_length", "2048",
            "--gradient_checkpointing", "True",
            "--dataloader_num_workers", "4",
            "--lazy_preprocess", "True",
        ])
    )


if __name__ == "__main__":
    main()
