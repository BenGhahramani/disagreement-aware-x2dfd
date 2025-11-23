#!/usr/bin/env python3
"""
Weak-supply: run a traditional pretrained detector to produce per-image scores.

This script mirrors the user-provided pipeline structure and writes an
intermediate JSON array with entries like:

  {
    "image_path": "/abs/path/to/img.png",
    "ground_truth_label": "real|fake",
    "blending_score": 0.873
  }

Requirements:
- Place the detector implementation in src/blending/detector.py (provided)
- Place weights at weights/blending_models/best_gf.pth (default in this repo)

Examples:
  python weak_supply.py \
    --real /path/to/real.json 500 \
    --fake /path/to/fake.json 1000 \
    --shuffle \
    --output_path results/weak_supply/intermediate_scores.json

Or run with config (no --real/--fake needed; defaults to config_small.yaml):
  python weak_supply.py
  python weak_supply.py --config config.yaml
"""

import os
import json
import argparse
import torch
import random
from tqdm import tqdm
import yaml

# Assuming model code is in the src directory
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(_ROOT / 'src'))
from utils.model_scoring import get_provider, resolve_abs_paths
from utils.score_and_augment import run_scoring_and_augment


def load_paths_from_single_json(json_path, label, limit=0, image_root_prefix=None):
    """Loads image paths from a single JSON file with optional random sampling."""
    potential_images = []
    if not os.path.isfile(json_path):
        print(f"Warning: JSON file not found at {json_path}. Skipping.")
        return potential_images

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        all_image_info = data.get("images", [])

        desc = f"  -> Parsing all paths from {os.path.basename(json_path)}"
        for image_info in tqdm(all_image_info, desc=desc, leave=False):
            path = image_info.get("path") or image_info.get("image_path")
            if not path:
                continue
            # Resolve to absolute path if needed
            abs_path = path if os.path.isabs(path) else (
                os.path.join(image_root_prefix, path) if image_root_prefix else os.path.abspath(path)
            )
            if os.path.isfile(abs_path):
                potential_images.append({"path": abs_path, "label": label})

    if limit > 0 and len(potential_images) > limit:
        print(f"  -> Randomly sampling {limit} images from the {len(potential_images)} valid paths found.")
        images_from_file = random.sample(potential_images, limit)
    else:
        images_from_file = potential_images

    print(f"  -> Loaded {len(images_from_file)} images from {os.path.basename(json_path)}.")
    return images_from_file


def _load_from_config(cfg_path: str):
    """Build (real_args, fake_args, image_root_prefix, limit) from YAML config."""
    try:
        with open(cfg_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f) or {}
    except Exception as e:
        raise SystemExit(f"Failed to read config: {cfg_path}: {e}")

    data_dir = (cfg.get('paths') or {}).get('data_dir') or 'data'
    image_root_prefix = (cfg.get('paths') or {}).get('image_root_prefix')
    files_cfg = (cfg.get('files') or {})
    real_files = list(files_cfg.get('real_files') or [])
    fake_files = list(files_cfg.get('fake_files') or [])
    limit = int(((cfg.get('run_params') or {}).get('max_images_per_dataset')) or 0)

    real_args = [(str(Path(data_dir) / f), str(limit)) for f in real_files]
    fake_args = [(str(Path(data_dir) / f), str(limit)) for f in fake_files]

    weak_cfg = (cfg.get('weak_supply') or {})
    annotations_input = weak_cfg.get('annotations_input')
    annotations_output = weak_cfg.get('annotations_output')

    return real_args, fake_args, image_root_prefix, limit, annotations_input, annotations_output


def main(args):
    # Prefer augmentation mode first (skip Phase 1 when possible)
    merged_path = args.annotations_input
    cfg_root_prefix = None
    if not merged_path and args.config:
        try:
            _, _, cfg_root_prefix, _, ann_in, ann_out = _load_from_config(args.config)
        except Exception:
            ann_in = None
            ann_out = None
        if not args.image_root_prefix and cfg_root_prefix:
            args.image_root_prefix = cfg_root_prefix
        if ann_in and not merged_path:
            merged_path = ann_in
        if ann_out and not args.annotations_output:
            args.annotations_output = ann_out

    if not merged_path:
        merged_path = _read_latest_merged_path()

    if merged_path and not (args.real or args.fake):
        out_path = args.annotations_output or (
            merged_path[:-5] + "_scored.json" if merged_path.endswith('.json') else merged_path + "_scored.json"
        )
        device = args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
        print("--- Augmentation Mode: scoring merged JSON and writing annotated output ---")
        print(f"Input : {merged_path}")
        print(f"Output: {out_path}")
        updated, missing = run_scoring_and_augment(
            annotations_path=merged_path,
            image_root_prefix=args.image_root_prefix,
            output_path=out_path,
            model_name=args.model_name,
            weights_path=args.weights_path,
            img_size=args.img_size,
            num_class=args.num_class,
            device=device,
            model_alias=args.provider.capitalize(),
            lo=args.lo,
            hi=args.hi,
        )
        print(f"✅ Done. Updated: {updated}, Missing: {missing}")
        return

    print("--- Phase 1: Preparing Image List (legacy) ---")

    real_images = []
    if args.real:
        print("Loading 'real' images source by source...")
        for path, limit_str in args.real:
            limit = int(limit_str)
            images = load_paths_from_single_json(path, 'real', limit=limit, image_root_prefix=args.image_root_prefix)
            real_images.extend(images)

    fake_images = []
    if args.fake:
        print("Loading 'fake' images source by source...")
        for path, limit_str in args.fake:
            limit = int(limit_str)
            images = load_paths_from_single_json(path, 'fake', limit=limit, image_root_prefix=args.image_root_prefix)
            fake_images.extend(images)

    # If no explicit inputs provided, fall back to config
    if not real_images and not fake_images:
        cfg_path = args.config or 'config_small.yaml'
        print(f"No --real/--fake specified. Loading inputs from {cfg_path} ...")
        real_args, fake_args, cfg_root_prefix, cfg_limit, ann_in, ann_out = _load_from_config(cfg_path)
        # If args.image_root_prefix not provided, use config one
        if not args.image_root_prefix:
            args.image_root_prefix = cfg_root_prefix
        # Fill annotations mode from config if provided
        if ann_in and not args.annotations_input:
            args.annotations_input = ann_in
        if ann_out and not args.annotations_output:
            args.annotations_output = ann_out
        # Load real/fake from config lists
        for path, limit_str in real_args:
            limit = int(limit_str)
            images = load_paths_from_single_json(path, 'real', limit=limit, image_root_prefix=args.image_root_prefix)
            real_images.extend(images)
        for path, limit_str in fake_args:
            limit = int(limit_str)
            images = load_paths_from_single_json(path, 'fake', limit=limit, image_root_prefix=args.image_root_prefix)
            fake_images.extend(images)

    # If annotations_input provided, run augmentation mode (direct scoring + write back)
    if args.annotations_input:
        merged_path = args.annotations_input
        out_path = args.annotations_output or (merged_path[:-5] + "_scored.json" if merged_path.endswith('.json') else merged_path + "_scored.json")
        print("\n--- Augmentation Mode: scoring merged JSON and writing annotated output ---")
        updated, missing = run_scoring_and_augment(
            annotations_path=merged_path,
            image_root_prefix=args.image_root_prefix,
            output_path=out_path,
            model_name=args.model_name,
            weights_path=args.weights_path,
            img_size=args.img_size,
            num_class=args.num_class,
            device=args.device,
            model_alias=args.provider.capitalize(),
            lo=args.lo,
            hi=args.hi,
        )
        print(f"✅ Done. Updated: {updated}, Missing: {missing}")
        print(f"   Input : {merged_path}")
        print(f"   Output: {out_path}")
        return

    combined_image_list = real_images + fake_images
    if args.shuffle:
        print("\nShuffling the final combined image list...")
        random.shuffle(combined_image_list)

    if not combined_image_list:
        print("Error: No valid image paths found. Exiting.")
        return

    print(f"\nTotal images to process: {len(combined_image_list)} ({len(real_images)} real, {len(fake_images)} fake)")

    print("\n--- Phase 2: Running Inference to Calculate Scores ---")
    device = args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    provider = get_provider(
        args.provider,
        model_name=args.model_name,
        weights_path=args.weights_path,
        img_size=args.img_size,
        num_class=args.num_class,
        device=device,
    )
    print(f"\n--- Provider '{args.provider}' initialized successfully ---")

    BATCH_SIZE = int(args.batch_size)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'w', encoding='utf-8') as f:
        f.write("[\n")

        total_processed = 0
        total_images = len(combined_image_list)

        for i in tqdm(range(0, total_images, BATCH_SIZE), desc=f"Processing images in batches of {BATCH_SIZE}"):
            batch_items = combined_image_list[i:i + BATCH_SIZE]
            batch_paths = [item['path'] for item in batch_items]

            # Provider can be any registered detector backend
            score_map = provider.compute_scores(batch_paths)

            for j, item in enumerate(batch_items):
                path = item['path']
                label = item['label']
                res = score_map.get(path)
                blending_score = None if res is None else res.score

                score_entry = {
                    "image_path": path,
                    "ground_truth_label": label,
                    "blending_score": blending_score,
                    "score": blending_score
                }

                f.write(json.dumps(score_entry, indent=4))
                if total_processed + j < total_images - 1:
                    f.write(",\n")
                else:
                    f.write("\n")

            total_processed += len(batch_items)

        f.write("]\n")

    print(f"\n✅ Process complete. Intermediate scores saved to: {args.output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate an intermediate file with model scores for a list of images, with per-file limits.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        '--real', metavar=('PATH', 'LIMIT'), nargs=2, action='append',
        help=(
            "Path to a REAL image JSON and the number of images to load.\n"
            "Set LIMIT to 0 to load all images from that file.\n"
            "This argument can be specified multiple times.\n"
            "Example: --real real_a.json 500 --real real_b.json 1000"
        )
    )
    parser.add_argument(
        '--fake', metavar=('PATH', 'LIMIT'), nargs=2, action='append',
        help=(
            "Path to a FAKE image JSON and the number of images to load.\n"
            "This argument can be specified multiple times.\n"
            "Example: --fake fake_a.json 200"
        )
    )
    parser.add_argument('--output_path', type=str, default="results/weak_supply/intermediate_scores.json",
                        help="Path to save the intermediate scores JSON file.")
    parser.add_argument('--shuffle', action='store_true', help="Shuffle the final combined list of images before processing.")
    parser.add_argument('--image_root_prefix', default=None, help='Prefix to resolve relative image paths in input JSON files.')
    parser.add_argument('--config', default='config_small.yaml', help='Config YAML to load inputs from when --real/--fake are omitted.')
    # Annotations augmentation mode
    parser.add_argument('--annotations_input', default=None, help='Merged annotations JSON to score and augment (alternative mode).')
    parser.add_argument('--annotations_output', default=None, help='Output path for augmented JSON when --annotations_input is provided.')

    # Provider (model) options
    parser.add_argument('--provider', default='blending', help='Score provider name (default: blending).')
    parser.add_argument('--model_name', default='swinv2_base_window16_256', help='Model name for provider (if applicable).')
    parser.add_argument('--weights_path', default=os.path.join(str(_ROOT), 'weights', 'blending_models', 'best_gf.pth'), help='Weights path for provider.')
    parser.add_argument('--img_size', type=int, default=256, help='Input resolution for provider model.')
    parser.add_argument('--num_class', type=int, default=2, help='Number of classes for provider model.')
    parser.add_argument('--device', default=None, help='Device string; default auto-select.')
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch size for outer loop grouping.')
    parser.add_argument('--lo', type=float, default=0.3, help='Low threshold (augmentation mode).')
    parser.add_argument('--hi', type=float, default=0.7, help='High threshold (augmentation mode).')

    args = parser.parse_args()

    # If neither provided, we will load from config in main()

    main(args)
