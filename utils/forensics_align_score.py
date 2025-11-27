#!/usr/bin/env python3
"""
Score images with the Forensics Aligner expert and write an intermediate JSON.

Mimics the user's batching/incremental-writing logic, but focuses on the
Aligner (ours-sync by default). Supports typical dataset JSON schemas:

1) Dataset schema with Description root:
   {
     "Description": "/abs/root",
     "images": [ {"image_path": "rel/or/abs"}, ... ]
   }

2) Dataset schema with absolute 'path' fields (no Description needed):
   { "images": [ {"path": "/abs/path.png"}, ... ] }

3) Conversation-style list with absolute 'image' fields:
   [ {"image": "/abs/path.png", ...}, ... ]

Output JSON is a flat list, entries like:
  {
    "image_path": "/abs/path/to/img.png",
    "ground_truth_label": "real|fake",
    "aligner_score": 0.573
  }

Usage examples:
  python utils/forensics_align_score.py \
    --real /path/Real.json 500 \
    --fake /path/Fake.json 1000 \
    --output_path results/weak_supply/intermediate_scores.json \
    --batch-size 1000 --shuffle --model ours-sync \
    --weights-dir /abs/weights/forensics_live
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import List, Dict

from tqdm import tqdm

# Ensure repo root on sys.path for `src` imports when executed as a script
import sys
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch  # type: ignore
from src.forensics.aligner import Aligner


def _resolve_abs(p: str, root_prefix: str | None) -> str:
    if not p:
        return p
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(root_prefix or "", p))


def load_paths_from_single_json(json_path: str, label: str, limit: int = 0) -> List[Dict[str, str]]:
    """Load image paths from a single JSON with optional random sampling.

    Supports Description+image_path, absolute path, or conversation list with 'image'.
    """
    potential: List[Dict[str, str]] = []
    if not os.path.isfile(json_path):
        print(f"Warning: JSON file not found at {json_path}. Skipping.")
        return potential

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Detect schema
    desc = None
    images = None
    if isinstance(data, dict):
        desc = data.get('Description') or data.get('description')
        images = data.get('images')
    elif isinstance(data, list):
        images = data  # conversation list style

    if isinstance(images, list):
        for it in tqdm(images, desc=f"  -> Parsing {os.path.basename(json_path)}", leave=False):
            p = None
            if isinstance(it, dict):
                p = it.get('path') or it.get('image_path') or it.get('image')
            elif isinstance(it, str):
                p = it
            if not p:
                continue
            ap = _resolve_abs(p, desc)
            if os.path.isfile(ap):
                potential.append({"path": ap, "label": label})
    else:
        print(f"Warning: Unrecognized JSON schema in {json_path}, skipping.")

    if limit > 0 and len(potential) > limit:
        print(f"  -> Randomly sampling {limit} images from {len(potential)} paths in {os.path.basename(json_path)}.")
        potential = random.sample(potential, limit)

    print(f"  -> Loaded {len(potential)} images from {os.path.basename(json_path)}.")
    return potential


def main(args: argparse.Namespace) -> None:
    print("--- Phase 1: Preparing Image List ---")
    random.seed(args.seed)

    real_images: List[Dict[str, str]] = []
    if args.real:
        print("Loading 'real' images source by source...")
        for path, limit_str in args.real:
            real_images.extend(load_paths_from_single_json(path, 'real', limit=int(limit_str)))

    fake_images: List[Dict[str, str]] = []
    if args.fake:
        print("Loading 'fake' images source by source...")
        for path, limit_str in args.fake:
            fake_images.extend(load_paths_from_single_json(path, 'fake', limit=int(limit_str)))

    items = real_images + fake_images
    if args.shuffle:
        print("\nShuffling the final combined image list...")
        random.shuffle(items)

    if not items:
        print("Error: No valid image paths found. Exiting.")
        return

    print(f"\nTotal images to process: {len(items)} ({len(real_images)} real, {len(fake_images)} fake)")

    # ---- Phase 2: Inference ----
    print("\n--- Phase 2: Running Inference to Calculate Scores (Aligner only) ---")
    device = args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Default weights_dir points to this repo's weights/forensics_live
    weights_dir = args.weights_dir or str(_ROOT / 'weights' / 'forensics_live')
    aligner = Aligner(
        models_list=[args.model],
        weights_dir=weights_dir,
        device=device,
    )
    print("\n--- Aligner loaded successfully ---")

    bs = int(args.batch_size)
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = len(items)
    processed = 0
    with out_path.open('w', encoding='utf-8') as f:
        f.write('[\n')
        for i in tqdm(range(0, total, bs), desc=f"Processing images in batches of {bs}"):
            batch = items[i:i+bs]
            paths = [it['path'] for it in batch]
            scores = aligner.infer(paths)

            for j, it in enumerate(batch):
                p = it['path']
                lbl = it['label']
                sc = None
                if isinstance(scores.get(p), dict):
                    sc = scores[p].get('score')
                entry = {
                    'image_path': p,
                    'ground_truth_label': lbl,
                    'aligner_score': sc,
                }
                f.write(json.dumps(entry, ensure_ascii=False, indent=2))
                if processed + j < total - 1:
                    f.write(',\n')
                else:
                    f.write('\n')
            processed += len(batch)
        f.write(']\n')

    print(f"\n✅ Done. Scores saved to: {str(out_path)}")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Score images with Forensics Aligner (ours-sync by default).')
    p.add_argument('--real', metavar=('PATH', 'LIMIT'), nargs=2, action='append', help='Real JSON path + per-file limit (0 for all). Can repeat.')
    p.add_argument('--fake', metavar=('PATH', 'LIMIT'), nargs=2, action='append', help='Fake JSON path + per-file limit (0 for all). Can repeat.')
    p.add_argument('--output_path', default='results/weak_supply/aligner_scores.json', help='Output JSON path.')
    p.add_argument('--shuffle', action='store_true', help='Shuffle final combined list.')
    p.add_argument('--batch-size', type=int, default=1000, help='Batch size for processing (logical grouping; Aligner runs per-image).')
    p.add_argument('--device', default=None, help='Torch device (default auto).')
    p.add_argument('--weights-dir', default=None, help='Root dir that contains <model>/config.yaml (default: weights/forensics_live).')
    p.add_argument('--model', default='ours-sync', help='Forensics sub-model folder name (default: ours-sync).')
    p.add_argument('--seed', type=int, default=1337, help='Random seed for sampling/shuffle.')
    args = p.parse_args()
    if not args.real and not args.fake:
        p.error('At least one --real or --fake source must be provided.')
    main(args)

