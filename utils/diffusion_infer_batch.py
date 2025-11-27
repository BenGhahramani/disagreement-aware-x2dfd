#!/usr/bin/env python3
"""
Batch inference for the minimal Diffusion/Aligner expert (src/diffusion).

This script mirrors utils/forensics_align_score.py but uses the in-repo
src/diffusion.core.Aligner which now supports native batched inference.

Input JSON schemas supported (any of):
  1) Dataset with Description root
     {
       "Description": "/abs/root",
       "images": [ {"image_path": "rel/or/abs"}, ... ]
     }

  2) Dataset with absolute 'path' fields (no Description required)
     { "images": [ {"path": "/abs/path.png"}, ... ] }

  3) Conversation-style list with absolute 'image' fields
     [ {"image": "/abs/path.png", ...}, ... ]

CLI styles:
  - Two-list mode (recommended; matches forensics_align_score):
      python utils/diffusion_infer_batch.py \\
        --real /path/Real.json 0 \\
        --fake /path/Deepfakes.json 0 \\
        --weights-dir weights \\
        --model ours-sync \\
        --batch-size 128 --num-workers 4 \\
        --out results/diffusion_scores.json

  - Single JSON mode:
      python utils/diffusion_infer_batch.py \\
        --json /path/Real.json --label real \\
        --weights-dir weights --model ours-sync --out out.json

Output: a flat JSON list with entries like
  {
    "image_path": "/abs/path/to/img.png",
    "ground_truth_label": "real|fake|auto",
    "diffusion_score": 0.037  # prob of fake from the expert
  }
If the expert directory contains multiple sub-models, the average over
configured models is written as diffusion_score, and per-model scores are
also included under a nested dict field "per_model": {name: score}.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch  # type: ignore

# Ensure repo root on sys.path for `src` imports when executed as a script
import sys
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.diffusion.core import Aligner  # type: ignore


def _resolve_abs(p: str, root_prefix: str | None) -> str:
    if not p:
        return p
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(root_prefix or "", p))


def _load_paths(json_path: str, label: str) -> List[Dict[str, str]]:
    paths: List[Dict[str, str]] = []
    if not os.path.isfile(json_path):
        print(f"Warning: JSON file not found at {json_path}. Skipping.")
        return paths
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    desc = None
    images = None
    if isinstance(data, dict):
        desc = data.get('Description') or data.get('description')
        images = data.get('images')
    elif isinstance(data, list):
        images = data

    if isinstance(images, list):
        for it in images:
            p = None
            if isinstance(it, dict):
                p = it.get('path') or it.get('image_path') or it.get('image')
            elif isinstance(it, str):
                p = it
            if not p:
                continue
            ap = _resolve_abs(p, desc)
            if os.path.isfile(ap):
                paths.append({"path": ap, "label": label})
    else:
        print(f"Warning: Unrecognized JSON schema in {json_path}, skipping.")

    return paths


def main() -> None:
    p = argparse.ArgumentParser(description='Batched scoring with Diffusion/Aligner expert (ours-sync by default).')
    # Two-list mode (can repeat)
    p.add_argument('--real', metavar=('PATH', 'LIMIT'), nargs=2, action='append', help='Real JSON path + per-file limit (0 for all). Repeatable.')
    p.add_argument('--fake', metavar=('PATH', 'LIMIT'), nargs=2, action='append', help='Fake JSON path + per-file limit (0 for all). Repeatable.')
    # Single JSON mode
    p.add_argument('--json', default=None, help='Single JSON in any supported schema.')
    p.add_argument('--label', default='auto', choices=['real', 'fake', 'auto'], help='Label used for single JSON mode (default: auto).')

    p.add_argument('--out', '--output', dest='output_path', default='results/diffusion_scores.json', help='Output JSON path.')
    p.add_argument('--weights-dir', default='weights', help='Root dir that contains <model>/config.yaml (default: weights).')
    p.add_argument('--model', default='ours-sync', help='Sub-model folder name (default: ours-sync).')
    p.add_argument('--device', default=None, help='Torch device (default auto).')
    p.add_argument('--batch-size', type=int, default=128, help='Batch size for DataLoader.')
    p.add_argument('--num-workers', type=int, default=4, help='DataLoader workers.')
    p.add_argument('--pin-memory', action='store_true', help='Enable pinned memory for DataLoader (default: disabled).')

    args = p.parse_args()

    items: List[Dict[str, str]] = []
    # Two-list mode
    if args.real or args.fake:
        if args.real:
            for path, limit_str in args.real:
                paths = _load_paths(path, 'real')
                lim = int(limit_str)
                if lim > 0 and len(paths) > lim:
                    import random
                    paths = random.sample(paths, lim)
                items.extend(paths)
        if args.fake:
            for path, limit_str in args.fake:
                paths = _load_paths(path, 'fake')
                lim = int(limit_str)
                if lim > 0 and len(paths) > lim:
                    import random
                    paths = random.sample(paths, lim)
                items.extend(paths)
    elif args.json:
        # Single JSON mode
        label = args.label
        if label == 'auto':
            base = os.path.basename(args.json).lower()
            label = 'real' if 'real' in base else 'fake'
        items = _load_paths(args.json, label)
    else:
        p.error('Provide either --json or at least one of --real/--fake inputs')

    if not items:
        print('No valid image paths found; exiting.')
        return

    device = args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    aligner = Aligner(models_list=[args.model], weights_dir=args.weights_dir, device=device)
    print("Aligner initialized.")

    # Run batched inference once per model inside Aligner.infer
    paths = [it['path'] for it in items]
    results = aligner.infer(paths, batch_size=int(args.batch_size), num_workers=int(args.num_workers), pin_memory=bool(args.pin_memory))

    # Compose flat list with aggregate diffusion_score and optional per_model
    out_list = []
    for it in items:
        pth = it['path']
        lbl = it.get('label')
        entry = {
            'image_path': pth,
            'ground_truth_label': lbl,
        }
        res = results.get(pth) or {}
        if isinstance(res, dict) and 'error' in res:
            entry['diffusion_score'] = None
            entry['per_model'] = res
        elif isinstance(res, dict):
            # compute mean over models in this run
            vals = [v for v in res.values() if isinstance(v, (int, float))]
            entry['diffusion_score'] = float(sum(vals)/len(vals)) if vals else None
            entry['per_model'] = {k: float(v) for k, v in res.items() if isinstance(v, (int, float))}
        else:
            entry['diffusion_score'] = None
        out_list.append(entry)

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_list, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"\n✅ Done. Scores saved to: {str(out_path)}")


if __name__ == '__main__':
    main()

