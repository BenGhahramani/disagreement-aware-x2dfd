#!/usr/bin/env python3
"""
One-step scoring and augmentation for merged-style annotations.

Input JSON format (merged annotations):
[
  {
    "id": "1",
    "image": "relative/or/abs/path.png",
    "conversations": [
      {"from":"human", "value": "<image>\nIs this image real or fake?"},
      {"from":"gpt",   "value": "This image is real/fake. ..."}
    ]
  }, ...
]

For each item, this script:
- Resolves the image to an absolute path using --image-root-prefix (when needed)
- Runs the BlendingDetector to get P(fake) score
- Appends to human: " By the observation of {model} model, the score is {score:.3f}."
- Appends to gpt:   " And the image shows {artifact}." where artifact depends on thresholds

Defaults are baked in so you can simply run:
  python utils/score_and_augment.py

By default it will:
- Read latest merged path from results/annotations/latest_run.json
- Use image root prefix: datasets/raw/images
- Use weights: weights/blending_models/best_gf.pth
- Write to <merged_path> with suffix _scored.json
- Use thresholds lo=0.3, hi=0.7 and alias "Blending"
You can still override via CLI flags if needed.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List, Tuple, Optional

import torch

import sys
from pathlib import Path

# Ensure repo root on sys.path for `src` imports when executed as a script
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.blending.detector import BlendingDetector
from utils.paths import DATASETS_ROOT

# ---- Built-in defaults (can be overridden by CLI) ----
DEFAULT_IMAGE_ROOT_PREFIX = str(DATASETS_ROOT / "raw" / "images")
DEFAULT_WEIGHTS_PATH = "weights/blending_models/best_gf.pth"
DEFAULT_MODEL_NAME = "swinv2_base_window16_256"
DEFAULT_ALIAS = "Blending"
DEFAULT_LO = 0.3
DEFAULT_HI = 0.7


def _resolve_abs(path_rel: str, root_prefix: str | None) -> str:
    if os.path.isabs(path_rel):
        return path_rel
    if root_prefix:
        return os.path.normpath(os.path.join(root_prefix, path_rel))
    return os.path.abspath(path_rel)


def _artifact_phrase(score: float, lo: float, hi: float) -> str:
    if score >= hi:
        return "obvious blending artifacts"
    if score <= lo:
        return "little or no blending artifacts"
    return "some blending artifacts"


def load_annotations(path: str) -> List[dict]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_annotations(path: str, data: List[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _read_latest_merged_path(latest_path: str = "results/annotations/latest_run.json") -> Optional[str]:
    try:
        with open(latest_path, 'r', encoding='utf-8') as f:
            info = json.load(f)
        return info.get("artefacts", {}).get("merged_annotations")
    except Exception:
        return None


def run_scoring_and_augment(
    annotations_path: str,
    image_root_prefix: str | None,
    output_path: str,
    model_name: str = 'swinv2_base_window16_256',
    weights_path: str = 'weights/blending_models/best_gf.pth',
    img_size: int = 256,
    num_class: int = 2,
    device: str | None = None,
    model_alias: str = 'Blending',
    lo: float = 0.3,
    hi: float = 0.7,
) -> Tuple[int, int]:
    data = load_annotations(annotations_path)

    # Collect absolute paths to score
    abs_paths: List[str] = []
    rel_paths: List[str] = []
    for it in data:
        p = it.get('image')
        if not p:
            continue
        rel_paths.append(p)
        abs_paths.append(_resolve_abs(p, image_root_prefix))

    if device is None:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    detector = BlendingDetector(
        model_name=model_name,
        weights_path=weights_path,
        img_size=img_size,
        num_class=num_class,
        device=device,
    )

    # Run inference (detector batches internally)
    scores_dict = detector.infer(abs_paths)

    # Apply augmentation
    updated, missing = 0, 0
    for it, abs_p in zip(data, abs_paths):
        conv = it.get('conversations') or []
        if len(conv) < 2:
            continue
        human_val = conv[0].get('value', '')
        gpt_val = conv[1].get('value', '')

        entry = scores_dict.get(abs_p)
        sc = None if entry is None else entry.get('score')
        if sc is None:
            # Leave as-is but mark N/A in human for visibility
            conv[0]['value'] = human_val + f" By the observation of {model_alias} model, the score is N/A."
            missing += 1
        else:
            conv[0]['value'] = human_val + f" By the observation of {model_alias} model, the score is {sc:.3f}."
            phrase = _artifact_phrase(float(sc), lo, hi)
            conv[1]['value'] = gpt_val + f" And the image shows {phrase}."
            updated += 1

    save_annotations(output_path, data)
    return updated, missing


def main():
    p = argparse.ArgumentParser(description='Score and augment merged annotations in one pass.')
    p.add_argument('--annotations', default=None, help='Path to merged annotations JSON to augment (default: latest_run).')
    p.add_argument('--output', default=None, help='Output path for augmented JSON (default: <merged>_scored.json).')
    p.add_argument('--image-root-prefix', default=DEFAULT_IMAGE_ROOT_PREFIX, help='Prefix to resolve relative image paths to absolute paths.')
    p.add_argument('--model-name', default=DEFAULT_MODEL_NAME, help='timm model name for BlendingDetector.')
    p.add_argument('--weights', default=DEFAULT_WEIGHTS_PATH, help='Path to detector weights.')
    p.add_argument('--img-size', type=int, default=256, help='Detector input size.')
    p.add_argument('--num-class', type=int, default=2, help='Number of classes in the classifier.')
    p.add_argument('--device', default=None, help='Device string (default auto).')
    p.add_argument('--alias', default=DEFAULT_ALIAS, help='Alias used in the appended text for the model.')
    p.add_argument('--lo', type=float, default=DEFAULT_LO, help='Low threshold for artifact description.')
    p.add_argument('--hi', type=float, default=DEFAULT_HI, help='High threshold for artifact description.')
    args = p.parse_args()

    merged_path = args.annotations or _read_latest_merged_path()
    if not merged_path:
        raise SystemExit("Could not resolve merged annotations path. Provide --annotations or ensure results/annotations/latest_run.json exists.")

    output_path = args.output or (merged_path[:-5] + "_scored.json" if merged_path.endswith('.json') else merged_path + "_scored.json")

    updated, missing = run_scoring_and_augment(
        annotations_path=merged_path,
        image_root_prefix=args.image_root_prefix,
        output_path=output_path,
        model_name=args.model_name,
        weights_path=args.weights,
        img_size=args.img_size,
        num_class=args.num_class,
        device=args.device,
        model_alias=args.alias,
        lo=args.lo,
        hi=args.hi,
    )

    print(f"✅ Done. Updated: {updated}, Missing: {missing}")
    print(f"   Input : {merged_path}")
    print(f"   Output: {output_path}")


if __name__ == '__main__':
    main()
