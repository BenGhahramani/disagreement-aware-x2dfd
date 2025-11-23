#!/usr/bin/env python3
"""
Convert existing annotations (with long expert prompt) to the binary-QA style.

Input format (legacy style):
[
  {
    "id": "1",
    "image": "manipulated_sequences/Deepfakes/c23/frames/071_054/379.png",
    "conversations": [
      {"from": "human", "value": "<image>\n<very long expert prompt>"},
      {"from": "gpt",   "value": "<original explanation>"}
    ]
  }, ...
]

Output format (binary-QA style):
[
  {
    "id": "1",
    "image": "...",
    "conversations": [
      {"from":"human", "value": "<image>\nIs this image real or fake?"},
      {"from":"gpt",   "value": "This image is real/fake. <original explanation>"}
    ]
  }, ...
]

Label resolution order:
1) --label override if provided
2) Infer from image path: keywords for fake (Deepfakes/Face2Face/FaceSwap/NeuralTextures/manipulated_sequences),
   real (original_sequences/Real)
3) Fallback to parsing original GPT text for 'fake'/'real'
4) Default: fake

This tool does NOT compute detector scores. Use utils/score_and_augment.py
or utils/add_score.py after this conversion if you want to append scores.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List, Dict, Any

from .annotation_utils import build_binary_question, compose_labeled_response


FAKE_HINTS = [
    'deepfakes', 'face2face', 'faceswap', 'neuraltextures', 'manipulated_sequences'
]
REAL_HINTS = [
    'original_sequences', 'real'
]


def _infer_label(image_path: str, original_gpt: str) -> str:
    p = (image_path or '').lower()
    for k in FAKE_HINTS:
        if k in p:
            return 'fake'
    for k in REAL_HINTS:
        if k in p:
            return 'real'
    g = (original_gpt or '').lower()
    if 'fake' in g:
        return 'fake'
    if 'real' in g:
        return 'real'
    return 'fake'


def convert(input_path: str, output_path: str, label_override: str | None = None, reindex: bool = False) -> None:
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    out: List[Dict[str, Any]] = []
    for idx, it in enumerate(data, start=1):
        image = it.get('image')
        conv = it.get('conversations') or []
        original_gpt = ''
        if len(conv) >= 2 and isinstance(conv[1], dict):
            original_gpt = conv[1].get('value') or ''

        label = (label_override or '').strip().lower() if label_override else None
        if label not in ('real', 'fake'):
            label = _infer_label(image or '', original_gpt)

        human_val = f"<image>\n{build_binary_question()}"
        gpt_val = compose_labeled_response(label, original_gpt)

        new_item = {
            'id': str(idx if reindex else it.get('id', idx)),
            'image': image,
            'conversations': [
                {'from': 'human', 'value': human_val},
                {'from': 'gpt', 'value': gpt_val},
            ],
        }
        out.append(new_item)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


def main():
    p = argparse.ArgumentParser(description='Convert legacy annotation JSON to binary-QA style.')
    p.add_argument('--input', required=True, help='Path to legacy JSON file.')
    p.add_argument('--output', required=True, help='Path to write binary-QA styled JSON.')
    p.add_argument('--label', default=None, help='Override label (real/fake). If not set, infer.')
    p.add_argument('--reindex', action='store_true', help='Reindex IDs from 1..N.')
    args = p.parse_args()

    convert(args.input, args.output, label_override=args.label, reindex=args.reindex)
    print(f"✅ Converted to binary-QA style: {args.output}")


if __name__ == '__main__':
    main()

