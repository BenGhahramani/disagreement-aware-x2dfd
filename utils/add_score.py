#!/usr/bin/env python3
"""
Augment merged annotations JSON with detector scores.

For each item in the merged annotations, append to:
- human value: " By the observation of {model_name} model, the score is {score:.3f}."
- gpt value:   " And the image shows {artifact_phrase}." where phrase depends on score:
    - score >= hi -> "obvious blending artifacts"
    - lo < score < hi -> "some blending artifacts"
    - score <= lo -> "little or no blending artifacts"

Score lookup is done by matching the end of detector image_path with the item's
relative image path (suffix match). If missing, the item is left unchanged unless
--mark-missing is set, in which case it appends N/A.

Defaults are baked in so you can simply run:
  python utils/add_score.py

By default it will:
- Read latest merged path from results/annotations/latest_run.json
- Read scores from results/weak_supply/intermediate_scores.json
- Write to <merged_path> with suffix _scored.json
- Use thresholds lo=0.3, hi=0.7 and alias "Blending"
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple, Optional

# ---- Built-in defaults ----
DEFAULT_ALIAS = 'Blending'
DEFAULT_LO = 0.3
DEFAULT_HI = 0.7
DEFAULT_SCORES_PATH = 'results/weak_supply/intermediate_scores.json'


def _index_scores_by_suffix(scores: List[dict]) -> Dict[str, float]:
    idx: Dict[str, float] = {}
    for s in scores:
        p = s.get("image_path") or s.get("path")
        sc = s.get("blending_score")
        if not p:
            continue
        if sc is None:
            continue
        idx[os.path.normpath(p)] = float(sc)
        # Also index by suffix relative to dataset roots if possible
        parts = os.path.normpath(p).split(os.sep)
        # Create cumulative suffix keys to improve matching robustness
        for k in range(len(parts) - 1):
            suffix = os.path.join(*parts[k:])
            idx.setdefault(suffix, float(sc))
        # Ensure we have the basename as final fallback
        idx.setdefault(os.path.basename(p), float(sc))
    return idx


def _find_score(path_rel: str, index: Dict[str, float]) -> Tuple[bool, float]:
    # Try exact relative, normalized
    key = os.path.normpath(path_rel)
    if key in index:
        return True, index[key]
    # Try suffix match among known keys (most robust across absolute/relative)
    for k in (key, os.path.basename(key)):
        if k in index:
            return True, index[k]
    # Slow path: search by endswith among full keys
    for k in index.keys():
        if k.endswith(key):
            return True, index[k]
    return False, 0.0


def _artifact_phrase(score: float, lo: float, hi: float) -> str:
    if score >= hi:
        return "obvious blending artifacts"
    if score <= lo:
        return "little or no blending artifacts"
    return "some blending artifacts"


def _read_latest_merged_path(latest_path: str = "results/annotations/latest_run.json") -> Optional[str]:
    try:
        with open(latest_path, 'r', encoding='utf-8') as f:
            info = json.load(f)
        return info.get("artefacts", {}).get("merged_annotations")
    except Exception:
        return None


def augment_annotations_with_scores(
    annotations_path: str,
    scores_path: str,
    output_path: str,
    model_name: str = "Blending",
    lo: float = 0.3,
    hi: float = 0.7,
    mark_missing: bool = False,
) -> None:
    with open(annotations_path, 'r', encoding='utf-8') as f:
        annotations = json.load(f)

    with open(scores_path, 'r', encoding='utf-8') as f:
        scores = json.load(f)

    score_index = _index_scores_by_suffix(scores)

    updated = 0
    missing = 0
    for it in annotations:
        path_rel = it.get("image")
        if not path_rel:
            continue
        ok, score = _find_score(path_rel, score_index)
        conv = it.get("conversations") or []
        if len(conv) < 2:
            continue
        human_val = conv[0].get("value", "")
        gpt_val = conv[1].get("value", "")

        if ok:
            human_tail = f" By the observation of {model_name} model, the score is {score:.3f}."
            phrase = _artifact_phrase(score, lo, hi)
            gpt_tail = f" And the image shows {phrase}."
            conv[0]["value"] = human_val + human_tail
            conv[1]["value"] = gpt_val + gpt_tail
            updated += 1
        else:
            if mark_missing:
                conv[0]["value"] = human_val + f" By the observation of {model_name} model, the score is N/A."
            missing += 1

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(annotations, f, ensure_ascii=False, indent=2)

    print(f"✅ Augmented annotations written to: {output_path}")
    print(f"   Updated items: {updated}; Missing scores: {missing}")


def main():
    p = argparse.ArgumentParser(description="Augment merged annotations JSON with detector scores.")
    p.add_argument('--annotations', default=None, help='Path to merged annotations JSON (default: latest_run).')
    p.add_argument('--scores', default=DEFAULT_SCORES_PATH, help='Path to intermediate scores JSON (from weak_supply.py).')
    p.add_argument('--output', default=None, help='Output path for the augmented merged JSON (default: <merged>_scored.json).')
    p.add_argument('--model-name', default=DEFAULT_ALIAS, help='Name of the detector model to mention in the text.')
    p.add_argument('--lo', type=float, default=DEFAULT_LO, help='Low threshold for artifact description.')
    p.add_argument('--hi', type=float, default=DEFAULT_HI, help='High threshold for artifact description.')
    p.add_argument('--mark-missing', action='store_true', help='Append N/A when score is missing.')
    args = p.parse_args()

    merged_path = args.annotations or _read_latest_merged_path()
    if not merged_path:
        raise SystemExit("Could not resolve merged annotations path. Provide --annotations or ensure results/annotations/latest_run.json exists.")

    output_path = args.output or (merged_path[:-5] + "_scored.json" if merged_path.endswith('.json') else merged_path + "_scored.json")

    augment_annotations_with_scores(
        annotations_path=merged_path,
        scores_path=args.scores,
        output_path=output_path,
        model_name=args.model_name,
        lo=args.lo,
        hi=args.hi,
        mark_missing=args.mark_missing,
    )
    print(f"   Input : {merged_path}")
    print(f"   Output: {output_path}")


if __name__ == '__main__':
    main()
