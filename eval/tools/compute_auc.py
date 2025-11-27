#!/usr/bin/env python3
"""Compute ROC AUC from LoRA inference outputs (fake score turns)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

from utils.paths import OUTPUT_ROOT


def _load_latest_run_info(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"Run info not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _classify_from_path(path: Path) -> int | None:
    """Return 0 for real, 1 for fake, or None when ambiguous.

    Rules (order matters):
    - If any parent directory equals 'real' or 'fake' (case-insensitive), use it.
    - Else use strict filename token match: (^|[_-])(real|fake)($|[_.-]).
      This prevents false hits like 'Real3DPortrait'.
    """
    name = path.name.lower()
    # Prefer directory cue
    for seg in path.parts:
        s = str(seg).lower()
        if s == "real":
            return 0
        if s == "fake":
            return 1
    import re
    # Strict token at end or separated by _ - . boundaries
    if re.search(r"(^|[_-])real($|[_.-])", name):
        return 0
    if re.search(r"(^|[_-])fake($|[_.-])", name):
        return 1
    return None


def _scan_dir(dir_path: Path) -> Tuple[List[Path], List[Path]]:
    real: List[Path] = []
    fake: List[Path] = []
    for fp in dir_path.rglob("*_result.json"):
        label = _classify_from_path(fp)
        if label == 0:
            real.append(fp)
        elif label == 1:
            fake.append(fp)
        else:
            # ignore ambiguous files instead of mislabeling
            continue
    return sorted(real), sorted(fake)


def _load_scores(path: Path, label: int) -> List[Tuple[float, int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return []
    pairs: List[Tuple[float, int]] = []
    for item in payload:
        conv = item.get("conversations") if isinstance(item, dict) else None
        if not isinstance(conv, list):
            continue
        score_val = None
        for turn in conv:
            if isinstance(turn, dict) and str(turn.get("from", "")).lower() == "fake score":
                val = turn.get("value")
                try:
                    score_val = float(val)
                except (TypeError, ValueError):
                    score_val = None
                break
        if score_val is None:
            continue
        pairs.append((score_val, label))
    return pairs


def compute_auc(pairs: List[Tuple[float, int]]) -> float:
    pos = sum(1 for _, lbl in pairs if lbl == 1)
    neg = sum(1 for _, lbl in pairs if lbl == 0)
    if pos == 0 or neg == 0:
        raise SystemExit("Need scores from both real and fake datasets to compute AUC")
    sorted_pairs = sorted(pairs, key=lambda x: x[0])
    rank = 1
    rank_sum_pos = 0.0
    idx = 0
    n = len(sorted_pairs)
    while idx < n:
        j = idx
        while j < n and sorted_pairs[j][0] == sorted_pairs[idx][0]:
            j += 1
        group = sorted_pairs[idx:j]
        group_size = len(group)
        avg_rank = (2 * rank + group_size - 1) / 2.0
        pos_in_group = sum(1 for _, lbl in group if lbl == 1)
        rank_sum_pos += avg_rank * pos_in_group
        rank += group_size
        idx = j
    auc = (rank_sum_pos - pos * (pos + 1) / 2.0) / (pos * neg)
    return float(auc)


def main() -> None:
    default_run_info = Path(OUTPUT_ROOT) / "infer" / "latest_run.json"
    ap = argparse.ArgumentParser(description="Compute AUC from LoRA inference outputs")
    ap.add_argument("--run-info", default=str(default_run_info), help="Path to latest_run.json from eval.infer.runner")
    ap.add_argument("--dir", default=None, help="Optional directory containing *_result.json files")
    ap.add_argument("--real", nargs="*", default=None, help="Explicit real result JSON files")
    ap.add_argument("--fake", nargs="*", default=None, help="Explicit fake result JSON files")
    ap.add_argument("--out", default=None, help="Optional path to write AUC metrics JSON")
    args = ap.parse_args()

    real_paths: List[Path] = []
    fake_paths: List[Path] = []
    run_dir: Path | None = None

    if args.dir:
        run_dir = Path(args.dir)
        r, f = _scan_dir(run_dir)
        real_paths.extend(r)
        fake_paths.extend(f)
    else:
        info = _load_latest_run_info(Path(args.run_info))
        datasets_dir = info.get("artefacts", {}).get("datasets_dir")
        if not datasets_dir:
            raise SystemExit("latest_run.json missing artefacts.datasets_dir")
        run_dir = Path(datasets_dir)
        r, f = _scan_dir(run_dir)
        real_paths.extend(r)
        fake_paths.extend(f)

    # If explicit lists are provided, prefer them and ignore auto-detected ones
    if args.real:
        real_paths = [Path(p) for p in args.real]
    if args.fake:
        fake_paths = [Path(p) for p in args.fake]

    if not real_paths or not fake_paths:
        raise SystemExit("No real/fake result files found. Provide --dir or explicit paths.")

    pairs: List[Tuple[float, int]] = []
    for rp in real_paths:
        pairs.extend(_load_scores(rp, 0))
    for fp in fake_paths:
        pairs.extend(_load_scores(fp, 1))
    auc = compute_auc(pairs)
    print(f"AUC (fake score): {auc:.6f}")

    out_path = Path(args.out) if args.out else None
    if not out_path and run_dir is not None:
        out_path = Path(run_dir).parent / "metrics" / "auc.json"
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "auc": auc,
            "real_files": [str(p) for p in real_paths],
            "fake_files": [str(p) for p in fake_paths],
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
