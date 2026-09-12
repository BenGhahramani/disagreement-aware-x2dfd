#!/usr/bin/env python3
"""Analyse FF++ c23 source-domain consistency scores (CPU only)."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from eval.x2dfd_ffpp_source_domain import (
    COMPARISON_MODE,
    DEFAULT_ADAPTER,
    DEFAULT_BASE,
    DEFAULT_EXPERTS,
    PAPER_IN_DOMAIN_AUC,
    SCRIPT_VERSION,
    build_provenance,
    compute_source_domain_metrics,
    write_json,
)

LOGGER = logging.getLogger("x2dfd.analyse_ffpp_source_domain")

EXIT_PASS = 0
EXIT_USAGE = 2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREP = PROJECT_ROOT / "datasets" / "evaluation" / "ffpp_c23_source"
DEFAULT_OUTPUT = PROJECT_ROOT / "eval" / "outputs" / "x2dfd_ffpp_source_domain"


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def merge_frames(
    prep_frames: Sequence[Dict[str, Any]],
    scored: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    by_id = {r["frame_id"]: dict(r) for r in scored if isinstance(r.get("frame_id"), str)}
    merged: List[Dict[str, Any]] = []
    for row in prep_frames:
        fid = row.get("frame_id")
        base = dict(row)
        if isinstance(fid, str) and fid in by_id:
            scored_row = by_id[fid]
            base.update(
                {
                    "status": scored_row.get("status", "scored"),
                    "fake_score": scored_row.get("fake_score"),
                    "real_score": scored_row.get("real_score"),
                    "label": scored_row.get("label"),
                    "failure_reason": scored_row.get("failure_reason")
                    or base.get("failure_reason"),
                }
            )
        elif base.get("status") != "failed":
            base["status"] = "failed"
            base["failure_reason"] = base.get("failure_reason") or "not_scored"
        merged.append(base)
    return merged


def analyse(*, prep_dir: Path, output_dir: Path) -> Dict[str, Any]:
    prep_dir = Path(prep_dir)
    output_dir = Path(output_dir)
    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    prep = json.loads((prep_dir / "preparation_manifest.json").read_text(encoding="utf-8"))
    scored = _load_jsonl(output_dir / "frame_scores.jsonl")
    frames = merge_frames(prep.get("frames") or [], scored)
    metrics = compute_source_domain_metrics(frames)

    summary = {
        "comparison_mode": COMPARISON_MODE,
        "script_version": SCRIPT_VERSION,
        "paper_comparison_metadata": {
            "table9_auc": PAPER_IN_DOMAIN_AUC,
            "not_a_pass_fail_threshold": True,
        },
        "metrics": {k: v for k, v in metrics.items() if k != "videos"},
        "n_videos": metrics["n_videos_total"],
        "n_usable_frames": metrics["n_usable_frames"],
        "n_failed_frames": metrics["n_failed_frames"],
        "frame_level_roc_auc": metrics["frame_level_roc_auc"],
        "video_level_roc_auc": metrics["video_level_roc_auc"],
    }
    write_json(analysis_dir / "metrics.json", {**metrics, "summary": summary})
    write_json(analysis_dir / "summary.json", summary)
    write_json(analysis_dir / "merged_frames.json", {"frames": frames})
    split_path = Path(
        prep.get("official_split_path")
        or (Path(prep.get("dataset_root") or prep_dir) / "test.json")
    )
    frames_per_video = int(prep.get("frames_per_video_target") or 8)
    prov = build_provenance(
        dataset_root=Path(prep.get("dataset_root") or prep_dir),
        split_path=split_path,
        output_dir=output_dir,
        quantisation="fp16",
        experts=DEFAULT_EXPERTS,
        adapter_path=DEFAULT_ADAPTER,
        base_path=DEFAULT_BASE,
        load_4bit=False,
        frames_per_video=frames_per_video,
        subset=prep.get("subset_selection") if isinstance(prep.get("subset_selection"), dict) else None,
        extra={"stage": "analysis", "metrics_summary": summary["metrics"]},
    )
    write_json(analysis_dir / "provenance.json", prov)
    return {"analysis_dir": str(analysis_dir.resolve()), "summary": summary}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m tools.analyse_ffpp_source_domain")
    p.add_argument("--prep-dir", type=Path, default=DEFAULT_PREP)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    try:
        result = analyse(prep_dir=args.prep_dir, output_dir=args.output_dir)
    except (OSError, json.JSONDecodeError, KeyError) as exc:
        LOGGER.error("%s", exc)
        return EXIT_USAGE
    print(json.dumps(result, indent=2))
    LOGGER.info(
        "frame AUC=%s video AUC=%s (paper Table9 FF++c23 metadata=%s)",
        result["summary"].get("frame_level_roc_auc"),
        result["summary"].get("video_level_roc_auc"),
        PAPER_IN_DOMAIN_AUC["FF++c23"],
    )
    return EXIT_PASS


if __name__ == "__main__":
    raise SystemExit(main())
