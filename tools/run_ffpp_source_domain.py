#!/usr/bin/env python3
"""Run FP16 X2-DFD inference for FF++ c23 source-domain consistency.

Uses inherited ``llava-v1.5-7b-lora-[ble-diff]`` with both specialists.
Resumes via ``frame_scores.jsonl``. Does not modify thesis labelled outputs.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

from eval.x2dfd_ffpp_source_domain import (
    COMPARISON_MODE,
    DEFAULT_ADAPTER,
    DEFAULT_BASE,
    DEFAULT_EXPERTS,
    SCRIPT_VERSION,
    build_provenance,
    write_json,
)
from tools.run_smoke_test import build_environment, run_process

LOGGER = logging.getLogger("x2dfd.run_ffpp_source_domain")

EXIT_PASS = 0
EXIT_USAGE = 2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREP = PROJECT_ROOT / "datasets" / "evaluation" / "ffpp_c23_source"
DEFAULT_OUTPUT = PROJECT_ROOT / "eval" / "outputs" / "x2dfd_ffpp_source_domain"
DEFAULT_CONFIG = PROJECT_ROOT / "eval" / "configs" / "infer_config.bunya.yaml"


def _load_scored_ids(scores_path: Path) -> Set[str]:
    done: Set[str] = set()
    if not scores_path.is_file():
        return done
    for line in scores_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        fid = row.get("frame_id")
        if isinstance(fid, str):
            done.add(fid)
    return done


def _append_scores(scores_path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    with scores_path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _parse_runner_output(payload: Any) -> List[Dict[str, Any]]:
    if not isinstance(payload, list):
        return []
    rows: List[Dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        image = item.get("image")
        real = fake = label = None
        for turn in item.get("conversations") or []:
            if not isinstance(turn, dict):
                continue
            src = turn.get("from")
            val = turn.get("value")
            if src == "real score":
                try:
                    real = float(val)
                except (TypeError, ValueError):
                    real = None
            elif src == "fake score":
                try:
                    fake = float(val)
                except (TypeError, ValueError):
                    fake = None
            elif src == "gpt" and isinstance(val, str):
                m = re.search(r"\b(real|fake)\b", val, flags=re.IGNORECASE)
                if m:
                    label = m.group(1).lower()
        rows.append(
            {
                "image": image,
                "real_score": real,
                "fake_score": fake,
                "label": label,
            }
        )
    return rows


def _chunk(items: Sequence[Dict[str, Any]], size: int) -> List[List[Dict[str, Any]]]:
    if size < 1:
        raise ValueError("chunk size must be >= 1")
    return [list(items[i : i + size]) for i in range(0, len(items), size)]


def run_source_domain(
    *,
    prep_dir: Path,
    output_dir: Path,
    project_root: Path,
    config_path: Path,
    experts: str = DEFAULT_EXPERTS,
    chunk_size: int = 64,
    resume: bool = True,
    dry_run: bool = False,
    timeout_s: float = 86400.0,
) -> Dict[str, Any]:
    prep_dir = Path(prep_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    man_path = prep_dir / "preparation_manifest.json"
    if not man_path.is_file():
        raise FileNotFoundError(f"missing preparation manifest: {man_path}")
    manifest = json.loads(man_path.read_text(encoding="utf-8"))
    frames = list(manifest.get("frames") or [])
    usable = [
        f
        for f in frames
        if f.get("status") in {"cropped", "extracted"}
        and (f.get("crop_rel_path") or f.get("frame_rel_path"))
    ]
    scores_path = output_dir / "frame_scores.jsonl"
    done = _load_scored_ids(scores_path) if resume else set()
    pending = [f for f in usable if f.get("frame_id") not in done]

    meta = {
        "comparison_mode": COMPARISON_MODE,
        "script_version": SCRIPT_VERSION,
        "prep_dir": str(prep_dir.resolve()),
        "n_usable": len(usable),
        "n_already_scored": len(done),
        "n_pending": len(pending),
        "experts": experts,
        "quantisation": "fp16",
        "load_4bit": False,
        "chunk_size": chunk_size,
        "resume": resume,
    }
    write_json(output_dir / "run_meta.json", meta)
    split_path = Path(
        manifest.get("official_split_path")
        or (Path(manifest.get("dataset_root") or prep_dir) / "test.json")
    )
    frames_per_video = int(manifest.get("frames_per_video_target") or 8)
    write_json(
        output_dir / "run_provenance.json",
        build_provenance(
            dataset_root=Path(manifest.get("dataset_root") or prep_dir),
            split_path=split_path,
            output_dir=output_dir,
            quantisation="fp16",
            experts=experts,
            adapter_path=DEFAULT_ADAPTER,
            base_path=DEFAULT_BASE,
            load_4bit=False,
            frames_per_video=frames_per_video,
            subset=(
                manifest.get("subset_selection")
                if isinstance(manifest.get("subset_selection"), dict)
                else None
            ),
            extra={"stage": "inference", **meta},
        ),
    )

    if dry_run:
        return {**meta, "dry_run": True, "output_dir": str(output_dir.resolve())}

    chunks_dir = output_dir / "infer_chunks"
    results_dir = output_dir / "infer_results"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    env = build_environment(load_4bit=False, load_8bit=False)
    python = sys.executable
    n_written = 0

    for chunk_idx, chunk in enumerate(_chunk(pending, chunk_size)):
        chunk_frames = []
        for row in chunk:
            rel = row.get("crop_rel_path") or row.get("frame_rel_path")
            chunk_frames.append(
                {
                    "image_path": str(rel).replace("\\", "/"),
                    "frame_id": row["frame_id"],
                    "video_id": row["video_id"],
                    "ground_truth": row["ground_truth"],
                    "manipulation": row.get("manipulation"),
                    "compression": row.get("compression"),
                    "split": row.get("split"),
                    "frame_index": row["frame_index"],
                    "sample_slot": row["sample_slot"],
                }
            )
        infer_payload = {
            "Description": str(prep_dir.resolve()).replace("\\", "/"),
            "images": [{"image_path": f["image_path"]} for f in chunk_frames],
        }
        in_path = chunks_dir / f"chunk_{chunk_idx:04d}.json"
        out_path = results_dir / f"chunk_{chunk_idx:04d}_out.json"
        write_json(in_path, infer_payload)
        write_json(chunks_dir / f"chunk_{chunk_idx:04d}_map.json", chunk_frames)

        if out_path.is_file() and resume:
            LOGGER.info("resume: reusing %s", out_path)
        else:
            cmd = [
                python,
                "-m",
                "eval.infer.runner",
                "--config",
                str(config_path),
                "--json",
                str(in_path),
                "--output",
                str(out_path),
                "--experts",
                experts,
                "--model-path",
                str(project_root / DEFAULT_ADAPTER),
                "--model-base",
                str(project_root / DEFAULT_BASE),
            ]
            LOGGER.info("running chunk %s (%s frames)", chunk_idx, len(chunk_frames))
            proc = run_process(cmd, cwd=project_root, env=env, timeout_s=timeout_s)
            if proc.returncode != 0:
                LOGGER.error("chunk %s failed: %s", chunk_idx, proc.stderr[-2000:])
                raise RuntimeError(
                    f"runner failed on chunk {chunk_idx} (rc={proc.returncode})"
                )

        payload = json.loads(out_path.read_text(encoding="utf-8"))
        parsed = _parse_runner_output(payload)
        if len(parsed) != len(chunk_frames):
            LOGGER.warning(
                "chunk %s: runner returned %s rows for %s inputs",
                chunk_idx,
                len(parsed),
                len(chunk_frames),
            )
        score_rows: List[Dict[str, Any]] = []
        for meta_row, scored in zip(chunk_frames, parsed):
            score_rows.append(
                {
                    **meta_row,
                    "status": "scored" if scored.get("fake_score") is not None else "failed",
                    "failure_reason": None
                    if scored.get("fake_score") is not None
                    else "missing_fake_score",
                    "real_score": scored.get("real_score"),
                    "fake_score": scored.get("fake_score"),
                    "label": scored.get("label"),
                    "runner_image": scored.get("image"),
                }
            )
        _append_scores(scores_path, score_rows)
        n_written += len(score_rows)

    return {
        **meta,
        "n_scores_written_this_run": n_written,
        "scores_path": str(scores_path.resolve()),
        "output_dir": str(output_dir.resolve()),
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m tools.run_ffpp_source_domain")
    p.add_argument("--prep-dir", type=Path, default=DEFAULT_PREP)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--experts", default=DEFAULT_EXPERTS)
    p.add_argument("--chunk-size", type=int, default=64)
    p.add_argument("--timeout", type=float, default=86400.0)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    try:
        result = run_source_domain(
            prep_dir=args.prep_dir,
            output_dir=args.output_dir,
            project_root=args.project_root,
            config_path=args.config,
            experts=args.experts,
            chunk_size=args.chunk_size,
            resume=not args.no_resume,
            dry_run=args.dry_run,
            timeout_s=args.timeout,
        )
    except (OSError, RuntimeError, json.JSONDecodeError) as exc:
        LOGGER.error("%s", exc)
        return EXIT_USAGE
    print(json.dumps(result, indent=2))
    return EXIT_PASS


if __name__ == "__main__":
    raise SystemExit(main())
