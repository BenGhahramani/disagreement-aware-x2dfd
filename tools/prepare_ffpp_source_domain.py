#!/usr/bin/env python3
"""Prepare FF++ c23 held-out test frames for source-domain sanity check.

CPU-only. Default: deterministic 100-video subset (50 real / 50 fake stratified)
× 8 frames. Use ``--full-test --frames-per-video 32`` for the fuller protocol.
Failed frames are recorded, not replaced.

Example::

    python -m tools.prepare_ffpp_source_domain \\
      --dataset-root /path/to/FaceForensics++ \\
      --output-dir datasets/evaluation/ffpp_c23_source
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from dashboard.live_analysis import STAGE3_CROP_KWARGS
from eval.reproducibility import utc_timestamp
from eval.x2dfd_ffpp_source_domain import (
    COMPARISON_MODE,
    COMPARISON_RATIONALE,
    COMPRESSION,
    DEFAULT_FRAMES_PER_VIDEO,
    DEFAULT_N_FAKE_VIDEOS,
    DEFAULT_N_REAL_VIDEOS,
    DEFAULT_SPLIT_DIR,
    DEFAULT_SUBSET_SEED,
    FULL_PROTOCOL_FRAMES_PER_VIDEO,
    SCRIPT_VERSION,
    FFPPSourceError,
    SubsetSelection,
    build_provenance,
    build_runner_manifest,
    count_frames_dir,
    enumerate_test_videos,
    expected_sanity_counts,
    expected_test_counts,
    frame_sampling_description,
    full_test_selection,
    load_official_pairs,
    plan_video_frames,
    probe_video_frame_count,
    select_sanity_subset,
    subset_selection_to_dict,
    write_json,
)
from tools.make_face_crop import FaceCropError, NoFaceFoundError, write_face_crop

LOGGER = logging.getLogger("x2dfd.prepare_ffpp_source_domain")

EXIT_PASS = 0
EXIT_USAGE = 2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "datasets" / "evaluation" / "ffpp_c23_source"


def _resolve_ffpp_root(dataset_root: Path) -> Path:
    root = Path(dataset_root).resolve()
    candidates = [
        root,
        root / "FaceForensics++",
        root / "FaceForensics",
        root / "ffpp",
    ]
    for cand in candidates:
        if (cand / "original_sequences").is_dir() or (cand / "manipulated_sequences").is_dir():
            return cand
    raise FFPPSourceError(
        "Could not locate FaceForensics++ tree (need original_sequences/ "
        f"or manipulated_sequences/ under {root})"
    )


def _read_frame_bgr_from_video(video_path: Path, index: int) -> Any:
    import cv2  # type: ignore

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FFPPSourceError(f"cannot open video: {video_path}")
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, float(index))
        ok, frame = capture.read()
        if not ok or frame is None or getattr(frame, "size", 0) == 0:
            raise FFPPSourceError(f"decode_failed frame={index} video={video_path}")
        return frame
    finally:
        capture.release()


def _read_frame_bgr_from_dir(frames_dir: Path, index: int) -> Any:
    import cv2  # type: ignore

    candidates = [
        frames_dir / f"{index:03d}.png",
        frames_dir / f"{index:04d}.png",
        frames_dir / f"{index}.png",
        frames_dir / f"{index:03d}.jpg",
        frames_dir / f"{index:04d}.jpg",
        frames_dir / f"{index}.jpg",
    ]
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        files = sorted(
            [p for p in frames_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}],
            key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem,
        )
        if 0 <= index < len(files):
            path = files[index]
    if path is None or not path.is_file():
        raise FFPPSourceError(f"frame file missing index={index} dir={frames_dir}")
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None or getattr(frame, "size", 0) == 0:
        raise FFPPSourceError(f"failed to read frame image: {path}")
    return frame


def _write_png(frame_bgr: Any, dest: Path) -> None:
    import cv2  # type: ignore

    dest.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(dest), frame_bgr, [int(getattr(cv2, "IMWRITE_PNG_COMPRESSION", 16)), 0])
    if not ok or not dest.is_file():
        raise FFPPSourceError(f"failed to write frame PNG: {dest}")


def _apply_max_videos(subset_meta: SubsetSelection, videos: List[Any], frames_per_video: int) -> SubsetSelection:
    data = subset_selection_to_dict(subset_meta)
    data["selected_video_ids"] = [v.video_id for v in videos]
    data["n_real_selected"] = sum(1 for v in videos if v.ground_truth == "real")
    data["n_fake_selected"] = sum(1 for v in videos if v.ground_truth == "fake")
    by_method: Dict[str, int] = {}
    for v in videos:
        if v.ground_truth == "fake":
            by_method[v.manipulation] = by_method.get(v.manipulation, 0) + 1
    data["n_fake_by_method"] = by_method
    data["planned_frames"] = len(videos) * frames_per_video
    return SubsetSelection(**data)


def prepare(
    *,
    dataset_root: Path,
    output_dir: Path,
    split_path: Optional[Path] = None,
    dry_run: bool = False,
    max_videos: Optional[int] = None,
    skip_crop: bool = False,
    full_test: bool = False,
    frames_per_video: Optional[int] = None,
    subset_seed: int = DEFAULT_SUBSET_SEED,
    n_real: int = DEFAULT_N_REAL_VIDEOS,
    n_fake: int = DEFAULT_N_FAKE_VIDEOS,
) -> Dict[str, Any]:
    root = _resolve_ffpp_root(dataset_root)
    split_file = Path(split_path) if split_path else DEFAULT_SPLIT_DIR / "test.json"
    pairs = load_official_pairs(split_file)
    catalogue = enumerate_test_videos(pairs, compression=COMPRESSION)
    full_counts = expected_test_counts(pairs)
    if frames_per_video is None:
        frames_per_video = (
            FULL_PROTOCOL_FRAMES_PER_VIDEO if full_test else DEFAULT_FRAMES_PER_VIDEO
        )
    if frames_per_video < 1:
        raise FFPPSourceError("frames_per_video must be >= 1")

    if full_test:
        videos, subset_meta = full_test_selection(
            catalogue, frames_per_video=frames_per_video
        )
    else:
        videos, subset_meta = select_sanity_subset(
            catalogue,
            seed=subset_seed,
            n_real=n_real,
            n_fake=n_fake,
            frames_per_video=frames_per_video,
        )
    if max_videos is not None:
        videos = videos[: max(0, int(max_videos))]
        subset_meta = _apply_max_videos(subset_meta, videos, frames_per_video)

    subset_dict = subset_selection_to_dict(subset_meta)
    sampling_desc = frame_sampling_description(frames_per_video)
    out = Path(output_dir)
    frames_root = out / "frames"
    crops_root = out / "crops"
    if not dry_run:
        frames_root.mkdir(parents=True, exist_ok=True)
        if not skip_crop:
            crops_root.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []
    n_probe_failed = 0

    for video in videos:
        abs_video = root / video.relative_video_path
        abs_frames = (
            root / video.relative_frames_dir if video.relative_frames_dir else None
        )
        source = None
        total = 0
        if abs_video.is_file():
            try:
                total = probe_video_frame_count(abs_video)
                source = "video"
            except FFPPSourceError as exc:
                LOGGER.warning("probe failed %s: %s", abs_video, exc)
        if total <= 0 and abs_frames is not None and abs_frames.is_dir():
            total = count_frames_dir(abs_frames)
            source = "frames_dir"
        if total <= 0 or source is None:
            n_probe_failed += 1
            records.append(
                {
                    "frame_id": f"{video.video_id}__probe_failed",
                    "video_id": video.video_id,
                    "video_relpath": video.relative_video_path,
                    "ground_truth": video.ground_truth,
                    "manipulation": video.manipulation,
                    "compression": video.compression,
                    "split": video.split,
                    "frame_index": -1,
                    "sample_slot": -1,
                    "total_frames": 0,
                    "status": "failed",
                    "failure_reason": "probe_failed:missing_video_or_frames",
                    "frame_rel_path": None,
                    "crop_rel_path": None,
                    "fake_score": None,
                    "real_score": None,
                    "label": None,
                }
            )
            continue

        plans = plan_video_frames(
            video, total_frames=total, frames_per_video=frames_per_video
        )
        for plan in plans:
            rec: Dict[str, Any] = {
                "frame_id": plan.frame_id,
                "video_id": plan.video_id,
                "video_relpath": plan.video_relpath,
                "ground_truth": plan.ground_truth,
                "manipulation": plan.manipulation,
                "compression": plan.compression,
                "split": plan.split,
                "frame_index": plan.frame_index,
                "sample_slot": plan.sample_slot,
                "total_frames": plan.total_frames,
                "status": "planned",
                "failure_reason": None,
                "frame_rel_path": None,
                "crop_rel_path": None,
                "fake_score": None,
                "real_score": None,
                "label": None,
                "frame_source": source,
            }
            if dry_run:
                records.append(rec)
                continue
            manip_dir = plan.manipulation
            frame_rel = (
                f"frames/{manip_dir}/{plan.video_id}/"
                f"frame_{plan.frame_index:06d}.png"
            )
            crop_rel = (
                f"crops/{manip_dir}/{plan.video_id}/"
                f"frame_{plan.frame_index:06d}.jpg"
            )
            frame_abs = out / frame_rel
            crop_abs = out / crop_rel
            try:
                if source == "video":
                    frame_bgr = _read_frame_bgr_from_video(abs_video, plan.frame_index)
                else:
                    assert abs_frames is not None
                    frame_bgr = _read_frame_bgr_from_dir(abs_frames, plan.frame_index)
                _write_png(frame_bgr, frame_abs)
                rec["frame_rel_path"] = frame_rel.replace("\\", "/")
                rec["status"] = "extracted"
                if skip_crop:
                    rec["crop_rel_path"] = None
                else:
                    write_face_crop(frame_abs, crop_abs, **STAGE3_CROP_KWARGS)
                    rec["crop_rel_path"] = crop_rel.replace("\\", "/")
                    rec["status"] = "cropped"
            except NoFaceFoundError as exc:
                rec["status"] = "failed"
                rec["failure_reason"] = f"no_face:{exc}"
                rec["crop_rel_path"] = None
            except FaceCropError as exc:
                rec["status"] = "failed"
                rec["failure_reason"] = f"crop_failed:{exc}"
                rec["crop_rel_path"] = None
            except FFPPSourceError as exc:
                rec["status"] = "failed"
                rec["failure_reason"] = str(exc)
                rec["crop_rel_path"] = None
            except OSError as exc:
                rec["status"] = "failed"
                rec["failure_reason"] = f"io_error:{exc}"
                rec["crop_rel_path"] = None
            records.append(rec)

    n_real_v = sum(1 for v in videos if v.ground_truth == "real")
    n_fake_v = sum(1 for v in videos if v.ground_truth == "fake")
    n_failed = sum(1 for r in records if r.get("status") == "failed")
    n_usable = sum(
        1
        for r in records
        if r.get("status") in {"cropped", "extracted"}
        and (skip_crop or r.get("crop_rel_path"))
    )

    by_manip: Dict[str, int] = {}
    for v in videos:
        by_manip[v.manipulation] = by_manip.get(v.manipulation, 0) + 1

    manifest = {
        "description": (
            "FaceForensics++ c23 held-out test for X2-DFD source-domain sanity "
            f"({subset_meta.mode}; {frames_per_video} frames/video)."
        ),
        "dataset": "FaceForensics++",
        "split": "official_test",
        "compression": COMPRESSION,
        "comparison_mode": COMPARISON_MODE,
        "comparison_rationale": COMPARISON_RATIONALE,
        "script_version": SCRIPT_VERSION,
        "generated_at": utc_timestamp(),
        "dataset_root": str(root),
        "official_split_path": str(split_file.resolve()),
        "frames_per_video_target": frames_per_video,
        "frame_sampling_strategy": sampling_desc,
        "subset_selection": subset_dict,
        "expected_sanity_counts": expected_sanity_counts(),
        "expected_full_split_counts": full_counts,
        "n_videos": len(videos),
        "n_real_videos": n_real_v,
        "n_fake_videos": n_fake_v,
        "n_videos_by_manipulation": by_manip,
        "n_probe_failed_videos": n_probe_failed,
        "n_frame_records": len(records),
        "n_usable_frames": n_usable,
        "n_failed_frames": n_failed,
        "dry_run": dry_run,
        "skip_crop": skip_crop,
        "frames": records,
    }

    written: Dict[str, str] = {}
    if not dry_run:
        written["preparation_manifest"] = str(
            write_json(out / "preparation_manifest.json", manifest)
        )
        runner_in = build_runner_manifest(records, description_root=out)
        written["infer_input"] = str(write_json(out / "infer_input.json", runner_in))
        prov = build_provenance(
            dataset_root=root,
            split_path=split_file,
            output_dir=out,
            quantisation="not_run_yet",
            experts="blending,diffusion_detector",
            adapter_path="weights/checkpoints/ckpt/llava-v1.5-7b-lora-[ble-diff]",
            base_path="weights/base/llava-v1.5-7b",
            load_4bit=False,
            frames_per_video=frames_per_video,
            subset=subset_dict,
            extra={"stage": "preparation", "n_usable_frames": n_usable},
        )
        written["provenance"] = str(write_json(out / "preparation_provenance.json", prov))
    else:
        written["dry_run_summary"] = str(
            write_json(out / "preparation_dry_run.json", manifest)
        )

    return {
        "output_dir": str(out.resolve()),
        "n_videos": len(videos),
        "n_real_videos": n_real_v,
        "n_fake_videos": n_fake_v,
        "n_usable_frames": n_usable,
        "n_failed_frames": n_failed,
        "subset_selection": subset_dict,
        "expected_full_split_counts": full_counts,
        "written": written,
        "dry_run": dry_run,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m tools.prepare_ffpp_source_domain")
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--split-json", type=Path, default=None)
    p.add_argument(
        "--full-test",
        action="store_true",
        help="use all official held-out test videos (default: 100-video sanity subset)",
    )
    p.add_argument(
        "--frames-per-video",
        type=int,
        default=None,
        help=(
            f"frames per video (default: {DEFAULT_FRAMES_PER_VIDEO} sanity, "
            f"{FULL_PROTOCOL_FRAMES_PER_VIDEO} with --full-test)"
        ),
    )
    p.add_argument("--subset-seed", type=int, default=DEFAULT_SUBSET_SEED)
    p.add_argument("--n-real", type=int, default=DEFAULT_N_REAL_VIDEOS)
    p.add_argument("--n-fake", type=int, default=DEFAULT_N_FAKE_VIDEOS)
    p.add_argument("--max-videos", type=int, default=None, help="optional cap after selection")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-crop", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    try:
        result = prepare(
            dataset_root=args.dataset_root,
            output_dir=args.output_dir,
            split_path=args.split_json,
            dry_run=args.dry_run,
            max_videos=args.max_videos,
            skip_crop=args.skip_crop,
            full_test=args.full_test,
            frames_per_video=args.frames_per_video,
            subset_seed=args.subset_seed,
            n_real=args.n_real,
            n_fake=args.n_fake,
        )
    except (FFPPSourceError, OSError) as exc:
        LOGGER.error("%s", exc)
        return EXIT_USAGE
    print(json.dumps(result, indent=2))
    return EXIT_PASS


if __name__ == "__main__":
    raise SystemExit(main())
