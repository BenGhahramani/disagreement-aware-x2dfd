"""Unit tests for Celeb-DF-v2 evaluation subset preparation (no real dataset)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

from eval.celebdf_v2_prepare import (
    CANDIDATE_FRACTIONS,
    DEFAULT_SEED,
    MIN_LAPLACIAN_VARIANCE,
    CelebDFPrepareError,
    CelebDFTestVideo,
    FrameExtraction,
    PreparedSample,
    UnusableVideoError,
    candidate_frame_indices,
    evaluate_frame_usability,
    face_sufficiently_inside,
    fill_class_with_usable_frames,
    locate_test_list,
    make_image_id,
    parse_test_list,
    prepare_samples,
    sample_to_manifest_entry,
    select_balanced_videos,
    validate_prepared,
    write_outputs,
)
from eval.labelled_manifest import load_labelled_manifest
from tools.make_face_crop import CropPlan

pytestmark = pytest.mark.unit


def _write_tiny_mp4(path: Path, *, n_frames: int = 10, size: int = 64) -> Path:
    """Write a tiny synthetic MP4 (no real faces — tests inject select_fn)."""

    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        5.0,
        (size, size),
    )
    assert writer.isOpened(), "VideoWriter failed to open (OpenCV build issue)"
    try:
        for i in range(n_frames):
            frame = np.full((size, size, 3), (i * 20) % 255, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()
    assert path.is_file() and path.stat().st_size > 0
    return path


def _fake_dataset(tmp_path: Path, *, n_real: int = 4, n_fake: int = 4) -> Path:
    root = tmp_path / "Celeb-DF"
    (root / "Celeb-real").mkdir(parents=True)
    (root / "YouTube-real").mkdir(parents=True)
    (root / "Celeb-synthesis").mkdir(parents=True)
    lines: List[str] = []
    for i in range(n_real):
        if i % 2 == 0:
            rel = f"YouTube-real/{i:05d}.mp4"
        else:
            rel = f"Celeb-real/id0_{i:04d}.mp4"
        _write_tiny_mp4(root / rel, n_frames=12 + i)
        lines.append(f"1 {rel}")
    for i in range(n_fake):
        rel = f"Celeb-synthesis/id0_id1_{i:04d}.mp4"
        _write_tiny_mp4(root / rel, n_frames=14 + i)
        lines.append(f"0 {rel}")
    (root / "List_of_testing_videos.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return root


def _always_usable_select(path: Path) -> Tuple[Any, FrameExtraction]:
    """Deterministic select_fn that accepts the 50% candidate without Haar."""

    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    frame[:, :] = (40, 80, 120)
    return frame, FrameExtraction(
        frame_index=5,
        total_frames=11,
        fps=5.0,
        timestamp_s=1.0,
        width=32,
        height=32,
        candidate_fraction=0.50,
        candidates=[],
        blur_laplacian_var=500.0,
        face_detection_xywh=[8, 8, 16, 16],
    )


def _fail_then_ok_factory(fail_paths: set[str]):
    def _select(path: Path) -> Tuple[Any, FrameExtraction]:
        if path.as_posix().replace("\\", "/") in fail_paths or any(
            p in path.as_posix().replace("\\", "/") for p in fail_paths
        ):
            raise UnusableVideoError(
                f"forced unusable: {path}",
                candidates=[],
            )
        return _always_usable_select(path)

    return _select


# --------------------------------------------------------------------------
# test-list parsing / selection
# --------------------------------------------------------------------------


def test_parse_test_list_classifies_official_labels(tmp_path: Path) -> None:
    root = _fake_dataset(tmp_path, n_real=3, n_fake=2)
    videos = parse_test_list(locate_test_list(root), root)
    assert len(videos) == 5
    assert sum(v.ground_truth == "real" for v in videos) == 3
    assert sum(v.ground_truth == "fake" for v in videos) == 2


def test_parse_rejects_label_directory_conflict(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    (root / "Celeb-synthesis").mkdir(parents=True)
    vid = root / "Celeb-synthesis" / "bad.mp4"
    _write_tiny_mp4(vid)
    (root / "List_of_testing_videos.txt").write_text(
        "1 Celeb-synthesis/bad.mp4\n", encoding="utf-8"
    )
    with pytest.raises(CelebDFPrepareError, match="conflicts"):
        parse_test_list(root / "List_of_testing_videos.txt", root)


def test_select_balanced_is_deterministic(tmp_path: Path) -> None:
    root = _fake_dataset(tmp_path, n_real=5, n_fake=5)
    videos = parse_test_list(locate_test_list(root), root)
    a, meta_a = select_balanced_videos(videos, n_real=2, n_fake=2, seed=DEFAULT_SEED)
    b, meta_b = select_balanced_videos(videos, n_real=2, n_fake=2, seed=DEFAULT_SEED)
    assert [v.relative_path for v in a] == [v.relative_path for v in b]
    assert meta_a["selected_real"] == meta_b["selected_real"]
    assert meta_a["selected_fake"] == meta_b["selected_fake"]


def test_select_balanced_changes_with_seed(tmp_path: Path) -> None:
    root = _fake_dataset(tmp_path, n_real=6, n_fake=6)
    videos = parse_test_list(locate_test_list(root), root)
    a, _ = select_balanced_videos(videos, n_real=3, n_fake=3, seed=1)
    b, _ = select_balanced_videos(videos, n_real=3, n_fake=3, seed=2)
    assert [v.relative_path for v in a] != [v.relative_path for v in b]


def test_no_duplicate_videos_across_classes(tmp_path: Path) -> None:
    root = _fake_dataset(tmp_path, n_real=4, n_fake=4)
    videos = parse_test_list(locate_test_list(root), root)
    selected, _ = select_balanced_videos(videos, n_real=2, n_fake=2, seed=7)
    paths = [v.relative_path for v in selected]
    assert len(paths) == len(set(paths))


# --------------------------------------------------------------------------
# usable-frame policy
# --------------------------------------------------------------------------


def test_candidate_fractions_order() -> None:
    assert CANDIDATE_FRACTIONS == (0.50, 0.40, 0.60, 0.30, 0.70)
    pairs = candidate_frame_indices(101)
    assert [f for f, _ in pairs] == list(CANDIDATE_FRACTIONS)
    assert pairs[0] == (0.50, 50)


def test_face_sufficiently_inside() -> None:
    assert face_sufficiently_inside((20, 20, 40, 40), 100, 100) is True
    assert face_sufficiently_inside((0, 20, 40, 40), 100, 100) is False
    assert face_sufficiently_inside((20, 0, 40, 40), 100, 100) is False


def test_no_face_rejection() -> None:
    frame = np.full((128, 128, 3), 127, dtype=np.uint8)

    def no_face_plan(*_a, **_k):
        return None

    verdict = evaluate_frame_usability(frame, plan_crop_fn=no_face_plan)
    assert verdict["usable"] is False
    assert verdict["rejection_reason"] == "no_face"


def test_blurred_frame_rejection() -> None:
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    # Smooth gradient → low Laplacian variance once a face plan is forced.
    for y in range(200):
        frame[y, :, :] = y

    def fake_plan(image_bgr, **_kwargs):
        return CropPlan(detection=(40, 40, 80, 80), square=(30, 30, 100, 100))

    verdict = evaluate_frame_usability(
        frame,
        min_laplacian_var=MIN_LAPLACIAN_VARIANCE,
        plan_crop_fn=fake_plan,
    )
    # Smooth ramp is usually below the documented blur threshold.
    assert verdict["face_inside_bounds"] is True
    assert verdict["usable"] is False
    assert verdict["rejection_reason"] == "severe_blur"
    assert verdict["blur_laplacian_var"] is not None
    assert verdict["blur_laplacian_var"] < MIN_LAPLACIAN_VARIANCE


def test_face_near_border_rejection() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    def edge_plan(image_bgr, **_kwargs):
        return CropPlan(detection=(0, 10, 30, 30), square=(0, 0, 50, 50))

    verdict = evaluate_frame_usability(frame, plan_crop_fn=edge_plan)
    assert verdict["usable"] is False
    assert verdict["rejection_reason"] == "face_near_border"


def test_usable_frame_acceptance() -> None:
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, size=(200, 200, 3), dtype=np.uint8)

    def ok_plan(image_bgr, **_kwargs):
        return CropPlan(detection=(40, 40, 80, 80), square=(30, 30, 100, 100))

    verdict = evaluate_frame_usability(
        frame,
        min_laplacian_var=1.0,  # noise frame is sharp enough
        plan_crop_fn=ok_plan,
    )
    assert verdict["usable"] is True
    assert verdict["rejection_reason"] is None
    assert verdict["face_detection_xywh"] == [40, 40, 80, 80]


# --------------------------------------------------------------------------
# replacement / prepare
# --------------------------------------------------------------------------


def test_deterministic_replacement_keeps_class_counts(tmp_path: Path) -> None:
    root = _fake_dataset(tmp_path, n_real=4, n_fake=4)
    videos = parse_test_list(locate_test_list(root), root)
    selected, meta = select_balanced_videos(videos, n_real=2, n_fake=2, seed=11)
    # Force first primary real video to fail.
    primary_real = [v for v in selected if v.ground_truth == "real"]
    fail_token = primary_real[0].relative_path
    out = tmp_path / "out"
    samples, extra = prepare_samples(
        selected,
        output_dir=out,
        all_videos=videos,
        n_real=2,
        n_fake=2,
        hash_videos=False,
        dry_run=False,
        select_fn=_fail_then_ok_factory({fail_token}),
    )
    validate_prepared(samples, n_real=2, n_fake=2)
    assert len(extra["replacements"]) == 1
    assert extra["replacements"][0]["rejected_video"] == fail_token
    assert fail_token not in [s.video.relative_path for s in samples]
    assert all(s.frame_abs_path and s.frame_abs_path.is_file() for s in samples)


def test_replacement_reproducible(tmp_path: Path) -> None:
    root = _fake_dataset(tmp_path, n_real=5, n_fake=5)
    videos = parse_test_list(locate_test_list(root), root)
    selected, _ = select_balanced_videos(videos, n_real=2, n_fake=2, seed=3)
    fail = [v for v in selected if v.ground_truth == "fake"][0].relative_path
    select_fn = _fail_then_ok_factory({fail})

    a, extra_a = prepare_samples(
        selected,
        output_dir=tmp_path / "a",
        all_videos=videos,
        n_real=2,
        n_fake=2,
        hash_videos=False,
        select_fn=select_fn,
    )
    b, extra_b = prepare_samples(
        selected,
        output_dir=tmp_path / "b",
        all_videos=videos,
        n_real=2,
        n_fake=2,
        hash_videos=False,
        select_fn=select_fn,
    )
    assert [s.video.relative_path for s in a] == [s.video.relative_path for s in b]
    assert extra_a["replacements"] == extra_b["replacements"]


def test_prepare_samples_writes_png_and_unique_ids(tmp_path: Path) -> None:
    root = _fake_dataset(tmp_path, n_real=3, n_fake=3)
    videos = parse_test_list(locate_test_list(root), root)
    selected, _ = select_balanced_videos(videos, n_real=2, n_fake=2, seed=11)
    out = tmp_path / "out"
    samples, _ = prepare_samples(
        selected,
        output_dir=out,
        all_videos=videos,
        n_real=2,
        n_fake=2,
        hash_videos=False,
        select_fn=_always_usable_select,
    )
    validate_prepared(samples, n_real=2, n_fake=2, dry_run=False)
    assert len({s.image_id for s in samples}) == 4


def test_fill_class_raises_when_pool_exhausted(tmp_path: Path) -> None:
    root = _fake_dataset(tmp_path, n_real=2, n_fake=1)
    videos = parse_test_list(locate_test_list(root), root)
    reals = [v for v in videos if v.ground_truth == "real"]

    def always_fail(_path: Path) -> Tuple[Any, FrameExtraction]:
        raise UnusableVideoError("nope", candidates=[])

    with pytest.raises(CelebDFPrepareError, match="could not obtain"):
        fill_class_with_usable_frames(
            ground_truth="real",
            primary=reals[:1],
            reserve=reals[1:],
            n_required=2,
            output_dir=tmp_path / "out",
            hash_videos=False,
            dry_run=False,
            select_fn=always_fail,
        )


# --------------------------------------------------------------------------
# manifest generation
# --------------------------------------------------------------------------


def test_manifest_loads_via_labelled_manifest_parser(tmp_path: Path) -> None:
    root = _fake_dataset(tmp_path, n_real=3, n_fake=3)
    videos = parse_test_list(locate_test_list(root), root)
    selected, meta = select_balanced_videos(videos, n_real=2, n_fake=2, seed=3)
    out = tmp_path / "celebdf_v2_final"
    samples, extra = prepare_samples(
        selected,
        output_dir=out,
        all_videos=videos,
        n_real=2,
        n_fake=2,
        hash_videos=False,
        select_fn=_always_usable_select,
    )
    validate_prepared(samples, n_real=2, n_fake=2)
    meta = {**meta, **extra}
    written = write_outputs(
        samples,
        output_dir=out,
        selection_meta=meta,
        test_list_path=locate_test_list(root),
        dataset_root=root,
        dry_run=False,
    )
    loaded = load_labelled_manifest(written["manifest"])
    assert len(loaded) == 4
    assert all(img.already_cropped is False for img in loaded)

    prov = json.loads(written["provenance"].read_text(encoding="utf-8"))
    assert prov["usability_policy"]["candidate_fractions"] == list(CANDIDATE_FRACTIONS)
    assert "samples" in prov


def test_sample_to_manifest_entry_stable_id() -> None:
    video = CelebDFTestVideo(
        relative_path="YouTube-real/00170.mp4",
        ground_truth="real",
        label_code=1,
        absolute_path=Path("YouTube-real/00170.mp4"),
    )
    assert make_image_id(video) == "celebdf_v2_real_YouTube-real__00170"
    entry = sample_to_manifest_entry(
        PreparedSample(
            image_id=make_image_id(video),
            video=video,
            frame_rel_path="frames/real/YouTube-real__00170.png",
            frame_abs_path=None,
            extraction=None,
            video_sha256=None,
            frame_sha256=None,
            notes="test",
        )
    )
    assert entry["already_cropped"] is False
    assert entry["dataset"] == "Celeb-DF-v2"


def test_dry_run_skips_frames(tmp_path: Path) -> None:
    root = _fake_dataset(tmp_path, n_real=3, n_fake=3)
    videos = parse_test_list(locate_test_list(root), root)
    selected, meta = select_balanced_videos(videos, n_real=1, n_fake=1, seed=5)
    out = tmp_path / "dry"
    samples, extra = prepare_samples(
        selected,
        output_dir=out,
        all_videos=videos,
        n_real=1,
        n_fake=1,
        dry_run=True,
        select_fn=_always_usable_select,
    )
    validate_prepared(samples, n_real=1, n_fake=1, dry_run=True)
    assert not list(out.glob("frames/**/*.png"))
    written = write_outputs(
        samples,
        output_dir=out,
        selection_meta={**meta, **extra},
        test_list_path=locate_test_list(root),
        dataset_root=root,
        dry_run=True,
    )
    assert written["provenance"].name.endswith("dry_run.json")
