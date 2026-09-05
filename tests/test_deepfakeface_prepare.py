"""Unit tests for DeepFakeFace evaluation subset preparation (no real ZIPs)."""
from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest

from eval.deepfakeface_prepare import (
    CATEGORY_SPECS,
    DEFAULT_SEED,
    DeepFakeFacePrepareError,
    UsabilityVerdict,
    image_output_relpath,
    list_image_members,
    make_image_id,
    prepare_samples,
    run_preparation,
    select_primary_and_reserve,
    sniff_output_suffix,
    validate_prepared,
    ZipMemberCandidate,
)
from eval.labelled_manifest import load_labelled_manifest

pytestmark = pytest.mark.unit



def _png_bytes(color: tuple[int, int, int] = (40, 80, 120), size: int = 64) -> bytes:
    import cv2

    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:, :] = color
    ok, encoded = cv2.imencode(".png", img)
    assert ok
    return encoded.tobytes()


def _jpeg_bytes(color: tuple[int, int, int] = (40, 80, 120), size: int = 64) -> bytes:
    import cv2

    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:, :] = color
    ok, encoded = cv2.imencode(".jpg", img)
    assert ok
    return encoded.tobytes()


def _write_category_zip(path: Path, *, root: str, n: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr(f"{root}/", "")
        zf.writestr(f"{root}/00/", "")
        for i in range(n):
            name = f"{root}/00/person{i:04d}_1990-01-01_2010.jpg"
            zf.writestr(name, _jpeg_bytes(color=((i * 3) % 255, (80 + i) % 200, 100)))
    return path


def _fake_dataset(tmp_path: Path, *, n_per: int = 8) -> Path:
    root = tmp_path / "DeepFakeFace"
    root.mkdir()
    for spec in CATEGORY_SPECS:
        cat = spec["category"]
        _write_category_zip(root / spec["zip_name"], root=cat, n=n_per)
    return root


def _always_usable(raw: bytes) -> UsabilityVerdict:
    return UsabilityVerdict(
        usable=True,
        rejection_reason=None,
        width=64,
        height=64,
        face_count=1,
        face_detection_xywh=[8, 8, 32, 32],
        face_inside_bounds=True,
    )


def _reject_then_accept(raw: bytes, *, reject_once: Dict[str, int]) -> UsabilityVerdict:
    # First call for any bytes hash rejects once then accepts — keyed by content.
    key = str(len(raw)) + raw[:8].hex()
    count = reject_once.get(key, 0)
    if count == 0:
        reject_once[key] = 1
        return UsabilityVerdict(usable=False, rejection_reason="no_face", width=64, height=64)
    return _always_usable(raw)


def test_list_image_members_sorted(tmp_path: Path) -> None:
    zpath = _write_category_zip(tmp_path / "wiki.zip", root="wiki", n=3)
    names = list_image_members(zpath)
    assert len(names) == 3
    assert names == sorted(names)


def test_select_primary_deterministic() -> None:
    members = [f"m{i:03d}.jpg" for i in range(20)]
    import random

    a, ra = select_primary_and_reserve(members, n=5, rng=random.Random(DEFAULT_SEED))
    b, rb = select_primary_and_reserve(members, n=5, rng=random.Random(DEFAULT_SEED))
    assert a == b
    assert ra == rb
    assert len(set(a) & set(ra)) == 0


def test_prepare_with_injected_usability(tmp_path: Path) -> None:
    root = _fake_dataset(tmp_path, n_per=10)
    out = tmp_path / "out"
    # Shrink quotas via monkeypatch of CATEGORY_SPECS is awkward; instead call
    # prepare_samples after temporarily reducing — use run with custom fill by
    # preparing only through lower-level with patched specs.
    from eval import deepfakeface_prepare as mod

    small_specs = tuple(
        {**spec, "n": 2 if spec["category"] == "wiki" else 1} for spec in CATEGORY_SPECS
    )
    original = mod.CATEGORY_SPECS
    try:
        mod.CATEGORY_SPECS = small_specs  # type: ignore[misc]
        samples, meta = prepare_samples(
            root,
            out,
            seed=DEFAULT_SEED,
            dry_run=False,
            evaluate_fn=_always_usable,
        )
        # validate_prepared expects 120/120 — call lightweight checks instead
        assert len(samples) == 5
        assert meta["seed"] == DEFAULT_SEED
        ids = [s.image_id for s in samples]
        assert len(ids) == len(set(ids))
        for s in samples:
            assert s.image_abs_path is not None and s.image_abs_path.is_file()
            assert s.image_sha256 == s.member_sha256
    finally:
        mod.CATEGORY_SPECS = original  # type: ignore[misc]


def test_replacement_on_unusable(tmp_path: Path) -> None:
    root = _fake_dataset(tmp_path, n_per=6)
    out = tmp_path / "out"
    from eval import deepfakeface_prepare as mod

    small_specs = tuple(
        {**spec, "n": 2 if spec["category"] == "wiki" else 1} for spec in CATEGORY_SPECS
    )
    reject_state: Dict[str, int] = {}

    def eval_fn(raw: bytes) -> UsabilityVerdict:
        # Reject every image once when first seen in this category run by using
        # a counter: first two wiki primary rejects, then accept — simpler:
        return _always_usable(raw)

    # Force first wiki primary member to fail by wrapping evaluate.
    original = mod.CATEGORY_SPECS
    calls = {"n": 0}

    def flaky(raw: bytes) -> UsabilityVerdict:
        calls["n"] += 1
        if calls["n"] == 1:
            return UsabilityVerdict(
                usable=False, rejection_reason="no_face", width=64, height=64, face_count=0
            )
        return _always_usable(raw)

    try:
        mod.CATEGORY_SPECS = small_specs  # type: ignore[misc]
        samples, meta = prepare_samples(
            root, out, seed=1, dry_run=False, evaluate_fn=flaky
        )
        assert len(meta["replacements"]) >= 1
        assert len(samples) == 5
    finally:
        mod.CATEGORY_SPECS = original  # type: ignore[misc]


def test_manifest_loads_with_labelled_parser(tmp_path: Path) -> None:
    root = _fake_dataset(tmp_path, n_per=10)
    out = tmp_path / "out"
    from eval import deepfakeface_prepare as mod

    small_specs = tuple(
        {**spec, "n": 2 if spec["category"] == "wiki" else 1} for spec in CATEGORY_SPECS
    )
    original = mod.CATEGORY_SPECS
    try:
        mod.CATEGORY_SPECS = small_specs  # type: ignore[misc]
        # Bypass validate_prepared 120/120 by writing manually after prepare_samples
        samples, meta = prepare_samples(
            root, out, seed=DEFAULT_SEED, dry_run=False, evaluate_fn=_always_usable
        )
        from eval.deepfakeface_prepare import write_outputs

        # Temporarily relax validate by writing outputs directly
        written = write_outputs(
            samples,
            output_dir=out,
            selection_meta=meta,
            dataset_root=root,
            dry_run=False,
        )
        images = load_labelled_manifest(written["manifest"])
        assert len(images) == 5
        assert all(im.already_cropped is False for im in images)
        assert all(im.dataset == "DeepFakeFace" for im in images)
    finally:
        mod.CATEGORY_SPECS = original  # type: ignore[misc]


def test_make_image_id_stable() -> None:
    from eval.deepfakeface_prepare import ZipMemberCandidate

    c = ZipMemberCandidate(
        category="wiki",
        zip_name="wiki.zip",
        member_name="wiki/00/abc_1990-01-01_2010.jpg",
        ground_truth="real",
        manipulation=None,
    )
    assert make_image_id(c) == "dff_wiki_abc_1990-01-01_2010"

def test_sniff_suffix_png_mislabeled_as_jpg() -> None:
    png = _png_bytes()
    assert sniff_output_suffix(png) == ".png"
    assert sniff_output_suffix(_jpeg_bytes()) == ".jpg"
    assert sniff_output_suffix(b"not-an-image") is None
    cand = ZipMemberCandidate(
        category="wiki",
        zip_name="wiki.zip",
        member_name="wiki/00/person_png_named.jpg",
        ground_truth="real",
        manipulation=None,
    )
    rel = image_output_relpath(cand, raw=png)
    assert rel.endswith(".png")
    assert ".jpg" not in Path(rel).suffix


def test_prepare_mislabeled_png_writes_png_suffix(tmp_path: Path) -> None:
    root = tmp_path / "DeepFakeFace"
    root.mkdir()
    # Minimal one-category zip with a PNG stored under .jpg name.
    zpath = root / "wiki.zip"
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("wiki/00/mislabeled.jpg", _png_bytes())
        for i in range(5):
            zf.writestr(f"wiki/00/ok{i:04d}.jpg", _jpeg_bytes(color=(i, 50, 90)))
    for spec in CATEGORY_SPECS:
        if spec["category"] == "wiki":
            continue
        _write_category_zip(root / spec["zip_name"], root=spec["category"], n=6)

    from eval import deepfakeface_prepare as mod

    small_specs = tuple(
        {**spec, "n": 2 if spec["category"] == "wiki" else 1} for spec in CATEGORY_SPECS
    )
    original = mod.CATEGORY_SPECS
    out = tmp_path / "out"
    try:
        mod.CATEGORY_SPECS = small_specs  # type: ignore[misc]
        samples, _meta = prepare_samples(
            root,
            out,
            seed=DEFAULT_SEED,
            dry_run=False,
            evaluate_fn=_always_usable,
        )
    finally:
        mod.CATEGORY_SPECS = original  # type: ignore[misc]

    png_samples = [s for s in samples if s.image_rel_path.endswith(".png")]
    # Depending on RNG, mislabeled may or may not be selected; force-check helper path
    # and ensure any written .jpg files are real JPEGs.
    for s in samples:
        path = out / s.image_rel_path
        assert path.is_file()
        header = path.read_bytes()[:8]
        if s.image_rel_path.endswith(".jpg"):
            assert header.startswith(b"\xff\xd8\xff")
        if s.image_rel_path.endswith(".png"):
            assert header.startswith(b"\x89PNG\r\n\x1a\n")
    # Explicit extract of mislabeled member path logic
    cand = ZipMemberCandidate(
        category="wiki",
        zip_name="wiki.zip",
        member_name="wiki/00/mislabeled.jpg",
        ground_truth="real",
        manipulation=None,
    )
    assert image_output_relpath(cand, raw=_png_bytes()).endswith(".png")

