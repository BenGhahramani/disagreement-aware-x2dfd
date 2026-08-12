"""Unit tests for dashboard/saved_examples.py — saved demo selector wiring."""
from __future__ import annotations

from pathlib import Path

import pytest

from dashboard.saved_examples import (
    FFPP_LIVE_RUN,
    SAVED_EXAMPLES,
    default_saved_example_id,
    load_saved_example,
    resolve_saved_example,
    saved_example_ids,
    saved_examples_by_id,
)
from dashboard.view_model import DEFAULT_MATRIX_DIR, DEFAULT_SUMMARY
from proof_of_concept.schema import Status

pytestmark = pytest.mark.unit


def test_saved_examples_registry_lists_nasa_then_ffpp() -> None:
    ids = saved_example_ids()
    assert ids == ("nasa", "ffpp_deepfakes")
    assert default_saved_example_id() == "nasa"
    lookup = saved_examples_by_id()
    assert set(lookup) == {"nasa", "ffpp_deepfakes"}


def test_resolve_saved_example_defaults_to_nasa() -> None:
    spec = resolve_saved_example(None)
    assert spec.example_id == "nasa"
    assert spec.matrix_dir == DEFAULT_MATRIX_DIR
    assert spec.summary_path == DEFAULT_SUMMARY
    assert spec.ground_truth_label == "Known authentic"


def test_resolve_saved_example_ffpp_paths() -> None:
    spec = resolve_saved_example("ffpp_deepfakes")
    assert spec.matrix_dir == FFPP_LIVE_RUN / "matrix"
    assert spec.summary_path == FFPP_LIVE_RUN / "summary.json"
    assert spec.ground_truth_label == "Known manipulated"
    assert "FaceForensics++" in spec.provenance_summary
    assert "Deepfakes" in spec.provenance_summary


def test_resolve_unknown_example_raises() -> None:
    with pytest.raises(KeyError, match="unknown saved example"):
        resolve_saved_example("missing")


@pytest.mark.skipif(
    not DEFAULT_MATRIX_DIR.is_dir(),
    reason="Stage 3 matrix outputs not present on this machine",
)
def test_load_nasa_saved_example_uses_stage3_outputs() -> None:
    view = load_saved_example("nasa")
    assert view.matrix_dir.resolve() == DEFAULT_MATRIX_DIR.resolve()
    assert view.summary_path is not None
    assert view.summary_path.resolve() == DEFAULT_SUMMARY.resolve()
    assert view.status is Status.UNCERTAIN
    assert view.image_path is not None
    assert view.image_path.name == "real_face_01_crop.jpg"
    combined = next(c for c in view.cards if c.run_name == "blending_diffusion")
    assert combined.expert_scores.get("blending") == pytest.approx(0.009)
    assert combined.expert_scores.get("diffusion") == pytest.approx(0.048)


@pytest.mark.skipif(
    not (FFPP_LIVE_RUN / "matrix").is_dir(),
    reason="FF++ live-analysis outputs not present on this machine",
)
def test_load_ffpp_saved_example_uses_live_analysis_outputs() -> None:
    view = load_saved_example("ffpp_deepfakes")
    assert view.matrix_dir.resolve() == (FFPP_LIVE_RUN / "matrix").resolve()
    assert view.summary_path is not None
    assert view.summary_path.resolve() == (FFPP_LIVE_RUN / "summary.json").resolve()
    assert view.status is Status.STABLE
    assert view.image_path is not None
    assert view.image_path.name == "face_crop.jpg"
    combined = next(c for c in view.cards if c.run_name == "blending_diffusion")
    assert combined.expert_scores.get("blending") == pytest.approx(1.0)
    assert combined.expert_scores.get("diffusion") == pytest.approx(0.969)
    assert view.evidence_conflicts == []


def test_load_saved_example_with_custom_fixture(tmp_path: Path) -> None:
    from tests.test_dashboard_view_model import _write_matrix

    directory = _write_matrix(tmp_path)
    summary = tmp_path / "summary.json"
    summary.write_text('{"quantisation": "4-bit", "runs": []}', encoding="utf-8")
    from dashboard.saved_examples import SavedExampleSpec

    custom = (
        SavedExampleSpec(
            example_id="custom",
            selector_label="Custom",
            matrix_dir=directory,
            summary_path=summary,
            ground_truth_label="Known authentic",
            provenance_summary="fixture",
        ),
    )
    view = load_saved_example("custom", examples=custom)
    assert view.matrix_dir.resolve() == directory.resolve()
    assert view.quantisation == "4-bit"
