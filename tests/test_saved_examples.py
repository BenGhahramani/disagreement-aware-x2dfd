"""Unit tests for dashboard/saved_examples.py — saved demo selector wiring."""
from __future__ import annotations

from pathlib import Path

import pytest

from dashboard.saved_examples import (
    FFPP_EXPECTED_IMAGE,
    FFPP_LIVE_RUN,
    NASA_EXPECTED_IMAGE,
    SavedExampleSpec,
    default_saved_example_id,
    load_saved_example,
    resolve_saved_example,
    saved_example_ids,
    saved_examples_by_id,
)
from dashboard.view_model import (
    DEFAULT_IMAGE,
    DEFAULT_MATRIX_DIR,
    DEFAULT_SUMMARY,
    EvidenceAgreement,
    PROJECT_ROOT,
    build_dashboard_view,
)
from proof_of_concept.schema import Status

pytestmark = pytest.mark.unit

_NASA_READY = DEFAULT_MATRIX_DIR.is_dir() and NASA_EXPECTED_IMAGE.is_file()
_FFPP_READY = (FFPP_LIVE_RUN / "matrix").is_dir() and FFPP_EXPECTED_IMAGE.is_file()


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
    assert spec.expected_image == NASA_EXPECTED_IMAGE


def test_resolve_saved_example_ffpp_paths() -> None:
    spec = resolve_saved_example("ffpp_deepfakes")
    assert spec.matrix_dir == FFPP_LIVE_RUN / "matrix"
    assert spec.summary_path == FFPP_LIVE_RUN / "summary.json"
    assert spec.ground_truth_label == "Known manipulated"
    assert spec.expected_image == FFPP_EXPECTED_IMAGE
    assert "FaceForensics++" in spec.provenance_summary
    assert "Deepfakes" in spec.provenance_summary


def test_saved_examples_declare_distinct_portable_images() -> None:
    nasa = resolve_saved_example("nasa")
    ffpp = resolve_saved_example("ffpp_deepfakes")
    assert nasa.expected_image is not None
    assert ffpp.expected_image is not None
    assert nasa.expected_image.resolve() != ffpp.expected_image.resolve()
    assert nasa.expected_image == PROJECT_ROOT / (
        "datasets/raw/images/poc/real_face_01_crop.jpg"
    )
    assert ffpp.expected_image == PROJECT_ROOT / (
        "eval/outputs/live_analysis/20260808T085812Z_ffpp_ex_deepfakes/face_crop.jpg"
    )


def test_resolve_unknown_example_raises() -> None:
    with pytest.raises(KeyError, match="unknown saved example"):
        resolve_saved_example("missing")


@pytest.mark.skipif(
    not _NASA_READY,
    reason="Stage 3 matrix outputs / NASA image not present on this machine",
)
def test_load_nasa_saved_example_uses_stage3_outputs() -> None:
    view = load_saved_example("nasa")
    assert view.matrix_dir.resolve() == DEFAULT_MATRIX_DIR.resolve()
    assert view.summary_path is not None
    assert view.summary_path.resolve() == DEFAULT_SUMMARY.resolve()
    assert view.status is Status.UNCERTAIN
    assert view.image_path is not None
    assert view.image_path.resolve() == NASA_EXPECTED_IMAGE.resolve()
    combined = next(c for c in view.cards if c.run_name == "blending_diffusion")
    assert combined.expert_scores.get("blending") == pytest.approx(0.009)
    assert combined.expert_scores.get("diffusion") == pytest.approx(0.048)


@pytest.mark.skipif(
    not _FFPP_READY,
    reason="FF++ live-analysis outputs / face crop not present on this machine",
)
def test_load_ffpp_saved_example_uses_live_analysis_outputs() -> None:
    view = load_saved_example("ffpp_deepfakes")
    assert view.matrix_dir.resolve() == (FFPP_LIVE_RUN / "matrix").resolve()
    assert view.summary_path is not None
    assert view.summary_path.resolve() == (FFPP_LIVE_RUN / "summary.json").resolve()
    assert view.status is Status.STABLE
    assert view.image_path is not None
    assert view.image_path.resolve() == FFPP_EXPECTED_IMAGE.resolve()
    combined = next(c for c in view.cards if c.run_name == "blending_diffusion")
    assert combined.expert_scores.get("blending") == pytest.approx(1.0)
    assert combined.expert_scores.get("diffusion") == pytest.approx(0.969)
    assert view.evidence_conflicts == []


@pytest.mark.skipif(
    not (_NASA_READY and _FFPP_READY),
    reason="Both saved-example image assets required",
)
def test_saved_examples_resolve_distinct_images_despite_stale_windows_paths() -> None:
    """Fresh-clone portability: stale Ben-machine absolute paths must not share NASA."""

    nasa = load_saved_example("nasa")
    ffpp = load_saved_example("ffpp_deepfakes")

    assert nasa.image_path is not None
    assert ffpp.image_path is not None
    assert nasa.image_path.resolve() == NASA_EXPECTED_IMAGE.resolve()
    assert ffpp.image_path.resolve() == FFPP_EXPECTED_IMAGE.resolve()
    assert nasa.image_path.resolve() != ffpp.image_path.resolve()
    assert ffpp.image_path.resolve() != DEFAULT_IMAGE.resolve()
    assert not any("Showing fallback" in w for w in ffpp.warnings)
    # Successful portable recovery must not surface a missing-image warning.
    for view in (nasa, ffpp):
        assert not any("not found" in w.lower() for w in view.warnings)
        assert not any("Using saved-example image" in w for w in view.warnings)


@pytest.mark.skipif(
    not (_NASA_READY and _FFPP_READY),
    reason="Both saved-example image assets required",
)
def test_ffpp_never_silently_falls_back_to_nasa_image() -> None:
    view = load_saved_example("ffpp_deepfakes")
    assert view.image_path is not None
    assert view.image_path.name == "face_crop.jpg"
    assert view.image_path.resolve() != NASA_EXPECTED_IMAGE.resolve()
    assert view.image_path.resolve() != DEFAULT_IMAGE.resolve()


@pytest.mark.skipif(
    not (_NASA_READY and _FFPP_READY),
    reason="Both saved examples required to lock scores/status",
)
def test_saved_example_scores_and_evidence_unchanged_with_image_fix() -> None:
    nasa = load_saved_example("nasa")
    ffpp = load_saved_example("ffpp_deepfakes")

    assert nasa.status is Status.UNCERTAIN
    assert nasa.evidence_agreement is EvidenceAgreement.CONFLICT
    assert nasa.model_assessment_label == "fake"
    assert nasa.model_assessment_score == pytest.approx(0.686, abs=0.01)

    assert ffpp.status is Status.STABLE
    assert ffpp.evidence_agreement is EvidenceAgreement.AGREEMENT
    assert ffpp.model_assessment_label == "fake"
    assert ffpp.model_assessment_score == pytest.approx(0.996, abs=0.01)
    assert ffpp.evidence_conflicts == []


def test_missing_expected_image_does_not_substitute_nasa(tmp_path: Path) -> None:
    from tests.test_dashboard_view_model import _write_matrix

    missing = tmp_path / "missing_face_crop.jpg"
    # Stale path must not remap onto a real repo file (no filename search).
    stale = Path(
        r"C:\Users\Ben\Desktop\UNI\REIT\root\disagreement-aware-x2dfd"
        r"\eval\outputs\live_analysis\nonexistent_run_zzzz\face_crop.jpg"
    )
    directory = _write_matrix(tmp_path, image=stale)
    summary = tmp_path / "summary.json"
    summary.write_text('{"quantisation": "4-bit", "runs": []}', encoding="utf-8")
    custom = (
        SavedExampleSpec(
            example_id="orphan_ffpp",
            selector_label="Orphan FF++",
            matrix_dir=directory,
            summary_path=summary,
            ground_truth_label="Known manipulated",
            provenance_summary="fixture missing image",
            expected_image=missing,
        ),
    )
    view = load_saved_example("orphan_ffpp", examples=custom)
    assert view.image_path is None
    assert view.warnings
    assert any("not found" in w.lower() or "missing" in w.lower() for w in view.warnings)
    assert "Showing fallback" not in " ".join(view.warnings)
    assert DEFAULT_IMAGE.name not in " ".join(view.warnings)


def test_expected_image_wins_over_stale_absolute_path(tmp_path: Path) -> None:
    from tests.test_dashboard_view_model import _write_matrix

    expected = tmp_path / "correct_crop.jpg"
    expected.write_bytes(b"\xff\xd8\xff\xe0correct")
    stale = Path(
        r"C:\Users\Ben\Desktop\UNI\REIT\root\disagreement-aware-x2dfd"
        r"\datasets\raw\images\poc\real_face_01_crop.jpg"
    )
    directory = _write_matrix(tmp_path, image=stale)
    view = build_dashboard_view(
        directory,
        summary_path=None,
        expected_image=expected,
    )
    assert view.image_path is not None
    assert view.image_path.resolve() == expected.resolve()
    assert view.image_path.resolve() != DEFAULT_IMAGE.resolve()
    assert not any("not found" in w.lower() for w in view.warnings)
    assert not any("Using saved-example image" in w for w in view.warnings)


def test_load_saved_example_with_custom_fixture(tmp_path: Path) -> None:
    from tests.test_dashboard_view_model import _write_matrix

    directory = _write_matrix(tmp_path)
    summary = tmp_path / "summary.json"
    summary.write_text('{"quantisation": "4-bit", "runs": []}', encoding="utf-8")

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
