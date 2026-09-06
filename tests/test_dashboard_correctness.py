"""Systematic dashboard correctness + robustness tests (no GPU / no inference)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from dashboard.saved_examples import load_saved_example
from dashboard.view_model import (
    CALIBRATION_NOTE,
    EXPERT_HI,
    EXPERT_LO,
    LABEL_BLENDING_DETECTOR,
    LABEL_DIFFUSION_DETECTOR,
    LABEL_MODEL_FAKE,
    LABEL_MODEL_REAL,
    EvidenceAgreement,
    build_dashboard_view,
    display_rationale,
)
from eval.dashboard_correctness import (
    FFPP_MATRIX,
    MAE_MATRIX,
    NASA_JONNY_MATRIX,
    check_agreement_boundaries,
    check_malformed_robustness,
    check_prototype_status_boundaries,
    check_representative_cases,
    check_saved_vs_live_consistency,
    check_score_fidelity,
    check_score_semantics,
    dashboard_facing_snapshot,
    read_demo_json_scores,
    run_all_checks,
)
from eval.experiment_configs import RUN_ORDER
from proof_of_concept.evaluator import CONF_HIGH
from proof_of_concept.schema import Status

pytestmark = pytest.mark.unit


def _assert_all_pass(checks, *, allow_skip: bool = False) -> None:
    for check in checks:
        if check.skipped:
            if allow_skip:
                continue
            pytest.skip(check.detail or check.name)
        assert check.passed, f"{check.category}/{check.name}: {check.detail}"


# --------------------------------------------------------------------------
# 1–2 score fidelity + semantics
# --------------------------------------------------------------------------


def test_score_semantics_labels_are_not_calibrated_probabilities() -> None:
    _assert_all_pass(check_score_semantics())
    blob = " ".join(
        [
            LABEL_MODEL_REAL,
            LABEL_MODEL_FAKE,
            LABEL_BLENDING_DETECTOR,
            LABEL_DIFFUSION_DETECTOR,
            CALIBRATION_NOTE,
            display_rationale("combined-experts confidence 0.69"),
        ]
    ).lower()
    assert "calibrated probability" not in blob
    assert "calibrated confidence" not in blob
    assert "score" in LABEL_MODEL_REAL.lower()
    assert "detector score" in LABEL_BLENDING_DETECTOR.lower()


@pytest.mark.skipif(not NASA_JONNY_MATRIX.is_dir(), reason="Stage 3 NASA matrix absent")
def test_score_fidelity_against_saved_nasa_json() -> None:
    summary = NASA_JONNY_MATRIX.parent.parent / "expert_matrix_summary.json"
    if not summary.is_file():
        summary = None
    else:
        # DEFAULT_SUMMARY lives next to expert_matrix/
        from eval.dashboard_correctness import NASA_JONNY_SUMMARY

        summary = NASA_JONNY_SUMMARY if NASA_JONNY_SUMMARY.is_file() else None
    _assert_all_pass(check_score_fidelity(NASA_JONNY_MATRIX, summary))

    for run_name in RUN_ORDER:
        demo = NASA_JONNY_MATRIX / f"demo_{run_name}.json"
        raw = read_demo_json_scores(demo)
        view = build_dashboard_view(NASA_JONNY_MATRIX, summary_path=summary)
        card = next(c for c in view.cards if c.run_name == run_name)
        assert card.real_score == pytest.approx(raw["real_score"])
        assert card.fake_score == pytest.approx(raw["fake_score"])
        assert card.label == raw["label"]


@pytest.mark.skipif(not FFPP_MATRIX.is_dir(), reason="FF++ live matrix absent")
def test_score_fidelity_against_saved_ffpp_json() -> None:
    from eval.dashboard_correctness import FFPP_SUMMARY

    summary = FFPP_SUMMARY if FFPP_SUMMARY.is_file() else None
    _assert_all_pass(check_score_fidelity(FFPP_MATRIX, summary))


# --------------------------------------------------------------------------
# 3–5 provenance, agreement, status
# --------------------------------------------------------------------------


def test_agreement_and_status_boundaries() -> None:
    assert EXPERT_LO == 0.30
    assert EXPERT_HI == 0.70
    assert CONF_HIGH == 0.70
    _assert_all_pass(check_agreement_boundaries())
    _assert_all_pass(check_prototype_status_boundaries())


def test_missing_evidence_does_not_become_stable(tmp_path: Path) -> None:
    checks = check_malformed_robustness(tmp_path / "malformed")
    by_name = {c.name: c for c in checks}
    assert by_name["missing_main_scores_not_valid_stable"].passed
    assert by_name["na_specialist_is_none_not_zero"].passed
    assert by_name["na_specialist_insufficient_evidence"].passed


# --------------------------------------------------------------------------
# 6 saved vs live consistency
# --------------------------------------------------------------------------


@pytest.mark.skipif(not NASA_JONNY_MATRIX.is_dir(), reason="Stage 3 NASA matrix absent")
def test_saved_and_live_paths_same_dashboard_snapshot() -> None:
    from eval.dashboard_correctness import NASA_JONNY_SUMMARY

    summary = NASA_JONNY_SUMMARY if NASA_JONNY_SUMMARY.is_file() else None
    _assert_all_pass(check_saved_vs_live_consistency(NASA_JONNY_MATRIX, summary))
    saved = load_saved_example("nasa")
    live_like = build_dashboard_view(NASA_JONNY_MATRIX, summary_path=summary)
    assert dashboard_facing_snapshot(saved) == dashboard_facing_snapshot(live_like)


# --------------------------------------------------------------------------
# 7 malformed / missing
# --------------------------------------------------------------------------


def test_malformed_and_partial_outputs(tmp_path: Path) -> None:
    _assert_all_pass(check_malformed_robustness(tmp_path / "malformed"))


# --------------------------------------------------------------------------
# 8 representative Mae + FF++
# --------------------------------------------------------------------------


@pytest.mark.skipif(not MAE_MATRIX.is_dir(), reason="Mae Jemison pilot matrix absent")
def test_mae_jemison_combined_is_conflict() -> None:
    from eval.dashboard_correctness import MAE_SUMMARY

    summary = MAE_SUMMARY if MAE_SUMMARY.is_file() else None
    view = build_dashboard_view(MAE_MATRIX, summary_path=summary)
    assert view.evidence_agreement is EvidenceAgreement.CONFLICT
    assert view.status is Status.UNCERTAIN
    assert view.model_assessment_label == "fake"
    combined = next(c for c in view.cards if c.run_name == "blending_diffusion")
    assert combined.expert_scores.get("blending") is not None
    assert combined.expert_scores.get("diffusion") is not None
    assert combined.expert_scores["blending"] < EXPERT_LO
    assert combined.expert_scores["diffusion"] < EXPERT_LO


@pytest.mark.skipif(not FFPP_MATRIX.is_dir(), reason="FF++ live matrix absent")
def test_ffpp_deepfake_combined_is_agreement() -> None:
    from eval.dashboard_correctness import FFPP_SUMMARY

    summary = FFPP_SUMMARY if FFPP_SUMMARY.is_file() else None
    view = build_dashboard_view(FFPP_MATRIX, summary_path=summary)
    assert view.evidence_agreement is EvidenceAgreement.AGREEMENT
    assert view.status is Status.STABLE
    assert view.model_assessment_label == "fake"
    assert view.evidence_conflicts == []
    combined = next(c for c in view.cards if c.run_name == "blending_diffusion")
    assert combined.expert_scores["blending"] > EXPERT_HI
    assert combined.expert_scores["diffusion"] > EXPERT_HI


def test_representative_case_runner(tmp_path: Path) -> None:
    # Allow skips when gitignored outputs are absent (e.g. fresh clone).
    checks = check_representative_cases()
    hard_fails = [c for c in checks if not c.passed and not c.skipped]
    assert not hard_fails, hard_fails


# --------------------------------------------------------------------------
# 9 validation report tool
# --------------------------------------------------------------------------


def test_run_all_checks_writes_summary_shape(tmp_path: Path) -> None:
    report = run_all_checks(scratch_dir=tmp_path / "scratch")
    payload = report.as_dict()
    assert payload["total_checks"] >= 30
    assert payload["failed"] == 0
    assert "categories" in payload
    assert payload["threshold_constants"]["evidence_expert_lo"] == 0.30
    assert payload["threshold_constants"]["evidence_expert_hi"] == 0.70
    assert payload["threshold_constants"]["prototype_stable_uncertain"] == 0.70
    out = tmp_path / "dashboard_validation_summary.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["ok"] is True
