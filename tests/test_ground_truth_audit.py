"""Unit tests for ground-truth audit and representative case selection."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from eval.ground_truth_audit import (
    CASE_CLOSEST_CORRECT,
    CASE_CLOSEST_INCORRECT,
    CASE_FALSE_NEGATIVE,
    CASE_FALSE_POSITIVE,
    CASE_STRONGEST_TN,
    CASE_STRONGEST_TP,
    OUTCOME_FN,
    OUTCOME_FP,
    OUTCOME_TN,
    OUTCOME_TP,
    build_audit_rows,
    load_validation_protocol_payload,
    outcome_category,
    resolve_balanced_threshold_from_protocol,
    select_representative_cases,
    write_ground_truth_audit_outputs,
)
from eval.threshold_sweep import REFERENCE_THRESHOLD, SCORE_SEMANTICS, predict_label

pytestmark = pytest.mark.unit


def _cfg(
    run_name: str,
    *,
    fake_score: float,
    real_score: Optional[float] = None,
    label: Optional[str] = None,
    status: str = "pass",
    blending: Optional[float] = None,
    diffusion: Optional[float] = None,
) -> Dict[str, Any]:
    if real_score is None:
        real_score = 1.0 - fake_score
    if label is None:
        label = "fake" if fake_score >= 0.5 else "real"
    row: Dict[str, Any] = {
        "run_name": run_name,
        "status": status,
        "label": label,
        "fake_score": fake_score,
        "real_score": real_score,
    }
    if blending is not None:
        row["blending_detector_score"] = blending
    if diffusion is not None:
        row["diffusion_detector_score"] = diffusion
    return row


def _image(
    image_id: str,
    ground_truth: str,
    fake_score: float,
    *,
    run_name: str = "blending",
    blending: Optional[float] = None,
    diffusion: Optional[float] = None,
    evidence_agreement: Optional[str] = None,
    prototype_status: Optional[str] = None,
    source_path: Optional[str] = None,
    dataset: str = "synthetic",
) -> Dict[str, Any]:
    return {
        "image_id": image_id,
        "source_path": source_path or f"/tmp/{image_id}.png",
        "ground_truth": ground_truth,
        "dataset": dataset,
        "evidence_agreement": evidence_agreement,
        "prototype_status": prototype_status,
        "configs": [
            _cfg(
                run_name,
                fake_score=fake_score,
                blending=blending,
                diffusion=diffusion,
            )
        ],
    }


def _aggregate(images: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"images": images, "n_images": len(images)}


def test_outcome_category_assignment() -> None:
    assert outcome_category("fake", "fake") == OUTCOME_TP
    assert outcome_category("real", "real") == OUTCOME_TN
    assert outcome_category("real", "fake") == OUTCOME_FP
    assert outcome_category("fake", "real") == OUTCOME_FN


def test_threshold_boundary_equals_threshold_is_fake() -> None:
    # Existing convention: fake_score >= threshold → fake
    assert predict_label(0.70, 0.70) == "fake"
    rows, meta = build_audit_rows(
        _aggregate(
            [
                _image("eq", "fake", 0.70),
            ]
        ),
        run_name="blending",
        active_threshold=0.70,
    )
    assert len(rows) == 1
    assert rows[0].active_prediction == "fake"
    assert rows[0].outcome == OUTCOME_TP
    assert rows[0].fake_score == pytest.approx(0.70)
    assert meta["active_threshold"] == pytest.approx(0.70)


def test_reference_and_active_correctness_and_change_flag() -> None:
    # Score 0.60: fake at 0.50, real at 0.70
    rows, _ = build_audit_rows(
        _aggregate([_image("s1", "fake", 0.60)]),
        run_name="blending",
        active_threshold=0.70,
    )
    row = rows[0]
    assert row.reference_prediction == "fake"
    assert row.active_prediction == "real"
    assert row.reference_correct is True
    assert row.active_correct is False
    assert row.prediction_changed_vs_reference is True
    assert row.outcome == OUTCOME_FN
    assert row.distance_from_active_threshold == pytest.approx(0.10)


def test_missing_specialist_scores_not_fabricated() -> None:
    rows, _ = build_audit_rows(
        _aggregate([_image("s1", "real", 0.2)]),
        run_name="blending",
        active_threshold=0.50,
    )
    row = rows[0]
    assert row.specialist_scores.get("blending") is None
    assert row.specialist_availability.get("blending") is False
    assert 0.0 not in row.specialist_scores.values()


def test_present_specialist_scores_preserved() -> None:
    rows, _ = build_audit_rows(
        _aggregate([_image("s1", "fake", 0.8, blending=0.66, diffusion=None)]),
        run_name="blending",
        active_threshold=0.50,
    )
    row = rows[0]
    assert row.specialist_scores["blending"] == pytest.approx(0.66)
    assert row.specialist_availability["blending"] is True
    assert row.specialist_scores["diffusion"] is None
    assert row.specialist_availability["diffusion"] is False


def test_missing_fake_score_skipped() -> None:
    images = [
        {
            "image_id": "ok",
            "ground_truth": "real",
            "dataset": "synthetic",
            "configs": [_cfg("blending", fake_score=0.2)],
        },
        {
            "image_id": "missing",
            "ground_truth": "fake",
            "dataset": "synthetic",
            "configs": [
                {
                    "run_name": "blending",
                    "status": "pass",
                    "label": "fake",
                    "fake_score": None,
                    "real_score": None,
                }
            ],
        },
    ]
    rows, meta = build_audit_rows(
        _aggregate(images), run_name="blending", active_threshold=0.5
    )
    assert len(rows) == 1
    assert rows[0].sample_id == "ok"
    assert meta["n_skipped"] == 1


def test_representative_selection_deterministic_and_covers_fp_fn() -> None:
    images = [
        _image("tp_strong", "fake", 0.95),
        _image("tp_weak", "fake", 0.72),
        _image("tn_strong", "real", 0.05),
        _image("tn_weak", "real", 0.40),
        _image("fp_a", "real", 0.80),
        _image("fp_b", "real", 0.65),
        _image("fn_a", "fake", 0.10),
        _image("fn_b", "fake", 0.30),
        _image("near_ok", "real", 0.49),  # TN close to 0.50
        _image("near_bad", "fake", 0.48),  # FN close to 0.50
    ]
    rows, _ = build_audit_rows(
        _aggregate(images), run_name="blending", active_threshold=0.50
    )
    cases_a = select_representative_cases(rows)
    cases_b = select_representative_cases(rows)
    assert cases_a == cases_b

    by_key = {c["case_key"]: c for c in cases_a}
    assert by_key[CASE_STRONGEST_TP]["sample_id"] == "tp_strong"
    assert by_key[CASE_STRONGEST_TN]["sample_id"] == "tn_strong"
    assert by_key[CASE_FALSE_POSITIVE]["sample_id"] == "fp_a"
    assert by_key[CASE_FALSE_NEGATIVE]["sample_id"] == "fn_a"
    assert by_key[CASE_CLOSEST_CORRECT]["sample_id"] == "near_ok"
    assert by_key[CASE_CLOSEST_INCORRECT]["sample_id"] == "near_bad"
    assert "selection_rule" in by_key[CASE_STRONGEST_TP]


def test_validation_metadata_absent_falls_back_to_reference() -> None:
    rows, meta = build_audit_rows(
        _aggregate([_image("s1", "real", 0.2)]),
        run_name="blending",
        protocol=None,
    )
    assert meta["active_threshold"] == pytest.approx(REFERENCE_THRESHOLD)
    assert meta["active_threshold_source"] == "reference_fallback"


def test_malformed_validation_protocol_does_not_crash(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    payload, err = load_validation_protocol_payload(bad)
    assert payload is None
    assert err is not None
    rows, meta = build_audit_rows(
        _aggregate([_image("s1", "fake", 0.9)]),
        run_name="blending",
        protocol=payload,
    )
    assert len(rows) == 1
    assert meta["active_threshold"] == pytest.approx(0.50)


def test_balanced_from_validation_sweep_not_held_out() -> None:
    protocol = {
        "results_by_run": {
            "blending": {
                "status": "ok",
                "validation_sweep": [
                    {
                        "threshold": 0.40,
                        "sensitivity_fake": 0.99,
                        "specificity_real": 0.40,
                        "balanced_accuracy": 0.695,
                    },
                    {
                        "threshold": 0.78,
                        "sensitivity_fake": 0.85,
                        "specificity_real": 0.90,
                        "balanced_accuracy": 0.875,
                    },
                    {
                        "threshold": 0.90,
                        "sensitivity_fake": 0.50,
                        "specificity_real": 0.98,
                        "balanced_accuracy": 0.74,
                    },
                ],
                # Poison held-out-looking fields that must not drive selection.
                "selections": {
                    "max_balanced_accuracy": {
                        "selection": {"selected_threshold": 0.11},
                        "held_out_test": {
                            "metrics": {"balanced_accuracy": 0.99},
                        },
                        "comparison": {
                            "test_balanced_accuracy": 0.99,
                            "val_balanced_accuracy": 0.10,
                        },
                    }
                },
            }
        }
    }
    thr, src = resolve_balanced_threshold_from_protocol(protocol, "blending")
    # Prefer dashboard_presets if present; here sweep-derived Balanced = 0.78
    assert thr == pytest.approx(0.78)
    assert src is not None
    assert "validation" in src.lower() or "sweep" in src.lower() or "preset" in src.lower()
    assert thr != pytest.approx(0.11)

    rows, meta = build_audit_rows(
        _aggregate([_image("s1", "fake", 0.80)]),
        run_name="blending",
        protocol=protocol,
    )
    assert meta["active_threshold"] == pytest.approx(0.78)
    assert rows[0].fake_score == pytest.approx(0.80)


def test_dashboard_presets_preferred_for_balanced() -> None:
    protocol = {
        "results_by_run": {
            "blending": {
                "dashboard_presets": {
                    "Balanced": {"threshold": 0.77, "validation_balanced_accuracy": 0.8}
                },
                "validation_sweep": [
                    {
                        "threshold": 0.50,
                        "sensitivity_fake": 0.7,
                        "specificity_real": 0.7,
                        "balanced_accuracy": 0.7,
                    }
                ],
            }
        }
    }
    thr, src = resolve_balanced_threshold_from_protocol(protocol, "blending")
    assert thr == pytest.approx(0.77)
    assert src == "dashboard_presets.Balanced"


def test_write_outputs_and_no_confidence_wording(tmp_path: Path) -> None:
    rows, meta = build_audit_rows(
        _aggregate(
            [
                _image("r1", "real", 0.1),
                _image("f1", "fake", 0.9),
            ]
        ),
        run_name="blending",
        active_threshold=0.5,
    )
    written = write_ground_truth_audit_outputs(rows, tmp_path, meta=meta)
    assert (tmp_path / "ground_truth_audit.json").is_file()
    assert (tmp_path / "ground_truth_audit.csv").is_file()
    assert (tmp_path / "representative_cases.md").is_file()
    blob = (tmp_path / "ground_truth_audit.json").read_text(encoding="utf-8").lower()
    assert "87% confidence" not in blob
    assert "probability of fake" not in blob
    assert "not a calibrated probability" in blob
    assert "score_semantics" in blob
    assert "not a calibrated" in SCORE_SEMANTICS.lower()
    assert "ground_truth_audit.json" in written


def test_raw_fake_score_unchanged_byte_value() -> None:
    score = 0.734512
    rows, _ = build_audit_rows(
        _aggregate([_image("x", "fake", score)]),
        run_name="blending",
        active_threshold=0.5,
    )
    assert rows[0].fake_score == score
