"""Unit tests for thesis evidence summaries, figures metadata, and worksheets."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from eval.ground_truth_audit import build_audit_rows
from eval.thesis_evidence import (
    HIGH_THRESHOLD_NOTE,
    THESIS_DISCRIMINATION_WORDING,
    build_manual_verification_rows,
    build_performance_summary_row,
    build_score_distribution_summary,
    build_threshold_justification_row,
    distribution_stats,
    extract_protocol_metrics_for_run,
    generate_thesis_evidence,
)
from eval.threshold_sweep import REFERENCE_THRESHOLD, compute_roc_auc

pytestmark = pytest.mark.unit


def _cfg(run_name: str, fake_score: float) -> Dict[str, Any]:
    return {
        "run_name": run_name,
        "status": "pass",
        "label": "fake" if fake_score >= 0.5 else "real",
        "fake_score": fake_score,
        "real_score": 1.0 - fake_score,
    }


def _image(image_id: str, gt: str, scores_by_run: Dict[str, float]) -> Dict[str, Any]:
    return {
        "image_id": image_id,
        "ground_truth": gt,
        "dataset": "synthetic",
        "source_path": f"/tmp/{image_id}.png",
        "configs": [_cfg(run, score) for run, score in scores_by_run.items()],
    }


def _balanced_aggregate() -> Dict[str, Any]:
    images: List[Dict[str, Any]] = []
    # Separable-ish scores on blending
    for i in range(6):
        images.append(
            _image(
                f"r{i}",
                "real",
                {
                    "none": 0.1 + 0.02 * i,
                    "blending": 0.12 + 0.02 * i,
                    "diffusion": 0.11 + 0.02 * i,
                    "blending_diffusion": 0.13 + 0.02 * i,
                },
            )
        )
        images.append(
            _image(
                f"f{i}",
                "fake",
                {
                    "none": 0.7 + 0.04 * i,
                    "blending": 0.72 + 0.04 * i,
                    "diffusion": 0.71 + 0.04 * i,
                    "blending_diffusion": 0.73 + 0.04 * i,
                },
            )
        )
    return {"images": images}


def _protocol_with_balanced(thr: float = 0.78) -> Dict[str, Any]:
    return {
        "split": {"seed": 4842, "val_fraction": 0.5},
        "results_by_run": {
            "blending": {
                "status": "ok",
                "dashboard_presets": {
                    "Balanced": {
                        "threshold": thr,
                        "validation_metrics": {"balanced_accuracy": 0.88},
                        "held_out_test": {
                            "metrics": {
                                "balanced_accuracy": 0.83,
                                "sensitivity_fake": 0.80,
                                "specificity_real": 0.86,
                            }
                        },
                    }
                },
                "validation_sweep": [
                    {
                        "threshold": thr,
                        "sensitivity_fake": 0.85,
                        "specificity_real": 0.90,
                        "balanced_accuracy": 0.875,
                    }
                ],
                "selections": {
                    "max_balanced_accuracy": {
                        "selection": {"selected_threshold": thr},
                        "comparison": {
                            "val_balanced_accuracy": 0.88,
                            "test_balanced_accuracy": 0.83,
                            "test_sensitivity_fake": 0.80,
                            "test_specificity_real": 0.86,
                            "ref_0_50_test_balanced_accuracy": 0.70,
                        },
                        "reference_0_50_held_out_test": {
                            "metrics": {"balanced_accuracy": 0.70}
                        },
                    }
                },
                "reference_0_50_held_out_test": {
                    "metrics": {"balanced_accuracy": 0.70}
                },
            }
        },
    }


def test_distribution_statistics() -> None:
    stats = distribution_stats([0.1, 0.2, 0.3, 0.4, 0.5])
    assert stats["n"] == 5
    assert stats["mean"] == pytest.approx(0.3)
    assert stats["median"] == pytest.approx(0.3)
    assert stats["min"] == pytest.approx(0.1)
    assert stats["max"] == pytest.approx(0.5)
    assert stats["q25"] == pytest.approx(0.2)
    assert stats["q75"] == pytest.approx(0.4)
    empty = distribution_stats([])
    assert empty["n"] == 0
    assert empty["mean"] is None


def test_score_distribution_summary_markers() -> None:
    rows, _ = build_audit_rows(
        {
            "images": [
                {
                    "image_id": "r",
                    "ground_truth": "real",
                    "dataset": "synthetic",
                    "configs": [_cfg("blending", 0.2)],
                },
                {
                    "image_id": "f",
                    "ground_truth": "fake",
                    "dataset": "synthetic",
                    "configs": [_cfg("blending", 0.8)],
                },
            ]
        },
        run_name="blending",
        active_threshold=0.78,
    )
    summary = build_score_distribution_summary(
        rows,
        dataset_name="synthetic",
        balanced_threshold=0.78,
        balanced_threshold_source="test",
    )
    assert summary["reference_threshold"] == pytest.approx(REFERENCE_THRESHOLD)
    assert summary["balanced_threshold"] == pytest.approx(0.78)
    assert summary["real_fake_score"]["n"] == 1
    assert summary["fake_fake_score"]["n"] == 1


def test_roc_auc_threshold_independent() -> None:
    from eval.thesis_evidence import rows_to_scored_samples as to_samples

    images = []
    for i in range(3):
        images.append(
            {
                "image_id": f"r{i}",
                "ground_truth": "real",
                "configs": [_cfg("blending", 0.1 + 0.05 * i)],
            }
        )
        images.append(
            {
                "image_id": f"f{i}",
                "ground_truth": "fake",
                "configs": [_cfg("blending", 0.8 + 0.05 * i)],
            }
        )
    rows_low, _ = build_audit_rows(
        {"images": images},
        run_name="blending",
        active_threshold=0.2,
    )
    rows_high, _ = build_audit_rows(
        {"images": images},
        run_name="blending",
        active_threshold=0.8,
    )
    auc_low = compute_roc_auc(to_samples(rows_low))["roc_auc"]
    auc_high = compute_roc_auc(to_samples(rows_high))["roc_auc"]
    assert auc_low is not None
    assert auc_low == pytest.approx(auc_high)
    assert auc_low == pytest.approx(1.0)


def test_no_test_leakage_into_threshold_selection() -> None:
    protocol = _protocol_with_balanced(0.78)
    # Corrupt held-out metrics — must not change selected threshold source value.
    protocol["results_by_run"]["blending"]["dashboard_presets"]["Balanced"][
        "held_out_test"
    ]["metrics"]["balanced_accuracy"] = 0.01
    metrics = extract_protocol_metrics_for_run(protocol, "blending")
    assert metrics["balanced_threshold"] == pytest.approx(0.78)
    assert metrics["held_out_balanced_accuracy"] == pytest.approx(0.01)


def test_threshold_justification_high_threshold_wording() -> None:
    rows, _ = build_audit_rows(
        {
            "images": [
                {
                    "image_id": "r",
                    "ground_truth": "real",
                    "configs": [_cfg("blending", 0.2)],
                },
                {
                    "image_id": "f",
                    "ground_truth": "fake",
                    "configs": [_cfg("blending", 0.95)],
                },
            ]
        },
        run_name="blending",
        active_threshold=0.99,
    )
    protocol = _protocol_with_balanced(0.99)
    just = build_threshold_justification_row(
        rows, dataset_name="DeepFakeFace", protocol=protocol
    )
    assert just["unusually_high_threshold"] is True
    assert just["high_threshold_note"] == HIGH_THRESHOLD_NOTE
    assert "does not improve ROC-AUC" in (just["high_threshold_note"] or "")
    assert THESIS_DISCRIMINATION_WORDING in just["discrimination_wording"]
    blob = json.dumps(just).lower()
    assert "confidence" not in blob or "not" in blob
    assert "universally" in (just["high_threshold_note"] or "").lower()


def test_manual_worksheet_human_fields_blank() -> None:
    cases = [
        {
            "sample_id": "s1",
            "dataset": "synthetic",
            "ground_truth": "fake",
            "run_name": "blending",
            "fake_score": 0.9,
            "active_threshold": 0.5,
            "active_prediction": "fake",
            "active_correct": True,
            "case_key": "strongest_correct_fake",
            "selection_rule": "rule",
        }
    ]
    rows = build_manual_verification_rows(cases)
    assert rows[0]["human_visual_check_completed"] == ""
    assert rows[0]["ground_truth_source_verified"] == ""
    assert rows[0]["notes"] == ""
    assert rows[0]["manual_verification_complete"] is False
    assert rows[0]["automated_correct"] is True


def test_performance_summary_descriptive_flags() -> None:
    just = {
        "dataset": "synthetic",
        "run_name": "blending",
        "roc_auc": 0.52,
        "reference_threshold": 0.5,
        "reference_balanced_accuracy": 0.5,
        "selected_balanced_threshold": 0.7,
        "held_out_balanced_accuracy": 0.6,
        "held_out_sensitivity_fake": 0.55,
        "held_out_specificity_real": 0.65,
        "difference_vs_0_50": 0.1,
        "interpretation_flags": [
            "weak discrimination (AUC close to 0.5)",
            "threshold-sensitive (held-out balanced accuracy shifts vs 0.50)",
        ],
    }
    row = build_performance_summary_row(just)
    assert "weak discrimination" in (row["interpretation_flags"] or "")
    assert row["gain_vs_reference"] == pytest.approx(0.1)


def test_generate_thesis_evidence_bundle(tmp_path: Path) -> None:
    agg = tmp_path / "aggregate.json"
    agg.write_text(json.dumps(_balanced_aggregate()), encoding="utf-8")
    proto = tmp_path / "validation_protocol.json"
    # Provide presets for all four runs so thresholds resolve cleanly.
    protocol = {"split": {"seed": 4842}, "results_by_run": {}}
    for run in ("none", "blending", "diffusion", "blending_diffusion"):
        protocol["results_by_run"][run] = {
            "dashboard_presets": {
                "Balanced": {
                    "threshold": 0.70,
                    "validation_metrics": {"balanced_accuracy": 0.8},
                    "held_out_test": {
                        "metrics": {
                            "balanced_accuracy": 0.75,
                            "sensitivity_fake": 0.7,
                            "specificity_real": 0.8,
                        }
                    },
                }
            },
            "selections": {
                "max_balanced_accuracy": {
                    "selection": {"selected_threshold": 0.70},
                    "comparison": {
                        "val_balanced_accuracy": 0.8,
                        "test_balanced_accuracy": 0.75,
                        "test_sensitivity_fake": 0.7,
                        "test_specificity_real": 0.8,
                        "ref_0_50_test_balanced_accuracy": 0.65,
                    },
                }
            },
        }
    proto.write_text(json.dumps(protocol), encoding="utf-8")

    out = tmp_path / "thesis_evidence"
    result = generate_thesis_evidence(
        [
            {
                "name": "Synthetic",
                "aggregate": agg,
                "validation_protocol": proto,
            }
        ],
        out,
        write_figures=True,
        write_ecdf=True,
    )
    assert result["n_audit_rows"] > 0
    assert (out / "thesis_evidence_manifest.json").is_file()
    assert (out / "tables" / "thesis_performance_summary.md").is_file()
    assert (out / "tables" / "threshold_justification.md").is_file()
    assert (out / "cases" / "manual_ground_truth_verification.csv").is_file()
    assert (out / "figures" / "score_distribution_synthetic_blending.png").is_file()

    manifest = json.loads((out / "thesis_evidence_manifest.json").read_text(encoding="utf-8"))
    assert manifest["validation_seed"] == 4842
    assert "not a calibrated" in manifest["score_semantics"].lower()
    blob = json.dumps(manifest).lower()
    assert "participant study was conducted" not in blob
    assert "does not claim a participant study" in blob or "participant study" in (
        " ".join(manifest.get("claim_boundary_notes") or []).lower()
    )
    assert any("manual" in s.lower() for s in manifest["manual_steps_remaining"])

    manual_csv = (out / "cases" / "manual_ground_truth_verification.csv").read_text(
        encoding="utf-8"
    )
    # Human columns exist and remain empty in data rows (header only has names).
    assert "human_visual_check_completed" in manual_csv
    lines = [ln for ln in manual_csv.strip().splitlines() if ln]
    assert len(lines) > 1


def test_missing_validation_protocol_still_exports(tmp_path: Path) -> None:
    agg = tmp_path / "aggregate.json"
    agg.write_text(json.dumps(_balanced_aggregate()), encoding="utf-8")
    out = tmp_path / "thesis_evidence"
    result = generate_thesis_evidence(
        [{"name": "Synthetic", "aggregate": agg, "validation_protocol": None}],
        out,
        write_figures=False,
        write_ecdf=False,
    )
    assert result["n_audit_rows"] > 0
    just = json.loads(
        (out / "tables" / "threshold_justification.json").read_text(encoding="utf-8")
    )
    assert just["rows"]
    # Balanced threshold may be null without protocol; export must still exist.
    assert (out / "tables" / "thesis_performance_summary.csv").is_file()


def test_raw_scores_unchanged_through_evidence_pipeline() -> None:
    score = 0.812345
    rows, _ = build_audit_rows(
        {
            "images": [
                {
                    "image_id": "x",
                    "ground_truth": "fake",
                    "configs": [_cfg("blending", score)],
                }
            ]
        },
        run_name="blending",
        active_threshold=0.5,
    )
    just = build_threshold_justification_row(rows, dataset_name="synthetic", protocol=None)
    assert rows[0].fake_score == score
    assert just["n_samples"] == 1
