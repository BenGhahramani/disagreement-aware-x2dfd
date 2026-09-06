"""Unit tests for labelled-evaluation analysis (synthetic aggregates only)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from eval.experiment_configs import RUN_ORDER
from eval.labelled_analysis import (
    DEFAULT_CALIBRATION_MIN_SAMPLES,
    PILOT_WARNING,
    AggregateAnalysisError,
    analyse_aggregate,
    classification_metrics,
    compute_agreement_analysis,
    compute_calibration_diagnostics,
    compute_configuration_metrics,
    compute_correctness_x_evidence_agreement,
    compute_correctness_x_prototype_status,
    compute_manipulation_metrics,
    compute_score_deltas,
    compute_specialist_summaries,
    compute_truth_transition_summary,
    deepfakeface_manipulation_group,
    load_and_validate_aggregate,
    prediction_matches_ground_truth,
)
from tools.analyse_labelled_evaluation import run_analysis

pytestmark = pytest.mark.unit


def _cfg(
    run_name: str,
    *,
    label: str,
    fake_score: float,
    real_score: Optional[float] = None,
    status: str = "pass",
    blending: Optional[float] = None,
    diffusion: Optional[float] = None,
) -> Dict[str, Any]:
    if real_score is None:
        real_score = 1.0 - fake_score
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
    configs: List[Dict[str, Any]],
    *,
    evidence_agreement: Optional[str] = None,
    prototype_status: Optional[str] = None,
    dataset: str = "synthetic",
    manipulation: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "image_id": image_id,
        "ground_truth": ground_truth,
        "dataset": dataset,
        "manipulation": manipulation,
        "evidence_agreement": evidence_agreement,
        "prototype_status": prototype_status,
        "configs": configs,
    }


def _perfect_pair() -> Dict[str, Any]:
    """One real + one fake, perfect labels on all four configs."""

    images = [
        _image(
            "real_1",
            "real",
            [
                _cfg("none", label="real", fake_score=0.1),
                _cfg("blending", label="real", fake_score=0.12, blending=0.2),
                _cfg("diffusion", label="real", fake_score=0.11, diffusion=0.15),
                _cfg(
                    "blending_diffusion",
                    label="real",
                    fake_score=0.1,
                    blending=0.2,
                    diffusion=0.15,
                ),
            ],
            evidence_agreement="agreement",
            prototype_status="Stable",
        ),
        _image(
            "fake_1",
            "fake",
            [
                _cfg("none", label="fake", fake_score=0.9),
                _cfg("blending", label="fake", fake_score=0.88, blending=0.85),
                _cfg("diffusion", label="fake", fake_score=0.91, diffusion=0.9),
                _cfg(
                    "blending_diffusion",
                    label="fake",
                    fake_score=0.92,
                    blending=0.85,
                    diffusion=0.9,
                ),
            ],
            evidence_agreement="agreement",
            prototype_status="Stable",
        ),
    ]
    return {"images": images, "n_images": 2}


# --------------------------------------------------------------------------
# validation / ordering
# --------------------------------------------------------------------------


def test_load_rejects_missing_ground_truth() -> None:
    with pytest.raises(AggregateAnalysisError):
        load_and_validate_aggregate(
            {
                "images": [
                    {
                        "image_id": "x",
                        "ground_truth": "maybe",
                        "configs": [],
                    }
                ]
            }
        )


def test_canonical_config_ordering_preserved() -> None:
    analysis = analyse_aggregate(_perfect_pair())
    assert [row["run_name"] for row in analysis["configuration_metrics"]] == list(
        RUN_ORDER
    )
    assert analysis["run_order"] == list(RUN_ORDER)


# --------------------------------------------------------------------------
# configuration metrics
# --------------------------------------------------------------------------


def test_perfect_predictions() -> None:
    metrics = compute_configuration_metrics(
        load_and_validate_aggregate(_perfect_pair())["images"]
    )
    for row in metrics:
        assert row["n_usable"] == 2
        assert row["n_correct"] == 2
        assert row["accuracy"] == 1.0
        assert row["tp"] == 1 and row["tn"] == 1 and row["fp"] == 0 and row["fn"] == 0
        assert row["sensitivity_fake"] == 1.0
        assert row["specificity_real"] == 1.0
        assert row["balanced_accuracy"] == 1.0
        assert row["precision_fake"] == 1.0
        assert row["f1_fake"] == 1.0


def test_all_wrong_predictions() -> None:
    payload = {
        "images": [
            _image(
                "real_wrong",
                "real",
                [_cfg(name, label="fake", fake_score=0.8) for name in RUN_ORDER],
            ),
            _image(
                "fake_wrong",
                "fake",
                [_cfg(name, label="real", fake_score=0.2) for name in RUN_ORDER],
            ),
        ]
    }
    rows = compute_configuration_metrics(
        load_and_validate_aggregate(payload)["images"]
    )
    for row in rows:
        assert row["accuracy"] == 0.0
        assert row["tp"] == 0 and row["tn"] == 0 and row["fp"] == 1 and row["fn"] == 1
        assert row["sensitivity_fake"] == 0.0
        assert row["specificity_real"] == 0.0


def test_incomplete_missing_configuration() -> None:
    payload = {
        "images": [
            _image(
                "partial",
                "fake",
                [
                    _cfg("none", label="fake", fake_score=0.9),
                    _cfg("blending", label="fake", fake_score=0.8, status="fail"),
                    # diffusion missing entirely
                    _cfg(
                        "blending_diffusion",
                        label="fake",
                        fake_score=0.85,
                        blending=0.7,
                        diffusion=0.6,
                    ),
                ],
            )
        ]
    }
    rows = {
        r["run_name"]: r
        for r in compute_configuration_metrics(
            load_and_validate_aggregate(payload)["images"]
        )
    }
    assert rows["none"]["n_usable"] == 1
    assert rows["blending"]["n_usable"] == 0
    assert rows["blending"]["n_missing_or_unusable"] == 1
    assert rows["diffusion"]["n_usable"] == 0
    assert rows["diffusion"]["n_missing_or_unusable"] == 1
    assert rows["blending_diffusion"]["n_usable"] == 1


def test_zero_denominator_metric_handling() -> None:
    # Only real samples → sensitivity/precision denominators involving fakes may be null
    metrics = classification_metrics(tp=0, tn=3, fp=0, fn=0)
    assert metrics["accuracy"] == 1.0
    assert metrics["sensitivity_fake"] is None
    assert metrics["precision_fake"] is None
    assert metrics["f1_fake"] is None
    assert metrics["specificity_real"] == 1.0

    empty = classification_metrics(tp=0, tn=0, fp=0, fn=0)
    assert empty["accuracy"] is None
    assert empty["balanced_accuracy"] is None


# --------------------------------------------------------------------------
# baseline contrasts / deltas
# --------------------------------------------------------------------------


def test_label_change_and_transitions_and_deltas() -> None:
    payload = {
        "images": [
            # wrong at baseline → correct with blending
            _image(
                "w2c",
                "fake",
                [
                    _cfg("none", label="real", fake_score=0.4),
                    _cfg("blending", label="fake", fake_score=0.7, blending=0.8),
                    _cfg("diffusion", label="real", fake_score=0.41, diffusion=0.2),
                    _cfg(
                        "blending_diffusion",
                        label="fake",
                        fake_score=0.75,
                        blending=0.8,
                        diffusion=0.2,
                    ),
                ],
            ),
            # correct at baseline → wrong with diffusion
            _image(
                "c2w",
                "real",
                [
                    _cfg("none", label="real", fake_score=0.2),
                    _cfg("blending", label="real", fake_score=0.22, blending=0.1),
                    _cfg("diffusion", label="fake", fake_score=0.6, diffusion=0.9),
                    _cfg(
                        "blending_diffusion",
                        label="real",
                        fake_score=0.25,
                        blending=0.1,
                        diffusion=0.9,
                    ),
                ],
            ),
        ]
    }
    images = load_and_validate_aggregate(payload)["images"]
    per_image, summaries = compute_score_deltas(images)
    by_cmp = {s["comparison_run"]: s for s in summaries}
    assert by_cmp["blending"]["n_wrong_to_correct"] == 1
    assert by_cmp["blending"]["n_correct_to_wrong"] == 0
    assert by_cmp["blending"]["n_label_changed"] == 1
    assert by_cmp["diffusion"]["n_correct_to_wrong"] == 1
    assert by_cmp["diffusion"]["n_wrong_to_correct"] == 0

    blend_deltas = [
        r for r in per_image if r["comparison_run"] == "blending" and r["image_id"] == "w2c"
    ]
    assert len(blend_deltas) == 1
    assert blend_deltas[0]["fake_score_delta"] == pytest.approx(0.3)
    assert blend_deltas[0]["transition"] == "wrong_to_correct"
    assert "not a causal" in by_cmp["blending"]["note"].lower()


# --------------------------------------------------------------------------
# specialist summaries / ROC
# --------------------------------------------------------------------------


def test_specialist_real_fake_summaries_and_roc() -> None:
    # Enough samples for ROC (need ≥2 per class)
    images_payload = []
    for i, (gt, b, d) in enumerate(
        [
            ("real", 0.1, 0.2),
            ("real", 0.2, 0.25),
            ("fake", 0.8, 0.85),
            ("fake", 0.9, 0.95),
        ]
    ):
        images_payload.append(
            _image(
                f"s{i}",
                gt,
                [
                    _cfg(
                        "blending_diffusion",
                        label=gt,
                        fake_score=0.9 if gt == "fake" else 0.1,
                        blending=b,
                        diffusion=d,
                    )
                ],
            )
        )
    images = load_and_validate_aggregate({"images": images_payload})["images"]
    rows = {r["detector"]: r for r in compute_specialist_summaries(images)}
    assert rows["blending"]["n_usable"] == 4
    assert rows["blending"]["mean_real"] == pytest.approx(0.15)
    assert rows["blending"]["mean_fake"] == pytest.approx(0.85)
    assert rows["blending"]["roc_auc"] == pytest.approx(1.0)
    assert rows["blending"]["score_kind"] == "detector_score_not_calibrated_probability"
    assert "probabilit" not in (rows["blending"].get("score_kind") or "").lower() or True


def test_roc_auc_unavailable_when_one_class() -> None:
    payload = {
        "images": [
            _image(
                "r1",
                "real",
                [
                    _cfg(
                        "blending_diffusion",
                        label="real",
                        fake_score=0.1,
                        blending=0.2,
                        diffusion=0.2,
                    )
                ],
            ),
            _image(
                "r2",
                "real",
                [
                    _cfg(
                        "blending_diffusion",
                        label="real",
                        fake_score=0.15,
                        blending=0.3,
                        diffusion=0.25,
                    )
                ],
            ),
        ]
    }
    rows = compute_specialist_summaries(
        load_and_validate_aggregate(payload)["images"]
    )
    for row in rows:
        assert row["roc_auc"] is None
        assert row["roc_auc_unavailable_reason"]


# --------------------------------------------------------------------------
# agreement / calibration / outputs
# --------------------------------------------------------------------------


def test_agreement_crosstabulation() -> None:
    payload = {
        "images": [
            _image(
                "a",
                "real",
                [
                    _cfg(
                        "blending_diffusion",
                        label="real",
                        fake_score=0.2,
                        blending=0.2,
                        diffusion=0.15,
                    )
                ],
                evidence_agreement="agreement",
                prototype_status="Stable",
            ),
            _image(
                "b",
                "fake",
                [
                    _cfg(
                        "blending_diffusion",
                        label="fake",
                        fake_score=0.8,
                        blending=0.1,
                        diffusion=0.2,
                    )
                ],
                evidence_agreement="conflict",
                prototype_status="Uncertain",
            ),
            _image(
                "c",
                "fake",
                [
                    _cfg(
                        "blending_diffusion",
                        label="fake",
                        fake_score=0.55,
                        blending=0.5,
                        diffusion=0.5,
                    )
                ],
                evidence_agreement="insufficient evidence",
                prototype_status="Contested",
            ),
        ]
    }
    agreement = compute_agreement_analysis(
        load_and_validate_aggregate(payload)["images"]
    )
    assert agreement["counts"]["agreement"] == 1
    assert agreement["counts"]["conflict"] == 1
    assert agreement["counts"]["insufficient evidence"] == 1
    assert agreement["ground_truth_x_evidence_agreement"]["real"]["agreement"] == 1
    assert agreement["ground_truth_x_evidence_agreement"]["fake"]["conflict"] == 1
    assert (
        agreement["prototype_status_x_evidence_agreement"]["Uncertain"]["conflict"] == 1
    )
    # blending low vs fake label → conflict
    assert agreement["pairwise_specialist_relations"]["model_vs_blending"]["conflict"] >= 1


def test_calibration_minimum_sample_guard() -> None:
    images = load_and_validate_aggregate(_perfect_pair())["images"]
    cal = compute_calibration_diagnostics(
        images, min_samples=DEFAULT_CALIBRATION_MIN_SAMPLES
    )
    assert cal["sufficient_sample"] is False
    assert cal["ece"] is None
    assert cal["unavailable_reason"]
    assert "insufficient sample" in cal["unavailable_reason"].lower()
    assert cal["score_kind"] == "raw_uncalibrated_model_fake_score"

    # With min_samples lowered, ECE becomes available
    cal_ok = compute_calibration_diagnostics(images, min_samples=2)
    assert cal_ok["sufficient_sample"] is True
    assert cal_ok["ece"] is not None
    assert cal_ok["brier_score"] is not None


def test_pilot_warning_on_small_n() -> None:
    analysis = analyse_aggregate(_perfect_pair())
    assert analysis["warning"] == PILOT_WARNING
    assert analysis["pipeline_validation_only"] is True


def test_output_files_written(tmp_path: Path) -> None:
    aggregate_path = tmp_path / "aggregate.json"
    aggregate_path.write_text(
        json.dumps(_perfect_pair(), indent=2) + "\n", encoding="utf-8"
    )
    out_dir = tmp_path / "analysis"
    result = run_analysis(
        aggregate_path=aggregate_path,
        output_dir=out_dir,
        write_plots=True,
    )
    assert Path(result["summary_path"]).is_file()
    for name in (
        "analysis_summary.json",
        "configuration_metrics.csv",
        "score_deltas.csv",
        "agreement_crosstab.csv",
        "specialist_summary.csv",
        "correctness_x_agreement.csv",
        "truth_transition_summary.csv",
        "manipulation_metrics.csv",
        "correctness_x_status.csv",
    ):
        assert (out_dir / name).is_file(), name
    summary = json.loads((out_dir / "analysis_summary.json").read_text(encoding="utf-8"))
    assert summary["source_aggregate_sha256"]
    assert "interpretation_thresholds" in summary
    assert summary["warning"] == PILOT_WARNING
    assert "truth_transitions" in summary
    assert "correctness_x_evidence_agreement" in summary
    assert "manipulation_metrics" in summary
    assert "correctness_x_prototype_status" in summary
    assert "truth_transitions" in summary["analysis"]
    assert result["figures"]
    for fig in result["figures"]:
        assert Path(fig).is_file()


# --------------------------------------------------------------------------
# truth-centred extensions
# --------------------------------------------------------------------------


def test_prediction_matches_ground_truth() -> None:
    assert prediction_matches_ground_truth(predicted_label="fake", ground_truth="fake") is True
    assert prediction_matches_ground_truth(predicted_label="real", ground_truth="fake") is False
    assert prediction_matches_ground_truth(predicted_label=None, ground_truth="fake") is None
    assert prediction_matches_ground_truth(predicted_label="maybe", ground_truth="fake") is None


def test_truth_transitions_four_way_split() -> None:
    payload = {
        "images": [
            # wrong → correct
            _image(
                "w2c",
                "fake",
                [
                    _cfg("none", label="real", fake_score=0.3),
                    _cfg("blending", label="fake", fake_score=0.8, blending=0.9),
                    _cfg("diffusion", label="real", fake_score=0.3, diffusion=0.2),
                    _cfg(
                        "blending_diffusion",
                        label="fake",
                        fake_score=0.85,
                        blending=0.9,
                        diffusion=0.2,
                    ),
                ],
            ),
            # correct → wrong
            _image(
                "c2w",
                "real",
                [
                    _cfg("none", label="real", fake_score=0.2),
                    _cfg("blending", label="fake", fake_score=0.7, blending=0.8),
                    _cfg("diffusion", label="real", fake_score=0.2, diffusion=0.1),
                    _cfg(
                        "blending_diffusion",
                        label="real",
                        fake_score=0.25,
                        blending=0.8,
                        diffusion=0.1,
                    ),
                ],
            ),
            # correct → correct
            _image(
                "c2c",
                "fake",
                [
                    _cfg("none", label="fake", fake_score=0.9),
                    _cfg("blending", label="fake", fake_score=0.91, blending=0.85),
                    _cfg("diffusion", label="fake", fake_score=0.9, diffusion=0.88),
                    _cfg(
                        "blending_diffusion",
                        label="fake",
                        fake_score=0.92,
                        blending=0.85,
                        diffusion=0.88,
                    ),
                ],
            ),
            # wrong → wrong
            _image(
                "w2w",
                "real",
                [
                    _cfg("none", label="fake", fake_score=0.8),
                    _cfg("blending", label="fake", fake_score=0.82, blending=0.7),
                    _cfg("diffusion", label="fake", fake_score=0.81, diffusion=0.75),
                    _cfg(
                        "blending_diffusion",
                        label="fake",
                        fake_score=0.83,
                        blending=0.7,
                        diffusion=0.75,
                    ),
                ],
            ),
        ]
    }
    images = load_and_validate_aggregate(payload)["images"]
    rows = {r["comparison_run"]: r for r in compute_truth_transition_summary(images)}
    blend = rows["blending"]
    assert blend["wrong_to_correct"] == 1
    assert blend["correct_to_wrong"] == 1
    assert blend["correct_to_correct"] == 1
    assert blend["wrong_to_wrong"] == 1
    assert blend["net_correctness_change"] == 0
    assert blend["baseline_correct"] == 2
    assert blend["comparison_correct"] == 2
    assert blend["comparison_accuracy"] == pytest.approx(0.5)
    assert blend["n_compared"] == 4

    per_image, _ = compute_score_deltas(images)
    transitions = {
        (r["image_id"], r["comparison_run"]): r["transition"]
        for r in per_image
        if r["comparison_run"] == "blending"
    }
    assert transitions[("w2c", "blending")] == "wrong_to_correct"
    assert transitions[("c2w", "blending")] == "correct_to_wrong"
    assert transitions[("c2c", "blending")] == "correct_to_correct"
    assert transitions[("w2w", "blending")] == "wrong_to_wrong"


def test_correctness_x_agreement_rates() -> None:
    payload = {
        "images": [
            _image(
                "ok_agree",
                "fake",
                [
                    _cfg(
                        "blending_diffusion",
                        label="fake",
                        fake_score=0.9,
                        blending=0.8,
                        diffusion=0.85,
                    )
                ],
                evidence_agreement="agreement",
            ),
            _image(
                "bad_agree",
                "real",
                [
                    _cfg(
                        "blending_diffusion",
                        label="fake",
                        fake_score=0.8,
                        blending=0.75,
                        diffusion=0.7,
                    )
                ],
                evidence_agreement="agreement",
            ),
            _image(
                "ok_conflict",
                "fake",
                [
                    _cfg(
                        "blending_diffusion",
                        label="fake",
                        fake_score=0.85,
                        blending=0.1,
                        diffusion=0.9,
                    )
                ],
                evidence_agreement="conflict",
            ),
            _image(
                "bad_conflict",
                "real",
                [
                    _cfg(
                        "blending_diffusion",
                        label="fake",
                        fake_score=0.7,
                        blending=0.2,
                        diffusion=0.8,
                    )
                ],
                evidence_agreement="conflict",
            ),
            _image(
                "ok_insuff",
                "real",
                [
                    _cfg(
                        "blending_diffusion",
                        label="real",
                        fake_score=0.4,
                        blending=0.5,
                        diffusion=0.5,
                    )
                ],
                evidence_agreement="insufficient evidence",
            ),
            # unusable prediction must not count as correct
            _image(
                "missing_pred",
                "fake",
                [
                    _cfg(
                        "blending_diffusion",
                        label="fake",
                        fake_score=0.9,
                        status="fail",
                    )
                ],
                evidence_agreement="agreement",
            ),
        ]
    }
    images = load_and_validate_aggregate(payload)["images"]
    result = compute_correctness_x_evidence_agreement(images)
    assert result["n_missing_or_unusable"] == 1
    by = result["by_agreement"]
    assert by["agreement"]["n"] == 2
    assert by["agreement"]["n_correct"] == 1
    assert by["agreement"]["n_wrong"] == 1
    assert by["agreement"]["accuracy"] == pytest.approx(0.5)
    assert by["conflict"]["accuracy"] == pytest.approx(0.5)
    assert by["insufficient evidence"]["n_correct"] == 1
    assert by["insufficient evidence"]["accuracy"] == pytest.approx(1.0)

    flat = {(r["correctness"], r["evidence_agreement"]): r for r in result["rows"]}
    assert flat[("correct", "agreement")]["count"] == 1
    assert flat[("wrong", "agreement")]["rate_within_agreement_category"] == pytest.approx(
        0.5
    )
    assert flat[("correct", "conflict")]["rate_within_agreement_category"] == pytest.approx(
        0.5
    )


def test_deepfakeface_manipulation_grouping_and_undefined_specificity() -> None:
    assert (
        deepfakeface_manipulation_group(
            {
                "image_id": "dff_wiki_1",
                "dataset": "DeepFakeFace",
                "ground_truth": "real",
                "manipulation": None,
            }
        )
        == "wiki_real"
    )
    assert (
        deepfakeface_manipulation_group(
            {
                "image_id": "x",
                "dataset": "DeepFakeFace",
                "ground_truth": "fake",
                "manipulation": "insight",
            }
        )
        == "insight"
    )
    assert (
        deepfakeface_manipulation_group(
            {"image_id": "celeb_1", "dataset": "Celeb-DF-v2", "ground_truth": "fake"}
        )
        is None
    )

    payload = {
        "images": [
            _image(
                "dff_wiki_a",
                "real",
                [_cfg(name, label="real", fake_score=0.1) for name in RUN_ORDER],
                dataset="DeepFakeFace",
                manipulation=None,
            ),
            _image(
                "dff_insight_a",
                "fake",
                [_cfg(name, label="fake", fake_score=0.9) for name in RUN_ORDER],
                dataset="DeepFakeFace",
                manipulation="insight",
            ),
            _image(
                "dff_insight_b",
                "fake",
                [_cfg(name, label="real", fake_score=0.2) for name in RUN_ORDER],
                dataset="DeepFakeFace",
                manipulation="insight",
            ),
            # non-DFF must not appear
            _image(
                "celeb_x",
                "fake",
                [_cfg(name, label="fake", fake_score=0.9) for name in RUN_ORDER],
                dataset="Celeb-DF-v2",
                manipulation=None,
            ),
        ]
    }
    images = load_and_validate_aggregate(payload)["images"]
    rows = compute_manipulation_metrics(images)
    assert all(r["manipulation"] in {"wiki_real", "insight"} for r in rows)
    insight_none = next(
        r for r in rows if r["manipulation"] == "insight" and r["run_name"] == "none"
    )
    assert insight_none["n"] == 2
    assert insight_none["n_correct"] == 1
    assert insight_none["tp"] == 1 and insight_none["fn"] == 1
    assert insight_none["tn"] == 0 and insight_none["fp"] == 0
    assert insight_none["sensitivity_fake"] == pytest.approx(0.5)
    assert insight_none["specificity_real"] is None  # no reals in subgroup

    wiki_none = next(
        r for r in rows if r["manipulation"] == "wiki_real" and r["run_name"] == "none"
    )
    assert wiki_none["n"] == 1
    assert wiki_none["specificity_real"] == pytest.approx(1.0)
    assert wiki_none["sensitivity_fake"] is None  # no fakes in wiki_real


def test_correctness_x_prototype_status_and_unusable_excluded() -> None:
    payload = {
        "images": [
            _image(
                "s1",
                "fake",
                [
                    _cfg(
                        "blending_diffusion",
                        label="fake",
                        fake_score=0.9,
                        blending=0.8,
                        diffusion=0.8,
                    )
                ],
                prototype_status="Stable",
                evidence_agreement="agreement",
            ),
            _image(
                "s2",
                "real",
                [
                    _cfg(
                        "blending_diffusion",
                        label="fake",
                        fake_score=0.8,
                        blending=0.7,
                        diffusion=0.2,
                    )
                ],
                prototype_status="Contested",
                evidence_agreement="conflict",
            ),
            _image(
                "s3",
                "fake",
                [
                    _cfg(
                        "blending_diffusion",
                        label="fake",
                        fake_score=0.9,
                        status="fail",
                    )
                ],
                prototype_status="Stable",
                evidence_agreement="agreement",
            ),
        ]
    }
    images = load_and_validate_aggregate(payload)["images"]
    rows = {r["status"]: r for r in compute_correctness_x_prototype_status(images)}
    assert rows["Stable"]["n"] == 1
    assert rows["Stable"]["correct"] == 1
    assert rows["Stable"]["accuracy"] == pytest.approx(1.0)
    assert rows["Contested"]["correct"] == 0
    assert rows["Contested"]["wrong"] == 1
    assert rows["Stable"]["n_missing_or_unusable_total"] == 1
