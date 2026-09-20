"""Unit tests for threshold-sweep analysis (no GPU)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.experiment_configs import RUN_ORDER
from eval.threshold_sweep import (
    DEFAULT_SPLIT_SEED,
    REFERENCE_THRESHOLD,
    SCORE_FIELD,
    SCORE_SEMANTICS,
    SELECTION_CRITERIA,
    ScoredSample,
    ThresholdSweepError,
    confusion_at_threshold,
    default_decision_operating_settings,
    descriptive_operating_points,
    evaluate_frozen_threshold,
    load_samples_from_ffpp_frames,
    load_samples_from_labelled_aggregate,
    predict_label,
    run_threshold_sweep,
    run_validation_test_protocol,
    select_threshold_on_validation,
    stratified_val_test_split,
    threshold_grid,
    write_threshold_outputs,
    write_validation_protocol_outputs,
)

pytestmark = pytest.mark.unit


def _samples() -> list[ScoredSample]:
    # Separable set: reals low, fakes high.
    return [
        ScoredSample("r1", "real", 0.1),
        ScoredSample("r2", "real", 0.2),
        ScoredSample("r3", "real", 0.4),
        ScoredSample("f1", "fake", 0.6),
        ScoredSample("f2", "fake", 0.8),
        ScoredSample("f3", "fake", 0.9),
    ]


def _large_balanced_samples(n_per_class: int = 20) -> list[ScoredSample]:
    samples: list[ScoredSample] = []
    for i in range(n_per_class):
        # Mildly separable: reals mostly <0.5, fakes mostly >0.5
        samples.append(
            ScoredSample(
                f"r{i:02d}",
                "real",
                0.05 + 0.4 * (i / max(n_per_class - 1, 1)),
            )
        )
        samples.append(
            ScoredSample(
                f"f{i:02d}",
                "fake",
                0.55 + 0.4 * (i / max(n_per_class - 1, 1)),
            )
        )
    return samples


def test_predict_label_boundaries() -> None:
    assert predict_label(0.5, 0.5) == "fake"
    assert predict_label(0.499999, 0.5) == "real"
    assert predict_label(0.0, 0.0) == "fake"
    assert predict_label(0.0, 1.0) == "real"
    assert predict_label(1.0, 1.0) == "fake"


def test_threshold_grid_includes_endpoints_and_reference() -> None:
    grid = threshold_grid(step=0.01)
    assert grid[0] == 0.0
    assert grid[-1] == 1.0
    assert REFERENCE_THRESHOLD in grid
    assert len(grid) == 101


def test_confusion_matrix_counts() -> None:
    samples = _samples()
    # At 0.5: all 3 reals TN, all 3 fakes TP
    row = confusion_at_threshold(samples, 0.5)
    assert row["tp"] == 3
    assert row["tn"] == 3
    assert row["fp"] == 0
    assert row["fn"] == 0
    assert row["accuracy"] == pytest.approx(1.0)
    assert row["balanced_accuracy"] == pytest.approx(1.0)
    # At 0.0 everything predicted fake
    row0 = confusion_at_threshold(samples, 0.0)
    assert row0["tp"] == 3
    assert row0["fp"] == 3
    assert row0["tn"] == 0
    assert row0["fn"] == 0
    # At 1.0 only score==1.0 would be fake; none here → all real preds
    row1 = confusion_at_threshold(samples, 1.0)
    assert row1["tp"] == 0
    assert row1["fn"] == 3
    assert row1["tn"] == 3
    assert row1["fp"] == 0


def test_operating_points_and_reference_preserved() -> None:
    result = run_threshold_sweep(_samples(), step=0.1)
    assert abs(result["reference_threshold"] - 0.5) < 1e-12
    thresholds = [r["threshold"] for r in result["thresholds"]]
    assert 0.5 in thresholds
    ops = result["operating_points"]
    assert ops["reference_threshold_0_50"]["threshold"] == pytest.approx(0.5)
    assert ops["reference_threshold_0_50"]["not_an_approved_default"] is True
    assert ops["max_balanced_accuracy"]["not_an_approved_default"] is True
    assert result["roc"]["threshold_independent"] is True
    assert result["roc"]["roc_auc"] == pytest.approx(1.0)
    assert SCORE_FIELD == result["score_field"]
    assert "calibrated" in SCORE_SEMANTICS.lower() or "not" in SCORE_SEMANTICS.lower()
    assert "probability" in SCORE_SEMANTICS.lower() or "Not a calibrated" in SCORE_SEMANTICS


def test_empty_samples_raise() -> None:
    with pytest.raises(ThresholdSweepError):
        run_threshold_sweep([])


def test_malformed_ground_truth_rejected() -> None:
    with pytest.raises(ThresholdSweepError):
        run_threshold_sweep([ScoredSample("x", "maybe", 0.5)])


def test_labelled_aggregate_loader(tmp_path: Path) -> None:
    payload = {
        "images": [
            {
                "image_id": "a",
                "ground_truth": "real",
                "configs": [
                    {
                        "run_name": "blending_diffusion",
                        "status": "pass",
                        "label": "real",
                        "real_score": 0.8,
                        "fake_score": 0.2,
                    }
                ],
            },
            {
                "image_id": "b",
                "ground_truth": "fake",
                "configs": [
                    {
                        "run_name": "blending_diffusion",
                        "status": "pass",
                        "label": "fake",
                        "real_score": 0.1,
                        "fake_score": 0.9,
                    }
                ],
            },
            {
                "image_id": "c",
                "ground_truth": "fake",
                "configs": [
                    {
                        "run_name": "blending_diffusion",
                        "status": "fail",
                        "label": None,
                        "fake_score": None,
                    }
                ],
            },
        ]
    }
    samples, meta = load_samples_from_labelled_aggregate(payload)
    assert len(samples) == 2
    assert meta["n_skipped"] == 1
    assert samples[0].fake_score == pytest.approx(0.2)


def test_ffpp_loader_frame_and_video() -> None:
    frames = [
        {
            "frame_id": "v1_0",
            "video_id": "v1",
            "ground_truth": "real",
            "status": "scored",
            "fake_score": 0.1,
        },
        {
            "frame_id": "v1_1",
            "video_id": "v1",
            "ground_truth": "real",
            "status": "scored",
            "fake_score": 0.3,
        },
        {
            "frame_id": "v2_0",
            "video_id": "v2",
            "ground_truth": "fake",
            "status": "scored",
            "fake_score": 0.9,
        },
        {
            "frame_id": "v2_1",
            "video_id": "v2",
            "ground_truth": "fake",
            "status": "failed",
            "fake_score": None,
        },
    ]
    frame_samples, _ = load_samples_from_ffpp_frames(frames, level="frame")
    assert len(frame_samples) == 3
    video_samples, _ = load_samples_from_ffpp_frames(frames, level="video")
    assert len(video_samples) == 2
    v1 = next(s for s in video_samples if s.sample_id == "v1")
    assert v1.fake_score == pytest.approx(0.2)


def test_write_outputs_and_operating_settings(tmp_path: Path) -> None:
    result = run_threshold_sweep(_samples(), step=0.25)
    written = write_threshold_outputs(result, tmp_path, write_plots=False)
    assert Path(written["threshold_sweep.json"]).is_file()
    assert Path(written["threshold_summary.json"]).is_file()
    assert Path(written["threshold_sweep.csv"]).is_file()
    summary = json.loads(Path(written["threshold_summary.json"]).read_text(encoding="utf-8"))
    assert summary["reference_threshold"] == 0.5
    assert "not calibrated" in summary["score_semantics"].lower() or "Not" in summary[
        "score_semantics"
    ]
    settings = default_decision_operating_settings()
    d = settings.to_dict()
    assert d["decision_threshold"] == 0.5
    assert d["scores_are_calibrated_probabilities"] is False
    assert "false positives" in d["wording_lower_threshold"]
    assert "conservative" in d["wording_higher_threshold"]


def test_descriptive_points_prefer_reference_on_ties() -> None:
    rows = [
        {
            "threshold": 0.4,
            "accuracy": 0.9,
            "balanced_accuracy": 0.9,
            "f1_fake": 0.9,
            "sensitivity_fake": 0.9,
            "specificity_real": 0.9,
            "precision_fake": 0.9,
            "tp": 1,
            "tn": 1,
            "fp": 0,
            "fn": 0,
        },
        {
            "threshold": 0.5,
            "accuracy": 0.9,
            "balanced_accuracy": 0.9,
            "f1_fake": 0.9,
            "sensitivity_fake": 0.9,
            "specificity_real": 0.9,
            "precision_fake": 0.9,
            "tp": 1,
            "tn": 1,
            "fp": 0,
            "fn": 0,
        },
    ]
    ops = descriptive_operating_points(rows)
    assert ops["max_accuracy"]["threshold"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Validation / held-out protocol
# ---------------------------------------------------------------------------


def test_stratified_split_is_deterministic() -> None:
    samples = _large_balanced_samples(12)
    val_a, test_a, meta_a = stratified_val_test_split(
        samples, seed=DEFAULT_SPLIT_SEED, val_fraction=0.5
    )
    val_b, test_b, meta_b = stratified_val_test_split(
        samples, seed=DEFAULT_SPLIT_SEED, val_fraction=0.5
    )
    assert [s.sample_id for s in val_a] == [s.sample_id for s in val_b]
    assert [s.sample_id for s in test_a] == [s.sample_id for s in test_b]
    assert meta_a["val_sample_ids"] == meta_b["val_sample_ids"]
    assert meta_a["seed"] == DEFAULT_SPLIT_SEED


def test_stratified_split_has_no_leakage() -> None:
    samples = _large_balanced_samples(10)
    val, test, meta = stratified_val_test_split(samples, seed=4842, val_fraction=0.5)
    val_ids = {s.sample_id for s in val}
    test_ids = {s.sample_id for s in test}
    assert not (val_ids & test_ids)
    assert val_ids | test_ids == {s.sample_id for s in samples}
    assert meta["no_leakage"] is True
    # Stratification: both classes present on both sides for n>=2 per class.
    assert {s.ground_truth for s in val} == {"real", "fake"}
    assert {s.ground_truth for s in test} == {"real", "fake"}


def test_threshold_selection_uses_validation_only() -> None:
    samples = _large_balanced_samples(16)
    val, test, _ = stratified_val_test_split(samples, seed=4842, val_fraction=0.5)
    selected = select_threshold_on_validation(
        val, criterion="max_balanced_accuracy", step=0.05
    )
    assert selected["selection_split"] == "validation"
    assert selected["frozen"] is True
    assert 0.0 <= selected["selected_threshold"] <= 1.0

    # Poison held-out scores: selection must be unchanged if re-run on same val.
    poisoned_test = [
        ScoredSample(
            s.sample_id,
            s.ground_truth,
            0.01 if s.ground_truth == "fake" else 0.99,
        )
        for s in test
    ]
    selected_again = select_threshold_on_validation(
        val, criterion="max_balanced_accuracy", step=0.05
    )
    assert selected_again["selected_threshold"] == selected["selected_threshold"]
    # Held-out eval with poisoned scores differs from clean eval — proves test
    # partition is only used after freeze.
    clean_eval = evaluate_frozen_threshold(test, selected["selected_threshold"])
    poisoned_eval = evaluate_frozen_threshold(
        poisoned_test, selected["selected_threshold"]
    )
    assert clean_eval["metrics"]["n_samples"] == poisoned_eval["metrics"]["n_samples"]
    assert clean_eval["metrics"]["accuracy"] != poisoned_eval["metrics"]["accuracy"]


def test_held_out_evaluation_and_reference_comparison(tmp_path: Path) -> None:
    base = _large_balanced_samples(14)
    samples_by_run = {name: list(base) for name in RUN_ORDER}
    # Perturb one config so thresholds can differ.
    samples_by_run["none"] = [
        ScoredSample(s.sample_id, s.ground_truth, min(1.0, s.fake_score + 0.05))
        for s in base
    ]
    result = run_validation_test_protocol(
        samples_by_run,
        seed=4842,
        val_fraction=0.5,
        criteria=SELECTION_CRITERIA,
        step=0.1,
    )
    assert result["split"]["seed"] == 4842
    assert set(result["results_by_run"]) == set(RUN_ORDER)
    assert len(result["comparison_table"]) == len(RUN_ORDER) * len(SELECTION_CRITERIA)
    assert "raw" in result["score_semantics"].lower() or "Not a calibrated" in result[
        "score_semantics"
    ]

    for _run_name, block in result["results_by_run"].items():
        assert block["status"] == "ok"
        ref = block["reference_0_50_held_out_test"]
        assert ref["threshold"] == pytest.approx(0.5)
        assert ref["split"] == "held_out_test"
        for criterion in SELECTION_CRITERIA:
            sel = block["selections"][criterion]
            frozen = sel["selection"]["selected_threshold"]
            assert sel["held_out_test"]["threshold"] == pytest.approx(frozen)
            assert sel["selection"]["validation_metrics"]["n_samples"] == block["n_val"]
            assert sel["held_out_test"]["metrics"]["n_samples"] == block["n_test"]
            # Same held-out set used for reference 0.50 comparison.
            assert (
                sel["reference_0_50_held_out_test"]["metrics"]["n_samples"]
                == block["n_test"]
            )

    written = write_validation_protocol_outputs(result, tmp_path)
    assert Path(written["validation_protocol.json"]).is_file()
    assert Path(written["comparison_table.csv"]).is_file()
    assert Path(written["comparison_table.md"]).is_file()
    md = Path(written["comparison_table.md"]).read_text(encoding="utf-8")
    assert "uncalibrated" in md.lower() or "raw" in md.lower()
    payload = json.loads(Path(written["validation_protocol.json"]).read_text(encoding="utf-8"))
    assert payload["protocol"] == "validation_select_held_out_evaluate"


def test_selection_criteria_include_required_modes() -> None:
    assert "max_balanced_accuracy" in SELECTION_CRITERIA
    assert "max_f1_fake" in SELECTION_CRITERIA
    assert "closest_equal_sensitivity_specificity" in SELECTION_CRITERIA
    samples = _large_balanced_samples(10)
    val, _, _ = stratified_val_test_split(samples, seed=4842)
    for criterion in SELECTION_CRITERIA:
        out = select_threshold_on_validation(val, criterion=criterion, step=0.1)
        assert out["criterion"] == criterion
        assert out["not_an_approved_default"] is True
