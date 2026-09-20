"""Unit tests for threshold-sweep analysis (no GPU)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.threshold_sweep import (
    REFERENCE_THRESHOLD,
    SCORE_FIELD,
    SCORE_SEMANTICS,
    ScoredSample,
    ThresholdSweepError,
    confusion_at_threshold,
    default_decision_operating_settings,
    descriptive_operating_points,
    load_samples_from_ffpp_frames,
    load_samples_from_labelled_aggregate,
    predict_label,
    run_threshold_sweep,
    threshold_grid,
    write_threshold_outputs,
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
