"""Threshold-sweep analysis over saved fake scores (no inference).

Applies a user-adjustable decision rule to raw model ``fake_score`` values
already stored in evaluation outputs. Scores are **not** calibrated
probabilities or confidence levels.

Default decision rule (documented, not a claim of optimality)::

    predict fake  if fake_score >= threshold
    predict real  otherwise

``threshold = 0.50`` is preserved as the explicit reference operating point
used by existing post-hoc score→label conventions in this repo.
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from eval.experiment_configs import PRIMARY_ASSESSMENT_RUN
from eval.labelled_analysis import (
    AggregateAnalysisError,
    _roc_auc,
    classification_metrics,
    load_and_validate_aggregate,
)
from eval.reproducibility import get_git_provenance, utc_timestamp

SCRIPT_VERSION = "1.0.0"
DEFAULT_THRESHOLD_STEP = 0.01
REFERENCE_THRESHOLD = 0.50
SCORE_FIELD = "fake_score"
SCORE_SEMANTICS = (
    "Raw model fake_score from saved inference outputs. Not a calibrated "
    "probability or confidence. Higher values indicate the model assigned "
    "relatively more mass to the fake token versus real at the scored step."
)
DECISION_RULE = "predict_fake_if_fake_score_ge_threshold"


class ThresholdSweepError(ValueError):
    """Threshold-sweep input is missing required fields or is malformed."""


@dataclass(frozen=True)
class ScoredSample:
    """One labelled sample with a usable raw fake score."""

    sample_id: str
    ground_truth: str  # real | fake
    fake_score: float
    source: str = ""
    run_name: Optional[str] = None
    extras: Dict[str, Any] = field(default_factory=dict)


def predict_label(fake_score: float, threshold: float) -> str:
    """Apply the documented decision rule to a raw fake score."""

    if not math.isfinite(fake_score):
        raise ThresholdSweepError(f"non-finite fake_score: {fake_score!r}")
    if not math.isfinite(threshold):
        raise ThresholdSweepError(f"non-finite threshold: {threshold!r}")
    return "fake" if float(fake_score) >= float(threshold) else "real"


def threshold_grid(
    *,
    start: float = 0.0,
    stop: float = 1.0,
    step: float = DEFAULT_THRESHOLD_STEP,
) -> List[float]:
    """Inclusive [start, stop] grid in ``step`` increments (float-safe)."""

    if step <= 0:
        raise ThresholdSweepError("step must be positive")
    if stop < start:
        raise ThresholdSweepError("stop must be >= start")
    n_steps = int(round((stop - start) / step))
    values = [round(start + i * step, 10) for i in range(n_steps + 1)]
    if values[-1] < stop - 1e-12:
        values.append(round(stop, 10))
    # Deduplicate while preserving order.
    out: List[float] = []
    seen = set()
    for value in values:
        key = round(value, 10)
        if key not in seen:
            seen.add(key)
            out.append(float(key))
    return out


def confusion_at_threshold(
    samples: Sequence[ScoredSample],
    threshold: float,
) -> Dict[str, Any]:
    """Confusion counts + classification metrics at one threshold."""

    tp = tn = fp = fn = 0
    for sample in samples:
        pred = predict_label(sample.fake_score, threshold)
        truth = sample.ground_truth
        if truth == "fake" and pred == "fake":
            tp += 1
        elif truth == "real" and pred == "real":
            tn += 1
        elif truth == "real" and pred == "fake":
            fp += 1
        elif truth == "fake" and pred == "real":
            fn += 1
        else:
            raise ThresholdSweepError(
                f"unexpected ground_truth={truth!r} for sample {sample.sample_id}"
            )
    metrics = classification_metrics(tp=tp, tn=tn, fp=fp, fn=fn)
    metrics.update(
        {
            "threshold": float(threshold),
            "n_samples": len(samples),
            "decision_rule": DECISION_RULE,
        }
    )
    return metrics


def _abs_sens_spec_gap(row: Dict[str, Any]) -> float:
    sens = row.get("sensitivity_fake")
    spec = row.get("specificity_real")
    if sens is None or spec is None:
        return float("inf")
    return abs(float(sens) - float(spec))


def _argmax_metric(
    rows: Sequence[Dict[str, Any]],
    key: str,
) -> Optional[Dict[str, Any]]:
    best: Optional[Dict[str, Any]] = None
    best_val = float("-inf")
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        value_f = float(value)
        if value_f > best_val + 1e-15:
            best_val = value_f
            best = row
        elif abs(value_f - best_val) <= 1e-15 and best is not None:
            # Tie-break: prefer threshold closer to REFERENCE_THRESHOLD.
            if abs(row["threshold"] - REFERENCE_THRESHOLD) < abs(
                best["threshold"] - REFERENCE_THRESHOLD
            ):
                best = row
    return best


def descriptive_operating_points(
    rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Descriptive extrema only — not approved production defaults."""

    max_bal = _argmax_metric(rows, "balanced_accuracy")
    max_acc = _argmax_metric(rows, "accuracy")
    max_f1 = _argmax_metric(rows, "f1_fake")
    equalish: Optional[Dict[str, Any]] = None
    best_gap = float("inf")
    for row in rows:
        gap = _abs_sens_spec_gap(row)
        if gap < best_gap - 1e-15:
            best_gap = gap
            equalish = row
        elif abs(gap - best_gap) <= 1e-15 and equalish is not None:
            if abs(row["threshold"] - REFERENCE_THRESHOLD) < abs(
                equalish["threshold"] - REFERENCE_THRESHOLD
            ):
                equalish = row

    def _point(row: Optional[Dict[str, Any]], *, role: str) -> Optional[Dict[str, Any]]:
        if row is None:
            return None
        return {
            "role": role,
            "threshold": row["threshold"],
            "accuracy": row.get("accuracy"),
            "balanced_accuracy": row.get("balanced_accuracy"),
            "sensitivity_fake": row.get("sensitivity_fake"),
            "specificity_real": row.get("specificity_real"),
            "precision_fake": row.get("precision_fake"),
            "f1_fake": row.get("f1_fake"),
            "tp": row.get("tp"),
            "tn": row.get("tn"),
            "fp": row.get("fp"),
            "fn": row.get("fn"),
            "not_an_approved_default": True,
        }

    ref = next((r for r in rows if abs(r["threshold"] - REFERENCE_THRESHOLD) < 1e-12), None)
    return {
        "note": (
            "Descriptive operating points from the analysed score set only. "
            "Do not treat max-* thresholds as chosen production defaults; "
            "they are not independently validated and may overfit this set."
        ),
        "reference_threshold_0_50": _point(ref, role="reference_0_50"),
        "max_balanced_accuracy": _point(max_bal, role="max_balanced_accuracy"),
        "max_accuracy": _point(max_acc, role="max_accuracy"),
        "max_f1_fake": _point(max_f1, role="max_f1_fake"),
        "closest_equal_sensitivity_specificity": _point(
            equalish, role="closest_equal_sens_spec"
        ),
    }


def compute_roc_auc(samples: Sequence[ScoredSample]) -> Dict[str, Any]:
    scores = [s.fake_score for s in samples]
    labels = [1 if s.ground_truth == "fake" else 0 for s in samples]
    auc, reason = _roc_auc(scores, labels)
    return {
        "roc_auc": auc,
        "roc_auc_unavailable_reason": reason,
        "threshold_independent": True,
        "n_samples": len(samples),
        "n_positive_fake": sum(labels),
        "n_negative_real": len(labels) - sum(labels),
        "score_field": SCORE_FIELD,
        "score_semantics": SCORE_SEMANTICS,
    }


def run_threshold_sweep(
    samples: Sequence[ScoredSample],
    *,
    thresholds: Optional[Sequence[float]] = None,
    step: float = DEFAULT_THRESHOLD_STEP,
) -> Dict[str, Any]:
    """Sweep thresholds and return rows + descriptive operating points + AUC."""

    if not samples:
        raise ThresholdSweepError("no usable scored samples")
    for sample in samples:
        if sample.ground_truth not in {"real", "fake"}:
            raise ThresholdSweepError(
                f"sample {sample.sample_id}: ground_truth must be real|fake"
            )
        if not isinstance(sample.fake_score, (int, float)) or not math.isfinite(
            float(sample.fake_score)
        ):
            raise ThresholdSweepError(
                f"sample {sample.sample_id}: fake_score must be finite"
            )

    grid = list(thresholds) if thresholds is not None else threshold_grid(step=step)
    if REFERENCE_THRESHOLD not in {round(t, 10) for t in grid}:
        grid = sorted(set(list(grid) + [REFERENCE_THRESHOLD]))

    rows = [confusion_at_threshold(samples, t) for t in grid]
    return {
        "script_version": SCRIPT_VERSION,
        "score_field": SCORE_FIELD,
        "score_semantics": SCORE_SEMANTICS,
        "decision_rule": DECISION_RULE,
        "reference_threshold": REFERENCE_THRESHOLD,
        "n_samples": len(samples),
        "n_real": sum(1 for s in samples if s.ground_truth == "real"),
        "n_fake": sum(1 for s in samples if s.ground_truth == "fake"),
        "thresholds": rows,
        "operating_points": descriptive_operating_points(rows),
        "roc": compute_roc_auc(samples),
    }


# ---------------------------------------------------------------------------
# Loaders / adapters for existing result layouts
# ---------------------------------------------------------------------------


def load_samples_from_labelled_aggregate(
    payload: Any,
    *,
    run_name: str = PRIMARY_ASSESSMENT_RUN,
) -> Tuple[List[ScoredSample], Dict[str, Any]]:
    """Adapter for labelled-evaluation ``aggregate.json`` layouts."""

    try:
        normalised = load_and_validate_aggregate(payload)
    except AggregateAnalysisError as exc:
        raise ThresholdSweepError(str(exc)) from exc

    samples: List[ScoredSample] = []
    skipped = 0
    for image in normalised["images"]:
        cfg = image.get("_configs_by_name", {}).get(run_name)
        if not isinstance(cfg, dict):
            skipped += 1
            continue
        score = cfg.get(SCORE_FIELD)
        if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
            skipped += 1
            continue
        gt = image.get("ground_truth")
        if gt not in {"real", "fake"}:
            skipped += 1
            continue
        sample_id = str(image.get("image_id") or image.get("source_path") or len(samples))
        samples.append(
            ScoredSample(
                sample_id=sample_id,
                ground_truth=gt,
                fake_score=float(score),
                source="labelled_aggregate",
                run_name=run_name,
                extras={
                    "saved_label": cfg.get("label"),
                    "real_score": cfg.get("real_score"),
                    "config_status": cfg.get("status"),
                    "dataset": image.get("dataset"),
                    "manipulation": image.get("manipulation"),
                },
            )
        )
    meta = {
        "loader": "labelled_aggregate",
        "run_name": run_name,
        "n_images_in_aggregate": len(normalised["images"]),
        "n_usable_samples": len(samples),
        "n_skipped": skipped,
    }
    return samples, meta


def load_samples_from_ffpp_frames(
    frames: Sequence[Dict[str, Any]],
    *,
    level: str = "frame",
) -> Tuple[List[ScoredSample], Dict[str, Any]]:
    """Adapter for FF++ preparation/score rows (frame or video mean)."""

    level_n = level.strip().lower()
    if level_n not in {"frame", "video"}:
        raise ThresholdSweepError("level must be 'frame' or 'video'")

    usable_frames: List[Dict[str, Any]] = []
    skipped = 0
    for row in frames:
        score = row.get(SCORE_FIELD)
        gt = row.get("ground_truth")
        status = row.get("status")
        if status == "failed" or score is None:
            skipped += 1
            continue
        if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
            skipped += 1
            continue
        if gt not in {"real", "fake"}:
            skipped += 1
            continue
        usable_frames.append(row)

    samples: List[ScoredSample] = []
    if level_n == "frame":
        for row in usable_frames:
            fid = str(row.get("frame_id") or len(samples))
            samples.append(
                ScoredSample(
                    sample_id=fid,
                    ground_truth=row["ground_truth"],
                    fake_score=float(row[SCORE_FIELD]),
                    source="ffpp_frame",
                    extras={
                        "video_id": row.get("video_id"),
                        "manipulation": row.get("manipulation"),
                        "saved_label": row.get("label"),
                    },
                )
            )
    else:
        by_video: Dict[str, List[Dict[str, Any]]] = {}
        for row in usable_frames:
            vid = row.get("video_id")
            if not isinstance(vid, str) or not vid:
                skipped += 1
                continue
            by_video.setdefault(vid, []).append(row)
        for vid, rows in sorted(by_video.items()):
            scores = [float(r[SCORE_FIELD]) for r in rows]
            gt = rows[0]["ground_truth"]
            mean_score = sum(scores) / len(scores)
            samples.append(
                ScoredSample(
                    sample_id=vid,
                    ground_truth=gt,
                    fake_score=float(mean_score),
                    source="ffpp_video_mean",
                    extras={
                        "n_frames": len(scores),
                        "manipulation": rows[0].get("manipulation"),
                        "aggregation": "mean_frame_fake_score",
                    },
                )
            )

    meta = {
        "loader": "ffpp_frames",
        "level": level_n,
        "n_input_rows": len(frames),
        "n_usable_samples": len(samples),
        "n_skipped": skipped,
    }
    return samples, meta


def load_ffpp_frame_rows(
    *,
    prep_dir: Optional[Path] = None,
    scores_path: Optional[Path] = None,
    merged_frames_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Merge FF++ preparation + scores, or load analysis merged_frames.json."""

    if merged_frames_path is not None:
        payload = json.loads(Path(merged_frames_path).read_text(encoding="utf-8"))
        frames = payload.get("frames") if isinstance(payload, dict) else None
        if not isinstance(frames, list):
            raise ThresholdSweepError("merged_frames.json must contain frames[]")
        return frames

    if prep_dir is None or scores_path is None:
        raise ThresholdSweepError(
            "provide merged_frames_path or both prep_dir and scores_path"
        )
    prep = json.loads((Path(prep_dir) / "preparation_manifest.json").read_text(encoding="utf-8"))
    prep_frames = prep.get("frames") or []
    scored_by_id: Dict[str, Dict[str, Any]] = {}
    for line in Path(scores_path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        fid = row.get("frame_id")
        if isinstance(fid, str):
            scored_by_id[fid] = row
    merged: List[Dict[str, Any]] = []
    for row in prep_frames:
        base = dict(row)
        fid = base.get("frame_id")
        if isinstance(fid, str) and fid in scored_by_id:
            scored = scored_by_id[fid]
            base.update(
                {
                    "status": scored.get("status", "scored"),
                    "fake_score": scored.get("fake_score"),
                    "real_score": scored.get("real_score"),
                    "label": scored.get("label"),
                    "failure_reason": scored.get("failure_reason")
                    or base.get("failure_reason"),
                }
            )
        elif base.get("status") != "failed":
            base["status"] = "failed"
            base["failure_reason"] = base.get("failure_reason") or "not_scored"
        merged.append(base)
    return merged


def write_threshold_outputs(
    result: Dict[str, Any],
    output_dir: Path,
    *,
    meta: Optional[Dict[str, Any]] = None,
    write_plots: bool = True,
) -> Dict[str, str]:
    """Write JSON/CSV (+ optional plots). Returns relative output paths."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": utc_timestamp(),
        "git": get_git_provenance(),
        "meta": meta or {},
        **result,
    }
    sweep_path = out / "threshold_sweep.json"
    summary = {
        "generated_at": payload["generated_at"],
        "git": payload["git"],
        "meta": payload["meta"],
        "script_version": result["script_version"],
        "score_field": result["score_field"],
        "score_semantics": result["score_semantics"],
        "decision_rule": result["decision_rule"],
        "reference_threshold": result["reference_threshold"],
        "n_samples": result["n_samples"],
        "n_real": result["n_real"],
        "n_fake": result["n_fake"],
        "roc": result["roc"],
        "operating_points": result["operating_points"],
        "reference_row": next(
            (
                r
                for r in result["thresholds"]
                if abs(r["threshold"] - REFERENCE_THRESHOLD) < 1e-12
            ),
            None,
        ),
    }
    summary_path = out / "threshold_summary.json"
    csv_path = out / "threshold_sweep.csv"

    sweep_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    fieldnames = [
        "threshold",
        "n_samples",
        "accuracy",
        "balanced_accuracy",
        "sensitivity_fake",
        "specificity_real",
        "precision_fake",
        "f1_fake",
        "tp",
        "tn",
        "fp",
        "fn",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in result["thresholds"]:
            writer.writerow({k: row.get(k) for k in fieldnames})

    written = {
        "threshold_sweep.json": str(sweep_path.resolve()),
        "threshold_summary.json": str(summary_path.resolve()),
        "threshold_sweep.csv": str(csv_path.resolve()),
    }
    if write_plots:
        written.update(_write_plots(result, out))
    return written


def _write_plots(result: Dict[str, Any], output_dir: Path) -> Dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = Path(output_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    rows = result["thresholds"]
    xs = [r["threshold"] for r in rows]
    written: Dict[str, str] = {}

    fig, ax = plt.subplots()
    ax.plot(xs, [r.get("accuracy") for r in rows], label="accuracy")
    ax.plot(xs, [r.get("balanced_accuracy") for r in rows], label="balanced accuracy")
    ax.axvline(REFERENCE_THRESHOLD, linestyle="--", linewidth=1, label="reference 0.50")
    ax.set_xlabel("decision threshold on raw fake_score")
    ax.set_ylabel("metric")
    ax.set_title("Threshold vs accuracy / balanced accuracy\n(raw fake_score; not calibrated)")
    ax.legend()
    path = fig_dir / "threshold_vs_accuracy.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    written["figures/threshold_vs_accuracy.png"] = str(path.resolve())

    fig, ax = plt.subplots()
    ax.plot(xs, [r.get("sensitivity_fake") for r in rows], label="sensitivity (fake)")
    ax.plot(xs, [r.get("specificity_real") for r in rows], label="specificity (real)")
    ax.axvline(REFERENCE_THRESHOLD, linestyle="--", linewidth=1, label="reference 0.50")
    ax.set_xlabel("decision threshold on raw fake_score")
    ax.set_ylabel("rate")
    ax.set_title(
        "Threshold vs sensitivity / specificity\n"
        "(lower threshold → more fake predictions; not calibrated confidence)"
    )
    ax.legend()
    path = fig_dir / "threshold_vs_sens_spec.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    written["figures/threshold_vs_sens_spec.png"] = str(path.resolve())

    # Optional ROC from ranks (simple staircase from sorted scores).
    samples_note = result.get("roc") or {}
    if samples_note.get("roc_auc") is not None:
        # Reconstruct a coarse ROC from threshold sweep confusion counts.
        fig, ax = plt.subplots()
        fpr = []
        tpr = []
        for row in rows:
            fp = row["fp"]
            tn = row["tn"]
            tp = row["tp"]
            fn = row["fn"]
            fpr.append(fp / (fp + tn) if (fp + tn) else 0.0)
            tpr.append(tp / (tp + fn) if (tp + fn) else 0.0)
        ax.plot(fpr, tpr, label=f"ROC (AUC≈{samples_note['roc_auc']:.3f})")
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="chance")
        ax.set_xlabel("false positive rate (1 − specificity)")
        ax.set_ylabel("true positive rate (sensitivity)")
        ax.set_title("ROC from threshold sweep on raw fake_score")
        ax.legend()
        path = fig_dir / "roc_curve.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        written["figures/roc_curve.png"] = str(path.resolve())

    return written


@dataclass(frozen=True)
class DecisionOperatingSettings:
    """User-adjustable operating settings for dashboard provenance (not calibration)."""

    decision_threshold: float = REFERENCE_THRESHOLD
    decision_rule: str = DECISION_RULE
    score_field: str = SCORE_FIELD
    expert_configuration: str = PRIMARY_ASSESSMENT_RUN
    scores_are_calibrated_probabilities: bool = False
    wording_lower_threshold: str = (
        "Lower decision thresholds classify more samples as fake "
        "(higher fake sensitivity) but may increase false positives on genuine media."
    )
    wording_higher_threshold: str = (
        "Higher decision thresholds are more conservative about calling fake "
        "and may miss more manipulated samples (lower fake sensitivity)."
    )
    wording_not_confidence: str = (
        "The decision threshold is a user-adjustable operating setting on the "
        "raw fake_score axis. It is not a calibrated confidence level."
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def default_decision_operating_settings(
    *,
    decision_threshold: float = REFERENCE_THRESHOLD,
    expert_configuration: str = PRIMARY_ASSESSMENT_RUN,
) -> DecisionOperatingSettings:
    return DecisionOperatingSettings(
        decision_threshold=float(decision_threshold),
        expert_configuration=expert_configuration,
    )
