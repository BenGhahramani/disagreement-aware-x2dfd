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
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from eval.experiment_configs import PRIMARY_ASSESSMENT_RUN, RUN_ORDER
from eval.labelled_analysis import (
    AggregateAnalysisError,
    _roc_auc,
    classification_metrics,
    load_and_validate_aggregate,
)
from eval.reproducibility import get_git_provenance, utc_timestamp

SCRIPT_VERSION = "1.1.0"
DEFAULT_THRESHOLD_STEP = 0.01
REFERENCE_THRESHOLD = 0.50
DEFAULT_SPLIT_SEED = 4842
DEFAULT_VAL_FRACTION = 0.5
SCORE_FIELD = "fake_score"
SCORE_SEMANTICS = (
    "Raw model fake_score from saved inference outputs. Not a calibrated "
    "probability or confidence. Higher values indicate the model assigned "
    "relatively more mass to the fake token versus real at the scored step."
)
DECISION_RULE = "predict_fake_if_fake_score_ge_threshold"
SELECTION_CRITERIA: Tuple[str, ...] = (
    "max_balanced_accuracy",
    "max_f1_fake",
    "closest_equal_sensitivity_specificity",
)
SELECTION_CRITERION_TO_OP_KEY: Dict[str, str] = {
    "max_balanced_accuracy": "max_balanced_accuracy",
    "max_f1_fake": "max_f1_fake",
    "closest_equal_sensitivity_specificity": "closest_equal_sensitivity_specificity",
}

# Prototype operating-target used only to form interpretable dashboard presets
# from a validation sweep. Not a scientifically universal cut-off or calibrated
# confidence level.
PRESET_TARGET_RATE = 0.95
PRESET_SENSITIVE = "Sensitive"
PRESET_BALANCED = "Balanced"
PRESET_CONSERVATIVE = "Conservative"
PRESET_NAMES: Tuple[str, ...] = (PRESET_SENSITIVE, PRESET_BALANCED, PRESET_CONSERVATIVE)
PRESET_TARGET_NOTE = (
    "PRESET_TARGET_RATE is a prototype operating target used to create "
    "interpretable Sensitive / Conservative presets from the validation sweep. "
    "It is not a scientifically universal threshold, an optimal default, or a "
    "calibrated confidence level."
)


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


def _row_metric(row: Mapping[str, Any], key: str) -> Optional[float]:
    value = row.get(key)
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return number


def _better_fallback_row(
    row: Dict[str, Any],
    best: Optional[Dict[str, Any]],
    *,
    primary: str,
    secondary: str,
) -> bool:
    """True if ``row`` should replace ``best`` for a max-primary then max-secondary fallback."""

    if best is None:
        return True
    p = _row_metric(row, primary)
    bp = _row_metric(best, primary)
    if p is None:
        return False
    if bp is None or p > bp + 1e-15:
        return True
    if abs(p - bp) > 1e-15:
        return False
    s = _row_metric(row, secondary)
    bs = _row_metric(best, secondary)
    if s is None:
        return False
    if bs is None or s > bs + 1e-15:
        return True
    if abs(s - bs) > 1e-15:
        return False
    return abs(float(row["threshold"]) - REFERENCE_THRESHOLD) < abs(
        float(best["threshold"]) - REFERENCE_THRESHOLD
    )


def _preset_payload(
    *,
    name: str,
    row: Optional[Dict[str, Any]],
    target: str,
    target_met: bool,
    used_fallback: bool,
    provenance: str,
) -> Dict[str, Any]:
    metrics = dict(row) if row is not None else {}
    threshold = _row_metric(metrics, "threshold") if metrics else None
    return {
        "name": name,
        "threshold": threshold,
        "validation_metrics": metrics,
        "validation_sensitivity_fake": _row_metric(metrics, "sensitivity_fake"),
        "validation_specificity_real": _row_metric(metrics, "specificity_real"),
        "validation_balanced_accuracy": _row_metric(metrics, "balanced_accuracy"),
        "target": target,
        "target_rate": PRESET_TARGET_RATE,
        "target_met": target_met,
        "used_fallback": used_fallback,
        "provenance": provenance,
        "not_an_approved_default": True,
        "selection_split": "validation",
    }


def select_operating_presets(
    rows: Sequence[Dict[str, Any]],
    *,
    target_rate: float = PRESET_TARGET_RATE,
) -> Dict[str, Any]:
    """Derive Sensitive / Balanced / Conservative from validation sweep rows only.

    Held-out test fields, if present on the rows, are ignored. These presets are
    operating trade-offs on raw ``fake_score``, not calibrated confidence.
    """

    usable = [dict(r) for r in rows if _row_metric(r, "threshold") is not None]
    balanced_row = _argmax_metric(usable, "balanced_accuracy")

    qualifying_sens: List[Dict[str, Any]] = []
    for row in usable:
        sens = _row_metric(row, "sensitivity_fake")
        if sens is not None and sens + 1e-15 >= float(target_rate):
            qualifying_sens.append(row)
    if qualifying_sens:
        sensitive_row = max(qualifying_sens, key=lambda r: float(r["threshold"]))
        sensitive_met, sensitive_fallback = True, False
    else:
        sensitive_row = None
        for row in usable:
            if _better_fallback_row(
                row, sensitive_row, primary="sensitivity_fake", secondary="specificity_real"
            ):
                sensitive_row = row
        sensitive_met, sensitive_fallback = False, True

    qualifying_spec: List[Dict[str, Any]] = []
    for row in usable:
        spec = _row_metric(row, "specificity_real")
        if spec is not None and spec + 1e-15 >= float(target_rate):
            qualifying_spec.append(row)
    if qualifying_spec:
        conservative_row = min(qualifying_spec, key=lambda r: float(r["threshold"]))
        conservative_met, conservative_fallback = True, False
    else:
        conservative_row = None
        for row in usable:
            if _better_fallback_row(
                row,
                conservative_row,
                primary="specificity_real",
                secondary="sensitivity_fake",
            ):
                conservative_row = row
        conservative_met, conservative_fallback = False, True

    presets = {
        PRESET_SENSITIVE: _preset_payload(
            name=PRESET_SENSITIVE,
            row=sensitive_row,
            target=f"fake_sensitivity_ge_{target_rate:.2f}",
            target_met=sensitive_met,
            used_fallback=sensitive_fallback,
            provenance=(
                "Highest validation threshold with fake sensitivity ≥ "
                f"{target_rate:.0%} (prototype operating target). "
                "Validation-only; not a calibrated confidence."
                if not sensitive_fallback
                else (
                    f"No validation threshold reached fake sensitivity ≥ {target_rate:.0%}; "
                    "using maximum validation fake sensitivity, then best specificity. "
                    "Not a calibrated confidence."
                )
            ),
        ),
        PRESET_BALANCED: _preset_payload(
            name=PRESET_BALANCED,
            row=balanced_row,
            target="max_validation_balanced_accuracy",
            target_met=balanced_row is not None,
            used_fallback=False,
            provenance=(
                "Validation-selected for balanced sensitivity/specificity performance"
            ),
        ),
        PRESET_CONSERVATIVE: _preset_payload(
            name=PRESET_CONSERVATIVE,
            row=conservative_row,
            target=f"real_specificity_ge_{target_rate:.2f}",
            target_met=conservative_met,
            used_fallback=conservative_fallback,
            provenance=(
                "Lowest validation threshold with real specificity ≥ "
                f"{target_rate:.0%} (prototype operating target). "
                "Validation-only; not a calibrated confidence."
                if not conservative_fallback
                else (
                    f"No validation threshold reached real specificity ≥ {target_rate:.0%}; "
                    "using maximum validation specificity, then best fake sensitivity. "
                    "Not a calibrated confidence."
                )
            ),
        ),
    }

    thresholds = [
        presets[name]["threshold"]
        for name in PRESET_NAMES
        if presets[name]["threshold"] is not None
    ]
    unique = {round(float(t), 10) for t in thresholds}
    collapsed = len(unique) < len(thresholds)
    s_t = presets[PRESET_SENSITIVE]["threshold"]
    b_t = presets[PRESET_BALANCED]["threshold"]
    c_t = presets[PRESET_CONSERVATIVE]["threshold"]
    unexpected_order = False
    if s_t is not None and b_t is not None and c_t is not None:
        unexpected_order = not (s_t - 1e-12 <= b_t <= c_t + 1e-12)

    collapse_note = None
    if collapsed:
        collapse_note = (
            "For this validation set, these operating goals resolve to the same threshold."
        )
    order_note = None
    if unexpected_order:
        order_note = (
            "Unexpected preset order on this validation sweep "
            "(Sensitive / Balanced / Conservative are not non-decreasing). "
            "True selected thresholds are kept; they are not perturbed."
        )

    return {
        "presets": presets,
        "collapsed": collapsed,
        "unexpected_order": unexpected_order,
        "collapse_note": collapse_note,
        "order_note": order_note,
        "target_rate": float(target_rate),
        "target_note": PRESET_TARGET_NOTE,
        "selection_split": "validation",
        "score_semantics": SCORE_SEMANTICS,
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
# Validation / held-out test protocol (threshold selection without leakage)
# ---------------------------------------------------------------------------


def stratified_val_test_split(
    samples: Sequence[ScoredSample],
    *,
    seed: int = DEFAULT_SPLIT_SEED,
    val_fraction: float = DEFAULT_VAL_FRACTION,
) -> Tuple[List[ScoredSample], List[ScoredSample], Dict[str, Any]]:
    """Deterministic real/fake-stratified validation vs held-out test split.

    Within each label class, samples are sorted by ``sample_id`` then shuffled
    with ``random.Random(seed)`` so repeats are identical. Partitions are
    disjoint; every input sample is assigned to exactly one side.
    """

    if not samples:
        raise ThresholdSweepError("no samples to split")
    if not (0.0 < float(val_fraction) < 1.0):
        raise ThresholdSweepError("val_fraction must be in (0, 1)")

    by_label: Dict[str, List[ScoredSample]] = {"real": [], "fake": []}
    for sample in samples:
        if sample.ground_truth not in by_label:
            raise ThresholdSweepError(
                f"sample {sample.sample_id}: ground_truth must be real|fake"
            )
        by_label[sample.ground_truth].append(sample)

    rng = random.Random(int(seed))
    val: List[ScoredSample] = []
    test: List[ScoredSample] = []
    per_label: Dict[str, Dict[str, int]] = {}
    for label in ("real", "fake"):
        pool = sorted(by_label[label], key=lambda s: s.sample_id)
        rng.shuffle(pool)
        n_val = int(round(len(pool) * float(val_fraction)))
        # Keep both sides non-empty when the class has at least 2 samples.
        if len(pool) >= 2:
            n_val = min(max(n_val, 1), len(pool) - 1)
        elif len(pool) == 1:
            n_val = 1 if val_fraction >= 0.5 else 0
        val.extend(pool[:n_val])
        test.extend(pool[n_val:])
        per_label[label] = {"n_total": len(pool), "n_val": n_val, "n_test": len(pool) - n_val}

    val_sorted = sorted(val, key=lambda s: s.sample_id)
    test_sorted = sorted(test, key=lambda s: s.sample_id)
    val_ids = {s.sample_id for s in val_sorted}
    test_ids = {s.sample_id for s in test_sorted}
    if val_ids & test_ids:
        raise ThresholdSweepError("split leakage: overlapping sample_ids")
    if len(val_ids) + len(test_ids) != len({s.sample_id for s in samples}):
        # Duplicate IDs are ambiguous for a clean protocol.
        raise ThresholdSweepError("split requires unique sample_ids")

    meta = {
        "seed": int(seed),
        "val_fraction": float(val_fraction),
        "stratify_by": "ground_truth",
        "n_val": len(val_sorted),
        "n_test": len(test_sorted),
        "per_label": per_label,
        "val_sample_ids": [s.sample_id for s in val_sorted],
        "test_sample_ids": [s.sample_id for s in test_sorted],
        "no_leakage": True,
    }
    return val_sorted, test_sorted, meta


def partition_by_ids(
    samples: Sequence[ScoredSample],
    *,
    val_ids: Sequence[str],
    test_ids: Sequence[str],
) -> Tuple[List[ScoredSample], List[ScoredSample]]:
    """Apply a shared ID partition to one config's scored samples."""

    val_set = set(val_ids)
    test_set = set(test_ids)
    if val_set & test_set:
        raise ThresholdSweepError("partition_by_ids: overlapping val/test ids")
    val = sorted(
        [s for s in samples if s.sample_id in val_set],
        key=lambda s: s.sample_id,
    )
    test = sorted(
        [s for s in samples if s.sample_id in test_set],
        key=lambda s: s.sample_id,
    )
    return val, test


def select_threshold_on_validation(
    val_samples: Sequence[ScoredSample],
    *,
    criterion: str,
    step: float = DEFAULT_THRESHOLD_STEP,
) -> Dict[str, Any]:
    """Select a frozen threshold using validation scores only."""

    if criterion not in SELECTION_CRITERION_TO_OP_KEY:
        raise ThresholdSweepError(
            f"unsupported selection criterion: {criterion!r}; "
            f"expected one of {list(SELECTION_CRITERIA)}"
        )
    if not val_samples:
        raise ThresholdSweepError("validation set is empty")

    sweep = run_threshold_sweep(val_samples, step=step)
    op_key = SELECTION_CRITERION_TO_OP_KEY[criterion]
    point = sweep["operating_points"].get(op_key)
    if not isinstance(point, dict) or point.get("threshold") is None:
        raise ThresholdSweepError(
            f"could not select threshold for criterion={criterion!r} on validation"
        )
    frozen = float(point["threshold"])
    val_metrics = confusion_at_threshold(val_samples, frozen)
    return {
        "criterion": criterion,
        "selected_threshold": frozen,
        "frozen": True,
        "selection_split": "validation",
        "selection_note": (
            "Threshold chosen on the validation partition only; "
            "held-out test scores were not used for selection."
        ),
        "score_field": SCORE_FIELD,
        "score_semantics": SCORE_SEMANTICS,
        "validation_metrics": val_metrics,
        "validation_operating_point": point,
        "validation_roc": sweep["roc"],
        "not_an_approved_default": True,
    }


def evaluate_frozen_threshold(
    test_samples: Sequence[ScoredSample],
    threshold: float,
) -> Dict[str, Any]:
    """Evaluate a previously frozen threshold on held-out test scores."""

    if not test_samples:
        raise ThresholdSweepError("held-out test set is empty")
    metrics = confusion_at_threshold(test_samples, float(threshold))
    return {
        "threshold": float(threshold),
        "split": "held_out_test",
        "metrics": metrics,
        "roc": compute_roc_auc(test_samples),
        "score_field": SCORE_FIELD,
        "score_semantics": SCORE_SEMANTICS,
    }


def _comparison_row(
    *,
    run_name: str,
    criterion: str,
    selected_threshold: float,
    validation_metrics: Dict[str, Any],
    test_metrics: Dict[str, Any],
    reference_test_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    test_bal = test_metrics.get("balanced_accuracy")
    ref_bal = reference_test_metrics.get("balanced_accuracy")
    delta = None
    if test_bal is not None and ref_bal is not None:
        delta = float(test_bal) - float(ref_bal)
    return {
        "run_name": run_name,
        "criterion": criterion,
        "selected_threshold": selected_threshold,
        "reference_threshold": REFERENCE_THRESHOLD,
        "val_n": validation_metrics.get("n_samples"),
        "val_accuracy": validation_metrics.get("accuracy"),
        "val_balanced_accuracy": validation_metrics.get("balanced_accuracy"),
        "val_f1_fake": validation_metrics.get("f1_fake"),
        "val_sensitivity_fake": validation_metrics.get("sensitivity_fake"),
        "val_specificity_real": validation_metrics.get("specificity_real"),
        "test_n": test_metrics.get("n_samples"),
        "test_accuracy": test_metrics.get("accuracy"),
        "test_balanced_accuracy": test_metrics.get("balanced_accuracy"),
        "test_f1_fake": test_metrics.get("f1_fake"),
        "test_sensitivity_fake": test_metrics.get("sensitivity_fake"),
        "test_specificity_real": test_metrics.get("specificity_real"),
        "test_precision_fake": test_metrics.get("precision_fake"),
        "ref_0_50_test_accuracy": reference_test_metrics.get("accuracy"),
        "ref_0_50_test_balanced_accuracy": reference_test_metrics.get("balanced_accuracy"),
        "ref_0_50_test_f1_fake": reference_test_metrics.get("f1_fake"),
        "delta_test_balanced_accuracy_vs_0_50": delta,
        "score_semantics": "raw_uncalibrated_fake_score",
    }


def run_validation_test_protocol(
    samples_by_run: Dict[str, Sequence[ScoredSample]],
    *,
    seed: int = DEFAULT_SPLIT_SEED,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    criteria: Sequence[str] = SELECTION_CRITERIA,
    step: float = DEFAULT_THRESHOLD_STEP,
    run_order: Sequence[str] = RUN_ORDER,
) -> Dict[str, Any]:
    """Select thresholds on validation; evaluate frozen thresholds on held-out test.

    A single stratified ID split is shared across configs so comparisons use the
    same images. Selection never sees held-out test scores.
    """

    if not samples_by_run:
        raise ThresholdSweepError("samples_by_run is empty")

    # Build a canonical labelled ID list from the union of runs (consistent GT).
    gt_by_id: Dict[str, str] = {}
    for run_name, samples in samples_by_run.items():
        for sample in samples:
            prev = gt_by_id.get(sample.sample_id)
            if prev is not None and prev != sample.ground_truth:
                raise ThresholdSweepError(
                    f"inconsistent ground_truth for {sample.sample_id}: "
                    f"{prev} vs {sample.ground_truth} ({run_name})"
                )
            gt_by_id[sample.sample_id] = sample.ground_truth

    canonical = [
        ScoredSample(sample_id=sid, ground_truth=gt, fake_score=0.0)
        for sid, gt in sorted(gt_by_id.items())
    ]
    _, _, split_meta = stratified_val_test_split(
        canonical, seed=seed, val_fraction=val_fraction
    )
    val_ids = list(split_meta["val_sample_ids"])
    test_ids = list(split_meta["test_sample_ids"])

    criteria_list = list(criteria)
    for name in criteria_list:
        if name not in SELECTION_CRITERION_TO_OP_KEY:
            raise ThresholdSweepError(f"unsupported selection criterion: {name!r}")

    results_by_run: Dict[str, Any] = {}
    comparison_rows: List[Dict[str, Any]] = []

    for run_name in run_order:
        samples = list(samples_by_run.get(run_name) or [])
        if not samples:
            results_by_run[run_name] = {
                "run_name": run_name,
                "status": "skipped_no_scores",
                "selections": {},
            }
            continue
        val_samples, test_samples = partition_by_ids(
            samples, val_ids=val_ids, test_ids=test_ids
        )
        if not val_samples or not test_samples:
            results_by_run[run_name] = {
                "run_name": run_name,
                "status": "skipped_empty_split",
                "n_val": len(val_samples),
                "n_test": len(test_samples),
                "selections": {},
            }
            continue

        reference_test = evaluate_frozen_threshold(test_samples, REFERENCE_THRESHOLD)
        sweep = run_threshold_sweep(val_samples, step=step)
        val_rows = list(sweep["thresholds"])
        ops = sweep["operating_points"]
        selections: Dict[str, Any] = {}
        for criterion in criteria_list:
            op_key = SELECTION_CRITERION_TO_OP_KEY[criterion]
            point = ops.get(op_key)
            if not isinstance(point, dict) or point.get("threshold") is None:
                continue
            frozen = float(point["threshold"])
            selected = {
                "criterion": criterion,
                "selected_threshold": frozen,
                "frozen": True,
                "selection_split": "validation",
                "selection_note": (
                    "Threshold chosen on the validation partition only; "
                    "held-out test scores were not used for selection."
                ),
                "score_field": SCORE_FIELD,
                "score_semantics": SCORE_SEMANTICS,
                "validation_metrics": confusion_at_threshold(val_samples, frozen),
                "validation_operating_point": point,
                "validation_roc": sweep["roc"],
                "not_an_approved_default": True,
            }
            held_out = evaluate_frozen_threshold(test_samples, frozen)
            row = _comparison_row(
                run_name=run_name,
                criterion=criterion,
                selected_threshold=frozen,
                validation_metrics=selected["validation_metrics"],
                test_metrics=held_out["metrics"],
                reference_test_metrics=reference_test["metrics"],
            )
            comparison_rows.append(row)
            selections[criterion] = {
                "selection": selected,
                "held_out_test": held_out,
                "reference_0_50_held_out_test": reference_test,
                "comparison": row,
            }

        preset_pack = select_operating_presets(val_rows)
        dashboard_presets: Dict[str, Any] = {}
        for name, preset in preset_pack["presets"].items():
            entry = dict(preset)
            thr = preset.get("threshold")
            if thr is not None:
                entry["held_out_test"] = evaluate_frozen_threshold(test_samples, float(thr))
            dashboard_presets[name] = entry

        results_by_run[run_name] = {
            "run_name": run_name,
            "status": "ok",
            "n_val": len(val_samples),
            "n_test": len(test_samples),
            "n_real_val": sum(1 for s in val_samples if s.ground_truth == "real"),
            "n_fake_val": sum(1 for s in val_samples if s.ground_truth == "fake"),
            "n_real_test": sum(1 for s in test_samples if s.ground_truth == "real"),
            "n_fake_test": sum(1 for s in test_samples if s.ground_truth == "fake"),
            "reference_0_50_held_out_test": reference_test,
            "selections": selections,
            "validation_sweep": val_rows,
            "dashboard_presets": dashboard_presets,
            "preset_notes": {
                "collapsed": preset_pack["collapsed"],
                "unexpected_order": preset_pack["unexpected_order"],
                "collapse_note": preset_pack["collapse_note"],
                "order_note": preset_pack["order_note"],
                "target_rate": preset_pack["target_rate"],
                "target_note": preset_pack["target_note"],
            },
        }

    return {
        "script_version": SCRIPT_VERSION,
        "protocol": "validation_select_held_out_evaluate",
        "score_field": SCORE_FIELD,
        "score_semantics": SCORE_SEMANTICS,
        "decision_rule": DECISION_RULE,
        "reference_threshold": REFERENCE_THRESHOLD,
        "split": split_meta,
        "criteria": criteria_list,
        "step": float(step),
        "run_order": list(run_order),
        "results_by_run": results_by_run,
        "comparison_table": comparison_rows,
        "notes": [
            "Thresholds are selected on the validation partition only.",
            "Held-out test metrics use the frozen validation-selected threshold.",
            "Fixed 0.50 reference is evaluated on the same held-out test partition.",
            "Scores are raw uncalibrated model fake_score values, not probabilities.",
            "Selected thresholds are analysis outputs, not approved production defaults.",
        ],
    }


def load_samples_by_run_from_labelled_aggregate(
    payload: Any,
    *,
    run_order: Sequence[str] = RUN_ORDER,
) -> Tuple[Dict[str, List[ScoredSample]], Dict[str, Any]]:
    """Load scored samples for every config cell in a labelled aggregate."""

    by_run: Dict[str, List[ScoredSample]] = {}
    meta_by_run: Dict[str, Any] = {}
    for run_name in run_order:
        samples, meta = load_samples_from_labelled_aggregate(payload, run_name=run_name)
        by_run[run_name] = samples
        meta_by_run[run_name] = meta
    return by_run, {
        "loader": "labelled_aggregate_all_runs",
        "run_order": list(run_order),
        "per_run": meta_by_run,
    }


def format_comparison_table_markdown(rows: Sequence[Dict[str, Any]]) -> str:
    """Concise markdown comparison table for validation-protocol results."""

    headers = [
        "run",
        "criterion",
        "thr",
        "val_bAcc",
        "test_bAcc",
        "test_F1",
        "ref0.50_bAcc",
        "ΔbAcc",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    def _fmt(value: Any, digits: int = 3) -> str:
        if value is None:
            return "—"
        if isinstance(value, float):
            return f"{value:.{digits}f}"
        return str(value)

    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("run_name", "")),
                    str(row.get("criterion", "")),
                    _fmt(row.get("selected_threshold"), 2),
                    _fmt(row.get("val_balanced_accuracy")),
                    _fmt(row.get("test_balanced_accuracy")),
                    _fmt(row.get("test_f1_fake")),
                    _fmt(row.get("ref_0_50_test_balanced_accuracy")),
                    _fmt(row.get("delta_test_balanced_accuracy_vs_0_50")),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append(
        "_Scores are raw uncalibrated `fake_score` values. "
        "Thresholds selected on validation only; metrics above include "
        "held-out test evaluation of the frozen threshold vs fixed 0.50._"
    )
    return "\n".join(lines) + "\n"


def write_validation_protocol_outputs(
    result: Dict[str, Any],
    output_dir: Path,
    *,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """Write validation-protocol JSON/CSV/markdown comparison artefacts."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": utc_timestamp(),
        "git": get_git_provenance(),
        "meta": meta or {},
        **result,
    }
    json_path = out / "validation_protocol.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    rows = list(result.get("comparison_table") or [])
    csv_path = out / "comparison_table.csv"
    fieldnames = [
        "run_name",
        "criterion",
        "selected_threshold",
        "reference_threshold",
        "val_n",
        "val_accuracy",
        "val_balanced_accuracy",
        "val_f1_fake",
        "val_sensitivity_fake",
        "val_specificity_real",
        "test_n",
        "test_accuracy",
        "test_balanced_accuracy",
        "test_f1_fake",
        "test_sensitivity_fake",
        "test_specificity_real",
        "test_precision_fake",
        "ref_0_50_test_accuracy",
        "ref_0_50_test_balanced_accuracy",
        "ref_0_50_test_f1_fake",
        "delta_test_balanced_accuracy_vs_0_50",
        "score_semantics",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})

    md_path = out / "comparison_table.md"
    md_path.write_text(format_comparison_table_markdown(rows), encoding="utf-8")

    # Full protocol CSV alias for tooling that expects protocol-named files.
    protocol_csv = out / "validation_protocol.csv"
    protocol_csv.write_text(csv_path.read_text(encoding="utf-8"), encoding="utf-8")

    return {
        "validation_protocol.json": str(json_path.resolve()),
        "validation_protocol.csv": str(protocol_csv.resolve()),
        "comparison_table.csv": str(csv_path.resolve()),
        "comparison_table.md": str(md_path.resolve()),
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
