"""Analysis helpers for labelled-evaluation aggregate outputs.

Pure functions over aggregate JSON. Does not run inference or modify raw
runner outputs. Metrics treat specialist detector scores as detector scores
(not calibrated probabilities). Calibration diagnostics apply only to main
model real/fake scores and are suppressed below a configurable sample floor.
"""
from __future__ import annotations

import statistics
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dashboard.view_model import EXPERT_HI, EXPERT_LO, _specialist_relation
from eval.experiment_configs import RUN_ORDER
from eval.reproducibility import interpretation_thresholds

GROUND_TRUTH_VALUES = frozenset({"real", "fake"})
BASELINE_RUN = "none"
COMBINED_RUN = "blending_diffusion"
DEFAULT_CALIBRATION_MIN_SAMPLES = 30
DEFAULT_ROC_MIN_PER_CLASS = 2

EVIDENCE_AGREEMENT_CATEGORIES: Tuple[str, ...] = (
    "agreement",
    "conflict",
    "insufficient evidence",
)
PROTOTYPE_STATUS_ORDER: Tuple[str, ...] = ("Stable", "Uncertain", "Contested")
DEEPFAKEFACE_MANIPULATION_ORDER: Tuple[str, ...] = (
    "wiki_real",
    "insight",
    "inpainting",
    "text2img",
)
DEEPFAKEFACE_DATASET_NAMES = frozenset({"deepfakeface", "deepfake-face", "deep fake face"})

PILOT_WARNING = (
    "Pilot contains 4 images and is for pipeline validation only; results are not "
    "sufficient for statistical conclusions."
)


class AggregateAnalysisError(ValueError):
    """Aggregate payload is missing required fields or is malformed."""


def _safe_div(num: float, den: float) -> Optional[float]:
    if den == 0:
        return None
    return num / den


def _median(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(statistics.median(values))


def _mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(statistics.fmean(values))


def _stdev(values: Sequence[float]) -> Optional[float]:
    if len(values) < 2:
        return None
    return float(statistics.stdev(values))


def classification_metrics(
    *,
    tp: int,
    tn: int,
    fp: int,
    fn: int,
) -> Dict[str, Any]:
    """Binary metrics with nulls when a denominator is zero."""

    usable = tp + tn + fp + fn
    accuracy = _safe_div(tp + tn, usable)
    sensitivity = _safe_div(tp, tp + fn)  # recall for fake
    specificity = _safe_div(tn, tn + fp)
    precision = _safe_div(tp, tp + fp)
    if sensitivity is None or precision is None:
        f1 = None
    elif precision + sensitivity == 0:
        f1 = None
    else:
        f1 = 2 * precision * sensitivity / (precision + sensitivity)
    if sensitivity is None or specificity is None:
        balanced = None
    else:
        balanced = 0.5 * (sensitivity + specificity)
    return {
        "n_usable": usable,
        "n_correct": tp + tn,
        "accuracy": accuracy,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "sensitivity_fake": sensitivity,
        "recall_fake": sensitivity,
        "specificity_real": specificity,
        "balanced_accuracy": balanced,
        "precision_fake": precision,
        "f1_fake": f1,
    }


def load_and_validate_aggregate(payload: Any) -> Dict[str, Any]:
    """Validate top-level aggregate shape; return a normalised copy."""

    if not isinstance(payload, dict):
        raise AggregateAnalysisError("aggregate root must be a JSON object")
    images = payload.get("images")
    if not isinstance(images, list) or not images:
        raise AggregateAnalysisError("aggregate.images must be a non-empty list")

    normalised: List[Dict[str, Any]] = []
    for index, raw in enumerate(images):
        if not isinstance(raw, dict):
            raise AggregateAnalysisError(f"images[{index}] must be an object")
        image_id = raw.get("image_id")
        if not isinstance(image_id, str) or not image_id.strip():
            raise AggregateAnalysisError(f"images[{index}].image_id is required")
        gt = raw.get("ground_truth")
        if not isinstance(gt, str) or gt.strip().lower() not in GROUND_TRUTH_VALUES:
            raise AggregateAnalysisError(
                f"images[{index}].ground_truth must be 'real' or 'fake'"
            )
        configs = raw.get("configs")
        if not isinstance(configs, list):
            raise AggregateAnalysisError(f"images[{index}].configs must be a list")
        by_name: Dict[str, Dict[str, Any]] = {}
        for c_index, cfg in enumerate(configs):
            if not isinstance(cfg, dict):
                raise AggregateAnalysisError(
                    f"images[{index}].configs[{c_index}] must be an object"
                )
            run_name = cfg.get("run_name")
            if not isinstance(run_name, str) or not run_name:
                raise AggregateAnalysisError(
                    f"images[{index}].configs[{c_index}].run_name is required"
                )
            for score_key in ("real_score", "fake_score"):
                value = cfg.get(score_key)
                if value is not None and not isinstance(value, (int, float)):
                    raise AggregateAnalysisError(
                        f"images[{index}].configs[{c_index}].{score_key} "
                        f"must be numeric or null"
                    )
                if isinstance(value, (int, float)) and not (0.0 <= float(value) <= 1.0):
                    raise AggregateAnalysisError(
                        f"images[{index}].configs[{c_index}].{score_key} "
                        f"out of [0, 1]: {value}"
                    )
            by_name[run_name] = cfg
        entry = dict(raw)
        entry["ground_truth"] = gt.strip().lower()
        entry["_configs_by_name"] = by_name
        normalised.append(entry)

    out = dict(payload)
    out["images"] = normalised
    return out


def config_usable(cfg: Optional[Dict[str, Any]]) -> bool:
    if cfg is None:
        return False
    if cfg.get("status") != "pass":
        return False
    return cfg.get("label") in {"real", "fake"}


def compute_configuration_metrics(
    images: Sequence[Dict[str, Any]],
    *,
    run_order: Sequence[str] = RUN_ORDER,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run_name in run_order:
        tp = tn = fp = fn = 0
        missing = 0
        for image in images:
            cfg = image.get("_configs_by_name", {}).get(run_name)
            if not config_usable(cfg):
                missing += 1
                continue
            assert cfg is not None
            pred = cfg["label"]
            truth = image["ground_truth"]
            if truth == "fake" and pred == "fake":
                tp += 1
            elif truth == "real" and pred == "real":
                tn += 1
            elif truth == "real" and pred == "fake":
                fp += 1
            else:
                fn += 1
        metrics = classification_metrics(tp=tp, tn=tn, fp=fp, fn=fn)
        metrics.update(
            {
                "run_name": run_name,
                "n_missing_or_unusable": missing,
            }
        )
        rows.append(metrics)
    return rows


def compute_score_deltas(
    images: Sequence[Dict[str, Any]],
    *,
    baseline: str = BASELINE_RUN,
    run_order: Sequence[str] = RUN_ORDER,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Per-image fake-score deltas and summary contrasts vs baseline.

    Describes observed changes / contrasts, not causal effects.
    """

    per_image: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    targets = [name for name in run_order if name != baseline]

    for run_name in targets:
        deltas: List[float] = []
        label_changed = 0
        wrong_to_correct = 0
        correct_to_wrong = 0
        compared = 0
        for image in images:
            by_name = image.get("_configs_by_name", {})
            base = by_name.get(baseline)
            other = by_name.get(run_name)
            if not config_usable(base) or not config_usable(other):
                continue
            assert base is not None and other is not None
            compared += 1
            truth = image["ground_truth"]
            base_label = base["label"]
            other_label = other["label"]
            base_fake = base.get("fake_score")
            other_fake = other.get("fake_score")
            delta = None
            if isinstance(base_fake, (int, float)) and isinstance(other_fake, (int, float)):
                delta = float(other_fake) - float(base_fake)
                deltas.append(delta)
            changed = base_label != other_label
            if changed:
                label_changed += 1
            base_correct = base_label == truth
            other_correct = other_label == truth
            if not base_correct and other_correct:
                wrong_to_correct += 1
                transition = "wrong_to_correct"
            elif base_correct and not other_correct:
                correct_to_wrong += 1
                transition = "correct_to_wrong"
            elif base_correct and other_correct:
                transition = "correct_to_correct"
            else:
                transition = "wrong_to_wrong"
            per_image.append(
                {
                    "image_id": image["image_id"],
                    "ground_truth": truth,
                    "dataset": image.get("dataset"),
                    "manipulation": image.get("manipulation"),
                    "baseline_run": baseline,
                    "comparison_run": run_name,
                    "baseline_label": base_label,
                    "comparison_label": other_label,
                    "label_changed": changed,
                    "transition": transition,
                    "baseline_fake_score": base_fake,
                    "comparison_fake_score": other_fake,
                    "fake_score_delta": delta,
                }
            )
        summaries.append(
            {
                "baseline_run": baseline,
                "comparison_run": run_name,
                "n_compared": compared,
                "n_label_changed": label_changed,
                "n_wrong_to_correct": wrong_to_correct,
                "n_correct_to_wrong": correct_to_wrong,
                "mean_fake_score_delta": _mean(deltas),
                "median_fake_score_delta": _median(deltas),
                "note": (
                    "Observed contrast relative to the no-expert baseline; "
                    "not a causal effect estimate."
                ),
            }
        )
    return per_image, summaries


def _roc_auc(scores: Sequence[float], labels: Sequence[int]) -> Tuple[Optional[float], Optional[str]]:
    """Trapezoidal ROC-AUC. ``labels`` are 1=positive(fake), 0=negative(real)."""

    if len(scores) != len(labels) or not scores:
        return None, "no usable scores"
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos < DEFAULT_ROC_MIN_PER_CLASS or n_neg < DEFAULT_ROC_MIN_PER_CLASS:
        return None, (
            f"need at least {DEFAULT_ROC_MIN_PER_CLASS} samples per class "
            f"(have pos={n_pos}, neg={n_neg})"
        )
    pairs = sorted(zip(scores, labels), key=lambda p: p[0])
    # Mann–Whitney / rank formulation
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + j + 1) / 2.0  # 1-based average rank
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j
    rank_sum_pos = sum(rank for rank, (_, lab) in zip(ranks, pairs) if lab == 1)
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc), None


def compute_specialist_summaries(
    images: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Summaries for blending / diffusion detector scores (not probabilities)."""

    rows: List[Dict[str, Any]] = []
    for detector, score_key in (
        ("blending", "blending_detector_score"),
        ("diffusion", "diffusion_detector_score"),
    ):
        real_scores: List[float] = []
        fake_scores: List[float] = []
        all_scores: List[float] = []
        labels: List[int] = []
        for image in images:
            # Prefer combined-experts cell; fall back to any config with the score.
            preferred = image.get("_configs_by_name", {}).get("blending_diffusion")
            candidates = [preferred] if preferred else []
            candidates.extend(
                cfg
                for name, cfg in image.get("_configs_by_name", {}).items()
                if name != "blending_diffusion"
            )
            score = None
            for cfg in candidates:
                if cfg is None:
                    continue
                value = cfg.get(score_key)
                if value is None and isinstance(cfg.get("expert_scores"), dict):
                    value = cfg["expert_scores"].get(detector)
                if isinstance(value, (int, float)):
                    score = float(value)
                    break
            if score is None:
                continue
            all_scores.append(score)
            if image["ground_truth"] == "real":
                real_scores.append(score)
                labels.append(0)
            else:
                fake_scores.append(score)
                labels.append(1)
        auc, auc_reason = _roc_auc(all_scores, labels)
        rows.append(
            {
                "detector": detector,
                "score_kind": "detector_score_not_calibrated_probability",
                "n_usable": len(all_scores),
                "n_real": len(real_scores),
                "n_fake": len(fake_scores),
                "mean_real": _mean(real_scores),
                "median_real": _median(real_scores),
                "mean_fake": _mean(fake_scores),
                "median_fake": _median(fake_scores),
                "min": min(all_scores) if all_scores else None,
                "max": max(all_scores) if all_scores else None,
                "stdev": _stdev(all_scores),
                "roc_auc": auc,
                "roc_auc_unavailable_reason": auc_reason,
            }
        )
    return rows


def compute_agreement_analysis(
    images: Sequence[Dict[str, Any]],
    *,
    lo: float = EXPERT_LO,
    hi: float = EXPERT_HI,
) -> Dict[str, Any]:
    counts = {"agreement": 0, "conflict": 0, "insufficient evidence": 0, "missing": 0}
    gt_x_agree: Dict[str, Dict[str, int]] = {
        "real": {"agreement": 0, "conflict": 0, "insufficient evidence": 0},
        "fake": {"agreement": 0, "conflict": 0, "insufficient evidence": 0},
    }
    status_x_agree: Dict[str, Dict[str, int]] = {}

    pairwise = {
        "model_vs_blending": {"support": 0, "conflict": 0, "inconclusive": 0, "missing": 0},
        "model_vs_diffusion": {"support": 0, "conflict": 0, "inconclusive": 0, "missing": 0},
        "blending_vs_diffusion_direction": {
            "same_support_direction": 0,
            "disagree_support_direction": 0,
            "either_inconclusive": 0,
            "missing": 0,
        },
    }

    for image in images:
        agree = image.get("evidence_agreement")
        if agree not in counts:
            counts["missing"] += 1
            agree_key = None
        else:
            counts[agree] += 1
            agree_key = agree
        gt = image["ground_truth"]
        if agree_key is not None:
            gt_x_agree[gt][agree_key] += 1
        status = image.get("prototype_status") or "unknown"
        status_x_agree.setdefault(
            status,
            {"agreement": 0, "conflict": 0, "insufficient evidence": 0},
        )
        if agree_key is not None and agree_key in status_x_agree[status]:
            status_x_agree[status][agree_key] += 1

        cfg = image.get("_configs_by_name", {}).get("blending_diffusion")
        if not config_usable(cfg):
            pairwise["model_vs_blending"]["missing"] += 1
            pairwise["model_vs_diffusion"]["missing"] += 1
            pairwise["blending_vs_diffusion_direction"]["missing"] += 1
            continue
        assert cfg is not None
        label = cfg["label"]
        blend = cfg.get("blending_detector_score")
        diff = cfg.get("diffusion_detector_score")
        if not isinstance(blend, (int, float)):
            pairwise["model_vs_blending"]["missing"] += 1
            blend_rel = None
        else:
            blend_rel = _specialist_relation(label, float(blend), lo=lo, hi=hi)
            pairwise["model_vs_blending"][blend_rel] += 1
        if not isinstance(diff, (int, float)):
            pairwise["model_vs_diffusion"]["missing"] += 1
            diff_rel = None
        else:
            diff_rel = _specialist_relation(label, float(diff), lo=lo, hi=hi)
            pairwise["model_vs_diffusion"][diff_rel] += 1

        if blend_rel is None or diff_rel is None:
            pairwise["blending_vs_diffusion_direction"]["missing"] += 1
        elif blend_rel == "inconclusive" or diff_rel == "inconclusive":
            pairwise["blending_vs_diffusion_direction"]["either_inconclusive"] += 1
        elif blend_rel == diff_rel:
            pairwise["blending_vs_diffusion_direction"]["same_support_direction"] += 1
        else:
            pairwise["blending_vs_diffusion_direction"]["disagree_support_direction"] += 1

    n = len(images) or 1
    rates = {
        key: (counts[key] / len(images) if images else None)
        for key in ("agreement", "conflict", "insufficient evidence", "missing")
    }
    return {
        "counts": counts,
        "rates": rates,
        "ground_truth_x_evidence_agreement": gt_x_agree,
        "prototype_status_x_evidence_agreement": status_x_agree,
        "pairwise_specialist_relations": pairwise,
        "interpretation_thresholds": {"evidence_expert_lo": lo, "evidence_expert_hi": hi},
        "n_images": len(images),
    }


def prediction_matches_ground_truth(
    *,
    predicted_label: Any,
    ground_truth: str,
) -> Optional[bool]:
    """Return True/False when both sides are usable labels; else None."""

    if not isinstance(predicted_label, str):
        return None
    pred = predicted_label.strip().lower()
    truth = ground_truth.strip().lower()
    if pred not in GROUND_TRUTH_VALUES or truth not in GROUND_TRUTH_VALUES:
        return None
    return pred == truth


def is_deepfakeface_image(image: Dict[str, Any]) -> bool:
    dataset = image.get("dataset")
    if isinstance(dataset, str) and dataset.strip().lower() in DEEPFAKEFACE_DATASET_NAMES:
        return True
    image_id = image.get("image_id")
    return isinstance(image_id, str) and image_id.lower().startswith("dff_")


def deepfakeface_manipulation_group(image: Dict[str, Any]) -> Optional[str]:
    """Map a DeepFakeFace aggregate row to wiki_real / insight / inpainting / text2img.

    Wiki reals often have ``manipulation: null`` in the manifest; classify them as
    ``wiki_real`` via dataset metadata and/or ``dff_wiki_`` image IDs.
    """

    if not is_deepfakeface_image(image):
        return None

    manip = image.get("manipulation")
    if isinstance(manip, str):
        key = manip.strip().lower()
        if key in {"insight", "inpainting", "text2img"}:
            return key
        if key in {"wiki", "wiki_real"}:
            return "wiki_real"

    image_id = (image.get("image_id") or "").strip().lower()
    for prefix, group in (
        ("dff_wiki_", "wiki_real"),
        ("dff_insight_", "insight"),
        ("dff_inpainting_", "inpainting"),
        ("dff_text2img_", "text2img"),
    ):
        if image_id.startswith(prefix):
            return group

    # Real DeepFakeFace rows with null/blank manipulation are wiki.
    if image.get("ground_truth") == "real" and (
        manip is None or (isinstance(manip, str) and not manip.strip())
    ):
        return "wiki_real"
    return None


def compute_correctness_x_evidence_agreement(
    images: Sequence[Dict[str, Any]],
    *,
    run_name: str = COMBINED_RUN,
) -> Dict[str, Any]:
    """Correctness of ``run_name`` predictions crossed with evidence agreement.

    Correctness is ``predicted label == ground_truth`` only (never score direction).
    Missing/unusable predictions and missing agreement labels are counted explicitly
    and excluded from within-category rates.
    """

    cells: Dict[str, Dict[str, int]] = {
        agree: {"correct": 0, "wrong": 0}
        for agree in EVIDENCE_AGREEMENT_CATEGORIES
    }
    n_missing_or_unusable = 0
    n_usable_missing_agreement = 0

    for image in images:
        cfg = image.get("_configs_by_name", {}).get(run_name)
        if not config_usable(cfg):
            n_missing_or_unusable += 1
            continue
        assert cfg is not None
        match = prediction_matches_ground_truth(
            predicted_label=cfg.get("label"),
            ground_truth=image["ground_truth"],
        )
        if match is None:
            n_missing_or_unusable += 1
            continue
        agree = image.get("evidence_agreement")
        if agree not in cells:
            n_usable_missing_agreement += 1
            continue
        cells[agree]["correct" if match else "wrong"] += 1

    rows: List[Dict[str, Any]] = []
    by_agreement: Dict[str, Any] = {}
    for agree in EVIDENCE_AGREEMENT_CATEGORIES:
        n_correct = cells[agree]["correct"]
        n_wrong = cells[agree]["wrong"]
        n = n_correct + n_wrong
        accuracy = _safe_div(n_correct, n)
        by_agreement[agree] = {
            "n": n,
            "n_correct": n_correct,
            "n_wrong": n_wrong,
            "accuracy": accuracy,
        }
        for correctness, count in (
            ("correct", n_correct),
            ("wrong", n_wrong),
        ):
            rows.append(
                {
                    "correctness": correctness,
                    "evidence_agreement": agree,
                    "count": count,
                    "rate_within_agreement_category": _safe_div(count, n),
                    "n_correct": n_correct,
                    "n_wrong": n_wrong,
                    "accuracy": accuracy,
                    "n_in_agreement_category": n,
                }
            )

    return {
        "run_name": run_name,
        "n_images": len(images),
        "n_missing_or_unusable": n_missing_or_unusable,
        "n_usable_missing_agreement": n_usable_missing_agreement,
        "by_agreement": by_agreement,
        "rows": rows,
        "note": (
            "Correctness uses predicted label == ground_truth for "
            f"{run_name}; rates are within each evidence-agreement category."
        ),
    }


def compute_truth_transition_summary(
    images: Sequence[Dict[str, Any]],
    *,
    baseline: str = BASELINE_RUN,
    run_order: Sequence[str] = RUN_ORDER,
) -> List[Dict[str, Any]]:
    """Baseline→comparison transitions verified against ground truth.

    Separates non-flips into ``correct_to_correct`` and ``wrong_to_wrong``
    (does not emit a single ``unchanged`` bucket).
    """

    rows: List[Dict[str, Any]] = []
    targets = [name for name in run_order if name != baseline]
    for run_name in targets:
        w2c = c2w = c2c = w2w = 0
        compared = 0
        missing = 0
        baseline_correct = 0
        comparison_correct = 0
        for image in images:
            by_name = image.get("_configs_by_name", {})
            base = by_name.get(baseline)
            other = by_name.get(run_name)
            if not config_usable(base) or not config_usable(other):
                missing += 1
                continue
            assert base is not None and other is not None
            truth = image["ground_truth"]
            base_match = prediction_matches_ground_truth(
                predicted_label=base.get("label"),
                ground_truth=truth,
            )
            other_match = prediction_matches_ground_truth(
                predicted_label=other.get("label"),
                ground_truth=truth,
            )
            if base_match is None or other_match is None:
                missing += 1
                continue
            compared += 1
            if base_match:
                baseline_correct += 1
            if other_match:
                comparison_correct += 1
            if not base_match and other_match:
                w2c += 1
            elif base_match and not other_match:
                c2w += 1
            elif base_match and other_match:
                c2c += 1
            else:
                w2w += 1
        rows.append(
            {
                "baseline_run": baseline,
                "comparison_run": run_name,
                "n_compared": compared,
                "n_missing_or_unusable": missing,
                "wrong_to_correct": w2c,
                "correct_to_wrong": c2w,
                "correct_to_correct": c2c,
                "wrong_to_wrong": w2w,
                "net_correctness_change": w2c - c2w,
                "baseline_correct": baseline_correct,
                "comparison_correct": comparison_correct,
                "comparison_accuracy": _safe_div(comparison_correct, compared),
            }
        )
    return rows


def compute_manipulation_metrics(
    images: Sequence[Dict[str, Any]],
    *,
    run_order: Sequence[str] = RUN_ORDER,
) -> List[Dict[str, Any]]:
    """Per-manipulation metrics for DeepFakeFace images only.

    Fake-only subgroups leave ``specificity_real`` null when there are no reals;
    ``wiki_real`` leaves ``sensitivity_fake`` null when there are no fakes.
    """

    grouped: Dict[str, List[Dict[str, Any]]] = {
        name: [] for name in DEEPFAKEFACE_MANIPULATION_ORDER
    }
    for image in images:
        group = deepfakeface_manipulation_group(image)
        if group is None:
            continue
        if group not in grouped:
            grouped[group] = []
        grouped[group].append(image)

    rows: List[Dict[str, Any]] = []
    for manip in list(DEEPFAKEFACE_MANIPULATION_ORDER) + [
        g for g in grouped if g not in DEEPFAKEFACE_MANIPULATION_ORDER
    ]:
        subset = grouped.get(manip) or []
        if not subset:
            continue
        for run_name in run_order:
            tp = tn = fp = fn = 0
            missing = 0
            for image in subset:
                cfg = image.get("_configs_by_name", {}).get(run_name)
                if not config_usable(cfg):
                    missing += 1
                    continue
                assert cfg is not None
                pred = cfg["label"]
                truth = image["ground_truth"]
                if truth == "fake" and pred == "fake":
                    tp += 1
                elif truth == "real" and pred == "real":
                    tn += 1
                elif truth == "real" and pred == "fake":
                    fp += 1
                else:
                    fn += 1
            metrics = classification_metrics(tp=tp, tn=tn, fp=fp, fn=fn)
            rows.append(
                {
                    "manipulation": manip,
                    "run_name": run_name,
                    "n": metrics["n_usable"],
                    "n_missing_or_unusable": missing,
                    "n_correct": metrics["n_correct"],
                    "accuracy": metrics["accuracy"],
                    "tp": metrics["tp"],
                    "tn": metrics["tn"],
                    "fp": metrics["fp"],
                    "fn": metrics["fn"],
                    "sensitivity_fake": metrics["sensitivity_fake"],
                    "specificity_real": metrics["specificity_real"],
                }
            )
    return rows


def compute_correctness_x_prototype_status(
    images: Sequence[Dict[str, Any]],
    *,
    run_name: str = COMBINED_RUN,
) -> List[Dict[str, Any]]:
    """Prototype status vs correctness of ``run_name`` predictions."""

    tallies: Dict[str, Dict[str, int]] = {}
    n_missing_or_unusable = 0
    for image in images:
        cfg = image.get("_configs_by_name", {}).get(run_name)
        if not config_usable(cfg):
            n_missing_or_unusable += 1
            continue
        assert cfg is not None
        match = prediction_matches_ground_truth(
            predicted_label=cfg.get("label"),
            ground_truth=image["ground_truth"],
        )
        if match is None:
            n_missing_or_unusable += 1
            continue
        status = image.get("prototype_status")
        if not isinstance(status, str) or not status.strip():
            status = "unknown"
        else:
            status = status.strip()
        bucket = tallies.setdefault(status, {"correct": 0, "wrong": 0})
        bucket["correct" if match else "wrong"] += 1

    ordered = list(PROTOTYPE_STATUS_ORDER) + [
        s for s in tallies if s not in PROTOTYPE_STATUS_ORDER
    ]
    rows: List[Dict[str, Any]] = []
    for status in ordered:
        if status not in tallies:
            continue
        n_correct = tallies[status]["correct"]
        n_wrong = tallies[status]["wrong"]
        n = n_correct + n_wrong
        rows.append(
            {
                "status": status,
                "n": n,
                "correct": n_correct,
                "wrong": n_wrong,
                "accuracy": _safe_div(n_correct, n),
                "run_name": run_name,
                "n_missing_or_unusable_total": n_missing_or_unusable,
            }
        )
    return rows


def compute_calibration_diagnostics(
    images: Sequence[Dict[str, Any]],
    *,
    min_samples: int = DEFAULT_CALIBRATION_MIN_SAMPLES,
    n_bins: int = 10,
    run_name: str = "blending_diffusion",
) -> Dict[str, Any]:
    """Brier / reliability / ECE for main-model fake scores only.

    These are diagnostics of raw, currently uncalibrated model scores.
    """

    scores: List[float] = []
    outcomes: List[int] = []  # 1 if GT fake
    for image in images:
        cfg = image.get("_configs_by_name", {}).get(run_name)
        if cfg is None:
            # fall back to any usable config with a fake_score
            for name in RUN_ORDER:
                candidate = image.get("_configs_by_name", {}).get(name)
                if config_usable(candidate) and isinstance(candidate.get("fake_score"), (int, float)):
                    cfg = candidate
                    break
        if cfg is None or not isinstance(cfg.get("fake_score"), (int, float)):
            continue
        scores.append(float(cfg["fake_score"]))
        outcomes.append(1 if image["ground_truth"] == "fake" else 0)

    result: Dict[str, Any] = {
        "score_kind": "raw_uncalibrated_model_fake_score",
        "run_name_preference": run_name,
        "n_samples": len(scores),
        "min_samples_required": min_samples,
        "n_bins": n_bins,
        "sufficient_sample": len(scores) >= min_samples,
        "brier_score": None,
        "ece": None,
        "reliability_bins": [],
        "unavailable_reason": None,
    }
    if len(scores) < min_samples:
        result["unavailable_reason"] = (
            f"insufficient sample size for calibration summaries "
            f"(n={len(scores)} < min_samples={min_samples})"
        )
        # Still compute Brier when both classes present and n>=2 for infrastructure,
        # but leave ECE null and mark insufficient.
        if len(scores) >= 2 and len(set(outcomes)) == 2:
            result["brier_score"] = float(
                sum((s - o) ** 2 for s, o in zip(scores, outcomes)) / len(scores)
            )
            result["brier_note"] = (
                "Brier computed for infrastructure checks only; "
                "sample size is below the configured calibration floor."
            )
        return result

    brier = sum((s - o) ** 2 for s, o in zip(scores, outcomes)) / len(scores)
    bins: List[Dict[str, Any]] = []
    ece = 0.0
    for b in range(n_bins):
        lo = b / n_bins
        hi = (b + 1) / n_bins
        idxs = [
            i
            for i, s in enumerate(scores)
            if (s >= lo and s < hi) or (b == n_bins - 1 and s == hi)
        ]
        if not idxs:
            bins.append(
                {
                    "bin": b,
                    "lo": lo,
                    "hi": hi,
                    "count": 0,
                    "mean_score": None,
                    "empirical_frequency": None,
                }
            )
            continue
        mean_score = sum(scores[i] for i in idxs) / len(idxs)
        freq = sum(outcomes[i] for i in idxs) / len(idxs)
        ece += (len(idxs) / len(scores)) * abs(mean_score - freq)
        bins.append(
            {
                "bin": b,
                "lo": lo,
                "hi": hi,
                "count": len(idxs),
                "mean_score": mean_score,
                "empirical_frequency": freq,
            }
        )
    result["brier_score"] = float(brier)
    result["ece"] = float(ece)
    result["reliability_bins"] = bins
    return result


def analyse_aggregate(
    payload: Dict[str, Any],
    *,
    calibration_min_samples: int = DEFAULT_CALIBRATION_MIN_SAMPLES,
) -> Dict[str, Any]:
    """Run the full analysis suite on a validated aggregate payload."""

    validated = load_and_validate_aggregate(payload)
    images = validated["images"]
    thresholds = interpretation_thresholds()
    config_metrics = compute_configuration_metrics(images)
    per_image_deltas, delta_summaries = compute_score_deltas(images)
    specialist = compute_specialist_summaries(images)
    agreement = compute_agreement_analysis(
        images,
        lo=thresholds["evidence_expert_lo"],
        hi=thresholds["evidence_expert_hi"],
    )
    calibration = compute_calibration_diagnostics(
        images,
        min_samples=calibration_min_samples,
    )
    correctness_x_agreement = compute_correctness_x_evidence_agreement(images)
    truth_transitions = compute_truth_transition_summary(images)
    manipulation_metrics = compute_manipulation_metrics(images)
    correctness_x_status = compute_correctness_x_prototype_status(images)
    n = len(images)
    return {
        "n_images": n,
        "pipeline_validation_only": n < 30,
        "warning": PILOT_WARNING if n <= 4 else None,
        "configuration_metrics": config_metrics,
        "baseline_contrasts": delta_summaries,
        "score_deltas": per_image_deltas,
        "specialist_summaries": specialist,
        "agreement": agreement,
        "calibration": calibration,
        "correctness_x_evidence_agreement": correctness_x_agreement,
        "truth_transitions": truth_transitions,
        "manipulation_metrics": manipulation_metrics,
        "correctness_x_prototype_status": correctness_x_status,
        "interpretation_thresholds": thresholds,
        "run_order": list(RUN_ORDER),
    }
