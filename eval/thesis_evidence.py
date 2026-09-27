"""Thesis evidence layer: distributions, justification, summaries, export.

Operates on saved labelled aggregates and optional validation_protocol.json.
Does not run inference, does not rewrite raw scores, and does not select
thresholds from held-out test data.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from eval.experiment_configs import PRIMARY_ASSESSMENT_RUN, RUN_ORDER
from eval.ground_truth_audit import (
    CLAIM_BOUNDARY_NOTES,
    AuditRow,
    build_audit_for_runs,
    load_aggregate_json,
    load_validation_protocol_payload,
    representative_cases_markdown,
    resolve_balanced_threshold_from_protocol,
    select_representative_cases_by_run,
    write_csv,
    write_ground_truth_audit_outputs,
    write_json,
)
from eval.reproducibility import get_git_provenance, utc_timestamp
from eval.threshold_sweep import (
    PRESET_BALANCED,
    REFERENCE_THRESHOLD,
    SCORE_FIELD,
    SCORE_SEMANTICS,
    DECISION_RULE,
    ScoredSample,
    compute_roc_auc,
    confusion_at_threshold,
    predict_label,
    run_threshold_sweep,
    select_operating_presets,
)

# re-export for tests / callers
__all__ = [
    "THESIS_DISCRIMINATION_WORDING",
    "HIGH_THRESHOLD_NOTE",
    "distribution_stats",
    "build_score_distribution_summary",
    "build_threshold_justification_row",
    "build_performance_summary_row",
    "build_manual_verification_rows",
    "generate_thesis_evidence",
    "extract_protocol_metrics_for_run",
]

SCRIPT_VERSION = "1.0.0"

THESIS_DISCRIMINATION_WORDING = (
    "Threshold-independent discrimination is reported using ROC-AUC. "
    "Threshold-dependent operating performance is reported separately using "
    "held-out balanced accuracy, sensitivity and specificity."
)

HIGH_THRESHOLD_NOTE = (
    "An unusually high decision threshold (for example above 0.90) indicates "
    "that the raw fake_score distribution for this configuration/dataset is "
    "shifted toward the fake token. Adjusting the threshold changes only the "
    "operating point on that score axis; it does not improve ROC-AUC or the "
    "underlying ranking of samples. Justification for such a threshold rests "
    "on validation selection plus held-out behaviour and remains "
    "dataset/application-specific — not a universally appropriate setting."
)

MANUAL_HUMAN_COLUMNS = (
    "human_visual_check_completed",
    "ground_truth_source_verified",
    "notes",
)


def _safe_float(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        f = float(value)
        return f if math.isfinite(f) else None
    return None


def _quantile(sorted_vals: Sequence[float], q: float) -> Optional[float]:
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = q * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_vals[lo])
    frac = pos - lo
    return float(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)


def distribution_stats(values: Sequence[float]) -> Dict[str, Any]:
    """Descriptive statistics for one score cohort (real or fake)."""

    vals = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    vals.sort()
    n = len(vals)
    if n == 0:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
            "q10": None,
            "q25": None,
            "q75": None,
            "q90": None,
        }
    mean = statistics.fmean(vals)
    median = statistics.median(vals)
    std = statistics.pstdev(vals) if n > 1 else 0.0
    return {
        "n": n,
        "mean": mean,
        "median": median,
        "std": std,
        "min": vals[0],
        "max": vals[-1],
        "q10": _quantile(vals, 0.10),
        "q25": _quantile(vals, 0.25),
        "q75": _quantile(vals, 0.75),
        "q90": _quantile(vals, 0.90),
    }


def rows_to_scored_samples(rows: Sequence[AuditRow]) -> List[ScoredSample]:
    return [
        ScoredSample(
            sample_id=r.sample_id,
            ground_truth=r.ground_truth,
            fake_score=r.fake_score,
            source="ground_truth_audit",
            run_name=r.run_name,
        )
        for r in rows
    ]


def score_cohorts(rows: Sequence[AuditRow]) -> Tuple[List[float], List[float]]:
    real_scores = [r.fake_score for r in rows if r.ground_truth == "real"]
    fake_scores = [r.fake_score for r in rows if r.ground_truth == "fake"]
    return real_scores, fake_scores


def build_score_distribution_summary(
    rows: Sequence[AuditRow],
    *,
    dataset_name: Optional[str] = None,
    balanced_threshold: Optional[float] = None,
    balanced_threshold_source: Optional[str] = None,
) -> Dict[str, Any]:
    real_scores, fake_scores = score_cohorts(rows)
    run_name = rows[0].run_name if rows else None
    active = rows[0].active_threshold if rows else None
    return {
        "dataset": dataset_name or (rows[0].dataset if rows else None),
        "run_name": run_name,
        "n_rows": len(rows),
        "reference_threshold": REFERENCE_THRESHOLD,
        "balanced_threshold": balanced_threshold,
        "balanced_threshold_source": balanced_threshold_source,
        "active_threshold": active,
        "real_fake_score": distribution_stats(real_scores),
        "fake_fake_score": distribution_stats(fake_scores),
        "score_field": SCORE_FIELD,
        "score_semantics": SCORE_SEMANTICS,
    }


def _write_score_distribution_figure(
    rows: Sequence[AuditRow],
    path: Path,
    *,
    balanced_threshold: Optional[float] = None,
    title: Optional[str] = None,
) -> Optional[str]:
    if not rows:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    real_scores, fake_scores = score_cohorts(rows)
    fig, ax = plt.subplots()
    bins = [i / 20.0 for i in range(21)]
    if real_scores:
        ax.hist(
            real_scores,
            bins=bins,
            alpha=0.55,
            label=f"ground-truth real (n={len(real_scores)})",
            density=True,
        )
    if fake_scores:
        ax.hist(
            fake_scores,
            bins=bins,
            alpha=0.55,
            label=f"ground-truth fake (n={len(fake_scores)})",
            density=True,
        )
    ax.axvline(
        REFERENCE_THRESHOLD,
        linestyle="--",
        linewidth=1.2,
        color="black",
        label="reference 0.50",
    )
    if balanced_threshold is not None:
        ax.axvline(
            float(balanced_threshold),
            linestyle=":",
            linewidth=1.4,
            color="tab:red",
            label=f"validation Balanced {float(balanced_threshold):.2f}",
        )
    ax.set_xlabel("raw fake_score (uncalibrated)")
    ax.set_ylabel("density")
    ax.set_title(
        title
        or (
            f"Score distribution · {rows[0].run_name}\n"
            "(raw fake_score; not probability/confidence)"
        )
    )
    ax.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return str(path.resolve())


def _write_threshold_tradeoff_figure(
    rows: Sequence[AuditRow],
    path: Path,
    *,
    balanced_threshold: Optional[float] = None,
    title: Optional[str] = None,
) -> Optional[str]:
    if len(rows) < 2:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    samples = rows_to_scored_samples(rows)
    try:
        sweep = run_threshold_sweep(samples)
    except Exception:
        return None
    thr_rows = sweep["thresholds"]
    xs = [r["threshold"] for r in thr_rows]
    fig, ax = plt.subplots()
    ax.plot(xs, [r.get("sensitivity_fake") for r in thr_rows], label="sensitivity (fake)")
    ax.plot(xs, [r.get("specificity_real") for r in thr_rows], label="specificity (real)")
    ax.axvline(REFERENCE_THRESHOLD, linestyle="--", linewidth=1.2, label="reference 0.50")
    if balanced_threshold is not None:
        ax.axvline(
            float(balanced_threshold),
            linestyle=":",
            linewidth=1.4,
            color="tab:red",
            label=f"validation Balanced {float(balanced_threshold):.2f}",
        )
    ax.set_xlabel("decision threshold on raw fake_score")
    ax.set_ylabel("rate")
    ax.set_title(
        title
        or (
            f"Threshold trade-off · {rows[0].run_name}\n"
            "(operating point only; not calibrated confidence)"
        )
    )
    ax.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return str(path.resolve())


def _write_optional_ecdf_figure(
    rows: Sequence[AuditRow],
    path: Path,
    *,
    balanced_threshold: Optional[float] = None,
) -> Optional[str]:
    if not rows:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    real_scores, fake_scores = score_cohorts(rows)

    def _ecdf(vals: Sequence[float]) -> Tuple[List[float], List[float]]:
        xs = sorted(float(v) for v in vals)
        n = len(xs)
        if n == 0:
            return [], []
        ys = [(i + 1) / n for i in range(n)]
        return xs, ys

    fig, ax = plt.subplots()
    rx, ry = _ecdf(real_scores)
    fx, fy = _ecdf(fake_scores)
    if rx:
        ax.step(rx, ry, where="post", label="ECDF real")
    if fx:
        ax.step(fx, fy, where="post", label="ECDF fake")
    ax.axvline(REFERENCE_THRESHOLD, linestyle="--", linewidth=1.2, label="reference 0.50")
    if balanced_threshold is not None:
        ax.axvline(
            float(balanced_threshold),
            linestyle=":",
            linewidth=1.4,
            color="tab:red",
            label=f"validation Balanced {float(balanced_threshold):.2f}",
        )
    ax.set_xlabel("raw fake_score (uncalibrated)")
    ax.set_ylabel("empirical CDF")
    ax.set_title(f"Empirical CDF · {rows[0].run_name}")
    ax.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return str(path.resolve())


def extract_protocol_metrics_for_run(
    protocol: Optional[Mapping[str, Any]],
    run_name: str,
) -> Dict[str, Any]:
    """Pull validation / held-out metrics without using held-out for selection."""

    out: Dict[str, Any] = {
        "balanced_threshold": None,
        "balanced_threshold_source": None,
        "validation_balanced_accuracy": None,
        "held_out_balanced_accuracy": None,
        "held_out_sensitivity_fake": None,
        "held_out_specificity_real": None,
        "reference_held_out_balanced_accuracy": None,
        "validation_seed": None,
        "protocol_available": False,
        "notes": [],
    }
    if protocol is None:
        out["notes"].append("validation protocol absent")
        return out

    split = protocol.get("split")
    if isinstance(split, Mapping):
        out["validation_seed"] = split.get("seed")

    bal, src = resolve_balanced_threshold_from_protocol(protocol, run_name)
    out["balanced_threshold"] = bal
    out["balanced_threshold_source"] = src
    if bal is None:
        out["notes"].append(src or "Balanced threshold unavailable")
        return out

    out["protocol_available"] = True
    results = protocol.get("results_by_run")
    if not isinstance(results, Mapping):
        return out
    block = results.get(run_name)
    if not isinstance(block, Mapping):
        return out

    presets = block.get("dashboard_presets")
    if isinstance(presets, Mapping):
        bal_preset = presets.get(PRESET_BALANCED)
        if isinstance(bal_preset, Mapping):
            val_metrics = bal_preset.get("validation_metrics")
            if isinstance(val_metrics, Mapping):
                out["validation_balanced_accuracy"] = _safe_float(
                    val_metrics.get("balanced_accuracy")
                )
            held = bal_preset.get("held_out_test")
            if isinstance(held, Mapping):
                metrics = held.get("metrics")
                if isinstance(metrics, Mapping):
                    out["held_out_balanced_accuracy"] = _safe_float(
                        metrics.get("balanced_accuracy")
                    )
                    out["held_out_sensitivity_fake"] = _safe_float(
                        metrics.get("sensitivity_fake")
                    )
                    out["held_out_specificity_real"] = _safe_float(
                        metrics.get("specificity_real")
                    )

    # Fallback: max_balanced_accuracy selection comparison / held-out block.
    selections = block.get("selections")
    if isinstance(selections, Mapping):
        sel = selections.get("max_balanced_accuracy")
        if isinstance(sel, Mapping):
            comparison = sel.get("comparison")
            if isinstance(comparison, Mapping):
                if out["validation_balanced_accuracy"] is None:
                    out["validation_balanced_accuracy"] = _safe_float(
                        comparison.get("val_balanced_accuracy")
                    )
                if out["held_out_balanced_accuracy"] is None:
                    out["held_out_balanced_accuracy"] = _safe_float(
                        comparison.get("test_balanced_accuracy")
                    )
                if out["held_out_sensitivity_fake"] is None:
                    out["held_out_sensitivity_fake"] = _safe_float(
                        comparison.get("test_sensitivity_fake")
                    )
                if out["held_out_specificity_real"] is None:
                    out["held_out_specificity_real"] = _safe_float(
                        comparison.get("test_specificity_real")
                    )
                out["reference_held_out_balanced_accuracy"] = _safe_float(
                    comparison.get("ref_0_50_test_balanced_accuracy")
                )
            if out["held_out_balanced_accuracy"] is None:
                held = sel.get("held_out_test")
                if isinstance(held, Mapping):
                    metrics = held.get("metrics")
                    if isinstance(metrics, Mapping):
                        out["held_out_balanced_accuracy"] = _safe_float(
                            metrics.get("balanced_accuracy")
                        )
                        out["held_out_sensitivity_fake"] = _safe_float(
                            metrics.get("sensitivity_fake")
                        )
                        out["held_out_specificity_real"] = _safe_float(
                            metrics.get("specificity_real")
                        )
            ref_block = sel.get("reference_0_50_held_out_test")
            if isinstance(ref_block, Mapping) and out["reference_held_out_balanced_accuracy"] is None:
                metrics = ref_block.get("metrics")
                if isinstance(metrics, Mapping):
                    out["reference_held_out_balanced_accuracy"] = _safe_float(
                        metrics.get("balanced_accuracy")
                    )

    ref0 = block.get("reference_0_50_held_out_test")
    if isinstance(ref0, Mapping) and out["reference_held_out_balanced_accuracy"] is None:
        metrics = ref0.get("metrics")
        if isinstance(metrics, Mapping):
            out["reference_held_out_balanced_accuracy"] = _safe_float(
                metrics.get("balanced_accuracy")
            )

    # If validation_sweep exists, also report Balanced validation metrics from presets pack.
    if out["validation_balanced_accuracy"] is None:
        sweep = block.get("validation_sweep")
        if isinstance(sweep, list) and sweep:
            pack = select_operating_presets([r for r in sweep if isinstance(r, dict)])
            bal_preset = pack["presets"].get(PRESET_BALANCED) or {}
            out["validation_balanced_accuracy"] = _safe_float(
                bal_preset.get("validation_balanced_accuracy")
            )

    return out


def descriptive_interpretation_flags(
    *,
    roc_auc: Optional[float],
    gain_vs_reference: Optional[float],
    balanced_threshold: Optional[float],
) -> List[str]:
    """Descriptive flags only — not quality grades or approval claims."""

    flags: List[str] = []
    if roc_auc is None:
        flags.append("auc_unavailable")
    elif roc_auc < 0.55:
        flags.append("weak discrimination (AUC close to 0.5)")
    elif roc_auc < 0.70:
        flags.append("moderate discrimination")
    else:
        flags.append("useful discrimination")

    if gain_vs_reference is not None and abs(float(gain_vs_reference)) >= 0.05:
        flags.append("threshold-sensitive (held-out balanced accuracy shifts vs 0.50)")
    elif gain_vs_reference is not None and abs(float(gain_vs_reference)) < 0.02:
        flags.append("limited threshold sensitivity vs 0.50 on held-out balanced accuracy")

    if balanced_threshold is not None and float(balanced_threshold) > 0.90:
        flags.append("high validation-selected threshold (>0.90); see high-threshold note")

    return flags


def build_threshold_justification_row(
    rows: Sequence[AuditRow],
    *,
    dataset_name: str,
    protocol: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    if not rows:
        return {
            "dataset": dataset_name,
            "run_name": None,
            "error": "no usable rows",
        }
    run_name = rows[0].run_name
    samples = rows_to_scored_samples(rows)
    roc = compute_roc_auc(samples)
    ref_metrics = confusion_at_threshold(samples, REFERENCE_THRESHOLD)
    proto = extract_protocol_metrics_for_run(protocol, run_name)

    bal_thr = proto["balanced_threshold"]
    # Full-set metrics at Balanced threshold are descriptive only when protocol absent.
    fullset_bal_metrics = None
    if bal_thr is not None:
        fullset_bal_metrics = confusion_at_threshold(samples, float(bal_thr))

    held_bal = proto["held_out_balanced_accuracy"]
    ref_held = proto["reference_held_out_balanced_accuracy"]
    # Prefer held-out gain; fall back to full-set descriptive delta if needed.
    gain = None
    gain_basis = None
    if held_bal is not None and ref_held is not None:
        gain = float(held_bal) - float(ref_held)
        gain_basis = "held_out_balanced_accuracy_vs_reference_0_50_held_out"
    elif (
        fullset_bal_metrics is not None
        and fullset_bal_metrics.get("balanced_accuracy") is not None
        and ref_metrics.get("balanced_accuracy") is not None
    ):
        gain = float(fullset_bal_metrics["balanced_accuracy"]) - float(
            ref_metrics["balanced_accuracy"]
        )
        gain_basis = "full_set_descriptive_only_not_held_out"

    high_thr = bal_thr is not None and float(bal_thr) > 0.90
    return {
        "dataset": dataset_name,
        "run_name": run_name,
        "n_samples": len(rows),
        "roc_auc": roc.get("roc_auc"),
        "roc_note": roc.get("roc_auc_unavailable_reason"),
        "reference_threshold": REFERENCE_THRESHOLD,
        "reference_balanced_accuracy": ref_metrics.get("balanced_accuracy"),
        "reference_sensitivity_fake": ref_metrics.get("sensitivity_fake"),
        "reference_specificity_real": ref_metrics.get("specificity_real"),
        "selected_balanced_threshold": bal_thr,
        "balanced_threshold_source": proto["balanced_threshold_source"],
        "validation_balanced_accuracy": proto["validation_balanced_accuracy"],
        "held_out_balanced_accuracy": held_bal,
        "held_out_sensitivity_fake": proto["held_out_sensitivity_fake"],
        "held_out_specificity_real": proto["held_out_specificity_real"],
        "reference_held_out_balanced_accuracy": ref_held,
        "difference_vs_0_50": gain,
        "difference_basis": gain_basis,
        "unusually_high_threshold": high_thr,
        "high_threshold_note": HIGH_THRESHOLD_NOTE if high_thr else None,
        "discrimination_wording": THESIS_DISCRIMINATION_WORDING,
        "score_semantics": SCORE_SEMANTICS,
        "protocol_notes": list(proto.get("notes") or []),
        "interpretation_flags": descriptive_interpretation_flags(
            roc_auc=_safe_float(roc.get("roc_auc")),
            gain_vs_reference=gain,
            balanced_threshold=bal_thr,
        ),
    }


def threshold_justification_markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# Threshold justification",
        "",
        THESIS_DISCRIMINATION_WORDING,
        "",
        "| Dataset | Config | ROC-AUC | Ref thr | Ref balAcc | Bal thr | Val balAcc | Held-out balAcc | Held-out sens | Held-out spec | Δ vs 0.50 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    def _fmt(v: Any, digits: int = 3) -> str:
        if v is None:
            return "—"
        try:
            return f"{float(v):.{digits}f}"
        except (TypeError, ValueError):
            return str(v)

    for row in rows:
        lines.append(
            "| {ds} | {cfg} | {auc} | {ref} | {rba} | {bthr} | {vba} | {hba} | {hs} | {hsp} | {dlt} |".format(
                ds=row.get("dataset", ""),
                cfg=row.get("run_name", ""),
                auc=_fmt(row.get("roc_auc")),
                ref=_fmt(row.get("reference_threshold"), 2),
                rba=_fmt(row.get("reference_balanced_accuracy")),
                bthr=_fmt(row.get("selected_balanced_threshold"), 2),
                vba=_fmt(row.get("validation_balanced_accuracy")),
                hba=_fmt(row.get("held_out_balanced_accuracy")),
                hs=_fmt(row.get("held_out_sensitivity_fake")),
                hsp=_fmt(row.get("held_out_specificity_real")),
                dlt=_fmt(row.get("difference_vs_0_50")),
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Raw `fake_score` is uncalibrated; thresholds are operating settings, not confidence levels.",
            "- Held-out metrics never select the threshold.",
            f"- {HIGH_THRESHOLD_NOTE}",
            "",
        ]
    )
    return "\n".join(lines)


def build_performance_summary_row(
    justification: Mapping[str, Any],
) -> Dict[str, Any]:
    flags = list(justification.get("interpretation_flags") or [])
    return {
        "dataset": justification.get("dataset"),
        "configuration": justification.get("run_name"),
        "roc_auc": justification.get("roc_auc"),
        "reference_threshold": justification.get("reference_threshold"),
        "reference_balanced_accuracy": justification.get("reference_balanced_accuracy"),
        "selected_balanced_threshold": justification.get("selected_balanced_threshold"),
        "held_out_balanced_accuracy": justification.get("held_out_balanced_accuracy"),
        "held_out_fake_sensitivity": justification.get("held_out_sensitivity_fake"),
        "held_out_real_specificity": justification.get("held_out_specificity_real"),
        "gain_vs_reference": justification.get("difference_vs_0_50"),
        "interpretation_flags": "; ".join(flags) if flags else None,
        "score_semantics": SCORE_SEMANTICS,
    }


def performance_summary_markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# Thesis performance summary",
        "",
        THESIS_DISCRIMINATION_WORDING,
        "",
        "| Dataset | Configuration | ROC-AUC | Ref thr | Ref balAcc | Selected Bal thr | Held-out balAcc | Held-out fake sens | Held-out real spec | Gain vs ref | Interpretation |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    def _fmt(v: Any, digits: int = 3) -> str:
        if v is None:
            return "—"
        try:
            return f"{float(v):.{digits}f}"
        except (TypeError, ValueError):
            return str(v)

    for row in rows:
        lines.append(
            "| {ds} | {cfg} | {auc} | {ref} | {rba} | {bthr} | {hba} | {hs} | {hsp} | {gain} | {flags} |".format(
                ds=row.get("dataset", ""),
                cfg=row.get("configuration", ""),
                auc=_fmt(row.get("roc_auc")),
                ref=_fmt(row.get("reference_threshold"), 2),
                rba=_fmt(row.get("reference_balanced_accuracy")),
                bthr=_fmt(row.get("selected_balanced_threshold"), 2),
                hba=_fmt(row.get("held_out_balanced_accuracy")),
                hs=_fmt(row.get("held_out_fake_sensitivity")),
                hsp=_fmt(row.get("held_out_real_specificity")),
                gain=_fmt(row.get("gain_vs_reference")),
                flags=(row.get("interpretation_flags") or "—").replace("|", "/"),
            )
        )
    lines.extend(
        [
            "",
            "_Interpretation flags are descriptive only; they are not quality grades or approvals._",
            "",
        ]
    )
    return "\n".join(lines)


def build_manual_verification_rows(
    cases: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Populate automated fields; leave human columns blank."""

    out: List[Dict[str, Any]] = []
    for case in cases:
        out.append(
            {
                "sample_id": case.get("sample_id"),
                "dataset": case.get("dataset"),
                "ground_truth": case.get("ground_truth"),
                "config": case.get("run_name") or case.get("config"),
                "fake_score": case.get("fake_score"),
                "active_threshold": case.get("active_threshold"),
                "predicted_label": case.get("active_prediction"),
                "automated_correct": case.get("active_correct"),
                "case_key": case.get("case_key"),
                "selection_rule": case.get("selection_rule"),
                "human_visual_check_completed": "",
                "ground_truth_source_verified": "",
                "notes": "",
                "manual_verification_complete": False,
                "disclaimer": (
                    "Automated worksheet only. Manual verification is not complete "
                    "until human columns are filled."
                ),
            }
        )
    return out


def manual_verification_markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# Manual ground-truth verification worksheet",
        "",
        "Automated fields are pre-filled. Human columns are intentionally blank.",
        "Do **not** claim manual verification is complete until human columns are filled.",
        "",
        "| Sample ID | Dataset | Ground truth | Config | fake_score | Active threshold | Predicted | Automated correct? | Human visual check completed? | Ground-truth source verified? | Notes |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    def _fmt(v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    for row in rows:
        lines.append(
            "| {sid} | {ds} | {gt} | {cfg} | {fs} | {thr} | {pred} | {ac} |  |  |  |".format(
                sid=_fmt(row.get("sample_id")),
                ds=_fmt(row.get("dataset")),
                gt=_fmt(row.get("ground_truth")),
                cfg=_fmt(row.get("config")),
                fs=_fmt(row.get("fake_score")),
                thr=_fmt(row.get("active_threshold")),
                pred=_fmt(row.get("predicted_label")),
                ac=_fmt(row.get("automated_correct")),
            )
        )
    lines.append("")
    return "\n".join(lines)


def generate_thesis_evidence(
    datasets: Sequence[Mapping[str, Any]],
    output_dir: Path | str,
    *,
    run_names: Sequence[str] = RUN_ORDER,
    write_figures: bool = True,
    write_ecdf: bool = True,
) -> Dict[str, Any]:
    """Generate the full thesis evidence bundle for one or more datasets.

    Each dataset mapping should include:
    - ``name``: display name
    - ``aggregate``: path to aggregate.json
    - ``validation_protocol``: optional path
    """

    out_root = Path(output_dir)
    tables = out_root / "tables"
    figures = out_root / "figures"
    cases_dir = out_root / "cases"
    metadata = out_root / "metadata"
    for d in (tables, figures, cases_dir, metadata):
        d.mkdir(parents=True, exist_ok=True)

    all_audit_rows: List[AuditRow] = []
    all_justification: List[Dict[str, Any]] = []
    all_performance: List[Dict[str, Any]] = []
    all_dist_summaries: List[Dict[str, Any]] = []
    all_cases: List[Dict[str, Any]] = []
    written: Dict[str, str] = {}
    input_files: List[Dict[str, Any]] = []
    unavailable: List[str] = []
    dataset_sizes: Dict[str, Any] = {}

    for spec in datasets:
        name = str(spec.get("name") or "dataset")
        aggregate_path = Path(spec["aggregate"])
        protocol_path = spec.get("validation_protocol")
        payload = load_aggregate_json(aggregate_path)
        protocol, protocol_error = load_validation_protocol_payload(protocol_path)
        if protocol_error:
            unavailable.append(f"{name}: {protocol_error}")

        input_files.append(
            {
                "dataset": name,
                "aggregate": str(aggregate_path.resolve()),
                "validation_protocol": (
                    str(Path(protocol_path).resolve()) if protocol_path else None
                ),
                "validation_protocol_loaded": protocol is not None,
                "validation_protocol_error": protocol_error,
            }
        )

        rows, audit_meta = build_audit_for_runs(
            payload,
            run_names=run_names,
            protocol=protocol,
        )
        # Stamp dataset name when aggregate omits it.
        stamped: List[AuditRow] = []
        for row in rows:
            stamped.append(row if row.dataset else replace(row, dataset=name))
        rows = stamped
        all_audit_rows.extend(rows)
        primary = run_names[0] if run_names else PRIMARY_ASSESSMENT_RUN
        dataset_sizes[name] = {
            "n_audit_rows": len(rows),
            "n_images_estimate": (audit_meta.get("per_run") or {})
            .get(primary, {})
            .get("n_images_in_aggregate"),
            "per_run": audit_meta.get("per_run"),
        }

        # Per-dataset audit + cases under cases/
        ds_case_dir = cases_dir / _slug(name)
        ds_case_dir.mkdir(parents=True, exist_ok=True)
        written.update(
            {
                f"cases/{_slug(name)}/{k}": v
                for k, v in write_ground_truth_audit_outputs(
                    rows, ds_case_dir, meta={"dataset": name, **audit_meta}
                ).items()
            }
        )

        by_run = select_representative_cases_by_run(rows)
        for run_name, case_list in by_run.items():
            for case in case_list:
                entry = dict(case)
                entry["dataset"] = entry.get("dataset") or name
                entry["config"] = run_name
                all_cases.append(entry)

        # Distributions + figures per config
        rows_by_run: Dict[str, List[AuditRow]] = {}
        for row in rows:
            rows_by_run.setdefault(row.run_name, []).append(row)

        for run_name in run_names:
            run_rows = rows_by_run.get(run_name, [])
            bal_thr, bal_src = resolve_balanced_threshold_from_protocol(protocol, run_name)
            dist = build_score_distribution_summary(
                run_rows,
                dataset_name=name,
                balanced_threshold=bal_thr,
                balanced_threshold_source=bal_src,
            )
            all_dist_summaries.append(dist)

            if write_figures and run_rows:
                fig_stem = f"{_slug(name)}_{run_name}"
                p_dist = figures / f"score_distribution_{fig_stem}.png"
                p_trade = figures / f"threshold_tradeoff_{fig_stem}.png"
                got = _write_score_distribution_figure(
                    run_rows,
                    p_dist,
                    balanced_threshold=bal_thr,
                    title=(
                        f"{name} · {run_name}\n"
                        "raw fake_score distribution (uncalibrated)"
                    ),
                )
                if got:
                    written[f"figures/score_distribution_{fig_stem}.png"] = got
                got = _write_threshold_tradeoff_figure(
                    run_rows,
                    p_trade,
                    balanced_threshold=bal_thr,
                )
                if got:
                    written[f"figures/threshold_tradeoff_{fig_stem}.png"] = got
                if write_ecdf:
                    p_ecdf = figures / f"score_ecdf_{fig_stem}.png"
                    got = _write_optional_ecdf_figure(
                        run_rows, p_ecdf, balanced_threshold=bal_thr
                    )
                    if got:
                        written[f"figures/score_ecdf_{fig_stem}.png"] = got

            just = build_threshold_justification_row(
                run_rows, dataset_name=name, protocol=protocol
            )
            all_justification.append(just)
            all_performance.append(build_performance_summary_row(just))

    # Consolidated tables
    p = write_json(
        tables / "score_distribution_summary.json",
        {
            "generated_at": utc_timestamp(),
            "rows": all_dist_summaries,
            "score_semantics": SCORE_SEMANTICS,
        },
    )
    written["tables/score_distribution_summary.json"] = str(p.resolve())
    flat_dist = []
    for row in all_dist_summaries:
        for cohort, key in (("real", "real_fake_score"), ("fake", "fake_fake_score")):
            stats = row.get(key) or {}
            flat_dist.append(
                {
                    "dataset": row.get("dataset"),
                    "run_name": row.get("run_name"),
                    "ground_truth_cohort": cohort,
                    "balanced_threshold": row.get("balanced_threshold"),
                    **stats,
                }
            )
    p = write_csv(tables / "score_distribution_summary.csv", flat_dist)
    written["tables/score_distribution_summary.csv"] = str(p.resolve())

    p = write_json(
        tables / "threshold_justification.json",
        {
            "generated_at": utc_timestamp(),
            "discrimination_wording": THESIS_DISCRIMINATION_WORDING,
            "high_threshold_note": HIGH_THRESHOLD_NOTE,
            "rows": all_justification,
        },
    )
    written["tables/threshold_justification.json"] = str(p.resolve())
    p = write_csv(tables / "threshold_justification.csv", all_justification)
    written["tables/threshold_justification.csv"] = str(p.resolve())
    md_path = tables / "threshold_justification.md"
    md_path.write_text(threshold_justification_markdown(all_justification), encoding="utf-8")
    written["tables/threshold_justification.md"] = str(md_path.resolve())

    p = write_json(
        tables / "thesis_performance_summary.json",
        {
            "generated_at": utc_timestamp(),
            "discrimination_wording": THESIS_DISCRIMINATION_WORDING,
            "rows": all_performance,
        },
    )
    written["tables/thesis_performance_summary.json"] = str(p.resolve())
    p = write_csv(tables / "thesis_performance_summary.csv", all_performance)
    written["tables/thesis_performance_summary.csv"] = str(p.resolve())
    md_path = tables / "thesis_performance_summary.md"
    md_path.write_text(performance_summary_markdown(all_performance), encoding="utf-8")
    written["tables/thesis_performance_summary.md"] = str(md_path.resolve())

    # Consolidated cases + manual worksheet
    p = write_json(
        cases_dir / "representative_cases.json",
        {
            "generated_at": utc_timestamp(),
            "cases": all_cases,
            "note": (
                "Deterministic automated selection. Not cherry-picked for drama. "
                "Manual verification is separate."
            ),
        },
    )
    written["cases/representative_cases.json"] = str(p.resolve())
    p = write_csv(cases_dir / "representative_cases.csv", all_cases)
    written["cases/representative_cases.csv"] = str(p.resolve())
    md_path = cases_dir / "representative_cases.md"
    md_path.write_text(
        representative_cases_markdown(all_cases, title="Representative cases (all datasets)"),
        encoding="utf-8",
    )
    written["cases/representative_cases.md"] = str(md_path.resolve())

    manual_rows = build_manual_verification_rows(all_cases)
    p = write_csv(cases_dir / "manual_ground_truth_verification.csv", manual_rows)
    written["cases/manual_ground_truth_verification.csv"] = str(p.resolve())
    md_path = cases_dir / "manual_ground_truth_verification.md"
    md_path.write_text(manual_verification_markdown(manual_rows), encoding="utf-8")
    written["cases/manual_ground_truth_verification.md"] = str(md_path.resolve())

    # Consolidated full audit
    p = write_json(
        tables / "ground_truth_audit.json",
        {
            "generated_at": utc_timestamp(),
            "n_rows": len(all_audit_rows),
            "rows": [r.to_dict() for r in all_audit_rows],
            "score_semantics": SCORE_SEMANTICS,
            "claim_boundary_notes": list(CLAIM_BOUNDARY_NOTES),
        },
    )
    written["tables/ground_truth_audit.json"] = str(p.resolve())
    from eval.ground_truth_audit import audit_rows_to_csv_dicts

    p = write_csv(tables / "ground_truth_audit.csv", audit_rows_to_csv_dicts(all_audit_rows))
    written["tables/ground_truth_audit.csv"] = str(p.resolve())

    validation_seed = None
    for spec_info in input_files:
        # best-effort from first loaded protocol
        pass
    for spec in datasets:
        protocol, _err = load_validation_protocol_payload(spec.get("validation_protocol"))
        if protocol and isinstance(protocol.get("split"), Mapping):
            validation_seed = protocol["split"].get("seed")
            if validation_seed is not None:
                break

    manifest = {
        "generated_at": utc_timestamp(),
        "script_version": SCRIPT_VERSION,
        "git": get_git_provenance(),
        "input_files": input_files,
        "dataset_sizes": dataset_sizes,
        "configs": list(run_names),
        "validation_seed": validation_seed,
        "score_field": SCORE_FIELD,
        "score_semantics": SCORE_SEMANTICS,
        "threshold_semantics": {
            "decision_rule": DECISION_RULE,
            "reference_threshold": REFERENCE_THRESHOLD,
            "balanced_preset": PRESET_BALANCED,
            "selection_note": (
                "Balanced thresholds are taken from validation_protocol validation "
                "data only; held-out metrics are evaluation information."
            ),
        },
        "discrimination_wording": THESIS_DISCRIMINATION_WORDING,
        "claim_boundary_notes": list(CLAIM_BOUNDARY_NOTES),
        "unavailable_information": unavailable,
        "generated_outputs": written,
        "manual_steps_remaining": [
            "Fill human columns in cases/manual_ground_truth_verification.csv",
            "Visually inspect representative cases against source media where available",
            "Confirm ground-truth labels against dataset provenance documentation",
            "Do not claim manual verification complete until human fields are filled",
        ],
    }
    p = write_json(out_root / "thesis_evidence_manifest.json", manifest)
    written["thesis_evidence_manifest.json"] = str(p.resolve())
    p = write_json(metadata / "thesis_evidence_manifest.json", manifest)
    written["metadata/thesis_evidence_manifest.json"] = str(p.resolve())

    return {
        "output_dir": str(out_root.resolve()),
        "manifest": manifest,
        "written": written,
        "n_audit_rows": len(all_audit_rows),
        "n_cases": len(all_cases),
        "unavailable_information": unavailable,
    }


def _slug(text: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text.strip())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_").lower() or "dataset"

