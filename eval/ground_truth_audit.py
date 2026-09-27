"""Ground-truth audit and representative case selection (no inference).

Builds per-sample prediction tables from labelled ``aggregate.json`` scores and
optional validation-protocol thresholds. Raw ``fake_score`` values are never
rewritten. Scores are not treated as calibrated probabilities or confidence.
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from eval.experiment_configs import PRIMARY_ASSESSMENT_RUN, RUN_ORDER
from eval.labelled_analysis import AggregateAnalysisError, load_and_validate_aggregate
from eval.reproducibility import get_git_provenance, utc_timestamp
from eval.threshold_sweep import (
    PRESET_BALANCED,
    REFERENCE_THRESHOLD,
    SCORE_FIELD,
    SCORE_SEMANTICS,
    DECISION_RULE,
    predict_label,
    select_operating_presets,
)

SCRIPT_VERSION = "1.0.0"

OUTCOME_TP = "TP"
OUTCOME_TN = "TN"
OUTCOME_FP = "FP"
OUTCOME_FN = "FN"
OUTCOME_VALUES = (OUTCOME_TP, OUTCOME_TN, OUTCOME_FP, OUTCOME_FN)

CASE_STRONGEST_TP = "strongest_correct_fake"
CASE_STRONGEST_TN = "strongest_correct_real"
CASE_FALSE_POSITIVE = "false_positive"
CASE_FALSE_NEGATIVE = "false_negative"
CASE_CLOSEST_CORRECT = "closest_to_threshold_correct"
CASE_CLOSEST_INCORRECT = "closest_to_threshold_incorrect"
CASE_DETECTOR_AGREEMENT = "detector_agreement"
CASE_DETECTOR_CONFLICT = "detector_conflict"

CORE_CASE_KEYS: Tuple[str, ...] = (
    CASE_STRONGEST_TP,
    CASE_STRONGEST_TN,
    CASE_FALSE_POSITIVE,
    CASE_FALSE_NEGATIVE,
    CASE_CLOSEST_CORRECT,
    CASE_CLOSEST_INCORRECT,
)
OPTIONAL_CASE_KEYS: Tuple[str, ...] = (
    CASE_DETECTOR_AGREEMENT,
    CASE_DETECTOR_CONFLICT,
)

CLAIM_BOUNDARY_NOTES: Tuple[str, ...] = (
    "Raw fake_score is an uncalibrated model score, not a probability or confidence.",
    "Decision rule: predict fake/manipulated if fake_score >= threshold.",
    "Reference threshold 0.50 is the existing post-hoc convention, not an optimal default.",
    "Validation-selected Balanced thresholds come from the validation partition only.",
    "Held-out metrics evaluate a frozen threshold; they never select the threshold.",
    "Stable / Uncertain / Contested (prototype status) is separate from the threshold decision.",
    "This audit does not claim a participant study or manual verification completion.",
)


class GroundTruthAuditError(ValueError):
    """Ground-truth audit input is missing required fields or is malformed."""


@dataclass(frozen=True)
class AuditRow:
    """One usable sample × configuration prediction record."""

    sample_id: str
    source_path: Optional[str]
    dataset: Optional[str]
    ground_truth: str
    run_name: str
    fake_score: float
    real_score: Optional[float]
    reference_threshold: float
    reference_prediction: str
    active_threshold: float
    active_prediction: str
    reference_correct: bool
    active_correct: bool
    outcome: str
    distance_from_active_threshold: float
    prediction_changed_vs_reference: bool
    specialist_scores: Dict[str, Optional[float]] = field(default_factory=dict)
    specialist_availability: Dict[str, bool] = field(default_factory=dict)
    evidence_agreement: Optional[str] = None
    prototype_status: Optional[str] = None
    saved_label: Optional[str] = None
    config_status: Optional[str] = None
    manipulation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def outcome_category(ground_truth: str, prediction: str) -> str:
    """Map truth/prediction to TP/TN/FP/FN (fake = positive class)."""

    gt = ground_truth.strip().lower()
    pred = prediction.strip().lower()
    if gt not in {"real", "fake"} or pred not in {"real", "fake"}:
        raise GroundTruthAuditError(
            f"invalid ground_truth/prediction: {ground_truth!r}/{prediction!r}"
        )
    if gt == "fake" and pred == "fake":
        return OUTCOME_TP
    if gt == "real" and pred == "real":
        return OUTCOME_TN
    if gt == "real" and pred == "fake":
        return OUTCOME_FP
    return OUTCOME_FN


def _safe_float(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        f = float(value)
        return f if math.isfinite(f) else None
    return None


def _extract_specialist_scores(cfg: Mapping[str, Any]) -> Tuple[Dict[str, Optional[float]], Dict[str, bool]]:
    """Return specialist scores without inventing missing values as 0."""

    scores: Dict[str, Optional[float]] = {}
    availability: Dict[str, bool] = {}

    for key, display in (
        ("blending_detector_score", "blending"),
        ("diffusion_detector_score", "diffusion"),
    ):
        val = _safe_float(cfg.get(key))
        scores[display] = val
        availability[display] = val is not None

    expert = cfg.get("expert_scores")
    if isinstance(expert, Mapping):
        for name, raw in expert.items():
            if not isinstance(name, str) or not name:
                continue
            if name in scores and scores[name] is not None:
                continue
            val = _safe_float(raw)
            scores[name] = val
            availability[name] = val is not None
    return scores, availability


def _config_has_usable_fake_score(cfg: Optional[Mapping[str, Any]]) -> bool:
    if cfg is None:
        return False
    score = _safe_float(cfg.get(SCORE_FIELD))
    return score is not None and 0.0 <= score <= 1.0


def load_validation_protocol_payload(
    path: Optional[Path | str],
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Load validation_protocol.json; return (payload, error). Never raises."""

    if path is None or str(path).strip() == "":
        return None, None
    path_obj = Path(path)
    if not path_obj.is_file():
        return None, f"validation protocol not found: {path_obj}"
    try:
        payload = json.loads(path_obj.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, f"malformed validation protocol: {type(exc).__name__}: {exc}"
    if not isinstance(payload, dict):
        return None, "validation protocol root must be a JSON object"
    return payload, None


def resolve_balanced_threshold_from_protocol(
    protocol: Optional[Mapping[str, Any]],
    run_name: str,
) -> Tuple[Optional[float], Optional[str]]:
    """Resolve Balanced threshold from validation data only (never held-out).

    Preference order:
    1. ``dashboard_presets.Balanced.threshold``
    2. ``select_operating_presets(validation_sweep).Balanced``
    3. ``selections[max_balanced_accuracy].selection.selected_threshold``
    """

    if protocol is None:
        return None, "no validation protocol loaded"
    results = protocol.get("results_by_run")
    if not isinstance(results, Mapping):
        return None, "validation protocol has no results_by_run"
    block = results.get(run_name)
    if not isinstance(block, Mapping):
        return None, f"no validation block for run {run_name!r}"

    presets = block.get("dashboard_presets")
    if isinstance(presets, Mapping):
        bal = presets.get(PRESET_BALANCED)
        if isinstance(bal, Mapping):
            thr = _safe_float(bal.get("threshold"))
            if thr is not None and 0.0 <= thr <= 1.0:
                return thr, "dashboard_presets.Balanced"

    sweep = block.get("validation_sweep")
    if isinstance(sweep, list) and sweep:
        rows = [r for r in sweep if isinstance(r, dict)]
        if rows:
            pack = select_operating_presets(rows)
            bal = pack["presets"].get(PRESET_BALANCED) or {}
            thr = _safe_float(bal.get("threshold"))
            if thr is not None and 0.0 <= thr <= 1.0:
                return thr, "select_operating_presets(validation_sweep).Balanced"

    selections = block.get("selections")
    if isinstance(selections, Mapping):
        sel = selections.get("max_balanced_accuracy")
        if isinstance(sel, Mapping):
            selection = sel.get("selection")
            if isinstance(selection, Mapping):
                thr = _safe_float(selection.get("selected_threshold"))
                if thr is not None and 0.0 <= thr <= 1.0:
                    return thr, "selections.max_balanced_accuracy (validation)"

    return None, f"Balanced threshold unavailable for run {run_name!r}"


def resolve_active_threshold(
    *,
    run_name: str,
    explicit_threshold: Optional[float] = None,
    protocol: Optional[Mapping[str, Any]] = None,
) -> Tuple[float, str]:
    """Resolve the active decision threshold for an audit/config."""

    if explicit_threshold is not None:
        thr = float(explicit_threshold)
        if not math.isfinite(thr) or not (0.0 <= thr <= 1.0):
            raise GroundTruthAuditError(f"threshold out of [0, 1]: {explicit_threshold!r}")
        return thr, "explicit"

    bal, note = resolve_balanced_threshold_from_protocol(protocol, run_name)
    if bal is not None:
        return float(bal), note or "validation Balanced"

    return float(REFERENCE_THRESHOLD), "reference_fallback"


def build_audit_rows(
    payload: Any,
    *,
    run_name: str = PRIMARY_ASSESSMENT_RUN,
    active_threshold: Optional[float] = None,
    protocol: Optional[Mapping[str, Any]] = None,
    reference_threshold: float = REFERENCE_THRESHOLD,
) -> Tuple[List[AuditRow], Dict[str, Any]]:
    """Build audit rows for one configuration from a labelled aggregate."""

    try:
        normalised = load_and_validate_aggregate(payload)
    except AggregateAnalysisError as exc:
        raise GroundTruthAuditError(str(exc)) from exc

    if run_name not in RUN_ORDER and not isinstance(run_name, str):
        raise GroundTruthAuditError(f"invalid run_name: {run_name!r}")

    thr, thr_source = resolve_active_threshold(
        run_name=run_name,
        explicit_threshold=active_threshold,
        protocol=protocol,
    )
    ref = float(reference_threshold)
    rows: List[AuditRow] = []
    skipped = 0

    for image in normalised["images"]:
        cfg = image.get("_configs_by_name", {}).get(run_name)
        if not _config_has_usable_fake_score(cfg):
            skipped += 1
            continue
        assert isinstance(cfg, dict)
        fake_score = float(cfg[SCORE_FIELD])
        gt = image["ground_truth"]
        sample_id = str(image.get("image_id") or image.get("source_path") or len(rows))
        source_path = image.get("source_path")
        if source_path is not None:
            source_path = str(source_path)

        ref_pred = predict_label(fake_score, ref)
        act_pred = predict_label(fake_score, thr)
        outcome = outcome_category(gt, act_pred)
        specialists, availability = _extract_specialist_scores(cfg)

        rows.append(
            AuditRow(
                sample_id=sample_id,
                source_path=source_path,
                dataset=(
                    str(image["dataset"]) if image.get("dataset") is not None else None
                ),
                ground_truth=gt,
                run_name=run_name,
                fake_score=fake_score,
                real_score=_safe_float(cfg.get("real_score")),
                reference_threshold=ref,
                reference_prediction=ref_pred,
                active_threshold=thr,
                active_prediction=act_pred,
                reference_correct=(ref_pred == gt),
                active_correct=(act_pred == gt),
                outcome=outcome,
                distance_from_active_threshold=abs(fake_score - thr),
                prediction_changed_vs_reference=(ref_pred != act_pred),
                specialist_scores=specialists,
                specialist_availability=availability,
                evidence_agreement=(
                    str(image["evidence_agreement"])
                    if image.get("evidence_agreement") is not None
                    else None
                ),
                prototype_status=(
                    str(image["prototype_status"])
                    if image.get("prototype_status") is not None
                    else None
                ),
                saved_label=(
                    str(cfg["label"]) if cfg.get("label") in {"real", "fake"} else None
                ),
                config_status=(
                    str(cfg["status"]) if cfg.get("status") is not None else None
                ),
                manipulation=(
                    str(image["manipulation"])
                    if image.get("manipulation") is not None
                    else None
                ),
            )
        )

    # Deterministic order for downstream selection.
    rows.sort(key=lambda r: (r.sample_id, r.run_name))
    meta = {
        "run_name": run_name,
        "n_images_in_aggregate": len(normalised["images"]),
        "n_usable_rows": len(rows),
        "n_skipped": skipped,
        "active_threshold": thr,
        "active_threshold_source": thr_source,
        "reference_threshold": ref,
        "score_field": SCORE_FIELD,
        "score_semantics": SCORE_SEMANTICS,
        "decision_rule": DECISION_RULE,
    }
    return rows, meta


def build_audit_for_runs(
    payload: Any,
    *,
    run_names: Sequence[str] = RUN_ORDER,
    thresholds_by_run: Optional[Mapping[str, float]] = None,
    protocol: Optional[Mapping[str, Any]] = None,
    reference_threshold: float = REFERENCE_THRESHOLD,
) -> Tuple[List[AuditRow], Dict[str, Any]]:
    """Build audit rows for multiple configurations."""

    all_rows: List[AuditRow] = []
    per_run: Dict[str, Any] = {}
    for run_name in run_names:
        explicit = None
        if thresholds_by_run and run_name in thresholds_by_run:
            explicit = float(thresholds_by_run[run_name])
        rows, meta = build_audit_rows(
            payload,
            run_name=run_name,
            active_threshold=explicit,
            protocol=protocol,
            reference_threshold=reference_threshold,
        )
        all_rows.extend(rows)
        per_run[run_name] = meta
    all_rows.sort(key=lambda r: (r.dataset or "", r.run_name, r.sample_id))
    summary = {
        "n_rows": len(all_rows),
        "run_names": list(run_names),
        "per_run": per_run,
        "score_semantics": SCORE_SEMANTICS,
        "decision_rule": DECISION_RULE,
        "claim_boundary_notes": list(CLAIM_BOUNDARY_NOTES),
    }
    return all_rows, summary


def _pick_extreme(
    candidates: Sequence[AuditRow],
    *,
    key_fn,
    prefer_max: bool,
) -> Optional[AuditRow]:
    if not candidates:
        return None
    best = candidates[0]
    best_key = key_fn(best)
    for row in candidates[1:]:
        k = key_fn(row)
        if prefer_max:
            if k > best_key or (k == best_key and row.sample_id < best.sample_id):
                best = row
                best_key = k
        else:
            if k < best_key or (k == best_key and row.sample_id < best.sample_id):
                best = row
                best_key = k
    # When keys tie on value, prefer lexicographically smaller sample_id
    # (already handled above). Re-resolve ties explicitly for stability.
    tied = [r for r in candidates if key_fn(r) == best_key]
    return sorted(tied, key=lambda r: r.sample_id)[0]


def _specialist_available(row: AuditRow) -> bool:
    return any(row.specialist_availability.values())


def _detector_supports_prediction(row: AuditRow) -> Optional[bool]:
    """Heuristic agreement: available specialists on the same side of 0.5 as prediction.

    Returns None when no specialist scores are available (do not invent).
    """

    available = [
        (name, score)
        for name, score in row.specialist_scores.items()
        if row.specialist_availability.get(name) and score is not None
    ]
    if not available:
        return None
    pred_fake = row.active_prediction == "fake"
    agreements = []
    for _name, score in available:
        # Specialist higher ⇒ more fake-leaning (existing detector convention).
        specialist_fake = float(score) >= 0.5
        agreements.append(specialist_fake == pred_fake)
    if all(agreements):
        return True
    if not any(agreements):
        return False
    return False  # mixed ⇒ treat as conflict for optional case


def select_representative_cases(
    rows: Sequence[AuditRow],
    *,
    include_optional_detector_cases: bool = True,
) -> List[Dict[str, Any]]:
    """Deterministically select representative thesis examples.

    Selection rules are recorded on each case. Missing categories are omitted
    rather than fabricating samples.
    """

    ordered = sorted(rows, key=lambda r: (r.run_name, r.sample_id))
    by_outcome: Dict[str, List[AuditRow]] = {k: [] for k in OUTCOME_VALUES}
    for row in ordered:
        by_outcome.setdefault(row.outcome, []).append(row)

    cases: List[Dict[str, Any]] = []

    def _add(case_key: str, row: Optional[AuditRow], rule: str) -> None:
        if row is None:
            return
        cases.append(
            {
                "case_key": case_key,
                "selection_rule": rule,
                "sample_id": row.sample_id,
                "source_path": row.source_path,
                "dataset": row.dataset,
                "ground_truth": row.ground_truth,
                "run_name": row.run_name,
                "fake_score": row.fake_score,
                "real_score": row.real_score,
                "active_threshold": row.active_threshold,
                "active_prediction": row.active_prediction,
                "outcome": row.outcome,
                "active_correct": row.active_correct,
                "distance_from_active_threshold": row.distance_from_active_threshold,
                "evidence_agreement": row.evidence_agreement,
                "prototype_status": row.prototype_status,
                "specialist_scores": dict(row.specialist_scores),
                "specialist_availability": dict(row.specialist_availability),
            }
        )

    _add(
        CASE_STRONGEST_TP,
        _pick_extreme(by_outcome[OUTCOME_TP], key_fn=lambda r: r.fake_score, prefer_max=True),
        "Among TP, maximise fake_score; tie-break smallest sample_id.",
    )
    _add(
        CASE_STRONGEST_TN,
        _pick_extreme(by_outcome[OUTCOME_TN], key_fn=lambda r: r.fake_score, prefer_max=False),
        "Among TN, minimise fake_score; tie-break smallest sample_id.",
    )
    _add(
        CASE_FALSE_POSITIVE,
        _pick_extreme(by_outcome[OUTCOME_FP], key_fn=lambda r: r.fake_score, prefer_max=True),
        "Among FP, maximise fake_score (most extreme false alarm); tie-break smallest sample_id.",
    )
    _add(
        CASE_FALSE_NEGATIVE,
        _pick_extreme(by_outcome[OUTCOME_FN], key_fn=lambda r: r.fake_score, prefer_max=False),
        "Among FN, minimise fake_score (most extreme miss); tie-break smallest sample_id.",
    )

    correct = [r for r in ordered if r.active_correct]
    incorrect = [r for r in ordered if not r.active_correct]
    _add(
        CASE_CLOSEST_CORRECT,
        _pick_extreme(
            correct,
            key_fn=lambda r: r.distance_from_active_threshold,
            prefer_max=False,
        ),
        "Among correct predictions, minimise |fake_score − active_threshold|; "
        "tie-break smallest sample_id.",
    )
    _add(
        CASE_CLOSEST_INCORRECT,
        _pick_extreme(
            incorrect,
            key_fn=lambda r: r.distance_from_active_threshold,
            prefer_max=False,
        ),
        "Among incorrect predictions, minimise |fake_score − active_threshold|; "
        "tie-break smallest sample_id.",
    )

    if include_optional_detector_cases:
        agree_rows = []
        conflict_rows = []
        for row in ordered:
            if not _specialist_available(row):
                continue
            flag = _detector_supports_prediction(row)
            if flag is True:
                agree_rows.append(row)
            elif flag is False:
                conflict_rows.append(row)
        _add(
            CASE_DETECTOR_AGREEMENT,
            _pick_extreme(
                agree_rows,
                key_fn=lambda r: r.distance_from_active_threshold,
                prefer_max=True,
            ),
            "Optional: available specialists unanimously agree with active prediction; "
            "prefer largest |fake_score − threshold|; tie-break smallest sample_id.",
        )
        _add(
            CASE_DETECTOR_CONFLICT,
            _pick_extreme(
                conflict_rows,
                key_fn=lambda r: r.distance_from_active_threshold,
                prefer_max=True,
            ),
            "Optional: available specialists disagree with active prediction (or mixed); "
            "prefer largest |fake_score − threshold|; tie-break smallest sample_id.",
        )

    return cases


def select_representative_cases_by_run(
    rows: Sequence[AuditRow],
    *,
    include_optional_detector_cases: bool = True,
) -> Dict[str, List[Dict[str, Any]]]:
    """Select representative cases independently per configuration."""

    by_run: Dict[str, List[AuditRow]] = {}
    for row in rows:
        by_run.setdefault(row.run_name, []).append(row)
    out: Dict[str, List[Dict[str, Any]]] = {}
    for run_name in sorted(by_run):
        out[run_name] = select_representative_cases(
            by_run[run_name],
            include_optional_detector_cases=include_optional_detector_cases,
        )
    return out


def audit_rows_to_csv_dicts(rows: Sequence[AuditRow]) -> List[Dict[str, Any]]:
    """Flatten audit rows for CSV export (JSON-encode nested dicts)."""

    out: List[Dict[str, Any]] = []
    for row in rows:
        d = row.to_dict()
        d["specialist_scores"] = json.dumps(d["specialist_scores"], sort_keys=True)
        d["specialist_availability"] = json.dumps(
            d["specialist_availability"], sort_keys=True
        )
        out.append(d)
    return out


def representative_cases_markdown(
    cases: Sequence[Mapping[str, Any]],
    *,
    title: str = "Representative cases",
) -> str:
    """Markdown table usable in supervisor slides / thesis."""

    lines = [
        f"# {title}",
        "",
        "| Sample | Ground truth | Config | fake_score | Threshold | Prediction | Outcome | Why selected |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for case in cases:
        lines.append(
            "| {sample} | {gt} | {cfg} | {score} | {thr} | {pred} | {out} | {why} |".format(
                sample=case.get("sample_id", ""),
                gt=case.get("ground_truth", ""),
                cfg=case.get("run_name", ""),
                score=(
                    f"{float(case['fake_score']):.4f}"
                    if case.get("fake_score") is not None
                    else ""
                ),
                thr=(
                    f"{float(case['active_threshold']):.4f}"
                    if case.get("active_threshold") is not None
                    else ""
                ),
                pred=case.get("active_prediction", ""),
                out=case.get("outcome", ""),
                why=(case.get("selection_rule") or case.get("case_key") or "").replace(
                    "|", "/"
                ),
            )
        )
    lines.append("")
    lines.append(
        "_Automated selection from saved raw scores. Not a claim of visual verification._"
    )
    lines.append("")
    return "\n".join(lines)


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})
    return path


def write_ground_truth_audit_outputs(
    rows: Sequence[AuditRow],
    output_dir: Path | str,
    *,
    meta: Optional[Mapping[str, Any]] = None,
    write_representative: bool = True,
) -> Dict[str, str]:
    """Write audit JSON/CSV and optional representative-case artefacts."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": utc_timestamp(),
        "script_version": SCRIPT_VERSION,
        "git": get_git_provenance(),
        "score_field": SCORE_FIELD,
        "score_semantics": SCORE_SEMANTICS,
        "decision_rule": DECISION_RULE,
        "reference_threshold": REFERENCE_THRESHOLD,
        "claim_boundary_notes": list(CLAIM_BOUNDARY_NOTES),
        "meta": dict(meta or {}),
        "n_rows": len(rows),
        "rows": [r.to_dict() for r in rows],
    }
    written: Dict[str, str] = {}
    p_json = write_json(out / "ground_truth_audit.json", payload)
    written["ground_truth_audit.json"] = str(p_json.resolve())
    p_csv = write_csv(out / "ground_truth_audit.csv", audit_rows_to_csv_dicts(rows))
    written["ground_truth_audit.csv"] = str(p_csv.resolve())

    if write_representative:
        by_run = select_representative_cases_by_run(rows)
        flat: List[Dict[str, Any]] = []
        for run_name, cases in by_run.items():
            for case in cases:
                entry = dict(case)
                entry["config"] = run_name
                flat.append(entry)
        rep_payload = {
            "generated_at": utc_timestamp(),
            "script_version": SCRIPT_VERSION,
            "selection_notes": (
                "Cases are selected deterministically from audit rows. "
                "They are not cherry-picked for visual drama and do not "
                "constitute completed manual verification."
            ),
            "cases_by_run": by_run,
            "cases": flat,
        }
        p_rep_json = write_json(out / "representative_cases.json", rep_payload)
        written["representative_cases.json"] = str(p_rep_json.resolve())
        p_rep_csv = write_csv(out / "representative_cases.csv", flat)
        written["representative_cases.csv"] = str(p_rep_csv.resolve())
        md = representative_cases_markdown(flat)
        p_md = out / "representative_cases.md"
        p_md.write_text(md, encoding="utf-8")
        written["representative_cases.md"] = str(p_md.resolve())

    return written


def load_aggregate_json(path: Path | str) -> Dict[str, Any]:
    path_obj = Path(path)
    try:
        payload = json.loads(path_obj.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GroundTruthAuditError(
            f"could not read aggregate {path_obj}: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise GroundTruthAuditError("aggregate root must be a JSON object")
    return payload
