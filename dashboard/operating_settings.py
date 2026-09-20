"""Dashboard operating-settings helpers (Streamlit-free).

Applies a user-adjustable decision threshold to already-saved raw ``fake_score``
values. Does not run inference and does not treat scores as calibrated
probabilities or confidence.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from eval.experiment_configs import PRIMARY_ASSESSMENT_RUN, RUN_ORDER, RUN_TITLES
from eval.threshold_sweep import (
    PRESET_BALANCED,
    PRESET_CONSERVATIVE,
    PRESET_NAMES,
    PRESET_SENSITIVE,
    PRESET_TARGET_NOTE,
    PRESET_TARGET_RATE,
    REFERENCE_THRESHOLD,
    SCORE_SEMANTICS,
    default_decision_operating_settings,
    predict_label,
    select_operating_presets,
)

# User-facing configuration labels (thesis 2x2).
CONFIG_DISPLAY_LABELS: Dict[str, str] = {
    "none": "Main model only",
    "blending": "Main + blending specialist",
    "diffusion": "Main + diffusion specialist",
    "blending_diffusion": "Main + blending + diffusion specialists",
}

THRESHOLD_SOURCE_REFERENCE = "reference"
THRESHOLD_SOURCE_VALIDATION = "validation-selected"
THRESHOLD_SOURCE_CUSTOM = "custom"

THRESHOLD_SOURCE_LABELS: Dict[str, str] = {
    THRESHOLD_SOURCE_REFERENCE: "Reference / default",
    THRESHOLD_SOURCE_VALIDATION: "Validation-selected",
    THRESHOLD_SOURCE_CUSTOM: "Custom / user-adjusted",
}

ENV_VALIDATION_PROTOCOL = "X2DFD_VALIDATION_PROTOCOL"

RAW_SCORE_NOTE = (
    "Raw detector scores are not calibrated probabilities or confidence values."
)
LOWER_THRESHOLD_COPY = (
    "Lower threshold: more sensitive to potential manipulation; "
    "may increase false positives on genuine media."
)
HIGHER_THRESHOLD_COPY = (
    "Higher threshold: more conservative before labelling media fake; "
    "may miss more manipulations."
)
TRADEOFF_NOTE = (
    "Sensitive / Balanced / Conservative are operating trade-offs on the raw "
    "fake_score axis, not confidence levels."
)
EVIDENCE_SEPARATION_NOTE = (
    "Detection threshold changes the real/manipulated decision. "
    "Evidence agreement describes whether the available detectors agree."
)
VALIDATION_SELECT_WORDING = (
    "Threshold selected on the validation partition only; held-out test scores "
    "were not used for selection."
)
HELD_OUT_WORDING = (
    "Held-out test metrics evaluate a frozen validation-selected threshold "
    "on a disjoint stratified partition (descriptive; not a universal default)."
)
COLLAPSE_NOTE = (
    "For this validation set, these operating goals resolve to the same threshold."
)
NO_SWEEP_REASON = (
    "Presets need the full validation threshold sweep. Load a "
    "validation_protocol.json that includes per-config validation_sweep rows "
    "(re-run the CPU validation protocol if this file is from an earlier run)."
)

PRESET_USER_COPY: Dict[str, str] = {
    PRESET_SENSITIVE: (
        "Catches more potentially manipulated media, but is more likely to flag genuine media."
    ),
    PRESET_BALANCED: (
        "Balances detection of manipulated media against false alarms on genuine media."
    ),
    PRESET_CONSERVATIVE: (
        "Requires stronger model evidence before labelling media manipulated, "
        "reducing false alarms but potentially missing more manipulations."
    ),
}

# Re-export names used by the Streamlit app / tests.
__all__ = [
    "CONFIG_DISPLAY_LABELS",
    "ENV_VALIDATION_PROTOCOL",
    "PRESET_BALANCED",
    "PRESET_CONSERVATIVE",
    "PRESET_NAMES",
    "PRESET_SENSITIVE",
    "PRESET_TARGET_RATE",
    "RAW_SCORE_NOTE",
    "THRESHOLD_SOURCE_CUSTOM",
    "THRESHOLD_SOURCE_REFERENCE",
    "THRESHOLD_SOURCE_VALIDATION",
]


@dataclass(frozen=True)
class ValidationOperatingPoint:
    """One dataset/config/criterion operating point from validation_protocol.json."""

    run_name: str
    criterion: str
    selected_threshold: float
    dataset_name: Optional[str] = None
    val_balanced_accuracy: Optional[float] = None
    test_balanced_accuracy: Optional[float] = None
    ref_0_50_test_balanced_accuracy: Optional[float] = None
    source_path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PresetInfo:
    """One Sensitive / Balanced / Conservative operating preset."""

    name: str
    threshold: Optional[float]
    user_copy: str
    provenance: str
    validation_sensitivity_fake: Optional[float] = None
    validation_specificity_real: Optional[float] = None
    validation_balanced_accuracy: Optional[float] = None
    target: str = ""
    target_met: Optional[bool] = None
    used_fallback: bool = False
    held_out_balanced_accuracy: Optional[float] = None
    held_out_sensitivity_fake: Optional[float] = None
    held_out_specificity_real: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationMetadata:
    """Optional validation-protocol payload for dashboard presets/provenance."""

    loaded: bool = False
    path: Optional[str] = None
    dataset_name: Optional[str] = None
    error: Optional[str] = None
    points: Dict[str, Dict[str, ValidationOperatingPoint]] = field(default_factory=dict)
    sweeps: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    stored_presets: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    score_semantics: str = SCORE_SEMANTICS
    notes: List[str] = field(default_factory=list)

    @property
    def has_sweeps(self) -> bool:
        return any(bool(rows) for rows in self.sweeps.values())

    @property
    def available(self) -> bool:
        return self.loaded and not self.error and (bool(self.points) or self.has_sweeps)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "loaded": self.loaded,
            "available": self.available,
            "has_sweeps": self.has_sweeps,
            "path": self.path,
            "dataset_name": self.dataset_name,
            "error": self.error,
            "score_semantics": self.score_semantics,
            "notes": list(self.notes),
            "points": {
                run: {crit: pt.to_dict() for crit, pt in by_crit.items()}
                for run, by_crit in self.points.items()
            },
            "sweep_run_names": list(self.sweeps),
        }


@dataclass(frozen=True)
class OperatingPrediction:
    """Threshold-based prediction for one saved sample (raw score unchanged)."""

    raw_fake_score: Optional[float]
    raw_real_score: Optional[float]
    decision_threshold: float
    predicted_label: Optional[str]
    saved_label: Optional[str]
    current_decision_word: str
    display_text: str
    saved_decision_provenance: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OperatingSettingsState:
    """UI-facing operating settings for the active sample/config."""

    active_run_name: str
    active_config_label: str
    decision_threshold: float
    threshold_source: str
    threshold_source_label: str
    prediction: OperatingPrediction
    available_run_names: List[str]
    unavailable_run_names: List[str]
    presets: Dict[str, Optional[float]]
    preset_details: Dict[str, PresetInfo]
    presets_available: bool
    presets_unavailable_reason: Optional[str]
    presets_collapsed: bool
    collapse_note: Optional[str]
    unexpected_order: bool
    order_note: Optional[str]
    matched_preset: Optional[str]
    validation: Optional[ValidationMetadata]
    validation_point: Optional[ValidationOperatingPoint]
    reference_threshold: float = REFERENCE_THRESHOLD
    raw_score_note: str = RAW_SCORE_NOTE
    lower_threshold_copy: str = LOWER_THRESHOLD_COPY
    higher_threshold_copy: str = HIGHER_THRESHOLD_COPY
    tradeoff_note: str = TRADEOFF_NOTE
    evidence_separation_note: str = EVIDENCE_SEPARATION_NOTE
    provenance_lines: List[str] = field(default_factory=list)
    decision_operating_settings: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "active_run_name": self.active_run_name,
            "active_config_label": self.active_config_label,
            "decision_threshold": self.decision_threshold,
            "threshold_source": self.threshold_source,
            "threshold_source_label": self.threshold_source_label,
            "prediction": self.prediction.to_dict(),
            "available_run_names": list(self.available_run_names),
            "unavailable_run_names": list(self.unavailable_run_names),
            "presets": dict(self.presets),
            "preset_details": {k: v.to_dict() for k, v in self.preset_details.items()},
            "presets_available": self.presets_available,
            "presets_unavailable_reason": self.presets_unavailable_reason,
            "presets_collapsed": self.presets_collapsed,
            "collapse_note": self.collapse_note,
            "unexpected_order": self.unexpected_order,
            "order_note": self.order_note,
            "matched_preset": self.matched_preset,
            "validation": None if self.validation is None else self.validation.to_dict(),
            "validation_point": (
                None if self.validation_point is None else self.validation_point.to_dict()
            ),
            "reference_threshold": self.reference_threshold,
            "raw_score_note": self.raw_score_note,
            "lower_threshold_copy": self.lower_threshold_copy,
            "higher_threshold_copy": self.higher_threshold_copy,
            "tradeoff_note": self.tradeoff_note,
            "evidence_separation_note": self.evidence_separation_note,
            "provenance_lines": list(self.provenance_lines),
            "decision_operating_settings": dict(self.decision_operating_settings),
        }


def config_display_label(run_name: str) -> str:
    return CONFIG_DISPLAY_LABELS.get(
        run_name, RUN_TITLES.get(run_name, run_name.replace("_", " "))
    )


def decision_display_word(label: Optional[str]) -> str:
    if label == "fake":
        return "MANIPULATED"
    if label == "real":
        return "REAL"
    return "—"


def format_rate_percent(value: Optional[float]) -> str:
    if value is None:
        return "—"
    return f"{float(value) * 100:.1f}%"


def resolve_validation_protocol_path(
    explicit: Optional[Path | str] = None,
    *,
    env: Optional[Mapping[str, str]] = None,
) -> Optional[Path]:
    """Resolve optional validation_protocol.json path (explicit > env > None)."""

    if explicit is not None and str(explicit).strip():
        return Path(explicit)
    environ = env if env is not None else os.environ
    raw = (environ.get(ENV_VALIDATION_PROTOCOL) or "").strip()
    if raw:
        return Path(raw)
    return None


def _infer_dataset_name(payload: Mapping[str, Any], path: Path) -> Optional[str]:
    meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
    for key in ("dataset_name", "dataset", "name"):
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    input_path = meta.get("input_path")
    if isinstance(input_path, str) and input_path.strip():
        lower = input_path.lower().replace("\\", "/")
        if "celeb" in lower:
            return "Celeb-DF-v2"
        if "deepfakeface" in lower or "dff" in lower:
            return "DeepFakeFace"
        return Path(input_path).parent.name or Path(input_path).stem
    parent = path.parent.name
    if parent and parent not in {".", ""}:
        return parent
    return None


def _safe_float(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return number


def _load_points_from_table(
    table: Any,
    *,
    dataset_name: Optional[str],
    source_path: str,
) -> Dict[str, Dict[str, ValidationOperatingPoint]]:
    points: Dict[str, Dict[str, ValidationOperatingPoint]] = {}
    if not isinstance(table, list):
        return points
    for row in table:
        if not isinstance(row, dict):
            continue
        run_name = row.get("run_name")
        criterion = row.get("criterion")
        thr = _safe_float(row.get("selected_threshold"))
        if not isinstance(run_name, str) or not isinstance(criterion, str) or thr is None:
            continue
        if not (0.0 <= thr <= 1.0):
            continue
        points.setdefault(run_name, {})[criterion] = ValidationOperatingPoint(
            run_name=run_name,
            criterion=criterion,
            selected_threshold=thr,
            dataset_name=dataset_name,
            val_balanced_accuracy=_safe_float(row.get("val_balanced_accuracy")),
            test_balanced_accuracy=_safe_float(row.get("test_balanced_accuracy")),
            ref_0_50_test_balanced_accuracy=_safe_float(
                row.get("ref_0_50_test_balanced_accuracy")
            ),
            source_path=source_path,
        )
    return points


def load_validation_protocol(
    path: Optional[Path | str] = None,
    *,
    env: Optional[Mapping[str, str]] = None,
) -> ValidationMetadata:
    """Load validation_protocol.json safely; missing/malformed never raises."""

    resolved = resolve_validation_protocol_path(path, env=env)
    if resolved is None:
        return ValidationMetadata(
            loaded=False,
            path=None,
            error=None,
            notes=["No validation protocol path configured."],
        )
    path_obj = Path(resolved)
    if not path_obj.is_file():
        return ValidationMetadata(
            loaded=False,
            path=str(path_obj),
            error=f"Validation protocol file not found: {path_obj}",
            notes=["Presets that need validation metadata are unavailable."],
        )
    try:
        payload = json.loads(path_obj.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return ValidationMetadata(
            loaded=False,
            path=str(path_obj.resolve()),
            error=f"Could not parse validation protocol: {type(exc).__name__}: {exc}",
            notes=["Malformed validation metadata ignored; using reference threshold."],
        )
    if not isinstance(payload, dict):
        return ValidationMetadata(
            loaded=False,
            path=str(path_obj.resolve()),
            error="validation_protocol.json root must be a JSON object",
        )

    source_path = str(path_obj.resolve())
    dataset_name = _infer_dataset_name(payload, path_obj)
    points = _load_points_from_table(
        payload.get("comparison_table"),
        dataset_name=dataset_name,
        source_path=source_path,
    )
    sweeps: Dict[str, List[Dict[str, Any]]] = {}
    stored_presets: Dict[str, Dict[str, Any]] = {}

    results = payload.get("results_by_run")
    if isinstance(results, dict):
        for run_name, block in results.items():
            if not isinstance(run_name, str) or not isinstance(block, dict):
                continue
            sweep = block.get("validation_sweep")
            if isinstance(sweep, list):
                rows = [row for row in sweep if isinstance(row, dict) and "threshold" in row]
                if rows:
                    sweeps[run_name] = rows
            stored = block.get("dashboard_presets")
            if isinstance(stored, dict):
                stored_presets[run_name] = stored
            selections = block.get("selections")
            if not isinstance(selections, dict):
                continue
            for criterion, sel_block in selections.items():
                if run_name in points and criterion in points[run_name]:
                    continue
                if not isinstance(criterion, str) or not isinstance(sel_block, dict):
                    continue
                selection = sel_block.get("selection")
                if not isinstance(selection, dict):
                    continue
                thr = _safe_float(selection.get("selected_threshold"))
                if thr is None or not (0.0 <= thr <= 1.0):
                    continue
                comparison = sel_block.get("comparison")
                comparison = comparison if isinstance(comparison, dict) else {}
                points.setdefault(run_name, {})[criterion] = ValidationOperatingPoint(
                    run_name=run_name,
                    criterion=criterion,
                    selected_threshold=thr,
                    dataset_name=dataset_name,
                    val_balanced_accuracy=_safe_float(
                        comparison.get("val_balanced_accuracy")
                    ),
                    test_balanced_accuracy=_safe_float(
                        comparison.get("test_balanced_accuracy")
                    ),
                    ref_0_50_test_balanced_accuracy=_safe_float(
                        comparison.get("ref_0_50_test_balanced_accuracy")
                    ),
                    source_path=source_path,
                )

    top_sweeps = payload.get("validation_sweeps")
    if isinstance(top_sweeps, dict):
        for run_name, sweep in top_sweeps.items():
            if isinstance(run_name, str) and isinstance(sweep, list) and run_name not in sweeps:
                rows = [row for row in sweep if isinstance(row, dict) and "threshold" in row]
                if rows:
                    sweeps[run_name] = rows

    semantics = payload.get("score_semantics")
    if not isinstance(semantics, str) or not semantics.strip():
        semantics = SCORE_SEMANTICS
    notes = payload.get("notes")
    note_list = [str(n) for n in notes] if isinstance(notes, list) else []
    if not points and not sweeps:
        return ValidationMetadata(
            loaded=False,
            path=source_path,
            dataset_name=dataset_name,
            error="validation_protocol.json contained no usable operating points or sweep rows",
            score_semantics=semantics,
            notes=note_list or ["No operating points found; presets unavailable."],
        )
    return ValidationMetadata(
        loaded=True,
        path=source_path,
        dataset_name=dataset_name,
        error=None,
        points=points,
        sweeps=sweeps,
        stored_presets=stored_presets,
        score_semantics=semantics,
        notes=note_list,
    )


def apply_decision_threshold(
    fake_score: Optional[float],
    threshold: float,
) -> Optional[str]:
    """Central decision rule: predict fake if fake_score >= threshold."""

    if fake_score is None:
        return None
    return predict_label(float(fake_score), float(threshold))


def build_operating_prediction(
    *,
    fake_score: Optional[float],
    real_score: Optional[float] = None,
    threshold: float,
    saved_label: Optional[str] = None,
) -> OperatingPrediction:
    """Build display prediction without mutating raw scores."""

    predicted = apply_decision_threshold(fake_score, threshold)
    word = decision_display_word(predicted)
    if fake_score is None:
        text = "Current decision: — (no raw fake score)"
    else:
        text = f"Current decision: {word}"
    saved_word = decision_display_word(saved_label)
    saved_provenance = (
        f"Original saved decision at reference threshold {REFERENCE_THRESHOLD:.2f}: "
        f"{saved_word}"
    )
    return OperatingPrediction(
        raw_fake_score=None if fake_score is None else float(fake_score),
        raw_real_score=None if real_score is None else float(real_score),
        decision_threshold=float(threshold),
        predicted_label=predicted,
        saved_label=saved_label,
        current_decision_word=word,
        display_text=text,
        saved_decision_provenance=saved_provenance,
    )


def available_run_names_from_cards(cards: Sequence[Any]) -> Tuple[List[str], List[str]]:
    """Return (available, unavailable) run names from ConfigCard-like objects."""

    available: List[str] = []
    unavailable: List[str] = []
    by_name = {getattr(c, "run_name", None): c for c in cards}
    for run_name in RUN_ORDER:
        card = by_name.get(run_name)
        if card is None:
            unavailable.append(run_name)
            continue
        error = getattr(card, "error", None)
        fake = getattr(card, "fake_score", None)
        if error or fake is None:
            unavailable.append(run_name)
        else:
            available.append(run_name)
    return available, unavailable


def _held_out_from_stored(stored: Any) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if not isinstance(stored, dict):
        return None, None, None
    held = stored.get("held_out_test")
    metrics = held.get("metrics") if isinstance(held, dict) else held
    if not isinstance(metrics, dict):
        return None, None, None
    return (
        _safe_float(metrics.get("balanced_accuracy")),
        _safe_float(metrics.get("sensitivity_fake")),
        _safe_float(metrics.get("specificity_real")),
    )


def resolve_preset_details_for_run(
    run_name: str,
    validation: Optional[ValidationMetadata],
) -> Tuple[Dict[str, PresetInfo], bool, Optional[str], bool, Optional[str], bool, Optional[str]]:
    """Return preset details computed from the validation sweep only."""

    empty = {
        name: PresetInfo(
            name=name,
            threshold=None,
            user_copy=PRESET_USER_COPY[name],
            provenance="",
        )
        for name in PRESET_NAMES
    }
    if validation is None or not validation.available:
        reason = (
            validation.error
            if validation and validation.error
            else "Validation metadata not loaded; presets unavailable until a "
            "validation_protocol.json path is configured."
        )
        return empty, False, reason, False, None, False, None
    rows = validation.sweeps.get(run_name) or []
    if not rows:
        return empty, False, NO_SWEEP_REASON, False, None, False, None

    pack = select_operating_presets(rows, target_rate=PRESET_TARGET_RATE)
    stored = validation.stored_presets.get(run_name) or {}
    details: Dict[str, PresetInfo] = {}
    for name in PRESET_NAMES:
        raw = pack["presets"].get(name) or {}
        held_b, held_s, held_p = _held_out_from_stored(stored.get(name))
        details[name] = PresetInfo(
            name=name,
            threshold=_safe_float(raw.get("threshold")),
            user_copy=PRESET_USER_COPY[name],
            provenance=str(raw.get("provenance") or ""),
            validation_sensitivity_fake=_safe_float(raw.get("validation_sensitivity_fake")),
            validation_specificity_real=_safe_float(raw.get("validation_specificity_real")),
            validation_balanced_accuracy=_safe_float(raw.get("validation_balanced_accuracy")),
            target=str(raw.get("target") or ""),
            target_met=bool(raw.get("target_met")) if raw.get("target_met") is not None else None,
            used_fallback=bool(raw.get("used_fallback")),
            held_out_balanced_accuracy=held_b,
            held_out_sensitivity_fake=held_s,
            held_out_specificity_real=held_p,
        )
    return (
        details,
        True,
        None,
        bool(pack.get("collapsed")),
        pack.get("collapse_note") or (COLLAPSE_NOTE if pack.get("collapsed") else None),
        bool(pack.get("unexpected_order")),
        pack.get("order_note"),
    )


def resolve_presets_for_run(
    run_name: str,
    validation: Optional[ValidationMetadata],
) -> Tuple[Dict[str, Optional[float]], bool, Optional[str]]:
    """Map Sensitive/Balanced/Conservative to validation-sweep operating points."""

    details, ok, reason, *_rest = resolve_preset_details_for_run(run_name, validation)
    return {name: details[name].threshold for name in PRESET_NAMES}, ok, reason


def preferred_validation_point(
    run_name: str,
    validation: Optional[ValidationMetadata],
    *,
    criterion: str = "max_balanced_accuracy",
) -> Optional[ValidationOperatingPoint]:
    if validation is None or not validation.available:
        return None
    by_crit = validation.points.get(run_name) or {}
    if criterion in by_crit:
        return by_crit[criterion]
    if "closest_equal_sensitivity_specificity" in by_crit:
        return by_crit["closest_equal_sensitivity_specificity"]
    if by_crit:
        return next(iter(by_crit.values()))
    return None


def matching_preset_name(
    threshold: float,
    details: Mapping[str, PresetInfo],
) -> Optional[str]:
    matches = [
        name
        for name, info in details.items()
        if info.threshold is not None and abs(float(info.threshold) - float(threshold)) <= 1e-9
    ]
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]
    # Prefer Balanced when several presets collapse onto the same threshold.
    if PRESET_BALANCED in matches:
        return PRESET_BALANCED
    return matches[0]


def build_provenance_lines(
    *,
    threshold: float,
    source: str,
    reference_threshold: float = REFERENCE_THRESHOLD,
    validation_point: Optional[ValidationOperatingPoint] = None,
    validation: Optional[ValidationMetadata] = None,
    matched_preset: Optional[str] = None,
    preset_details: Optional[Mapping[str, PresetInfo]] = None,
) -> List[str]:
    lines = [
        f"Reference threshold: {reference_threshold:.2f}",
    ]
    details = preset_details or {}
    balanced = details.get(PRESET_BALANCED)
    if balanced is not None and balanced.threshold is not None:
        lines.append(f"Validation-selected (Balanced) threshold: {balanced.threshold:.2f}")
    elif validation_point is not None:
        lines.append(
            f"Validation-selected threshold: {validation_point.selected_threshold:.2f}"
        )
    if source == THRESHOLD_SOURCE_CUSTOM:
        extra = f" (matches {matched_preset} preset)" if matched_preset else ""
        lines.append(f"Custom threshold: {threshold:.2f}{extra}")
    elif source == THRESHOLD_SOURCE_VALIDATION:
        label = matched_preset or "validation-selected"
        lines.append(f"Active threshold: {threshold:.2f} ({label})")
    elif source == THRESHOLD_SOURCE_REFERENCE:
        lines.append(f"Active threshold: {threshold:.2f} (reference)")
    else:
        lines.append(f"Active threshold: {threshold:.2f}")

    if validation is not None and validation.available:
        if validation.dataset_name:
            lines.append(f"Validation dataset: {validation.dataset_name}")
        active = details.get(matched_preset) if matched_preset else None
        if active is not None and source != THRESHOLD_SOURCE_CUSTOM:
            lines.append(f"Selection criterion / target: {active.target or active.provenance}")
        lines.append(VALIDATION_SELECT_WORDING)
        if source == THRESHOLD_SOURCE_CUSTOM:
            lines.append(
                "The current slider value is a custom operating setting; it was not "
                "selected by the validation protocol unless it happens to match a preset."
            )
        if active is not None and active.held_out_balanced_accuracy is not None:
            lines.append(HELD_OUT_WORDING)
            lines.append(
                "Held-out balanced accuracy at this preset: "
                f"{active.held_out_balanced_accuracy:.3f}"
            )
        if validation.path:
            lines.append(f"Validation metadata: {validation.path}")
    lines.append(PRESET_TARGET_NOTE)
    return lines


def build_operating_settings_state(
    *,
    cards: Sequence[Any],
    active_run_name: Optional[str] = None,
    decision_threshold: Optional[float] = None,
    threshold_source: str = THRESHOLD_SOURCE_REFERENCE,
    validation: Optional[ValidationMetadata] = None,
) -> OperatingSettingsState:
    """Assemble operating-settings state for the dashboard UI."""

    available, unavailable = available_run_names_from_cards(cards)
    run_name = active_run_name or PRIMARY_ASSESSMENT_RUN
    if run_name not in available:
        run_name = available[0] if available else PRIMARY_ASSESSMENT_RUN

    by_name = {getattr(c, "run_name", None): c for c in cards}
    card = by_name.get(run_name)
    fake_score = getattr(card, "fake_score", None) if card is not None else None
    real_score = getattr(card, "real_score", None) if card is not None else None
    saved_label = getattr(card, "label", None) if card is not None else None
    if fake_score is not None:
        fake_score = float(fake_score)
    if real_score is not None:
        real_score = float(real_score)

    details, presets_ok, presets_reason, collapsed, collapse_note, unexpected, order_note = (
        resolve_preset_details_for_run(run_name, validation)
    )
    val_point = preferred_validation_point(run_name, validation)
    balanced_thr = details[PRESET_BALANCED].threshold if presets_ok else None

    if decision_threshold is None:
        if threshold_source == THRESHOLD_SOURCE_VALIDATION and balanced_thr is not None:
            decision_threshold = balanced_thr
        elif threshold_source == THRESHOLD_SOURCE_VALIDATION and val_point is not None:
            decision_threshold = val_point.selected_threshold
        else:
            decision_threshold = REFERENCE_THRESHOLD
            threshold_source = THRESHOLD_SOURCE_REFERENCE

    source = threshold_source
    if source not in THRESHOLD_SOURCE_LABELS:
        source = THRESHOLD_SOURCE_CUSTOM

    matched = matching_preset_name(float(decision_threshold), details)
    if source == THRESHOLD_SOURCE_CUSTOM and matched is None:
        pass
    elif source == THRESHOLD_SOURCE_VALIDATION and matched is None and balanced_thr is not None:
        # Validation source without an exact preset match: still a user/analysis value.
        if abs(float(decision_threshold) - float(balanced_thr)) > 1e-9:
            matched = None

    presets = {name: details[name].threshold for name in PRESET_NAMES}
    prediction = build_operating_prediction(
        fake_score=fake_score,
        real_score=real_score,
        threshold=float(decision_threshold),
        saved_label=saved_label,
    )
    provenance = build_provenance_lines(
        threshold=float(decision_threshold),
        source=source,
        validation_point=val_point,
        validation=validation,
        matched_preset=matched if source != THRESHOLD_SOURCE_CUSTOM else matched,
        preset_details=details,
    )
    base_settings = default_decision_operating_settings(
        decision_threshold=float(decision_threshold),
        expert_configuration=run_name,
    ).to_dict()
    base_settings.update(
        {
            "threshold_source": source,
            "threshold_source_label": THRESHOLD_SOURCE_LABELS[source],
            "active_config_label": config_display_label(run_name),
            "raw_score_note": RAW_SCORE_NOTE,
            "presets_available": presets_ok,
            "matched_preset": matched,
            "saved_decision_provenance": prediction.saved_decision_provenance,
        }
    )
    return OperatingSettingsState(
        active_run_name=run_name,
        active_config_label=config_display_label(run_name),
        decision_threshold=float(decision_threshold),
        threshold_source=source,
        threshold_source_label=THRESHOLD_SOURCE_LABELS[source],
        prediction=prediction,
        available_run_names=available,
        unavailable_run_names=unavailable,
        presets=presets,
        preset_details=details,
        presets_available=presets_ok,
        presets_unavailable_reason=presets_reason,
        presets_collapsed=collapsed,
        collapse_note=collapse_note,
        unexpected_order=unexpected,
        order_note=order_note,
        matched_preset=matched,
        validation=validation,
        validation_point=val_point,
        provenance_lines=provenance,
        decision_operating_settings=base_settings,
    )
