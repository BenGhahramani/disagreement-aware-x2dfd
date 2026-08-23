"""Streamlit-free view-model for the Stage 3 supervisor dashboard.

Loads saved expert-matrix outputs only. Official status always comes from
``proof_of_concept.evaluator.evaluate`` — this module never invents a new
classification. Evidence-conflict notes are factual observations about
specialist detector scores versus the X2-DFD / LLaVA label.

User-facing score labels use score wording (not calibrated probability).
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# Allow ``streamlit run dashboard/app.py`` (script dir on sys.path) to import
# the repository packages when launched from the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from proof_of_concept.evaluator import evaluate
from proof_of_concept.normaliser import _extract_experts_from_human, load_scenario, primary_image
from proof_of_concept.schema import RunRecord, Status

PROJECT_ROOT = _REPO_ROOT
DEFAULT_MATRIX_DIR = PROJECT_ROOT / "eval" / "outputs" / "expert_matrix" / "demo_one_crop"
DEFAULT_SUMMARY = PROJECT_ROOT / "eval" / "outputs" / "expert_matrix_summary.json"
DEFAULT_IMAGE = PROJECT_ROOT / "datasets" / "raw" / "images" / "poc" / "real_face_01_crop.jpg"

RUN_ORDER: tuple[str, ...] = ("none", "blending", "diffusion", "blending_diffusion")

RUN_TITLES: Dict[str, str] = {
    "none": "No expert",
    "blending": "Blending",
    "diffusion": "Diffusion",
    "blending_diffusion": "Blending + Diffusion",
}

# Same lo/hi band as infer_config.yaml. Prototype evidence-interpretation
# thresholds for specialist-versus-model support / conflict / inconclusive —
# never used to change official Stable / Uncertain / Contested Status.
EXPERT_LO: float = 0.30
EXPERT_HI: float = 0.70

DISCLAIMER = (
    "These outputs are decision-support evidence and are not proof that media "
    "is authentic or manipulated."
)

CALIBRATION_NOTE = (
    "Scores are model outputs and are not currently calibrated probabilities."
)

THRESHOLD_NOTE = (
    "The 0.70 Stable / Uncertain cutoff is a prototype threshold used by this "
    "dashboard, not a scientifically established confidence boundary."
)

EVIDENCE_BAND_NOTE = (
    "The 0.30 / 0.70 specialist-score bands used for evidence agreement are "
    "prototype evidence-interpretation thresholds, not scientifically "
    "validated cut-offs."
)

NF4_NOTE = (
    "Inference used 4-bit NF4 quantisation so LLaVA-1.5-7B fits an RTX 3080 "
    "(10 GiB). Merging the LoRA adapter into 4-bit weights can introduce small "
    "rounding differences versus a full-precision run. Model scores and "
    "specialist detector scores are not interchangeable."
)

# User-facing score labels (presentation only — numeric values unchanged).
LABEL_MODEL_REAL = "Model real score"
LABEL_MODEL_FAKE = "Model fake score"
LABEL_BLENDING_DETECTOR = "Blending detector score"
LABEL_DIFFUSION_DETECTOR = "Diffusion detector score"

DETECTOR_SCORE_LABELS: Dict[str, str] = {
    "blending": LABEL_BLENDING_DETECTOR,
    "diffusion": LABEL_DIFFUSION_DETECTOR,
}

PROVENANCE_MODEL = "X2-DFD / LLaVA (main model score)"
PROVENANCE_BLENDING = "Blending specialist detector"
PROVENANCE_DIFFUSION = "Diffusion specialist detector"

EVIDENCE_PROVENANCE: tuple[str, ...] = (
    PROVENANCE_MODEL,
    PROVENANCE_BLENDING,
    PROVENANCE_DIFFUSION,
)


def detector_score_label(name: str) -> str:
    """User-facing label for a specialist detector score key."""

    return DETECTOR_SCORE_LABELS.get(name, f"{name.replace('_', ' ').title()} detector score")


def display_rationale(rationale: str) -> str:
    """Rewrite evaluator rationale wording for the dashboard without changing rules.

    Keeps numeric thresholds and status logic intact; only softens
    'confidence' / 'probability' phrasing for presentation.
    """

    text = rationale or ""
    text = text.replace("combined-experts confidence", "combined-experts model score")
    text = text.replace("combined-experts probability", "combined-experts model score")
    return text


class EvidenceAgreement(str, Enum):
    """Descriptive second axis — independent of Stable / Uncertain / Contested."""

    AGREEMENT = "agreement"
    CONFLICT = "conflict"
    INSUFFICIENT = "insufficient evidence"

    @property
    def display_label(self) -> str:
        return {
            EvidenceAgreement.AGREEMENT: "Agreement",
            EvidenceAgreement.CONFLICT: "Conflict",
            EvidenceAgreement.INSUFFICIENT: "Insufficient evidence",
        }[self]


PRIMARY_ASSESSMENT_RUN = "blending_diffusion"


@dataclass(frozen=True)
class ConfigCard:
    """One of the four expert-configuration columns."""

    run_name: str
    title: str
    label: Optional[str]
    real_score: Optional[float]
    fake_score: Optional[float]
    expert_scores: Dict[str, Optional[float]]
    explanation: str
    runtime_s: Optional[float]
    peak_vram_mib: Optional[int]
    output_path: Optional[str]
    error: Optional[str]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "run_name": self.run_name,
            "title": self.title,
            "label": self.label,
            "real_score": self.real_score,
            "fake_score": self.fake_score,
            "expert_scores": dict(self.expert_scores),
            "explanation": self.explanation,
            "runtime_s": self.runtime_s,
            "peak_vram_mib": self.peak_vram_mib,
            "output_path": self.output_path,
            "error": self.error,
        }


@dataclass
class DashboardView:
    """Everything the Streamlit page needs for one saved matrix directory."""

    image_path: Optional[Path]
    status: Status
    rationale: str
    cards: List[ConfigCard]
    evidence_conflicts: List[str]
    quantisation: str
    matrix_dir: Path
    summary_path: Optional[Path]
    model_assessment_label: Optional[str] = None
    model_assessment_score: Optional[float] = None
    model_assessment_text: str = "Model assessment unavailable"
    evidence_agreement: EvidenceAgreement = EvidenceAgreement.INSUFFICIENT
    disclaimer: str = DISCLAIMER
    nf4_note: str = NF4_NOTE
    calibration_note: str = CALIBRATION_NOTE
    threshold_note: str = THRESHOLD_NOTE
    evidence_band_note: str = EVIDENCE_BAND_NOTE
    evidence_provenance: List[str] = field(default_factory=lambda: list(EVIDENCE_PROVENANCE))
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def as_dict(self) -> Dict[str, Any]:
        return {
            "image_path": str(self.image_path) if self.image_path else None,
            "status": self.status.value,
            "rationale": self.rationale,
            "cards": [c.as_dict() for c in self.cards],
            "evidence_conflicts": list(self.evidence_conflicts),
            "quantisation": self.quantisation,
            "matrix_dir": str(self.matrix_dir),
            "summary_path": str(self.summary_path) if self.summary_path else None,
            "model_assessment_label": self.model_assessment_label,
            "model_assessment_score": self.model_assessment_score,
            "model_assessment_text": self.model_assessment_text,
            "evidence_agreement": self.evidence_agreement.value,
            "disclaimer": self.disclaimer,
            "nf4_note": self.nf4_note,
            "calibration_note": self.calibration_note,
            "threshold_note": self.threshold_note,
            "evidence_band_note": self.evidence_band_note,
            "evidence_provenance": list(self.evidence_provenance),
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


def parse_probability(value: Any) -> Optional[float]:
    """Parse a score that should live in ``[0, 1]``. Invalid values become None."""

    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text or text.upper() == "N/A":
            return None
        value = text
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN
        return None
    if number < 0.0 or number > 1.0:
        return None
    return number


def bar_fraction(value: Optional[float]) -> Optional[float]:
    """Clamp a parsed probability for a progress bar; None if unavailable."""

    parsed = parse_probability(value)
    return parsed


def extract_expert_scores(human_prompt: str) -> Dict[str, Optional[float]]:
    """Pull specialist scores from a runner human-turn prompt."""

    raw = _extract_experts_from_human(human_prompt or "")
    return {name: parse_probability(score) if score is not None else None for name, score in raw.items()}


def load_summary(path: Path | str | None) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def expert_scores_from_run_file(path: Path) -> Dict[str, Optional[float]]:
    """Read specialist scores from the human prompt of a saved runner JSON."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, list) or not payload or not isinstance(payload[0], dict):
        return {}
    for turn in payload[0].get("conversations") or []:
        if isinstance(turn, dict) and turn.get("from") == "human":
            value = turn.get("value")
            if isinstance(value, str):
                return extract_expert_scores(value)
    return {}


def _summary_run_index(summary: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    if not summary:
        return {}
    runs = summary.get("runs")
    if not isinstance(runs, list):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for entry in runs:
        if isinstance(entry, dict) and isinstance(entry.get("run_name"), str):
            out[entry["run_name"]] = entry
    return out


def _format_score_list(names: List[str], scores: Dict[str, Optional[float]]) -> str:
    parts = []
    for name in names:
        value = scores.get(name)
        if value is None:
            parts.append(f"{name} unavailable")
        else:
            parts.append(f"{name} {value:.3f}")
    return ", ".join(parts)


def describe_evidence_conflicts(
    cards: List[ConfigCard],
    *,
    lo: float = EXPERT_LO,
    hi: float = EXPERT_HI,
) -> List[str]:
    """Factual notes when specialist detector scores conflict with the model label.

    Does **not** change Status. Uses the pipeline's lo/hi bands only to phrase
    "low" / "high". If nothing conflicts, returns an empty list.
    """

    preferred = next((c for c in cards if c.run_name == PRIMARY_ASSESSMENT_RUN), None)
    candidates = [preferred] if preferred is not None else []
    candidates.extend(c for c in cards if c is not preferred)

    for card in candidates:
        if card is None or card.label is None or not card.expert_scores:
            continue
        low = sorted(
            name for name, score in card.expert_scores.items()
            if score is not None and score < lo
        )
        high = sorted(
            name for name, score in card.expert_scores.items()
            if score is not None and score > hi
        )
        if card.label == "fake" and low:
            detail = _format_score_list(low, card.expert_scores)
            if len(low) == 2:
                who = "Both specialist detectors"
            elif len(low) == 1:
                who = f"The {low[0]} detector"
            else:
                who = "Specialist detectors (" + ", ".join(low) + ")"
            verb = "assigns" if len(low) == 1 else "assign"
            notes = (
                f"{who} {verb} very low detector scores ({detail}) "
                f"while the X2-DFD / LLaVA verdict is fake."
            )
            return [notes]
        if card.label == "real" and high:
            detail = _format_score_list(high, card.expert_scores)
            if len(high) == 2:
                who = "Both specialist detectors"
            elif len(high) == 1:
                who = f"The {high[0]} detector"
            else:
                who = "Specialist detectors (" + ", ".join(high) + ")"
            verb = "assigns" if len(high) == 1 else "assign"
            notes = (
                f"{who} {verb} high detector scores ({detail}) "
                f"while the X2-DFD / LLaVA verdict is real."
            )
            return [notes]
    return []


def primary_assessment_card(cards: Sequence[ConfigCard]) -> Optional[ConfigCard]:
    """Prefer the combined-experts card; otherwise the first card with a label."""

    preferred = next((c for c in cards if c.run_name == PRIMARY_ASSESSMENT_RUN), None)
    if preferred is not None:
        return preferred
    return next((c for c in cards if c.label is not None), None)


def _has_specialist_scores(card: Optional[ConfigCard]) -> bool:
    if card is None or not card.expert_scores:
        return False
    return any(score is not None for score in card.expert_scores.values())


def build_model_assessment(
    cards: Sequence[ConfigCard],
) -> tuple[Optional[str], Optional[float], str]:
    """Return ``(label, score, display_text)`` from X2-DFD / LLaVA outputs only."""

    card = primary_assessment_card(cards)
    if card is None or card.label is None:
        return None, None, "Model assessment unavailable"
    label = card.label
    if label == "fake":
        score = card.fake_score
    elif label == "real":
        score = card.real_score
    else:
        score = None
    title = label.capitalize()
    if score is None:
        return label, None, title
    return label, score, f"{title} — model score {score:.3f}"


def _specialist_relation(
    label: str,
    score: float,
    *,
    lo: float = EXPERT_LO,
    hi: float = EXPERT_HI,
) -> str:
    """Map one specialist score to support / conflict / inconclusive vs model label.

    Uses the existing prototype 0.30 / 0.70 bands only (no new cut-offs).
    """

    if label == "fake":
        if score > hi:
            return "support"
        if score < lo:
            return "conflict"
        return "inconclusive"
    if label == "real":
        if score < lo:
            return "support"
        if score > hi:
            return "conflict"
        return "inconclusive"
    return "inconclusive"


def _usable_specialist_scores(
    cards: Sequence[ConfigCard],
) -> tuple[Optional[str], List[float]]:
    """Return ``(model_label, specialist scores)`` for agreement classification."""

    card = primary_assessment_card(cards)
    if card is None or card.label is None:
        return None, []

    if _has_specialist_scores(card):
        scores = [s for s in card.expert_scores.values() if s is not None]
        return card.label, scores

    for other in cards:
        if other is card or not _has_specialist_scores(other):
            continue
        scores = [s for s in other.expert_scores.values() if s is not None]
        if scores:
            return card.label, scores
    return card.label, []


def derive_evidence_agreement(
    cards: Sequence[ConfigCard],
    *,
    conflicts: Optional[List[str]] = None,
) -> EvidenceAgreement:
    """Classify specialist-versus-model agreement without changing evaluator Status.

    Categories (prototype 0.30 / 0.70 bands; same constants as conflict phrasing):

    - ``conflict`` — any usable specialist score conflicts with the model label
    - ``agreement`` — every usable specialist score supports the model label
    - ``insufficient evidence`` — no model label, no specialists, or any
      middling/inconclusive specialist score (including mixed support + inconclusive)

    ``conflicts`` is accepted for call-site compatibility; classification is always
    derived from specialist scores. When conflict notes are supplied and non-empty,
    the result is still ``conflict`` (aligned with :func:`describe_evidence_conflicts`).
    """

    label, scores = _usable_specialist_scores(cards)
    if label is None:
        return EvidenceAgreement.INSUFFICIENT
    if not scores:
        return EvidenceAgreement.INSUFFICIENT

    # Prefer explicit conflict notes when the caller already computed them.
    if conflicts:
        return EvidenceAgreement.CONFLICT

    relations = [_specialist_relation(label, score) for score in scores]
    if any(rel == "conflict" for rel in relations):
        return EvidenceAgreement.CONFLICT
    if relations and all(rel == "support" for rel in relations):
        return EvidenceAgreement.AGREEMENT
    return EvidenceAgreement.INSUFFICIENT


def build_config_card(
    run_name: str,
    record: RunRecord,
    *,
    expert_scores: Optional[Dict[str, Optional[float]]] = None,
    peak_vram_mib: Optional[int] = None,
    output_path: Optional[str] = None,
) -> ConfigCard:
    return ConfigCard(
        run_name=run_name,
        title=RUN_TITLES.get(run_name, run_name),
        label=record.prediction,
        real_score=parse_probability(record.real_score),
        fake_score=parse_probability(record.fake_score),
        expert_scores={
            name: parse_probability(score) if score is not None else None
            for name, score in (expert_scores or {}).items()
        },
        explanation=record.explanation,
        runtime_s=record.runtime_seconds,
        peak_vram_mib=peak_vram_mib,
        output_path=output_path,
        error=record.error,
    )


def _resolve_image(
    image_str: str,
    *,
    fallback: Path = DEFAULT_IMAGE,
) -> tuple[Optional[Path], Optional[str]]:
    if image_str and image_str != "<unknown>":
        candidate = Path(image_str)
        if candidate.is_file():
            return candidate, None
        warning = f"Image referenced by the matrix outputs was not found: {candidate}"
        if fallback.is_file():
            return fallback, warning + f". Showing fallback {fallback.name}."
        return None, warning
    if fallback.is_file():
        return fallback, None
    return None, f"Analysed image not found at {fallback}"


def build_dashboard_view(
    matrix_dir: Path | str = DEFAULT_MATRIX_DIR,
    *,
    summary_path: Path | str | None = DEFAULT_SUMMARY,
) -> DashboardView:
    """Assemble a dashboard view from a saved expert-matrix directory.

    Never raises for missing or malformed saved outputs: problems are collected
    on ``errors`` / ``warnings`` so the UI can render a clear message.
    """

    directory = Path(matrix_dir)
    errors: List[str] = []
    warnings: List[str] = []

    if not directory.is_dir():
        errors.append(f"Expert-matrix directory not found: {directory}")

    resolved_summary: Optional[Path] = None
    summary: Optional[Dict[str, Any]] = None
    if summary_path is not None:
        summary_candidate = Path(summary_path)
        if summary_candidate.is_file():
            summary = load_summary(summary_candidate)
            if summary is None:
                warnings.append(f"Summary JSON could not be parsed: {summary_candidate}")
            else:
                resolved_summary = summary_candidate.resolve()
        else:
            warnings.append(f"Summary JSON not found: {summary_candidate}")

    summary_runs = _summary_run_index(summary)

    try:
        runs = load_scenario(directory)
        status, rationale = evaluate(runs)
        image_str = primary_image(runs, directory)
    except Exception as exc:
        errors.append(f"Failed to load saved runs: {type(exc).__name__}: {exc}")
        runs = {}
        status, rationale = Status.FAILED, "No saved expert-matrix runs could be loaded."
        image_str = "<unknown>"

    image_path, image_warning = _resolve_image(image_str)
    if image_warning:
        warnings.append(image_warning)

    cards: List[ConfigCard] = []
    for run_name in RUN_ORDER:
        record = runs.get(run_name)
        if record is None:
            record = RunRecord(
                run_name=run_name,
                experts_used=[],
                prediction=None,
                real_score=None,
                fake_score=None,
                confidence=None,
                explanation="",
                runtime_seconds=None,
                error=f"file missing: demo_{run_name}.json",
            )
        demo_path = directory / f"demo_{run_name}.json"
        expert_scores: Dict[str, Optional[float]] = {}
        if demo_path.is_file():
            expert_scores = expert_scores_from_run_file(demo_path)
        summary_entry = summary_runs.get(run_name, {})
        if not expert_scores and isinstance(summary_entry.get("expert_scores"), dict):
            expert_scores = {
                str(k): parse_probability(v)
                for k, v in summary_entry["expert_scores"].items()
            }
        peak = summary_entry.get("peak_vram_mib")
        peak_i = int(peak) if isinstance(peak, (int, float)) else None
        out_path = summary_entry.get("output_path")
        if not isinstance(out_path, str) and demo_path.is_file():
            out_path = str(demo_path.resolve())
        cards.append(
            build_config_card(
                run_name,
                record,
                expert_scores=expert_scores,
                peak_vram_mib=peak_i,
                output_path=out_path if isinstance(out_path, str) else None,
            )
        )

    quantisation = "unknown"
    if summary and isinstance(summary.get("quantisation"), str):
        quantisation = summary["quantisation"]

    conflicts = describe_evidence_conflicts(cards)
    assessment_label, assessment_score, assessment_text = build_model_assessment(cards)
    agreement = derive_evidence_agreement(cards, conflicts=conflicts)

    return DashboardView(
        image_path=image_path,
        status=status,
        rationale=display_rationale(rationale),
        cards=cards,
        evidence_conflicts=conflicts,
        quantisation=quantisation,
        matrix_dir=directory.resolve(),
        summary_path=resolved_summary,
        model_assessment_label=assessment_label,
        model_assessment_score=assessment_score,
        model_assessment_text=assessment_text,
        evidence_agreement=agreement,
        errors=errors,
        warnings=warnings,
    )
