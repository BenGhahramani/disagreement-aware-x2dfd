"""Dashboard correctness / robustness checks (no inference, no GPU).

Pure helpers used by unit tests and ``tools.validate_dashboard_correctness``.
Verifies that the Streamlit-free view-model faithfully represents saved
expert-matrix JSON: score fidelity, semantics, provenance, agreement/status
thresholds, saved/live consistency, and safe handling of missing fields.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dashboard.view_model import (
    CALIBRATION_NOTE,
    DETECTOR_SCORE_LABELS,
    EXPERT_HI,
    EXPERT_LO,
    EVIDENCE_PROVENANCE,
    LABEL_BLENDING_DETECTOR,
    LABEL_DIFFUSION_DETECTOR,
    LABEL_MODEL_FAKE,
    LABEL_MODEL_REAL,
    PROVENANCE_BLENDING,
    PROVENANCE_DIFFUSION,
    PROVENANCE_MODEL,
    DashboardView,
    EvidenceAgreement,
    _specialist_relation,
    build_dashboard_view,
    derive_evidence_agreement,
    describe_evidence_conflicts,
    detector_score_label,
    display_rationale,
    parse_probability,
)
from eval.experiment_configs import RUN_ORDER
from eval.reproducibility import get_git_provenance, utc_timestamp
from proof_of_concept.evaluator import CONF_HIGH, evaluate
from proof_of_concept.normaliser import load_scenario
from proof_of_concept.schema import RunRecord, Status

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Representative saved outputs (gitignored under eval/outputs/ — optional on CI).
NASA_JONNY_MATRIX = PROJECT_ROOT / "eval" / "outputs" / "expert_matrix" / "demo_one_crop"
NASA_JONNY_SUMMARY = PROJECT_ROOT / "eval" / "outputs" / "expert_matrix_summary.json"
MAE_MATRIX = (
    PROJECT_ROOT
    / "eval"
    / "outputs"
    / "labelled_evaluation"
    / "pilot"
    / "nasa_mae_jemison"
    / "matrix"
)
MAE_SUMMARY = (
    PROJECT_ROOT
    / "eval"
    / "outputs"
    / "labelled_evaluation"
    / "pilot"
    / "nasa_mae_jemison"
    / "summary.json"
)
FFPP_MATRIX = (
    PROJECT_ROOT
    / "eval"
    / "outputs"
    / "live_analysis"
    / "20260808T085812Z_ffpp_ex_deepfakes"
    / "matrix"
)
FFPP_SUMMARY = (
    PROJECT_ROOT
    / "eval"
    / "outputs"
    / "live_analysis"
    / "20260808T085812Z_ffpp_ex_deepfakes"
    / "summary.json"
)

FORBIDDEN_SCORE_WORDS = (
    "calibrated probability",
    "calibrated probabilities",
    "calibrated confidence",
    "probability/confidence",
)


@dataclass
class CheckResult:
    category: str
    name: str
    passed: bool
    detail: str = ""
    skipped: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationReport:
    checks: List[CheckResult] = field(default_factory=list)
    generated_at: str = ""
    git_commit: Optional[str] = None
    git_dirty: Optional[bool] = None

    def add(self, result: CheckResult) -> None:
        self.checks.append(result)

    @property
    def total(self) -> int:
        return len(self.checks)

    @property
    def passed(self) -> int:
        return sum(1 for c in self.checks if c.passed and not c.skipped)

    @property
    def failed(self) -> int:
        return sum(1 for c in self.checks if not c.passed and not c.skipped)

    @property
    def skipped(self) -> int:
        return sum(1 for c in self.checks if c.skipped)

    def as_dict(self) -> Dict[str, Any]:
        categories: Dict[str, Dict[str, int]] = {}
        for check in self.checks:
            bucket = categories.setdefault(
                check.category, {"total": 0, "passed": 0, "failed": 0, "skipped": 0}
            )
            bucket["total"] += 1
            if check.skipped:
                bucket["skipped"] += 1
            elif check.passed:
                bucket["passed"] += 1
            else:
                bucket["failed"] += 1
        return {
            "generated_at": self.generated_at,
            "git_commit": self.git_commit,
            "git_dirty": self.git_dirty,
            "total_checks": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "ok": self.failed == 0,
            "categories": categories,
            "threshold_constants": {
                "evidence_expert_lo": EXPERT_LO,
                "evidence_expert_hi": EXPERT_HI,
                "prototype_stable_uncertain": CONF_HIGH,
            },
            "boundary_cases_tested": [
                0.2999,
                0.30,
                0.50,
                0.70,
                0.7001,
                CONF_HIGH - 1e-9,
                CONF_HIGH,
                CONF_HIGH + 1e-9,
            ],
            "representative_cases": {
                "nasa_jonny_kim": str(NASA_JONNY_MATRIX),
                "nasa_mae_jemison": str(MAE_MATRIX),
                "ffpp_ex_deepfakes": str(FFPP_MATRIX),
            },
            "checks": [c.as_dict() for c in self.checks],
        }


def _turn_value(payload: Any, from_key: str) -> Optional[str]:
    if not isinstance(payload, list) or not payload:
        return None
    item = payload[0]
    if not isinstance(item, dict):
        return None
    for turn in item.get("conversations") or []:
        if isinstance(turn, dict) and turn.get("from") == from_key:
            value = turn.get("value")
            return value if isinstance(value, str) else None
    return None


def read_demo_json_scores(path: Path) -> Dict[str, Any]:
    """Extract model + specialist scores from a saved runner demo_*.json."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    human = _turn_value(payload, "human") or ""
    gpt = _turn_value(payload, "gpt") or ""
    real_raw = _turn_value(payload, "real score")
    fake_raw = _turn_value(payload, "fake score")
    specialists: Dict[str, Optional[float]] = {}
    for match in re.finditer(
        r"the\s+(?P<alias>[A-Za-z0-9_]+)\s+score\s+is\s+(?P<score>[0-9]*\.?[0-9]+|N/A)",
        human,
        flags=re.IGNORECASE,
    ):
        alias = match.group("alias").lower()
        raw = match.group("score")
        specialists[alias] = None if raw.upper() == "N/A" else float(raw)
    pred_match = re.search(r"\b(real|fake)\b", gpt, flags=re.IGNORECASE)
    return {
        "label": pred_match.group(1).lower() if pred_match else None,
        "real_score": parse_probability(real_raw),
        "fake_score": parse_probability(fake_raw),
        "specialists": specialists,
        "human": human,
        "gpt": gpt,
    }


def dashboard_facing_snapshot(view: DashboardView) -> Dict[str, Any]:
    """Canonical fields compared across saved vs live view-model paths."""

    cards = {}
    for card in view.cards:
        cards[card.run_name] = {
            "label": card.label,
            "real_score": card.real_score,
            "fake_score": card.fake_score,
            "expert_scores": dict(card.expert_scores),
            "error": card.error,
        }
    return {
        "status": view.status.value,
        "evidence_agreement": view.evidence_agreement.value,
        "model_assessment_label": view.model_assessment_label,
        "model_assessment_score": view.model_assessment_score,
        "evidence_provenance": list(view.evidence_provenance),
        "cards": cards,
    }


def _approx_equal(a: Optional[float], b: Optional[float], *, tol: float = 1e-9) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return abs(float(a) - float(b)) <= tol


def check_score_fidelity(matrix_dir: Path, summary_path: Optional[Path] = None) -> List[CheckResult]:
    results: List[CheckResult] = []
    if not matrix_dir.is_dir():
        return [
            CheckResult(
                "score_fidelity",
                f"matrix_present:{matrix_dir.name}",
                False,
                f"missing {matrix_dir}",
                skipped=True,
            )
        ]
    view = build_dashboard_view(matrix_dir, summary_path=summary_path)
    by_name = {c.run_name: c for c in view.cards}
    for run_name in RUN_ORDER:
        demo = matrix_dir / f"demo_{run_name}.json"
        card = by_name.get(run_name)
        if not demo.is_file():
            results.append(
                CheckResult(
                    "score_fidelity",
                    f"{run_name}:demo_file",
                    card is not None and card.error is not None,
                    "demo missing; card should surface error not fabricated scores",
                )
            )
            continue
        raw = read_demo_json_scores(demo)
        assert card is not None
        ok_label = card.label == raw["label"]
        ok_real = _approx_equal(card.real_score, raw["real_score"])
        ok_fake = _approx_equal(card.fake_score, raw["fake_score"])
        results.append(
            CheckResult(
                "score_fidelity",
                f"{run_name}:model_scores",
                ok_label and ok_real and ok_fake,
                (
                    f"json label={raw['label']} real={raw['real_score']} fake={raw['fake_score']} "
                    f"vs view label={card.label} real={card.real_score} fake={card.fake_score}"
                ),
            )
        )
        for specialist in ("blending", "diffusion"):
            expected = raw["specialists"].get(specialist)
            actual = card.expert_scores.get(specialist)
            if expected is None and specialist not in raw["specialists"]:
                # Absent from prompt: must not invent a numeric zero.
                ok = specialist not in card.expert_scores or actual is None
                results.append(
                    CheckResult(
                        "score_fidelity",
                        f"{run_name}:{specialist}_absent",
                        ok,
                        f"view expert_scores={card.expert_scores}",
                    )
                )
            else:
                ok = _approx_equal(actual, expected)
                # Explicit N/A / null must stay None, never 0.0.
                if expected is None:
                    ok = actual is None
                results.append(
                    CheckResult(
                        "score_fidelity",
                        f"{run_name}:{specialist}_score",
                        ok,
                        f"json={expected} view={actual}",
                    )
                )
    return results


def check_score_semantics() -> List[CheckResult]:
    results: List[CheckResult] = []
    labels = [
        LABEL_MODEL_REAL,
        LABEL_MODEL_FAKE,
        LABEL_BLENDING_DETECTOR,
        LABEL_DIFFUSION_DETECTOR,
        *DETECTOR_SCORE_LABELS.values(),
        # CALIBRATION_NOTE validated separately (mentions calibrated as disclaimer)
        display_rationale("combined-experts confidence 0.69 is below 0.70"),
        display_rationale("combined-experts probability 0.80"),
    ]
    for text in labels:
        lowered = text.lower()
        bad = [w for w in FORBIDDEN_SCORE_WORDS if w in lowered]
        # Labels must not claim calibrated probability/confidence.
        has_calib_prob = "calibrated" in lowered and (
            "probabilit" in lowered or "confidence" in lowered
        )
        results.append(
            CheckResult(
                "score_semantics",
                f"label_ok:{text[:48]}",
                not bad and not has_calib_prob,
                text,
            )
        )
    results.append(
        CheckResult(
            "score_semantics",
            "calibration_note_present",
            "not currently calibrated" in CALIBRATION_NOTE.lower(),
            CALIBRATION_NOTE,
        )
    )
    results.append(
        CheckResult(
            "score_semantics",
            "model_labels_are_scores",
            "score" in LABEL_MODEL_REAL.lower() and "score" in LABEL_MODEL_FAKE.lower(),
            f"{LABEL_MODEL_REAL!r} / {LABEL_MODEL_FAKE!r}",
        )
    )
    results.append(
        CheckResult(
            "score_semantics",
            "specialist_labels_are_detector_scores",
            "detector score" in LABEL_BLENDING_DETECTOR.lower()
            and "detector score" in LABEL_DIFFUSION_DETECTOR.lower(),
            f"{LABEL_BLENDING_DETECTOR!r} / {LABEL_DIFFUSION_DETECTOR!r}",
        )
    )
    results.append(
        CheckResult(
            "score_semantics",
            "display_rationale_softens_confidence_word",
            "model score" in display_rationale("combined-experts confidence 0.69").lower(),
            display_rationale("combined-experts confidence 0.69"),
        )
    )
    return results


def check_evidence_provenance(view: Optional[DashboardView] = None) -> List[CheckResult]:
    results: List[CheckResult] = []
    results.append(
        CheckResult(
            "evidence_provenance",
            "blending_identified",
            "blending" in PROVENANCE_BLENDING.lower() and "detector" in PROVENANCE_BLENDING.lower(),
            PROVENANCE_BLENDING,
        )
    )
    results.append(
        CheckResult(
            "evidence_provenance",
            "diffusion_identified",
            "diffusion" in PROVENANCE_DIFFUSION.lower() and "detector" in PROVENANCE_DIFFUSION.lower(),
            PROVENANCE_DIFFUSION,
        )
    )
    results.append(
        CheckResult(
            "evidence_provenance",
            "model_identified",
            "llava" in PROVENANCE_MODEL.lower() or "x2-dfd" in PROVENANCE_MODEL.lower(),
            PROVENANCE_MODEL,
        )
    )
    results.append(
        CheckResult(
            "evidence_provenance",
            "detector_score_label_blending",
            detector_score_label("blending") == LABEL_BLENDING_DETECTOR,
            detector_score_label("blending"),
        )
    )
    results.append(
        CheckResult(
            "evidence_provenance",
            "detector_score_label_diffusion",
            detector_score_label("diffusion") == LABEL_DIFFUSION_DETECTOR,
            detector_score_label("diffusion"),
        )
    )
    if view is not None:
        results.append(
            CheckResult(
                "evidence_provenance",
                "view_preserves_provenance_tuple",
                list(view.evidence_provenance) == list(EVIDENCE_PROVENANCE),
                str(view.evidence_provenance),
            )
        )
        none_card = next((c for c in view.cards if c.run_name == "none"), None)
        if none_card is not None:
            # none config: absent specialists must not be numeric zero
            zeros = [
                name
                for name, score in none_card.expert_scores.items()
                if score == 0.0
            ]
            results.append(
                CheckResult(
                    "evidence_provenance",
                    "none_config_no_fabricated_zero_specialists",
                    not zeros,
                    f"expert_scores={none_card.expert_scores}",
                )
            )
    return results


def check_agreement_boundaries() -> List[CheckResult]:
    from dashboard.view_model import ConfigCard

    results: List[CheckResult] = []
    results.append(
        CheckResult(
            "evidence_agreement",
            "thresholds_unchanged",
            EXPERT_LO == 0.30 and EXPERT_HI == 0.70,
            f"lo={EXPERT_LO} hi={EXPERT_HI}",
        )
    )

    def card(label: str, experts: Dict[str, Optional[float]]) -> ConfigCard:
        return ConfigCard(
            run_name="blending_diffusion",
            title="Blending + Diffusion",
            label=label,
            real_score=0.2 if label == "fake" else 0.8,
            fake_score=0.8 if label == "fake" else 0.2,
            expert_scores=experts,
            explanation="",
            runtime_s=1.0,
            peak_vram_mib=None,
            output_path=None,
            error=None,
        )

    # Boundary specialist relations for fake label
    cases = [
        (0.2999, "conflict"),
        (0.30, "inconclusive"),  # not < lo
        (0.50, "inconclusive"),
        (0.70, "inconclusive"),  # not > hi
        (0.7001, "support"),
    ]
    for score, expected in cases:
        rel = _specialist_relation("fake", score, lo=EXPERT_LO, hi=EXPERT_HI)
        results.append(
            CheckResult(
                "evidence_agreement",
                f"fake_label_score_{score}",
                rel == expected,
                f"got {rel}, expected {expected}",
            )
        )

    # Real-label mirrored boundaries
    real_cases = [
        (0.2999, "support"),
        (0.30, "inconclusive"),
        (0.50, "inconclusive"),
        (0.70, "inconclusive"),
        (0.7001, "conflict"),
    ]
    for score, expected in real_cases:
        rel = _specialist_relation("real", score, lo=EXPERT_LO, hi=EXPERT_HI)
        results.append(
            CheckResult(
                "evidence_agreement",
                f"real_label_score_{score}",
                rel == expected,
                f"got {rel}, expected {expected}",
            )
        )

    # Full agreement classifications
    agree = derive_evidence_agreement(
        [card("fake", {"blending": 0.85, "diffusion": 0.90})]
    )
    conflict = derive_evidence_agreement(
        [card("fake", {"blending": 0.009, "diffusion": 0.048})]
    )
    insuff = derive_evidence_agreement(
        [card("fake", {"blending": 0.45, "diffusion": 0.55})]
    )
    missing = derive_evidence_agreement([card("fake", {})])
    null_spec = derive_evidence_agreement(
        [card("fake", {"blending": None, "diffusion": None})]
    )
    results.append(
        CheckResult(
            "evidence_agreement",
            "agreement_case",
            agree is EvidenceAgreement.AGREEMENT,
            agree.value,
        )
    )
    results.append(
        CheckResult(
            "evidence_agreement",
            "conflict_case",
            conflict is EvidenceAgreement.CONFLICT,
            conflict.value,
        )
    )
    results.append(
        CheckResult(
            "evidence_agreement",
            "insufficient_middling",
            insuff is EvidenceAgreement.INSUFFICIENT,
            insuff.value,
        )
    )
    results.append(
        CheckResult(
            "evidence_agreement",
            "insufficient_missing_specialists",
            missing is EvidenceAgreement.INSUFFICIENT,
            missing.value,
        )
    )
    results.append(
        CheckResult(
            "evidence_agreement",
            "insufficient_null_specialists",
            null_spec is EvidenceAgreement.INSUFFICIENT,
            null_spec.value,
        )
    )

    # describe_evidence_conflicts + derive_evidence_agreement alignment
    conflict_card = card("fake", {"blending": 0.01, "diffusion": 0.02})
    notes = describe_evidence_conflicts([conflict_card])
    derived = derive_evidence_agreement([conflict_card], conflicts=notes)
    results.append(
        CheckResult(
            "evidence_agreement",
            "conflicts_notes_align_with_agreement",
            bool(notes) and derived is EvidenceAgreement.CONFLICT,
            f"notes={notes!r} agreement={derived.value}",
        )
    )
    return results


def check_prototype_status_boundaries() -> List[CheckResult]:
    results: List[CheckResult] = []
    results.append(
        CheckResult(
            "prototype_status",
            "conf_high_unchanged",
            CONF_HIGH == 0.70,
            f"CONF_HIGH={CONF_HIGH}",
        )
    )

    def runs_all(
        *,
        fake: float,
        label: str = "fake",
        flip_none: bool = False,
    ) -> Dict[str, RunRecord]:
        out: Dict[str, RunRecord] = {}
        for name in RUN_ORDER:
            pred = "real" if (flip_none and name == "none") else label
            real = 1.0 - fake if pred == label else fake
            fake_s = fake if pred == label else 1.0 - fake
            # confidence = max(real, fake) in normaliser — approximate via scores
            conf = max(real, fake_s)
            out[name] = RunRecord(
                run_name=name,
                experts_used=[],
                prediction=pred,
                real_score=real,
                fake_score=fake_s,
                confidence=conf,
                explanation=pred,
                runtime_seconds=1.0,
                error=None,
            )
        return out

    # Just below / at / above Stable threshold (all agree)
    for score, expected in (
        (0.6999, Status.UNCERTAIN),
        (0.70, Status.STABLE),
        (0.7001, Status.STABLE),
    ):
        status, rationale = evaluate(runs_all(fake=score))
        results.append(
            CheckResult(
                "prototype_status",
                f"agree_fake_{score}",
                status is expected,
                f"got {status.value}; {rationale}",
            )
        )

    contested, cont_r = evaluate(runs_all(fake=0.95, flip_none=True))
    results.append(
        CheckResult(
            "prototype_status",
            "disagreement_is_contested",
            contested is Status.CONTESTED,
            cont_r,
        )
    )

    # Missing combined run must not silently become Stable
    incomplete = runs_all(fake=0.99)
    incomplete["blending_diffusion"] = RunRecord(
        run_name="blending_diffusion",
        experts_used=["blending", "diffusion"],
        prediction=None,
        real_score=None,
        fake_score=None,
        confidence=None,
        explanation="",
        runtime_seconds=None,
        error="file missing",
    )
    failed, failed_r = evaluate(incomplete)
    results.append(
        CheckResult(
            "prototype_status",
            "missing_combined_not_stable",
            failed is Status.FAILED,
            failed_r,
        )
    )

    # Determinism
    a1, r1 = evaluate(runs_all(fake=0.91))
    a2, r2 = evaluate(runs_all(fake=0.91))
    results.append(
        CheckResult(
            "prototype_status",
            "deterministic",
            a1 is a2 and r1 == r2,
            f"{a1.value}: {r1}",
        )
    )
    return results


def check_saved_vs_live_consistency(
    matrix_dir: Path,
    summary_path: Optional[Path],
) -> List[CheckResult]:
    """Saved-example path and live-analysis path both call ``build_dashboard_view``."""

    if not matrix_dir.is_dir():
        return [
            CheckResult(
                "saved_live_consistency",
                f"present:{matrix_dir.name}",
                False,
                "matrix missing",
                skipped=True,
            )
        ]
    # Saved path
    saved = build_dashboard_view(matrix_dir, summary_path=summary_path)
    # Live path (identical call site used by dashboard.live_analysis after writing matrix)
    live = build_dashboard_view(matrix_dir, summary_path=summary_path)
    snap_a = dashboard_facing_snapshot(saved)
    snap_b = dashboard_facing_snapshot(live)
    return [
        CheckResult(
            "saved_live_consistency",
            f"snapshot_identical:{matrix_dir.name}",
            snap_a == snap_b,
            "saved vs live view-model snapshots differ"
            if snap_a != snap_b
            else "identical dashboard-facing snapshot",
        )
    ]


def check_malformed_robustness(tmp_matrix: Path) -> List[CheckResult]:
    results: List[CheckResult] = []
    tmp_matrix.mkdir(parents=True, exist_ok=True)

    # Malformed JSON
    (tmp_matrix / "demo_none.json").write_text("{broken", encoding="utf-8")
    view = build_dashboard_view(tmp_matrix, summary_path=None)
    results.append(
        CheckResult(
            "malformed_robustness",
            "malformed_json_no_crash",
            view is not None and view.status is Status.FAILED,
            f"status={view.status.value} errors={view.errors}",
        )
    )
    none_card = view.cards[0]
    results.append(
        CheckResult(
            "malformed_robustness",
            "malformed_no_fabricated_scores",
            none_card.real_score is None and none_card.fake_score is None,
            f"real={none_card.real_score} fake={none_card.fake_score}",
        )
    )

    # Partial configuration (only none present, valid)
    partial = tmp_matrix.parent / "partial_matrix"
    partial.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "id": "1",
            "image": "missing.jpg",
            "conversations": [
                {"from": "human", "value": "<image>\nIs this image real or fake?"},
                {"from": "gpt", "value": "This image is fake"},
                {"from": "real score", "value": "0.20"},
                {"from": "fake score", "value": "0.80"},
            ],
        }
    ]
    (partial / "demo_none.json").write_text(json.dumps(payload), encoding="utf-8")
    (partial / "runtimes.json").write_text(json.dumps({"none": 1.0}), encoding="utf-8")
    partial_view = build_dashboard_view(partial, summary_path=None)
    results.append(
        CheckResult(
            "malformed_robustness",
            "partial_configs_failed_status",
            partial_view.status is Status.FAILED,
            partial_view.rationale,
        )
    )
    # Missing specialist fields: empty expert_scores, not zeros
    for card in partial_view.cards:
        if card.run_name == "none":
            results.append(
                CheckResult(
                    "malformed_robustness",
                    "missing_specialist_not_zero",
                    all(v is None for v in card.expert_scores.values())
                    or card.expert_scores == {},
                    str(card.expert_scores),
                )
            )
        else:
            results.append(
                CheckResult(
                    "malformed_robustness",
                    f"missing_run_{card.run_name}_has_error",
                    card.error is not None,
                    str(card.error),
                )
            )

    # Null specialist via N/A
    na_dir = tmp_matrix.parent / "na_specialist"
    na_dir.mkdir(parents=True, exist_ok=True)
    for name in RUN_ORDER:
        human = "<image>\nIs this image real or fake?"
        if name != "none":
            human += " And the blending score is N/A, and the diffusion score is N/A."
        body = [
            {
                "id": "1",
                "image": "x.jpg",
                "conversations": [
                    {"from": "human", "value": human},
                    {"from": "gpt", "value": "This image is fake"},
                    {"from": "real score", "value": "0.10"},
                    {"from": "fake score", "value": "0.90"},
                ],
            }
        ]
        (na_dir / f"demo_{name}.json").write_text(json.dumps(body), encoding="utf-8")
    (na_dir / "runtimes.json").write_text(
        json.dumps({n: 1.0 for n in RUN_ORDER}), encoding="utf-8"
    )
    na_view = build_dashboard_view(na_dir, summary_path=None)
    combined = next(c for c in na_view.cards if c.run_name == "blending_diffusion")
    results.append(
        CheckResult(
            "malformed_robustness",
            "na_specialist_is_none_not_zero",
            combined.expert_scores.get("blending") is None
            and combined.expert_scores.get("diffusion") is None,
            str(combined.expert_scores),
        )
    )
    results.append(
        CheckResult(
            "malformed_robustness",
            "na_specialist_insufficient_evidence",
            na_view.evidence_agreement is EvidenceAgreement.INSUFFICIENT,
            na_view.evidence_agreement.value,
        )
    )

    # Missing main scores
    miss_main = tmp_matrix.parent / "missing_main"
    miss_main.mkdir(parents=True, exist_ok=True)
    for name in RUN_ORDER:
        body = [
            {
                "id": "1",
                "image": "x.jpg",
                "conversations": [
                    {"from": "human", "value": "<image>\nIs this image real or fake?"},
                    {"from": "gpt", "value": "This image is fake"},
                ],
            }
        ]
        (miss_main / f"demo_{name}.json").write_text(json.dumps(body), encoding="utf-8")
    (miss_main / "runtimes.json").write_text(
        json.dumps({n: 1.0 for n in RUN_ORDER}), encoding="utf-8"
    )
    miss_view = build_dashboard_view(miss_main, summary_path=None)
    results.append(
        CheckResult(
            "malformed_robustness",
            "missing_main_scores_not_valid_stable",
            miss_view.status is Status.FAILED,
            miss_view.rationale,
        )
    )

    # Unknown run file ignored; view still builds four canonical cards
    results.append(
        CheckResult(
            "malformed_robustness",
            "canonical_four_cards",
            [c.run_name for c in miss_view.cards] == list(RUN_ORDER),
            str([c.run_name for c in miss_view.cards]),
        )
    )
    return results


def check_representative_cases() -> List[CheckResult]:
    results: List[CheckResult] = []

    def _case(
        name: str,
        matrix: Path,
        summary: Path,
        *,
        expect_agreement: EvidenceAgreement,
        expect_status: Optional[Status] = None,
    ) -> None:
        if not matrix.is_dir():
            results.append(
                CheckResult(
                    "representative_cases",
                    name,
                    False,
                    f"missing {matrix}",
                    skipped=True,
                )
            )
            return
        summary_path = summary if summary.is_file() else None
        view = build_dashboard_view(matrix, summary_path=summary_path)
        ok_agree = view.evidence_agreement is expect_agreement
        ok_status = expect_status is None or view.status is expect_status
        results.append(
            CheckResult(
                "representative_cases",
                name,
                ok_agree and ok_status and view.ok,
                (
                    f"agreement={view.evidence_agreement.value} "
                    f"status={view.status.value} "
                    f"label={view.model_assessment_label} "
                    f"score={view.model_assessment_score}"
                ),
            )
        )
        # Fidelity on this real matrix
        results.extend(check_score_fidelity(matrix, summary_path))
        results.extend(check_saved_vs_live_consistency(matrix, summary_path))
        results.extend(check_evidence_provenance(view))

    # Mae (known authentic) — combined specialists conflict with fake model verdict
    _case(
        "nasa_mae_jemison_conflict",
        MAE_MATRIX,
        MAE_SUMMARY,
        expect_agreement=EvidenceAgreement.CONFLICT,
        expect_status=Status.UNCERTAIN,
    )
    # Legacy Stage-3 NASA crop (Jonny Kim) — also conflict; kept as secondary authentic case
    _case(
        "nasa_jonny_kim_conflict",
        NASA_JONNY_MATRIX,
        NASA_JONNY_SUMMARY,
        expect_agreement=EvidenceAgreement.CONFLICT,
        expect_status=Status.UNCERTAIN,
    )
    # FF++ Deepfakes — agreement + Stable
    _case(
        "ffpp_deepfakes_agreement",
        FFPP_MATRIX,
        FFPP_SUMMARY,
        expect_agreement=EvidenceAgreement.AGREEMENT,
        expect_status=Status.STABLE,
    )
    return results


def run_all_checks(*, scratch_dir: Optional[Path] = None) -> ValidationReport:
    """Execute the full dashboard correctness suite and return a report."""

    git = get_git_provenance()
    report = ValidationReport(
        generated_at=utc_timestamp(),
        git_commit=git.get("git_commit"),
        git_dirty=git.get("git_dirty"),
    )
    for check in check_score_semantics():
        report.add(check)
    for check in check_agreement_boundaries():
        report.add(check)
    for check in check_prototype_status_boundaries():
        report.add(check)
    for check in check_evidence_provenance():
        report.add(check)
    for check in check_representative_cases():
        report.add(check)

    scratch = scratch_dir or (PROJECT_ROOT / "eval" / "outputs" / "_dashboard_correctness_scratch")
    # Prefer an ephemeral directory under pytest tmp when provided.
    try:
        for check in check_malformed_robustness(scratch / "malformed"):
            report.add(check)
    finally:
        # Leave scratch in place when under eval/outputs (gitignored); callers
        # using tmp_path should delete themselves.
        pass
    return report
