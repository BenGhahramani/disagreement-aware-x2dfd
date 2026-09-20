"""Unit tests for dashboard operating-settings helpers (no Streamlit/GPU)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pytest

from dashboard.operating_settings import (
    CONFIG_DISPLAY_LABELS,
    ENV_VALIDATION_PROTOCOL,
    PRESET_BALANCED,
    PRESET_CONSERVATIVE,
    PRESET_SENSITIVE,
    PRESET_TARGET_RATE,
    RAW_SCORE_NOTE,
    THRESHOLD_SOURCE_CUSTOM,
    THRESHOLD_SOURCE_REFERENCE,
    THRESHOLD_SOURCE_VALIDATION,
    apply_decision_threshold,
    available_run_names_from_cards,
    build_operating_prediction,
    build_operating_settings_state,
    config_display_label,
    load_validation_protocol,
    resolve_preset_details_for_run,
    resolve_presets_for_run,
)
from dashboard.view_model import ConfigCard, build_dashboard_view
from eval.threshold_sweep import REFERENCE_THRESHOLD
from proof_of_concept.schema import Status

pytestmark = pytest.mark.unit


def _card(
    run_name: str,
    *,
    label: Optional[str] = "fake",
    fake: Optional[float] = 0.87,
    real: Optional[float] = 0.13,
    error: Optional[str] = None,
    experts: Optional[dict] = None,
) -> ConfigCard:
    return ConfigCard(
        run_name=run_name,
        title=run_name,
        label=label,
        real_score=real,
        fake_score=fake,
        expert_scores=experts or {},
        explanation="",
        runtime_s=1.0,
        peak_vram_mib=1000,
        output_path=None,
        error=error,
    )


def _four_cards() -> list[ConfigCard]:
    return [
        _card("none", fake=0.40, real=0.60, label="real"),
        _card("blending", fake=0.80, real=0.20, label="fake", experts={"blending": 0.9}),
        _card("diffusion", fake=0.70, real=0.30, label="fake", experts={"diffusion": None}),
        _card(
            "blending_diffusion",
            fake=0.75,
            real=0.25,
            label="fake",
            experts={"blending": 0.85, "diffusion": 0.7},
        ),
    ]


def _sweep_rows() -> list[dict]:
    return [
        {
            "threshold": 0.50,
            "sensitivity_fake": 1.00,
            "specificity_real": 0.40,
            "balanced_accuracy": 0.70,
            "test_balanced_accuracy": 0.99,
        },
        {
            "threshold": 0.63,
            "sensitivity_fake": 0.967,
            "specificity_real": 0.517,
            "balanced_accuracy": 0.742,
            "test_balanced_accuracy": 0.11,
        },
        {
            "threshold": 0.78,
            "sensitivity_fake": 0.867,
            "specificity_real": 0.833,
            "balanced_accuracy": 0.850,
            "test_balanced_accuracy": 0.01,
        },
        {
            "threshold": 0.91,
            "sensitivity_fake": 0.60,
            "specificity_real": 0.967,
            "balanced_accuracy": 0.7835,
            "test_balanced_accuracy": 0.50,
        },
    ]


def _write_protocol(
    path: Path,
    *,
    dataset_hint: str = "celebdf_v2_fp16/aggregate.json",
    include_sweep: bool = True,
    collapse: bool = False,
) -> Path:
    rows = (
        [
            {
                "threshold": 0.5,
                "sensitivity_fake": 1.0,
                "specificity_real": 1.0,
                "balanced_accuracy": 1.0,
            }
        ]
        if collapse
        else _sweep_rows()
    )
    payload = {
        "score_semantics": (
            "Raw model fake_score from saved inference outputs. Not a calibrated "
            "probability or confidence."
        ),
        "notes": ["Thresholds selected on validation only."],
        "meta": {"input_path": dataset_hint},
        "comparison_table": [
            {
                "run_name": "blending",
                "criterion": "max_balanced_accuracy",
                "selected_threshold": 0.78 if include_sweep else 0.77,
                "val_balanced_accuracy": 0.85,
                "test_balanced_accuracy": 0.833,
                "ref_0_50_test_balanced_accuracy": 0.700,
            },
            {
                "run_name": "blending_diffusion",
                "criterion": "max_balanced_accuracy",
                "selected_threshold": 0.78,
                "test_balanced_accuracy": 0.767,
                "ref_0_50_test_balanced_accuracy": 0.633,
            },
        ],
        "results_by_run": {},
    }
    if include_sweep:
        for run_name in ("none", "blending", "diffusion", "blending_diffusion"):
            payload["results_by_run"][run_name] = {
                "status": "ok",
                "validation_sweep": rows,
                "dashboard_presets": {
                    PRESET_BALANCED: {
                        "threshold": 0.78 if not collapse else 0.5,
                        "held_out_test": {
                            "metrics": {
                                "balanced_accuracy": 0.767,
                                "sensitivity_fake": 0.80,
                                "specificity_real": 0.73,
                            }
                        },
                    }
                },
            }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_threshold_boundary_behaviour() -> None:
    assert apply_decision_threshold(0.5, 0.5) == "fake"
    assert apply_decision_threshold(0.499, 0.5) == "real"
    assert apply_decision_threshold(0.0, 0.0) == "fake"
    assert apply_decision_threshold(1.0, 1.0) == "fake"
    assert apply_decision_threshold(None, 0.5) is None


def test_custom_threshold_changes_prediction_only() -> None:
    cards = _four_cards()
    raw_before = cards[1].fake_score
    state_lo = build_operating_settings_state(
        cards=cards,
        active_run_name="blending",
        decision_threshold=0.50,
        threshold_source=THRESHOLD_SOURCE_REFERENCE,
    )
    state_hi = build_operating_settings_state(
        cards=cards,
        active_run_name="blending",
        decision_threshold=0.90,
        threshold_source=THRESHOLD_SOURCE_CUSTOM,
    )
    assert state_lo.prediction.predicted_label == "fake"
    assert state_hi.prediction.predicted_label == "real"
    assert state_lo.prediction.current_decision_word == "MANIPULATED"
    assert state_hi.prediction.current_decision_word == "REAL"
    assert state_hi.prediction.raw_fake_score == pytest.approx(raw_before)
    assert cards[1].fake_score == pytest.approx(raw_before)
    assert state_hi.threshold_source == THRESHOLD_SOURCE_CUSTOM
    assert state_hi.prediction.saved_label == "fake"
    assert "reference threshold 0.50" in state_hi.prediction.saved_decision_provenance
    assert "Current decision: REAL" in state_hi.prediction.display_text
    assert "Saved model label" not in state_hi.prediction.display_text


def test_raw_score_unchanged_in_prediction_helper() -> None:
    pred = build_operating_prediction(
        fake_score=0.87, real_score=0.13, threshold=0.77, saved_label="fake"
    )
    assert pred.raw_fake_score == pytest.approx(0.87)
    assert pred.raw_real_score == pytest.approx(0.13)
    assert pred.predicted_label == "fake"
    assert pred.current_decision_word == "MANIPULATED"
    assert "0.87" not in pred.display_text  # score is shown separately, not as a percent
    assert "confidence" not in pred.display_text.lower()
    assert "probability" not in pred.display_text.lower()


def test_missing_validation_metadata() -> None:
    meta = load_validation_protocol(None)
    assert meta.loaded is False
    assert meta.available is False
    assert meta.error is None
    presets, ok, reason = resolve_presets_for_run("blending", meta)
    assert ok is False
    assert presets[PRESET_SENSITIVE] is None
    assert reason is not None


def test_missing_validation_file(tmp_path: Path) -> None:
    meta = load_validation_protocol(tmp_path / "nope.json")
    assert meta.available is False
    assert meta.error is not None
    assert "not found" in meta.error.lower()


def test_malformed_validation_metadata(tmp_path: Path) -> None:
    bad = tmp_path / "validation_protocol.json"
    bad.write_text("{not-json", encoding="utf-8")
    meta = load_validation_protocol(bad)
    assert meta.available is False
    assert meta.error is not None
    assert "parse" in meta.error.lower() or "JSON" in meta.error


def test_protocol_without_sweep_does_not_invent_presets(tmp_path: Path) -> None:
    path = _write_protocol(tmp_path / "validation_protocol.json", include_sweep=False)
    meta = load_validation_protocol(path)
    assert meta.available is True  # comparison_table still loads
    presets, ok, reason = resolve_presets_for_run("blending", meta)
    assert ok is False
    assert presets[PRESET_SENSITIVE] is None
    assert reason is not None
    assert "sweep" in reason.lower()


def test_load_validation_protocol_and_presets_from_sweep(tmp_path: Path) -> None:
    path = _write_protocol(tmp_path / "validation_protocol.json")
    meta = load_validation_protocol(path)
    assert meta.available is True
    assert meta.has_sweeps is True
    assert meta.dataset_name == "Celeb-DF-v2"
    presets, ok, reason = resolve_presets_for_run("blending", meta)
    assert ok is True
    assert reason is None
    assert PRESET_TARGET_RATE == pytest.approx(0.95)
    assert presets[PRESET_SENSITIVE] == pytest.approx(0.63)
    assert presets[PRESET_BALANCED] == pytest.approx(0.78)
    assert presets[PRESET_CONSERVATIVE] == pytest.approx(0.91)
    details, *_ = resolve_preset_details_for_run("blending", meta)
    assert details[PRESET_SENSITIVE].validation_sensitivity_fake >= 0.95
    assert details[PRESET_CONSERVATIVE].validation_specificity_real >= 0.95
    # Held-out values may be attached after the fact; they must not change selection.
    assert details[PRESET_BALANCED].threshold == pytest.approx(0.78)


def test_no_held_out_leakage_into_preset_selection(tmp_path: Path) -> None:
    path = _write_protocol(tmp_path / "validation_protocol.json")
    meta = load_validation_protocol(path)
    details, ok, *_ = resolve_preset_details_for_run("blending_diffusion", meta)
    assert ok is True
    # Sweep rows have misleading test_balanced_accuracy; Balanced still uses val bAcc.
    assert details[PRESET_BALANCED].threshold == pytest.approx(0.78)
    assert details[PRESET_SENSITIVE].threshold == pytest.approx(0.63)


def test_env_validation_protocol_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _write_protocol(tmp_path / "validation_protocol.json")
    monkeypatch.setenv(ENV_VALIDATION_PROTOCOL, str(path))
    meta = load_validation_protocol(None)
    assert meta.available is True
    assert meta.path == str(path.resolve())


def test_config_display_mapping() -> None:
    assert config_display_label("none") == "Main model only"
    assert config_display_label("blending") == "Main + blending specialist"
    assert config_display_label("diffusion") == "Main + diffusion specialist"
    assert (
        config_display_label("blending_diffusion")
        == "Main + blending + diffusion specialists"
    )
    assert set(CONFIG_DISPLAY_LABELS) == {
        "none",
        "blending",
        "diffusion",
        "blending_diffusion",
    }


def test_config_switching_updates_score_and_presets(tmp_path: Path) -> None:
    path = _write_protocol(tmp_path / "validation_protocol.json")
    meta = load_validation_protocol(path)
    cards = _four_cards()
    none_state = build_operating_settings_state(
        cards=cards,
        active_run_name="none",
        decision_threshold=0.50,
        validation=meta,
    )
    blend_state = build_operating_settings_state(
        cards=cards,
        active_run_name="blending",
        decision_threshold=0.50,
        validation=meta,
    )
    assert none_state.prediction.raw_fake_score == pytest.approx(0.40)
    assert blend_state.prediction.raw_fake_score == pytest.approx(0.80)
    assert none_state.active_run_name == "none"
    assert blend_state.active_run_name == "blending"
    assert none_state.presets_available is True
    assert blend_state.presets[PRESET_BALANCED] == pytest.approx(0.78)


def test_unavailable_config_handling() -> None:
    cards = [
        _card("none", fake=0.4),
        _card("blending", fake=None, error="file missing"),
        _card("diffusion", fake=0.7),
        _card("blending_diffusion", fake=None, label=None),
    ]
    available, unavailable = available_run_names_from_cards(cards)
    assert available == ["none", "diffusion"]
    assert "blending" in unavailable
    assert "blending_diffusion" in unavailable
    state = build_operating_settings_state(
        cards=cards,
        active_run_name="blending",
        decision_threshold=0.5,
    )
    assert state.active_run_name in available
    assert "blending" in state.unavailable_run_names


def test_missing_specialist_score_stays_unavailable() -> None:
    cards = _four_cards()
    diffusion = next(c for c in cards if c.run_name == "diffusion")
    assert diffusion.expert_scores.get("diffusion") is None
    assert diffusion.expert_scores.get("diffusion") != 0.0


def test_preset_collapse_note(tmp_path: Path) -> None:
    path = _write_protocol(tmp_path / "validation_protocol.json", collapse=True)
    meta = load_validation_protocol(path)
    state = build_operating_settings_state(
        cards=_four_cards(),
        active_run_name="blending",
        decision_threshold=0.5,
        validation=meta,
    )
    assert state.presets_collapsed is True
    assert state.collapse_note is not None
    assert "same threshold" in state.collapse_note.lower()
    assert state.presets[PRESET_SENSITIVE] == state.presets[PRESET_BALANCED]


def test_custom_slider_provenance_does_not_claim_validation(tmp_path: Path) -> None:
    path = _write_protocol(tmp_path / "validation_protocol.json")
    meta = load_validation_protocol(path)
    state = build_operating_settings_state(
        cards=_four_cards(),
        active_run_name="blending",
        decision_threshold=0.82,
        threshold_source=THRESHOLD_SOURCE_CUSTOM,
        validation=meta,
    )
    assert state.threshold_source == THRESHOLD_SOURCE_CUSTOM
    joined = " ".join(state.provenance_lines).lower()
    assert "custom threshold: 0.82" in joined
    assert "was not selected by the validation protocol" in joined
    # Matching a preset at 0.82 should not happen with this sweep.
    assert state.matched_preset is None


def test_slider_matching_preset_is_reported(tmp_path: Path) -> None:
    path = _write_protocol(tmp_path / "validation_protocol.json")
    meta = load_validation_protocol(path)
    state = build_operating_settings_state(
        cards=_four_cards(),
        active_run_name="blending",
        decision_threshold=0.78,
        threshold_source=THRESHOLD_SOURCE_CUSTOM,
        validation=meta,
    )
    assert state.matched_preset == PRESET_BALANCED
    assert state.threshold_source == THRESHOLD_SOURCE_CUSTOM


def test_preset_provenance_and_wording(tmp_path: Path) -> None:
    path = _write_protocol(tmp_path / "validation_protocol.json")
    meta = load_validation_protocol(path)
    state = build_operating_settings_state(
        cards=_four_cards(),
        active_run_name="blending",
        decision_threshold=0.78,
        threshold_source=THRESHOLD_SOURCE_VALIDATION,
        validation=meta,
    )
    joined = " ".join(state.provenance_lines).lower()
    assert "reference threshold: 0.50" in joined
    assert "celeb-df-v2" in joined
    assert "validation partition" in joined or "validation-selected" in joined
    assert "calibrated" in RAW_SCORE_NOTE.lower()
    blob = json.dumps(state.to_dict()).lower()
    assert "87% confidence" not in blob
    assert "probability of fake" not in blob
    assert "scientifically optimal" not in blob
    assert "best threshold" not in blob
    assert PRESET_TARGET_RATE == pytest.approx(0.95)
    assert "prototype operating target" in blob


def test_technical_details_consistent_with_visible_label(tmp_path: Path) -> None:
    path = _write_protocol(tmp_path / "validation_protocol.json")
    meta = load_validation_protocol(path)
    state = build_operating_settings_state(
        cards=_four_cards(),
        active_run_name="blending_diffusion",
        decision_threshold=0.78,
        threshold_source=THRESHOLD_SOURCE_VALIDATION,
        validation=meta,
    )
    # fake 0.75 < 0.78 → real / REAL
    assert state.prediction.predicted_label == "real"
    assert state.prediction.current_decision_word == "REAL"
    assert state.prediction.display_text == "Current decision: REAL"
    assert state.decision_operating_settings["decision_threshold"] == pytest.approx(0.78)
    assert state.decision_operating_settings["expert_configuration"] == "blending_diffusion"
    assert state.active_config_label == config_display_label("blending_diffusion")
    assert "Original saved decision at reference threshold 0.50" in (
        state.prediction.saved_decision_provenance
    )


def test_evidence_status_unaffected_by_threshold(tmp_path: Path) -> None:
    directory = tmp_path / "matrix"
    directory.mkdir()
    image = tmp_path / "face.jpg"
    image.write_bytes(b"\xff\xd8\xff\xe0fakejpeg")
    names = ("none", "blending", "diffusion", "blending_diffusion")
    for name in names:
        payload = [
            {
                "id": "1",
                "image": str(image),
                "conversations": [
                    {"from": "human", "value": "<image>\nIs this image real or fake?"},
                    {"from": "gpt", "value": "This image is fake"},
                    {"from": "real score", "value": "0.20"},
                    {"from": "fake score", "value": "0.80"},
                ],
            }
        ]
        (directory / f"demo_{name}.json").write_text(json.dumps(payload), encoding="utf-8")
    view = build_dashboard_view(directory, summary_path=None)
    original_status = view.status
    assert original_status in {Status.STABLE, Status.UNCERTAIN, Status.CONTESTED, Status.FAILED}
    lo = build_operating_settings_state(cards=view.cards, decision_threshold=0.01)
    hi = build_operating_settings_state(cards=view.cards, decision_threshold=0.99)
    assert lo.prediction.predicted_label != hi.prediction.predicted_label
    assert view.status is original_status
    assert view.status == build_dashboard_view(directory, summary_path=None).status


def test_reference_threshold_constant() -> None:
    assert REFERENCE_THRESHOLD == pytest.approx(0.50)
    state = build_operating_settings_state(
        cards=_four_cards(),
        active_run_name="none",
        decision_threshold=None,
        threshold_source=THRESHOLD_SOURCE_REFERENCE,
    )
    assert state.decision_threshold == pytest.approx(0.50)
    assert state.threshold_source_label == "Reference / default"
