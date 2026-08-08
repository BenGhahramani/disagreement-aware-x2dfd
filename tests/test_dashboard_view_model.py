"""Unit tests for dashboard/view_model.py.

No Streamlit, no GPU, no network. Uses the real Stage 3 matrix directory when
present, and tmp_path fixtures otherwise.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import pytest

from dashboard.view_model import (
    DEFAULT_MATRIX_DIR,
    DEFAULT_SUMMARY,
    DISCLAIMER,
    ConfigCard,
    bar_fraction,
    build_config_card,
    build_dashboard_view,
    describe_evidence_conflicts,
    expert_scores_from_run_file,
    extract_expert_scores,
    load_summary,
    parse_probability,
)
from proof_of_concept.schema import RunRecord, Status

pytestmark = pytest.mark.unit


def _write_run(
    directory: Path,
    run_name: str,
    *,
    answer: str = "This image is fake",
    real: str = "0.30",
    fake: str = "0.70",
    prompt_tail: str = "",
    image: str = "C:/images/face.jpg",
) -> Path:
    human = "<image>\nIs this image real or fake?" + prompt_tail
    payload = [
        {
            "id": "1",
            "image": image,
            "conversations": [
                {"from": "human", "value": human},
                {"from": "gpt", "value": answer},
                {"from": "real score", "value": real},
                {"from": "fake score", "value": fake},
            ],
        }
    ]
    path = directory / f"demo_{run_name}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_matrix(
    tmp_path: Path,
    *,
    image: Optional[Path] = None,
    answers: Optional[Dict[str, str]] = None,
    scores: Optional[Dict[str, tuple[str, str]]] = None,
    prompts: Optional[Dict[str, str]] = None,
) -> Path:
    directory = tmp_path / "matrix"
    directory.mkdir()
    if image is None:
        image = tmp_path / "face.jpg"
        image.write_bytes(b"\xff\xd8\xff\xe0fakejpeg")
    names = ("none", "blending", "diffusion", "blending_diffusion")
    answers = answers or {name: "This image is fake" for name in names}
    scores = scores or {name: ("0.31", "0.69") for name in names}
    prompts = prompts or {
        "none": "",
        "blending": " And the blending score is 0.009.",
        "diffusion": " And the diffusion score is 0.048.",
        "blending_diffusion": (
            " And the blending score is 0.009, and the diffusion score is 0.048."
        ),
    }
    for name in names:
        real, fake = scores[name]
        _write_run(
            directory,
            name,
            answer=answers[name],
            real=real,
            fake=fake,
            prompt_tail=prompts[name],
            image=str(image),
        )
    (directory / "runtimes.json").write_text(
        json.dumps(
            {"none": 10.0, "blending": 11.0, "diffusion": 12.0, "blending_diffusion": 13.0}
        ),
        encoding="utf-8",
    )
    return directory


# --------------------------------------------------------------------------
# probability parsing
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        (0.3141, 0.3141),
        ("0.6859", 0.6859),
        (" 0.5 ", 0.5),
        (0, 0.0),
        (1, 1.0),
        ("N/A", None),
        ("", None),
        (None, None),
        (True, None),
        ("high", None),
        (1.5, None),
        (-0.1, None),
        (float("nan"), None),
    ],
)
def test_parse_probability(raw, expected) -> None:
    result = parse_probability(raw)
    if expected is None:
        assert result is None
    else:
        assert result == pytest.approx(expected)


def test_bar_fraction_matches_valid_probability() -> None:
    assert bar_fraction(0.64) == pytest.approx(0.64)
    assert bar_fraction("not-a-number") is None


# --------------------------------------------------------------------------
# expert score extraction
# --------------------------------------------------------------------------


def test_extract_expert_scores_from_prompt() -> None:
    prompt = (
        "<image>\nIs this image real or fake? And the blending score is 0.009, "
        "and the diffusion score is 0.048."
    )
    assert extract_expert_scores(prompt) == {"blending": 0.009, "diffusion": 0.048}


def test_extract_expert_scores_handles_na() -> None:
    assert extract_expert_scores("And the blending score is N/A.") == {"blending": None}


def test_extract_expert_scores_empty_prompt() -> None:
    assert extract_expert_scores("") == {}
    assert extract_expert_scores("<image>\nIs this image real or fake?") == {}


def test_expert_scores_from_run_file(tmp_path: Path) -> None:
    path = _write_run(
        tmp_path,
        "blending_diffusion",
        prompt_tail=" And the blending score is 0.009, and the diffusion score is 0.048.",
    )
    assert expert_scores_from_run_file(path) == {"blending": 0.009, "diffusion": 0.048}


def test_expert_scores_from_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "demo_none.json"
    path.write_text("{not json", encoding="utf-8")
    assert expert_scores_from_run_file(path) == {}


# --------------------------------------------------------------------------
# evidence conflict
# --------------------------------------------------------------------------


def test_describe_conflict_uses_actual_saved_values() -> None:
    card = ConfigCard(
        run_name="blending_diffusion",
        title="Blending + Diffusion",
        label="fake",
        real_score=0.3141,
        fake_score=0.6859,
        expert_scores={"blending": 0.009, "diffusion": 0.048},
        explanation="This image is fake",
        runtime_s=22.9,
        peak_vram_mib=9125,
        output_path=None,
        error=None,
    )
    notes = describe_evidence_conflicts([card])
    assert len(notes) == 1
    assert "very low fake likelihood" in notes[0]
    assert "blending 0.009" in notes[0]
    assert "diffusion 0.048" in notes[0]
    assert "language-model verdict is fake" in notes[0]
    assert "Both specialist detectors" in notes[0]


def test_describe_conflict_high_experts_vs_real_label() -> None:
    card = ConfigCard(
        run_name="blending_diffusion",
        title="Blending + Diffusion",
        label="real",
        real_score=0.80,
        fake_score=0.20,
        expert_scores={"blending": 0.91},
        explanation="real",
        runtime_s=1.0,
        peak_vram_mib=None,
        output_path=None,
        error=None,
    )
    notes = describe_evidence_conflicts([card])
    assert len(notes) == 1
    assert "high fake likelihood" in notes[0]
    assert "language-model verdict is real" in notes[0]


def test_describe_conflict_empty_when_aligned() -> None:
    card = ConfigCard(
        run_name="blending_diffusion",
        title="Blending + Diffusion",
        label="fake",
        real_score=0.10,
        fake_score=0.90,
        expert_scores={"blending": 0.85, "diffusion": 0.80},
        explanation="fake",
        runtime_s=1.0,
        peak_vram_mib=None,
        output_path=None,
        error=None,
    )
    assert describe_evidence_conflicts([card]) == []


def test_describe_conflict_empty_when_no_experts() -> None:
    card = ConfigCard(
        run_name="none",
        title="No expert",
        label="fake",
        real_score=0.2,
        fake_score=0.8,
        expert_scores={},
        explanation="fake",
        runtime_s=1.0,
        peak_vram_mib=None,
        output_path=None,
        error=None,
    )
    assert describe_evidence_conflicts([card]) == []


# --------------------------------------------------------------------------
# loading four saved-result structures / view-model
# --------------------------------------------------------------------------


def test_load_summary_missing_returns_none(tmp_path: Path) -> None:
    assert load_summary(tmp_path / "nope.json") is None


def test_load_summary_invalid_json_returns_none(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    path.write_text("{broken", encoding="utf-8")
    assert load_summary(path) is None


def test_build_dashboard_view_loads_four_configs(tmp_path: Path) -> None:
    directory = _write_matrix(
        tmp_path,
        scores={
            "none": ("0.24", "0.76"),
            "blending": ("0.36", "0.64"),
            "diffusion": ("0.29", "0.71"),
            "blending_diffusion": ("0.31", "0.69"),
        },
    )
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "quantisation": "4-bit",
                "runs": [
                    {
                        "run_name": "blending_diffusion",
                        "peak_vram_mib": 9125,
                        "output_path": str(directory / "demo_blending_diffusion.json"),
                        "expert_scores": {"blending": 0.009, "diffusion": 0.048},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    view = build_dashboard_view(directory, summary_path=summary)

    assert view.status is Status.UNCERTAIN
    assert "below 0.70" in view.rationale
    assert [c.run_name for c in view.cards] == [
        "none",
        "blending",
        "diffusion",
        "blending_diffusion",
    ]
    assert [c.title for c in view.cards] == [
        "No expert",
        "Blending",
        "Diffusion",
        "Blending + Diffusion",
    ]
    combined = view.cards[3]
    assert combined.label == "fake"
    assert combined.real_score == pytest.approx(0.31)
    assert combined.fake_score == pytest.approx(0.69)
    assert combined.expert_scores == {"blending": 0.009, "diffusion": 0.048}
    assert combined.peak_vram_mib == 9125
    assert combined.runtime_s == pytest.approx(13.0)
    assert view.quantisation == "4-bit"
    assert view.evidence_conflicts
    assert "0.009" in view.evidence_conflicts[0]
    assert view.disclaimer == DISCLAIMER
    assert view.image_path is not None and view.image_path.is_file()
    assert view.ok is True


def test_missing_matrix_directory_is_a_ui_error_not_an_exception(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"
    view = build_dashboard_view(missing, summary_path=tmp_path / "also-missing.json")
    assert view.errors
    assert any("not found" in message.lower() for message in view.errors)
    assert view.status is Status.FAILED
    assert len(view.cards) == 4


def test_malformed_run_json_does_not_raise(tmp_path: Path) -> None:
    directory = tmp_path / "matrix"
    directory.mkdir()
    (directory / "demo_none.json").write_text("{broken", encoding="utf-8")
    view = build_dashboard_view(directory, summary_path=None)
    assert view.cards[0].error is not None
    assert view.status is Status.FAILED


def test_build_dashboard_view_does_not_override_failed_status(tmp_path: Path) -> None:
    directory = tmp_path / "broken"
    directory.mkdir()
    _write_run(directory, "none")
    view = build_dashboard_view(directory, summary_path=None)
    assert view.status is Status.FAILED


def test_build_config_card_preserves_error() -> None:
    record = RunRecord(
        run_name="none",
        experts_used=[],
        prediction=None,
        real_score=None,
        fake_score=None,
        confidence=None,
        explanation="",
        runtime_seconds=None,
        error="file missing: demo_none.json",
    )
    card = build_config_card("none", record)
    assert card.error == "file missing: demo_none.json"
    assert card.label is None


@pytest.mark.skipif(
    not DEFAULT_MATRIX_DIR.is_dir(),
    reason="Stage 3 matrix outputs not present on this machine",
)
def test_real_stage3_matrix_renders_uncertain_with_conflict() -> None:
    summary = DEFAULT_SUMMARY if DEFAULT_SUMMARY.is_file() else None
    view = build_dashboard_view(DEFAULT_MATRIX_DIR, summary_path=summary)

    assert view.status is Status.UNCERTAIN
    assert view.disclaimer == DISCLAIMER
    assert view.image_path is not None and view.image_path.is_file()
    assert view.image_path.name == "real_face_01_crop.jpg"
    assert len(view.cards) == 4
    assert all(card.label == "fake" for card in view.cards)
    combined = next(c for c in view.cards if c.run_name == "blending_diffusion")
    assert combined.expert_scores.get("blending") == pytest.approx(0.009)
    assert combined.expert_scores.get("diffusion") == pytest.approx(0.048)
    assert view.evidence_conflicts
    assert "Both specialist detectors" in view.evidence_conflicts[0]
    assert "very low fake likelihood" in view.evidence_conflicts[0]
    assert view.quantisation == "4-bit"
    assert view.ok is True
