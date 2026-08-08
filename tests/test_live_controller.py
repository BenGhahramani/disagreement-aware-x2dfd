"""Unit tests for dashboard/live_controller.py — no Streamlit, no GPU."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from dashboard.live_analysis import (
    DEFAULT_INFER_CONFIG,
    DEFAULT_LIVE_ROOT,
    LiveAnalysisError,
    LiveAnalysisResult,
    ProgressEvent,
)
from dashboard.live_controller import (
    DEFAULT_UPLOAD_ROOT,
    live_analysis_call_spec,
    persist_upload,
    progress_fraction,
    progress_label,
    run_live_analysis,
    sanitize_upload_name,
    structured_error_messages,
    unexpected_error_message,
    validate_upload_payload,
)

pytestmark = pytest.mark.unit

MINIMAL_JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 32
MINIMAL_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 24


def test_sanitize_strips_directories_and_unsafe_characters() -> None:
    assert sanitize_upload_name("../temp/Mae Jemison!.JPG") == "Mae_Jemison.jpg"
    assert sanitize_upload_name("Mae Jemison!.JPG") == "Mae_Jemison.jpg"
    assert sanitize_upload_name("folder/photo.PNG") == "photo.png"


def test_sanitize_rejects_unsupported_types() -> None:
    with pytest.raises(LiveAnalysisError, match="unsupported"):
        sanitize_upload_name("notes.txt")
    with pytest.raises(LiveAnalysisError, match="no filename"):
        sanitize_upload_name("   ")


def test_validate_upload_payload_checks_magic_bytes() -> None:
    assert validate_upload_payload("face.jpg", MINIMAL_JPEG) == "face.jpg"
    assert validate_upload_payload("face.png", MINIMAL_PNG) == "face.png"
    with pytest.raises(LiveAnalysisError, match="not a JPEG"):
        validate_upload_payload("face.jpg", b"nope")
    with pytest.raises(LiveAnalysisError, match="empty"):
        validate_upload_payload("face.jpg", b"")


def test_persist_upload_writes_under_given_root(tmp_path: Path) -> None:
    clock = lambda: datetime(2026, 8, 8, 8, 24, 0, tzinfo=timezone.utc)
    dest = persist_upload(
        MINIMAL_JPEG,
        "portrait.jpg",
        dest_root=tmp_path / "uploads",
        clock=clock,
    )
    assert dest.is_file()
    assert dest.read_bytes() == MINIMAL_JPEG
    assert dest.parent.name == "20260808T082400Z_portrait"
    assert dest.name == "portrait.jpg"
    assert dest.is_relative_to(tmp_path / "uploads")


def test_persist_upload_rejects_bad_payload_before_writing(tmp_path: Path) -> None:
    with pytest.raises(LiveAnalysisError):
        persist_upload(b"not-an-image", "face.jpg", dest_root=tmp_path)
    assert list(tmp_path.rglob("*")) == []


def test_default_upload_root_is_gitignored_eval_outputs() -> None:
    relative = DEFAULT_UPLOAD_ROOT.relative_to(DEFAULT_UPLOAD_ROOT.parents[2])
    assert relative.parts[:2] == ("eval", "outputs")
    assert DEFAULT_UPLOAD_ROOT.parts[-1] == "live_uploads"
    assert DEFAULT_LIVE_ROOT.parts[-1] == "live_analysis"


def test_live_analysis_call_spec_is_windows_4bit() -> None:
    spec = live_analysis_call_spec()
    assert spec["config_path"] == DEFAULT_INFER_CONFIG
    assert spec["config_path"].name == "infer_config.windows.yaml"
    assert spec["load_4bit"] is True
    assert spec["sample_vram"] is True


def test_run_live_analysis_forwards_verified_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: dict = {}

    def fake_analyse(image_path, **kwargs):
        seen["image_path"] = Path(image_path)
        seen["kwargs"] = kwargs
        return LiveAnalysisResult(ok=True, source_path=Path(image_path))

    monkeypatch.setattr("dashboard.live_controller.analyse_image", fake_analyse)
    image = tmp_path / "face.jpg"
    image.write_bytes(MINIMAL_JPEG)
    result = run_live_analysis(image)
    assert result.ok is True
    assert seen["image_path"] == image
    assert seen["kwargs"]["config_path"] == DEFAULT_INFER_CONFIG
    assert seen["kwargs"]["load_4bit"] is True
    assert seen["kwargs"]["sample_vram"] is True
    assert "process_runner" not in seen["kwargs"]
    assert "crop_fn" not in seen["kwargs"]


def test_progress_helpers() -> None:
    event = ProgressEvent("crop", "Wrote face_crop.jpg", 3, 8, ok=True, detail="256x256")
    assert progress_fraction(event) == pytest.approx(0.375)
    assert "Wrote face_crop.jpg" in progress_label(event)
    assert "256x256" in progress_label(event)
    failed = ProgressEvent("crop", "no face detected", 3, 8, ok=False)
    assert progress_label(failed).startswith("Failed — ")
    assert progress_fraction(ProgressEvent("x", "m", 0, 0)) == 0.0


def test_structured_errors_never_invent_a_traceback() -> None:
    result = LiveAnalysisResult(
        ok=False,
        source_path=Path("face.jpg"),
        errors=["no face detected in source"],
    )
    assert structured_error_messages(result) == ["no face detected in source"]
    empty = LiveAnalysisResult(ok=False, source_path=Path("face.jpg"))
    assert structured_error_messages(empty) == ["Live analysis did not complete successfully."]
    ok = LiveAnalysisResult(ok=True, source_path=Path("face.jpg"))
    assert structured_error_messages(ok) == []
    assert "Traceback" not in unexpected_error_message(RuntimeError("boom"))
    assert unexpected_error_message(RuntimeError("boom")) == "RuntimeError: boom"
