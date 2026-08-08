"""Unit tests for dashboard/live_analysis.py.

Inference subprocesses and Haar cropping are mocked. No GPU, no weights.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence

import pytest

from dashboard import live_analysis as live
from dashboard.view_model import DISCLAIMER
from proof_of_concept.schema import Status
from tests.test_expert_matrix import MatrixRunner, result_payload
from tools.make_face_crop import CropPlan, NoFaceFoundError
from tools.run_smoke_test import load_manifest

pytestmark = pytest.mark.unit

MINIMAL_JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 32
MINIMAL_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 24


def write_jpeg(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(MINIMAL_JPEG)
    return path


def fake_cropper(source: Path, dest: Path, **kwargs: Any) -> CropPlan:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(MINIMAL_JPEG)
    fake_cropper.calls.append({"source": source, "dest": dest, **kwargs})  # type: ignore[attr-defined]
    return CropPlan(detection=(10, 12, 80, 90), square=(0, 0, 120, 120))


fake_cropper.calls = []  # type: ignore[attr-defined]


class RecordingRunner(MatrixRunner):
    """MatrixRunner that also records env and peak in-flight subprocesses."""

    def __init__(self, *args: Any, crop_image: Optional[Path] = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.envs: List[Dict[str, str]] = []
        self.inflight = 0
        self.max_inflight = 0
        self.crop_image = crop_image

    def __call__(self, command: Sequence[str], **kwargs: Any) -> SimpleNamespace:
        self.commands.append(list(command))
        self.inflight += 1
        self.max_inflight = max(self.max_inflight, self.inflight)
        self.envs.append(dict(kwargs.get("env") or {}))
        try:
            spec = {**self.default, **self.scenarios.get(self.run_name(command), {})}
            if spec.get("raises") is not None:
                raise spec["raises"]
            experts = self.requested_experts(command)
            image = str(self.crop_image) if self.crop_image is not None else "C:/images/face.jpg"
            payload = spec.get("payload", "UNSET")
            if payload == "UNSET":
                payload = result_payload(
                    experts=experts,
                    scores={name: 0.5 for name in experts},
                    answer="fake",
                    real="0.31",
                    fake="0.69",
                )
                if payload and isinstance(payload, list):
                    payload[0]["image"] = image
            elif isinstance(payload, list) and payload and isinstance(payload[0], dict):
                payload[0]["image"] = image
            if payload is not None:
                output = Path(self._flag(command, "--output") or "")
                output.parent.mkdir(parents=True, exist_ok=True)
                text = payload if isinstance(payload, str) else json.dumps(payload)
                output.write_text(text, encoding="utf-8")
            return SimpleNamespace(
                returncode=spec.get("returncode", 0),
                stdout=spec.get("stdout", ""),
                stderr=spec.get("stderr", ""),
            )
        finally:
            self.inflight -= 1


@pytest.fixture(autouse=True)
def _reset_crop_calls() -> None:
    fake_cropper.calls = []  # type: ignore[attr-defined]


def _run(
    tmp_path: Path,
    *,
    source: Optional[Path] = None,
    runner: Optional[RecordingRunner] = None,
    crop_fn: Any = fake_cropper,
    **kwargs: Any,
) -> live.LiveAnalysisResult:
    image = source or write_jpeg(tmp_path / "incoming" / "portrait.jpg")
    fake = runner or RecordingRunner()
    if fake.crop_image is None:
        # crop path is known only after analyse_image creates the work dir;
        # RecordingRunner patches image after dest exists via a wrapper below.
        pass

    captured: List[Path] = []

    def crop_and_remember(src: Path, dest: Path, **crop_kwargs: Any) -> CropPlan:
        plan = crop_fn(src, dest, **crop_kwargs)
        captured.append(dest)
        fake.crop_image = dest
        return plan

    return live.analyse_image(
        image,
        work_root=tmp_path / "live",
        project_root=tmp_path,
        process_runner=fake,
        crop_fn=crop_and_remember,
        sample_vram=False,
        **kwargs,
    )


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------


def test_missing_image_is_a_structured_error(tmp_path: Path) -> None:
    missing = tmp_path / "nope.jpg"
    result = live.analyse_image(missing, work_root=tmp_path / "live", crop_fn=fake_cropper)
    assert result.ok is False
    assert result.view is None
    assert any("not found" in err.lower() for err in result.errors)
    assert fake_cropper.calls == []  # type: ignore[attr-defined]


def test_unsupported_extension_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("hello", encoding="utf-8")
    result = live.analyse_image(path, work_root=tmp_path / "live", crop_fn=fake_cropper)
    assert result.ok is False
    assert "unsupported" in result.errors[0].lower()


def test_jpeg_magic_bytes_are_checked(tmp_path: Path) -> None:
    path = tmp_path / "fake.jpg"
    path.write_bytes(b"not-a-jpeg")
    with pytest.raises(live.LiveAnalysisError, match="not a JPEG"):
        live.validate_source_image(path)


def test_png_source_is_accepted(tmp_path: Path) -> None:
    path = tmp_path / "face.png"
    path.write_bytes(MINIMAL_PNG)
    assert live.validate_source_image(path) == path.resolve()


# --------------------------------------------------------------------------
# crop + manifest + work dir
# --------------------------------------------------------------------------


def test_creates_gitignored_style_workdir_and_stage3_crop_kwargs(tmp_path: Path) -> None:
    result = _run(tmp_path)
    assert result.work_dir is not None
    assert result.work_dir.is_dir()
    assert result.work_dir.parent == (tmp_path / "live").resolve()
    assert result.crop_path is not None and result.crop_path.is_file()
    assert result.crop_path.name == "face_crop.jpg"
    assert fake_cropper.calls  # type: ignore[attr-defined]
    # crop_fn in analyse_image is a wrapper; Stage 3 defaults live on write_face_crop kwargs
    assert live.STAGE3_CROP_KWARGS["size"] == 256
    assert live.STAGE3_CROP_KWARGS["margin"] == 1.3
    assert live.DEFAULT_INFER_CONFIG.name == "infer_config.windows.yaml"
    assert live.DEFAULT_LIVE_ROOT == live.PROJECT_ROOT / "eval" / "outputs" / "live_analysis"


def test_manifest_is_one_image_and_passes_existing_loader(tmp_path: Path) -> None:
    result = _run(tmp_path)
    assert result.manifest_path is not None
    info = load_manifest(result.manifest_path)
    assert len(info.images) == 1
    assert info.images[0] == result.crop_path
    payload = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert payload["images"][0]["image_path"] == "face_crop.jpg"


def test_no_face_stops_before_inference(tmp_path: Path) -> None:
    fake = RecordingRunner()

    def boom(source: Path, dest: Path, **kwargs: Any) -> CropPlan:
        raise NoFaceFoundError("no face detected in source")

    result = _run(tmp_path, runner=fake, crop_fn=boom)
    assert result.ok is False
    assert fake.commands == []
    assert any("no face" in err.lower() for err in result.errors)
    assert result.work_dir is not None and result.work_dir.is_dir()
    assert (result.work_dir / "meta.json").is_file()


def test_run_directory_disambiguates_collisions(tmp_path: Path) -> None:
    root = tmp_path / "live"
    fixed = datetime(2026, 8, 8, 7, 0, 0, tzinfo=timezone.utc)
    first = live.create_run_directory(root, "portrait", clock=lambda: fixed)
    second = live.create_run_directory(root, "portrait", clock=lambda: fixed)
    assert first != second
    assert first.is_dir() and second.is_dir()


# --------------------------------------------------------------------------
# four sequential 4-bit configs
# --------------------------------------------------------------------------


def test_runs_four_configs_sequentially_with_windows_4bit(tmp_path: Path) -> None:
    fake = RecordingRunner()
    result = _run(tmp_path, runner=fake)

    assert result.ok is True
    assert len(fake.commands) == 4
    assert fake.max_inflight == 1
    experts = [cmd[cmd.index("--experts") + 1] for cmd in fake.commands]
    assert experts == ["none", "blending", "diffusion", "blending,diffusion"]
    configs = [cmd[cmd.index("--config") + 1] for cmd in fake.commands]
    assert all(Path(path).name == "infer_config.windows.yaml" for path in configs)
    assert all(env.get("X2DFD_LOAD_4BIT") == "1" for env in fake.envs)
    assert all("X2DFD_LOAD_8BIT" not in env for env in fake.envs)


def test_outputs_are_poc_shaped_and_preserved(tmp_path: Path) -> None:
    result = _run(tmp_path)
    assert result.matrix_dir is not None
    names = sorted(p.name for p in result.matrix_dir.glob("demo_*.json"))
    assert names == [
        "demo_blending.json",
        "demo_blending_diffusion.json",
        "demo_diffusion.json",
        "demo_none.json",
    ]
    assert (result.matrix_dir / "runtimes.json").is_file()
    assert result.summary_path is not None and result.summary_path.is_file()
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["quantisation"] == "4-bit"
    assert summary["status"] == "pass"
    # still on disk after the function returns
    assert result.work_dir is not None
    assert (result.work_dir / "meta.json").is_file()
    assert (result.work_dir / "crop_plan.json").is_file()


def test_does_not_write_into_stage3_demo_directory(tmp_path: Path) -> None:
    stage3 = live.STAGE3_MATRIX_DIR
    before = {p.name: p.stat().st_mtime for p in stage3.glob("demo_*.json")} if stage3.is_dir() else {}
    result = _run(tmp_path)
    assert result.matrix_dir is not None
    assert result.matrix_dir.resolve() != stage3.resolve()
    if before:
        after = {p.name: p.stat().st_mtime for p in stage3.glob("demo_*.json")}
        assert after == before


# --------------------------------------------------------------------------
# view-model + partial failure
# --------------------------------------------------------------------------


def test_successful_run_builds_dashboard_view(tmp_path: Path) -> None:
    result = _run(tmp_path)
    assert result.view is not None
    view = result.view
    assert view.status is Status.UNCERTAIN
    assert "below 0.70" in view.rationale
    assert [card.run_name for card in view.cards] == [
        "none",
        "blending",
        "diffusion",
        "blending_diffusion",
    ]
    assert view.disclaimer == DISCLAIMER
    assert view.quantisation == "4-bit"
    assert view.image_path == result.crop_path
    assert view.matrix_dir == result.matrix_dir.resolve()


def test_invalid_output_is_recorded_without_raising(tmp_path: Path) -> None:
    fake = RecordingRunner(
        scenarios={
            "diffusion": {
                "payload": [{"id": "1", "image": "x", "conversations": [{"from": "human", "value": "hi"}]}],
                "returncode": 0,
            }
        }
    )
    result = _run(tmp_path, runner=fake)
    assert result.ok is False
    assert any(err.startswith("diffusion:") for err in result.errors)
    assert result.view is not None
    assert result.summary_path is not None
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "fail"
    assert summary["totals"]["failed"] == 1
    # remaining three configs still ran
    assert len(fake.commands) == 4


def test_progress_callback_receives_structured_events(tmp_path: Path) -> None:
    seen: List[live.ProgressEvent] = []
    result = _run(tmp_path, progress_callback=seen.append)
    assert result.ok is True
    stages = [event.stage for event in seen]
    assert stages[0] == "validate"
    assert "crop" in stages
    assert "manifest" in stages
    assert "config:none" in stages
    assert "config:blending_diffusion" in stages
    assert all(isinstance(event.current, int) and isinstance(event.total, int) for event in seen)
    assert result.events == seen


def test_default_crop_path_uses_stage3_write_face_crop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: Dict[str, Any] = {}
    fake = RecordingRunner()

    def fake_write(source: Path, dest: Path, **kwargs: Any) -> CropPlan:
        seen.update(kwargs)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(MINIMAL_JPEG)
        fake.crop_image = dest
        return CropPlan(detection=(1, 1, 8, 8), square=(0, 0, 16, 16))

    monkeypatch.setattr(live, "write_face_crop", fake_write)
    image = write_jpeg(tmp_path / "incoming" / "portrait.jpg")
    result = live.analyse_image(
        image,
        work_root=tmp_path / "live",
        project_root=tmp_path,
        process_runner=fake,
        sample_vram=False,
    )
    assert seen["size"] == 256
    assert seen["margin"] == 1.3
    assert seen["detect_width"] == 1024
    assert seen["quality"] == 95
    assert seen["overwrite"] is True
    assert result.ok is True


def test_default_live_root_is_under_gitignored_eval_outputs() -> None:
    relative = live.DEFAULT_LIVE_ROOT.relative_to(live.PROJECT_ROOT)
    assert relative.parts[:2] == ("eval", "outputs")
    assert relative.parts[2] == "live_analysis"
