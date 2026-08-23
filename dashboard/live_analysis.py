"""Backend service: analyse one local image with the verified Stage 3 pipeline.

Does not touch Streamlit. The saved Stage 3 dashboard continues to load
``eval/outputs/expert_matrix/demo_one_crop`` unchanged. This module only
prepares a new per-run directory and, when asked, drives the existing crop
tool + expert-matrix subprocesses.

Each expert configuration is a separate ``eval.infer.runner`` subprocess
(via ``tools.run_expert_matrix.execute_config``), so at most one
configuration occupies GPU memory at a time.

The Streamlit UI (later) should call :func:`analyse_image` with an optional
``progress_callback``.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dashboard.view_model import DashboardView, build_dashboard_view
from eval.experiment_configs import DEFAULT_CONFIGS
from tools.make_face_crop import (
    DEFAULT_DETECT_WIDTH,
    DEFAULT_MARGIN,
    DEFAULT_MIN_NEIGHBOURS,
    DEFAULT_QUALITY,
    DEFAULT_SCALE_FACTOR,
    DEFAULT_SIZE,
    CropPlan,
    FaceCropError,
    NoFaceFoundError,
    write_face_crop,
)
from tools.run_expert_matrix import (
    DEFAULT_TIMEOUT_S,
    ExpertConfig,
    RunOutcome,
    build_summary,
    execute_config,
    normalise_configs,
    write_json,
    write_runtimes_sidecar,
)
from tools.run_smoke_test import ManifestError, load_manifest

PROJECT_ROOT = _REPO_ROOT
DEFAULT_INFER_CONFIG = PROJECT_ROOT / "eval" / "configs" / "infer_config.windows.yaml"
DEFAULT_LIVE_ROOT = PROJECT_ROOT / "eval" / "outputs" / "live_analysis"
STAGE3_MATRIX_DIR = PROJECT_ROOT / "eval" / "outputs" / "expert_matrix" / "demo_one_crop"

LIVE_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
JPEG_MAGIC = b"\xff\xd8\xff"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
CROP_FILENAME = "face_crop.jpg"
MANIFEST_FILENAME = "manifest.json"
SUMMARY_FILENAME = "summary.json"
META_FILENAME = "meta.json"
CROP_PLAN_FILENAME = "crop_plan.json"

STAGE3_CROP_KWARGS: Dict[str, Any] = {
    "size": DEFAULT_SIZE,
    "margin": DEFAULT_MARGIN,
    "detect_width": DEFAULT_DETECT_WIDTH,
    "scale_factor": DEFAULT_SCALE_FACTOR,
    "min_neighbours": DEFAULT_MIN_NEIGHBOURS,
    "quality": DEFAULT_QUALITY,
    "overwrite": True,
}

ProgressCallback = Callable[["ProgressEvent"], None]
CropFn = Callable[..., CropPlan]
ProcessRunner = Callable[..., Any]


class LiveAnalysisError(ValueError):
    """Input or setup problem that stops the run before inference."""


@dataclass(frozen=True)
class ProgressEvent:
    """One structured step the UI can display without parsing logs."""

    stage: str
    message: str
    current: int
    total: int
    ok: bool = True
    detail: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "message": self.message,
            "current": self.current,
            "total": self.total,
            "ok": self.ok,
            "detail": self.detail,
        }


@dataclass
class LiveAnalysisResult:
    """Everything produced by one live run — success or structured failure."""

    ok: bool
    source_path: Path
    work_dir: Optional[Path] = None
    crop_path: Optional[Path] = None
    manifest_path: Optional[Path] = None
    matrix_dir: Optional[Path] = None
    summary_path: Optional[Path] = None
    view: Optional[DashboardView] = None
    events: List[ProgressEvent] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    outcomes: List[Dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "source_path": str(self.source_path),
            "work_dir": str(self.work_dir) if self.work_dir else None,
            "crop_path": str(self.crop_path) if self.crop_path else None,
            "manifest_path": str(self.manifest_path) if self.manifest_path else None,
            "matrix_dir": str(self.matrix_dir) if self.matrix_dir else None,
            "summary_path": str(self.summary_path) if self.summary_path else None,
            "view": None if self.view is None else self.view.as_dict(),
            "events": [event.as_dict() for event in self.events],
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "outcomes": list(self.outcomes),
        }


def validate_source_image(path: Path | str) -> Path:
    """Require an existing local JPEG or PNG. Does not run inference."""

    candidate = Path(path)
    if not candidate.exists():
        raise LiveAnalysisError(f"image not found: {candidate}")
    if not candidate.is_file():
        raise LiveAnalysisError(f"image path is not a file: {candidate}")
    suffix = candidate.suffix.lower()
    if suffix not in LIVE_IMAGE_SUFFIXES:
        raise LiveAnalysisError(
            f"unsupported image type {suffix!r}; expected one of "
            f"{sorted(LIVE_IMAGE_SUFFIXES)}"
        )
    try:
        header = candidate.read_bytes()[:16]
    except OSError as exc:
        raise LiveAnalysisError(f"cannot read image: {candidate}: {exc}") from exc
    if suffix in {".jpg", ".jpeg"} and not header.startswith(JPEG_MAGIC):
        raise LiveAnalysisError(f"file is not a JPEG: {candidate}")
    if suffix == ".png" and not header.startswith(PNG_MAGIC):
        raise LiveAnalysisError(f"file is not a PNG: {candidate}")
    return candidate.resolve()


def create_run_directory(work_root: Path, source_stem: str, *, clock: Callable[[], datetime] | None = None) -> Path:
    """Create a unique per-run folder under the gitignored live-output root."""

    stamp_fn = clock or (lambda: datetime.now(timezone.utc))
    stamp = stamp_fn().strftime("%Y%m%dT%H%M%SZ")
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", source_stem).strip("._") or "image"
    slug = slug[:40]
    base = work_root / f"{stamp}_{slug}"
    directory = base
    suffix = 2
    while directory.exists():
        directory = work_root / f"{stamp}_{slug}_{suffix}"
        suffix += 1
    directory.mkdir(parents=True, exist_ok=False)
    return directory.resolve()


def write_one_image_manifest(crop_path: Path, dest: Path) -> Path:
    """Write a runner manifest whose Description root is the crop's directory."""

    crop_path = crop_path.resolve()
    payload = {
        "Description": str(crop_path.parent),
        "images": [{"image_path": crop_path.name}],
    }
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    load_manifest(dest)
    return dest.resolve()


def _emit(
    events: List[ProgressEvent],
    callback: Optional[ProgressCallback],
    event: ProgressEvent,
) -> None:
    events.append(event)
    if callback is not None:
        callback(event)


def _default_crop(source: Path, dest: Path) -> CropPlan:
    return write_face_crop(source, dest, **STAGE3_CROP_KWARGS)


def _bind_crop_to_view(view: DashboardView, crop_path: Path) -> None:
    """Prefer the live crop; never silently substitute the Stage 3 demo face."""

    if crop_path.is_file():
        view.image_path = crop_path.resolve()
        view.warnings = [
            warning
            for warning in view.warnings
            if "fallback" not in warning.lower() and "Showing fallback" not in warning
        ]


def analyse_image(
    image_path: Path | str,
    *,
    work_root: Path | str | None = None,
    project_root: Path | str | None = None,
    config_path: Path | str | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    load_4bit: bool = True,
    sample_vram: bool = True,
    process_runner: Optional[ProcessRunner] = None,
    crop_fn: Optional[CropFn] = None,
    progress_callback: Optional[ProgressCallback] = None,
    configs: Optional[Sequence[str]] = None,
) -> LiveAnalysisResult:
    """Crop one image, run the four expert configs sequentially, return a view.

    ``process_runner`` and ``crop_fn`` are test seams. Production callers omit
    them so the real Haar crop and ``subprocess.run`` inference path are used.
    """

    events: List[ProgressEvent] = []
    root = Path(project_root) if project_root is not None else PROJECT_ROOT
    infer_config = Path(config_path) if config_path is not None else DEFAULT_INFER_CONFIG
    live_root = Path(work_root) if work_root is not None else DEFAULT_LIVE_ROOT
    requested = list(configs) if configs is not None else list(DEFAULT_CONFIGS)
    cropper = crop_fn or _default_crop

    try:
        source = validate_source_image(image_path)
    except LiveAnalysisError as exc:
        result = LiveAnalysisResult(ok=False, source_path=Path(image_path), errors=[str(exc)])
        _emit(
            result.events,
            progress_callback,
            ProgressEvent("validate", str(exc), 0, 1, ok=False, detail=str(exc)),
        )
        return result

    result = LiveAnalysisResult(ok=False, source_path=source)
    total_steps = 4 + len(requested)
    _emit(
        events,
        progress_callback,
        ProgressEvent("validate", f"Accepted source image {source.name}", 1, total_steps),
    )

    try:
        work_dir = create_run_directory(live_root, source.stem)
    except OSError as exc:
        result.errors.append(f"could not create working directory: {exc}")
        result.events = events
        _emit(
            events,
            progress_callback,
            ProgressEvent("workdir", result.errors[-1], 2, total_steps, ok=False),
        )
        return result

    result.work_dir = work_dir
    matrix_dir = work_dir / "matrix"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    result.matrix_dir = matrix_dir
    _emit(
        events,
        progress_callback,
        ProgressEvent("workdir", f"Working directory {work_dir}", 2, total_steps),
    )

    crop_path = work_dir / CROP_FILENAME
    try:
        plan = cropper(source, crop_path)
    except NoFaceFoundError as exc:
        result.errors.append(str(exc))
        result.events = events
        _emit(
            events,
            progress_callback,
            ProgressEvent("crop", str(exc), 3, total_steps, ok=False, detail=str(exc)),
        )
        _write_meta(result, infer_config=infer_config, load_4bit=load_4bit)
        return result
    except FaceCropError as exc:
        result.errors.append(str(exc))
        result.events = events
        _emit(
            events,
            progress_callback,
            ProgressEvent("crop", str(exc), 3, total_steps, ok=False, detail=str(exc)),
        )
        _write_meta(result, infer_config=infer_config, load_4bit=load_4bit)
        return result

    if not crop_path.is_file():
        result.errors.append(f"crop function did not write {crop_path}")
        result.events = events
        _emit(
            events,
            progress_callback,
            ProgressEvent("crop", result.errors[-1], 3, total_steps, ok=False),
        )
        _write_meta(result, infer_config=infer_config, load_4bit=load_4bit)
        return result

    result.crop_path = crop_path.resolve()
    plan_payload = plan.as_dict() if isinstance(plan, CropPlan) else {"crop": str(plan)}
    plan_payload.update({"source": str(source), "output": str(result.crop_path), **{
        key: STAGE3_CROP_KWARGS[key]
        for key in ("size", "margin", "detect_width", "scale_factor", "min_neighbours", "quality")
        if key in STAGE3_CROP_KWARGS
    }})
    write_json(work_dir / CROP_PLAN_FILENAME, plan_payload)
    _emit(
        events,
        progress_callback,
        ProgressEvent("crop", f"Wrote {CROP_FILENAME} ({DEFAULT_SIZE}x{DEFAULT_SIZE})", 3, total_steps),
    )

    try:
        manifest_path = write_one_image_manifest(crop_path, work_dir / MANIFEST_FILENAME)
    except (OSError, ManifestError) as exc:
        result.errors.append(f"manifest failed: {exc}")
        result.events = events
        _emit(
            events,
            progress_callback,
            ProgressEvent("manifest", result.errors[-1], 4, total_steps, ok=False),
        )
        _write_meta(result, infer_config=infer_config, load_4bit=load_4bit)
        return result

    result.manifest_path = manifest_path
    _emit(
        events,
        progress_callback,
        ProgressEvent("manifest", f"Wrote {MANIFEST_FILENAME}", 4, total_steps),
    )

    expert_configs: List[ExpertConfig] = list(normalise_configs(requested))
    log_dir = matrix_dir / "logs"
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    outcomes: List[RunOutcome] = []

    for index, config in enumerate(expert_configs, start=1):
        step = 4 + index
        _emit(
            events,
            progress_callback,
            ProgressEvent(
                f"config:{config.run_name}",
                f"Running configuration {config.experts_arg} ({index}/{len(expert_configs)})",
                step,
                total_steps,
            ),
        )
        outcome = execute_config(
            config,
            manifest_path=manifest_path,
            config_path=infer_config,
            output_dir=matrix_dir,
            log_dir=log_dir,
            project_root=root,
            timeout_s=timeout_s,
            load_4bit=load_4bit,
            load_8bit=False,
            sample_vram=sample_vram,
            process_runner=process_runner or subprocess.run,
        )
        outcomes.append(outcome)
        if outcome.status != "pass":
            detail = "; ".join(outcome.failures) or "configuration failed"
            result.errors.append(f"{config.run_name}: {detail}")
            _emit(
                events,
                progress_callback,
                ProgressEvent(
                    f"config:{config.run_name}",
                    f"{config.run_name} failed validation",
                    step,
                    total_steps,
                    ok=False,
                    detail=detail,
                ),
            )
        else:
            _emit(
                events,
                progress_callback,
                ProgressEvent(
                    f"config:{config.run_name}",
                    f"{config.run_name} passed ({outcome.label or 'unlabelled'})",
                    step,
                    total_steps,
                    detail=None if outcome.runtime_s is None else f"{outcome.runtime_s:.1f}s",
                ),
            )

    quantisation = "4-bit" if load_4bit else "fp16"
    summary = build_summary(
        outcomes,
        manifest=manifest_path,
        images=[crop_path.resolve()],
        config_path=infer_config,
        output_dir=matrix_dir,
        quantisation=quantisation,
        started_at=started_at,
        finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    summary_path = work_dir / SUMMARY_FILENAME
    write_json(summary_path, summary)
    write_runtimes_sidecar(matrix_dir, outcomes)
    result.summary_path = summary_path.resolve()
    result.outcomes = [outcome.as_dict() for outcome in outcomes]

    view = build_dashboard_view(matrix_dir, summary_path=summary_path)
    _bind_crop_to_view(view, crop_path)
    result.view = view
    result.warnings.extend(view.warnings)
    if view.errors:
        result.errors.extend(view.errors)

    result.ok = (
        summary.get("status") == "pass"
        and view.ok
        and all(outcome.status == "pass" for outcome in outcomes)
    )
    result.events = events
    _write_meta(result, infer_config=infer_config, load_4bit=load_4bit)
    return result


def _write_meta(
    result: LiveAnalysisResult,
    *,
    infer_config: Path,
    load_4bit: bool,
) -> None:
    if result.work_dir is None:
        return
    payload = {
        "ok": result.ok,
        "source_path": str(result.source_path),
        "crop_path": str(result.crop_path) if result.crop_path else None,
        "manifest_path": str(result.manifest_path) if result.manifest_path else None,
        "matrix_dir": str(result.matrix_dir) if result.matrix_dir else None,
        "summary_path": str(result.summary_path) if result.summary_path else None,
        "errors": list(result.errors),
        "warnings": list(result.warnings),
        "infer_config": str(infer_config),
        "load_4bit": load_4bit,
        "stage3_matrix_untouched": str(STAGE3_MATRIX_DIR),
    }
    try:
        write_json(result.work_dir / META_FILENAME, payload)
    except OSError:
        return
