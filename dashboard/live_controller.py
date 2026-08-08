"""UI-independent helpers for the Streamlit live-analysis tab.

No Streamlit imports. The dashboard calls these to persist an upload and to
invoke :func:`dashboard.live_analysis.analyse_image` with the verified
Windows 4-bit defaults. Tests cover this module without a browser or GPU.
"""
from __future__ import annotations

import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dashboard.live_analysis import (
    DEFAULT_INFER_CONFIG,
    DEFAULT_LIVE_ROOT,
    JPEG_MAGIC,
    LIVE_IMAGE_SUFFIXES,
    PNG_MAGIC,
    LiveAnalysisError,
    LiveAnalysisResult,
    ProgressEvent,
    analyse_image,
)

PROJECT_ROOT = _REPO_ROOT
DEFAULT_UPLOAD_ROOT = PROJECT_ROOT / "eval" / "outputs" / "live_uploads"

_UNSAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_upload_name(filename: str) -> str:
    """Return a single-path-segment JPEG/PNG name, or raise LiveAnalysisError."""

    if not filename or not str(filename).strip():
        raise LiveAnalysisError("upload has no filename")
    name = Path(str(filename).replace("\\", "/")).name
    suffix = Path(name).suffix.lower()
    if suffix not in LIVE_IMAGE_SUFFIXES:
        raise LiveAnalysisError(
            f"unsupported image type {suffix!r}; expected one of "
            f"{sorted(LIVE_IMAGE_SUFFIXES)}"
        )
    stem = _UNSAFE_NAME.sub("_", Path(name).stem).strip("._") or "upload"
    return f"{stem[:80]}{suffix}"


def validate_upload_payload(filename: str, data: bytes) -> str:
    """Check filename + magic bytes before anything is written to disk."""

    safe_name = sanitize_upload_name(filename)
    if not data:
        raise LiveAnalysisError("upload is empty")
    suffix = Path(safe_name).suffix.lower()
    if suffix in {".jpg", ".jpeg"} and not data.startswith(JPEG_MAGIC):
        raise LiveAnalysisError("file is not a JPEG")
    if suffix == ".png" and not data.startswith(PNG_MAGIC):
        raise LiveAnalysisError("file is not a PNG")
    return safe_name


def persist_upload(
    data: bytes,
    filename: str,
    *,
    dest_root: Path | str | None = None,
    clock: Callable[[], datetime] | None = None,
) -> Path:
    """Save an upload under the gitignored live-uploads tree and return its path."""

    safe_name = validate_upload_payload(filename, data)
    root = Path(dest_root) if dest_root is not None else DEFAULT_UPLOAD_ROOT
    stamp_fn = clock or (lambda: datetime.now(timezone.utc))
    stamp = stamp_fn().strftime("%Y%m%dT%H%M%SZ")
    folder = root / f"{stamp}_{Path(safe_name).stem}"
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / safe_name
    dest.write_bytes(data)
    return dest.resolve()


def live_analysis_call_spec() -> Dict[str, Any]:
    """Explicit kwargs so the UI cannot drift from the verified E2E settings."""

    return {
        "config_path": DEFAULT_INFER_CONFIG,
        "load_4bit": True,
        "sample_vram": True,
    }


def progress_fraction(event: ProgressEvent) -> float:
    if event.total <= 0:
        return 0.0
    value = event.current / float(event.total)
    return max(0.0, min(1.0, value))


def progress_label(event: ProgressEvent) -> str:
    prefix = "" if event.ok else "Failed — "
    label = f"{prefix}{event.message}"
    if event.detail:
        label = f"{label} ({event.detail})"
    return label


def structured_error_messages(result: LiveAnalysisResult) -> List[str]:
    """Human-readable errors for the UI. Never a traceback."""

    messages = [str(item) for item in result.errors if str(item).strip()]
    if result.ok:
        return messages
    if not messages:
        messages.append("Live analysis did not complete successfully.")
    return messages


def run_live_analysis(
    image_path: Path | str,
    *,
    progress_callback: Optional[Callable[[ProgressEvent], None]] = None,
) -> LiveAnalysisResult:
    """Run one live analysis with the verified Windows 4-bit defaults."""

    return analyse_image(
        image_path,
        progress_callback=progress_callback,
        **live_analysis_call_spec(),
    )


def unexpected_error_message(exc: BaseException) -> str:
    """Format an unexpected exception without a traceback."""

    return f"{type(exc).__name__}: {exc}"
