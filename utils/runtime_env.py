"""Cross-platform runtime path helpers for local and HPC execution."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Literal, Optional

PlatformName = Literal["windows", "linux", "bunya"]

INFER_CONFIG_BY_PLATFORM: dict[str, str] = {
    "windows": "infer_config.windows.yaml",
    "linux": "infer_config.yaml",
    "bunya": "infer_config.bunya.yaml",
}


def is_windows() -> bool:
    """True when running on Windows (including Git Bash reporting as win32)."""

    return os.name == "nt" or sys.platform.startswith("win")


def detect_platform(explicit: Optional[PlatformName] = None) -> PlatformName:
    """Return an explicit platform label or infer windows vs linux."""

    if explicit is not None:
        return explicit
    return "windows" if is_windows() else "linux"


def venv_python(project_root: Path | str) -> Path:
    """Return the expected venv interpreter path for the current OS."""

    root = Path(project_root)
    if is_windows():
        return root / ".venv" / "Scripts" / "python.exe"
    return root / ".venv" / "bin" / "python"


def infer_config_filename(platform: Optional[PlatformName] = None) -> str:
    """Map a platform label to the canonical eval infer config filename."""

    name = detect_platform(platform)
    if name == "bunya":
        return INFER_CONFIG_BY_PLATFORM["bunya"]
    if name == "windows":
        return INFER_CONFIG_BY_PLATFORM["windows"]
    return INFER_CONFIG_BY_PLATFORM["linux"]


def infer_config_path(
    project_root: Path | str,
    *,
    platform: Optional[PlatformName] = None,
) -> Path:
    """Absolute path to the platform-appropriate infer config under eval/configs/."""

    root = Path(project_root)
    return root / "eval" / "configs" / infer_config_filename(platform)
