"""Unit tests for utils/runtime_env.py."""
from __future__ import annotations

from pathlib import Path

import pytest

from utils import runtime_env as re

pytestmark = pytest.mark.unit


def test_infer_config_filename_windows() -> None:
    assert re.infer_config_filename("windows") == "infer_config.windows.yaml"


def test_infer_config_filename_bunya() -> None:
    assert re.infer_config_filename("bunya") == "infer_config.bunya.yaml"


def test_infer_config_filename_linux() -> None:
    assert re.infer_config_filename("linux") == "infer_config.yaml"


def test_infer_config_path_under_eval_configs(tmp_path: Path) -> None:
    path = re.infer_config_path(tmp_path, platform="bunya")
    assert path == tmp_path / "eval" / "configs" / "infer_config.bunya.yaml"


def test_venv_python_windows_layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(re, "is_windows", lambda: True)
    assert re.venv_python(tmp_path) == tmp_path / ".venv" / "Scripts" / "python.exe"


def test_venv_python_linux_layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(re, "is_windows", lambda: False)
    assert re.venv_python(tmp_path) == tmp_path / ".venv" / "bin" / "python"


def test_detect_platform_explicit_overrides_auto() -> None:
    assert re.detect_platform("bunya") == "bunya"
    assert re.detect_platform("linux") == "linux"
