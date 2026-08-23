"""Unit tests for tools/check_bunya_environment.py."""
from __future__ import annotations

from pathlib import Path

import pytest

from tools import check_bunya_environment as cbe
from tools import check_environment as ce

pytestmark = pytest.mark.unit


def test_default_config_points_at_bunya_yaml(tmp_path: Path) -> None:
    expected = tmp_path / "eval" / "configs" / "infer_config.bunya.yaml"
    assert cbe.DEFAULT_CONFIG_RELATIVE == Path("eval/configs/infer_config.bunya.yaml")
    assert (tmp_path / cbe.DEFAULT_CONFIG_RELATIVE) == expected


def test_require_bitsandbytes_upgrades_warn_to_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    base_results = [
        ce.CheckResult("import.torch", ce.Status.PASS, "ok"),
        ce.CheckResult(
            "import.bitsandbytes",
            ce.Status.WARN,
            "not installed - needed for: optional 4/8-bit loading",
            required=False,
        ),
    ]
    monkeypatch.setattr(ce, "run_checks", lambda *args, **kwargs: list(base_results))

    upgraded = cbe.run_checks(
        project_root=Path("."),
        config_path=Path("eval/configs/infer_config.bunya.yaml"),
        require_bitsandbytes=True,
        skip_weights=True,
        skip_datasets=True,
    )
    bnb = next(r for r in upgraded if r.name == "import.bitsandbytes")
    assert bnb.status is ce.Status.FAIL
    assert bnb.required is True
    assert "4-bit" in bnb.detail
