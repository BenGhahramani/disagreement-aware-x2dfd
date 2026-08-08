"""Compatibility tests: proof_of_concept.dashboard_data re-exports the view-model."""
from __future__ import annotations

import pytest

from dashboard import view_model as vm
from proof_of_concept import dashboard_data as poc

pytestmark = pytest.mark.unit


def test_poc_dashboard_data_reexports_view_model() -> None:
    assert poc.build_dashboard_view is vm.build_dashboard_view
    assert poc.DISCLAIMER == vm.DISCLAIMER
    assert poc.parse_probability is vm.parse_probability
