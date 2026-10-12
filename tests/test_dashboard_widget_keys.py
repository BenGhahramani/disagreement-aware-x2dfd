"""Regression tests: shared dashboard renderer must not emit duplicate widget keys.

The saved-example tab and the live-analysis tab both render
``render_dashboard_view`` in the same Streamlit run (``st.tabs`` renders all
tab bodies). Explicit widget keys must therefore be namespaced per view.
No GPU inference: the saved "nasa" example stands in for a live result view.
"""
from __future__ import annotations

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

pytestmark = pytest.mark.unit


def _render_two_views_script() -> None:
    import streamlit as st

    from dashboard.app import render_dashboard_view
    from dashboard.saved_examples import load_saved_example

    saved_tab, live_tab = st.tabs(["Saved example", "Analyse new image"])
    with saved_tab:
        render_dashboard_view(load_saved_example("nasa"), key_prefix="saved")
    with live_tab:
        # Same view rendered again as a stand-in for a completed live result.
        render_dashboard_view(load_saved_example("nasa"), key_prefix="live")


def _render_two_views_same_prefix_script() -> None:
    import streamlit as st

    from dashboard.app import render_dashboard_view
    from dashboard.saved_examples import load_saved_example

    saved_tab, live_tab = st.tabs(["Saved example", "Analyse new image"])
    with saved_tab:
        render_dashboard_view(load_saved_example("nasa"), key_prefix="dup")
    with live_tab:
        render_dashboard_view(load_saved_example("nasa"), key_prefix="dup")


def test_saved_and_live_views_render_together_without_duplicate_keys() -> None:
    at = AppTest.from_function(_render_two_views_script, default_timeout=60)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]

    # Both instances expose independent, namespaced controls.
    slider_keys = {s.key for s in at.slider}
    assert "saved_ops_threshold_slider" in slider_keys
    assert "live_ops_threshold_slider" in slider_keys

    select_keys = {s.key for s in at.selectbox}
    assert "saved_ops_run_select" in select_keys
    assert "live_ops_run_select" in select_keys

    text_keys = {t.key for t in at.text_input}
    assert "saved_ops_validation_path_input" in text_keys
    assert "live_ops_validation_path_input" in text_keys

    # Session state is namespaced per view (no shared ``ops_*`` keys).
    assert "saved_ops_threshold" in at.session_state
    assert "live_ops_threshold" in at.session_state
    assert "ops_threshold" not in at.session_state


def test_views_keep_independent_threshold_state() -> None:
    at = AppTest.from_function(_render_two_views_script, default_timeout=60)
    at.run()
    assert not at.exception

    live_slider = next(s for s in at.slider if s.key == "live_ops_threshold_slider")
    live_slider.set_value(0.80).run()
    assert not at.exception, [str(e) for e in at.exception]

    assert at.session_state["live_ops_threshold"] == pytest.approx(0.80)
    # Saved view untouched.
    assert at.session_state["saved_ops_threshold"] == pytest.approx(0.50)
    saved_slider = next(s for s in at.slider if s.key == "saved_ops_threshold_slider")
    assert saved_slider.value == pytest.approx(0.50)


def test_duplicate_prefix_is_detected_by_harness() -> None:
    """Sanity check: the harness really surfaces duplicate-key collisions."""

    at = AppTest.from_function(_render_two_views_same_prefix_script, default_timeout=60)
    at.run()
    assert at.exception
    assert any("ops_validation_path_input" in str(e) or "multiple" in str(e).lower() for e in at.exception)
