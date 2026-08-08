"""Minimal Streamlit dashboard over saved Stage 3 expert-matrix outputs.

Decision-support view only — no upload, no live inference. Launch from the
repository root::

    .venv\\Scripts\\python.exe -m streamlit run dashboard/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st

from dashboard.view_model import (
    DEFAULT_MATRIX_DIR,
    DEFAULT_SUMMARY,
    DashboardView,
    bar_fraction,
    build_dashboard_view,
)
from proof_of_concept.schema import Status

_STATUS_STYLE = {
    Status.STABLE: ("#1f6f4a", "#e8f5ee"),
    Status.CONTESTED: ("#8a3b12", "#fcefe6"),
    Status.UNCERTAIN: ("#6b5a1e", "#fbf6e3"),
    Status.FAILED: ("#6e1f2a", "#f8e8eb"),
}

_PAGE_CSS = """
<style>
    [data-testid="stMainBlockContainer"],
    .block-container {
        max-width: 100% !important;
        padding-top: 1.05rem;
        padding-bottom: 1.4rem;
        padding-left: 1.6rem !important;
        padding-right: 1.6rem !important;
    }
    h1 { font-size: 1.85rem !important; padding-top: 0.1rem !important; }
    .status-banner {
        border-left: 6px solid var(--accent);
        background: var(--bg);
        padding: 0.75rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.35rem 0 0.65rem 0;
    }
    .status-banner h2 {
        margin: 0 0 0.3rem 0;
        font-size: 1.32rem;
        color: var(--accent);
    }
    .status-banner p { margin: 0; color: #222; line-height: 1.42; font-size: 0.98rem; }
    .disclaimer {
        background: #f4f4f2;
        border: 1px solid #d8d8d2;
        border-radius: 8px;
        padding: 0.65rem 1rem;
        font-size: 0.95rem;
        color: #333;
        margin-bottom: 0.75rem;
    }
    .conflict-box {
        background: #f7f1ea;
        border: 1px solid #e0c9b0;
        border-radius: 8px;
        padding: 0.7rem 0.9rem;
        margin: 0.2rem 0 0.4rem 0;
    }
    .conflict-box h4 { margin: 0 0 0.35rem 0; color: #6a3e16; font-size: 1.02rem; }
    .conflict-box p { margin: 0 0 0.3rem 0; line-height: 1.42; }
    .bar-label { font-size: 0.92rem; color: #2c2c2c; margin-bottom: 0.1rem; }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        min-height: 100%;
        padding: 0.45rem 0.85rem 0.55rem 0.85rem;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] h4 { margin-bottom: 0.25rem; }
</style>
"""


def _fmt(value: Optional[float], *, digits: int = 3) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def _score_bar(label: str, value: Optional[float], *, help_text: Optional[str] = None) -> None:
    st.markdown(f"<div class='bar-label'>{label}: <strong>{_fmt(value)}</strong></div>", unsafe_allow_html=True)
    fraction = bar_fraction(value)
    if fraction is None:
        st.progress(0, text="unavailable")
    else:
        st.progress(fraction)
    if help_text:
        st.caption(help_text)


def _render_status(view: DashboardView) -> None:
    accent, background = _STATUS_STYLE.get(view.status, ("#333", "#eee"))
    st.markdown(
        f"""
        <div class="status-banner" style="--accent:{accent};--bg:{background}">
          <h2>Evidence status: {view.status.value}</h2>
          <p>{view.rationale}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_card(card) -> None:
    st.markdown(f"#### {card.title}")
    if card.error:
        st.error(card.error)
        return
    verdict = (card.label or "—").upper()
    st.markdown(f"**Final verdict:** `{verdict}`")
    _score_bar("LM real probability", card.real_score)
    _score_bar("LM fake probability", card.fake_score)
    if card.expert_scores:
        st.markdown("**Specialist detector score(s)**")
        st.caption("Detector fake-likelihood, not an LM token probability.")
        for name, score in card.expert_scores.items():
            _score_bar(name.title(), score)
    else:
        st.caption("No specialist detector score for this configuration.")
    if card.explanation:
        st.caption(f"Model text: {card.explanation}")


def _render_conflicts(view: DashboardView) -> None:
    st.subheader("Evidence conflict")
    if not view.evidence_conflicts:
        st.info(
            "No specialist-versus-language-model conflict noted for this saved run. "
            "Official status above is unchanged."
        )
        return
    body = "".join(f"<p>{note}</p>" for note in view.evidence_conflicts)
    st.markdown(
        f"""
        <div class="conflict-box">
          <h4>Observed specialist / language-model tension</h4>
          {body}
          <p style="margin:0.5rem 0 0 0;font-size:0.9rem;color:#555">
            This note is descriptive only. The official evidence status remains
            <strong>{view.status.value}</strong> as returned by the existing evaluator.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_technical(view: DashboardView) -> None:
    with st.expander("Technical details", expanded=False):
        st.markdown(f"**Quantisation:** {view.quantisation}")
        st.caption(view.nf4_note)
        rows = []
        for card in view.cards:
            rows.append(
                {
                    "configuration": card.title,
                    "run_name": card.run_name,
                    "runtime_s": None if card.runtime_s is None else round(card.runtime_s, 3),
                    "peak_vram_mib": card.peak_vram_mib,
                    "source_json": card.output_path or "—",
                }
            )
        st.dataframe(rows, width="stretch", hide_index=True)
        st.markdown(f"**Matrix directory:** `{view.matrix_dir}`")
        if view.summary_path:
            st.markdown(f"**Summary JSON:** `{view.summary_path}`")


def main() -> None:
    st.set_page_config(
        page_title="X2DFD disagreement dashboard",
        page_icon="◈",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(_PAGE_CSS, unsafe_allow_html=True)

    st.title("Disagreement-aware X2DFD")
    st.caption("Supervisor demo · saved Stage 3 expert-matrix outputs only")

    try:
        view = build_dashboard_view(DEFAULT_MATRIX_DIR, summary_path=DEFAULT_SUMMARY)
    except Exception as exc:
        st.error(f"Could not build the dashboard view: {type(exc).__name__}: {exc}")
        st.stop()

    st.markdown(f'<div class="disclaimer">{view.disclaimer}</div>', unsafe_allow_html=True)

    for message in view.errors:
        st.error(message)
    for message in view.warnings:
        st.warning(message)

    if view.errors and not view.cards:
        st.stop()

    left, right = st.columns([0.92, 2.08], gap="large")
    with left:
        st.subheader("Analysed image")
        if view.image_path is not None:
            st.image(str(view.image_path), width=300)
            st.caption(view.image_path.name)
        else:
            st.error("Analysed image is not available on disk.")
        _render_status(view)
        _render_conflicts(view)

    with right:
        st.subheader("Four expert configurations")
        row1 = st.columns(2, gap="medium")
        row2 = st.columns(2, gap="medium")
        for column, card in zip(list(row1) + list(row2), view.cards):
            with column:
                with st.container(border=True):
                    _render_card(card)

    _render_technical(view)


if __name__ == "__main__":
    main()
