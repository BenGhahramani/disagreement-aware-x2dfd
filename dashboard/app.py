"""Streamlit dashboard: saved Stage 3 demo (default) plus optional live analysis.

Launch from the repository root::

    .venv\\Scripts\\python.exe -m streamlit run dashboard/app.py
"""
from __future__ import annotations

import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st

from dashboard.live_analysis import LiveAnalysisResult
from dashboard.live_controller import (
    persist_upload,
    progress_fraction,
    progress_label,
    run_live_analysis,
    structured_error_messages,
    unexpected_error_message,
)
from dashboard.saved_examples import (
    SAVED_EXAMPLES,
    default_saved_example_id,
    load_saved_example,
    resolve_saved_example,
)
from dashboard.view_model import (
    DashboardView,
    bar_fraction,
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
    /* Streamlit's fixed header (~3.75rem) sits over the main block; offset
       content without hiding the header. rem tracks user font scaling. */
    [data-testid="stMainBlockContainer"],
    .block-container {
        max-width: 100% !important;
        padding-top: calc(1.05rem + 2.75rem) !important;
        padding-bottom: 1.4rem;
        padding-left: 1.6rem !important;
        padding-right: 1.6rem !important;
    }
    h1 { font-size: 1.85rem !important; margin-top: 0 !important; }
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
    quant = Decimal("1").scaleb(-digits)
    rounded = Decimal(str(value)).quantize(quant, rounding=ROUND_HALF_UP)
    return f"{rounded:.{digits}f}"


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
            "No specialist-versus-language-model conflict noted for this run. "
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


def render_dashboard_view(view: DashboardView) -> None:
    """Shared result UI for the saved Stage 3 example and a live run."""

    st.markdown(f'<div class="disclaimer">{view.disclaimer}</div>', unsafe_allow_html=True)

    for message in view.errors:
        st.error(message)
    for message in view.warnings:
        st.warning(message)

    if view.errors and not view.cards:
        return

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


def _render_saved_example() -> None:
    labels = [spec.selector_label for spec in SAVED_EXAMPLES]
    ids = [spec.example_id for spec in SAVED_EXAMPLES]
    default_index = ids.index(default_saved_example_id()) if default_saved_example_id() in ids else 0
    choice = st.radio(
        "Saved demonstration",
        labels,
        index=default_index,
        horizontal=True,
        help="Pre-computed expert-matrix outputs. Switching examples does not run GPU inference.",
    )
    spec = resolve_saved_example(ids[labels.index(choice)])
    st.markdown(
        f"""
        <div class="disclaimer" style="margin-top:0.35rem">
          <strong>{spec.ground_truth_label}.</strong> {spec.provenance_summary}
        </div>
        """,
        unsafe_allow_html=True,
    )
    try:
        view = load_saved_example(spec.example_id)
    except Exception as exc:
        st.error(unexpected_error_message(exc))
        return
    render_dashboard_view(view)


def _render_live_errors(result: LiveAnalysisResult) -> None:
    for message in structured_error_messages(result):
        st.error(message)


def _render_live_tab() -> None:
    st.info(
        "A complete live analysis runs the real 4-bit X2DFD pipeline on the GPU "
        "(four sequential expert configurations) and typically takes **1–2 minutes**. "
        "Nothing starts until you click **Analyse image**."
    )

    uploaded = st.file_uploader(
        "Upload a JPEG or PNG face photograph",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
    )
    if uploaded is None:
        result = st.session_state.get("live_result")
        if isinstance(result, LiveAnalysisResult):
            _render_live_errors(result)
            if result.view is not None:
                render_dashboard_view(result.view)
            if result.work_dir is not None:
                st.caption(f"Results preserved at `{result.work_dir}`")
        return

    file_id = f"{uploaded.name}:{uploaded.size}"
    if st.session_state.get("live_file_id") != file_id:
        st.session_state.live_file_id = file_id
        st.session_state.pop("live_result", None)

    data = uploaded.getvalue()
    st.subheader("Upload preview")
    st.image(data, width=300)
    st.caption(uploaded.name)

    analyse = st.button("Analyse image", type="primary")
    if analyse:
        st.session_state.pop("live_result", None)
        progress_bar = st.progress(0.0, text="Starting…")
        status = st.empty()

        def on_progress(event) -> None:
            progress_bar.progress(progress_fraction(event), text=progress_label(event))
            status.caption(event.detail or event.message)

        try:
            saved_path = persist_upload(data, uploaded.name)
            result = run_live_analysis(saved_path, progress_callback=on_progress)
        except Exception as exc:
            st.session_state.live_result = None
            st.error(unexpected_error_message(exc))
            return
        st.session_state.live_result = result

    result = st.session_state.get("live_result")
    if not isinstance(result, LiveAnalysisResult):
        return

    _render_live_errors(result)
    if result.view is not None:
        render_dashboard_view(result.view)
    if result.work_dir is not None:
        st.caption(f"Results preserved at `{result.work_dir}`")


def main() -> None:
    st.set_page_config(
        page_title="X2DFD disagreement dashboard",
        page_icon="◈",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(_PAGE_CSS, unsafe_allow_html=True)

    st.title("Disagreement-aware X2DFD")
    st.caption(
        "Default view is the saved supervisor demo (two labelled examples). "
        "Live analysis is optional and does not replace those outputs."
    )

    saved_tab, live_tab = st.tabs(["Saved example", "Analyse new image"])
    with saved_tab:
        _render_saved_example()
    with live_tab:
        _render_live_tab()


if __name__ == "__main__":
    main()
