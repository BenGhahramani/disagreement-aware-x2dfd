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
    CALIBRATION_NOTE,
    EVIDENCE_BAND_NOTE,
    EVIDENCE_PROVENANCE,
    LABEL_MODEL_FAKE,
    LABEL_MODEL_REAL,
    THRESHOLD_NOTE,
    DashboardView,
    bar_fraction,
    detector_score_label,
)
from dashboard.operating_settings import (
    ENV_VALIDATION_PROTOCOL,
    PRESET_BALANCED,
    PRESET_NAMES,
    THRESHOLD_SOURCE_CUSTOM,
    THRESHOLD_SOURCE_REFERENCE,
    THRESHOLD_SOURCE_VALIDATION,
    build_operating_settings_state,
    config_display_label,
    format_rate_percent,
    load_validation_protocol,
)
from eval.experiment_configs import PRIMARY_ASSESSMENT_RUN, RUN_ORDER
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
    .assessment-panel {
        background: #f7f8f6;
        border: 1px solid #d9ddd4;
        border-radius: 8px;
        padding: 0.7rem 0.9rem;
        margin: 0.25rem 0 0.55rem 0;
    }
    .assessment-panel h4 {
        margin: 0 0 0.25rem 0;
        font-size: 0.92rem;
        color: #555;
        font-weight: 600;
        text-transform: none;
    }
    .assessment-panel .value {
        margin: 0 0 0.65rem 0;
        font-size: 1.05rem;
        color: #1a1a1a;
        line-height: 1.35;
    }
    .assessment-panel .value:last-child { margin-bottom: 0; }
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


def _render_status(view: DashboardView, *, operating_state=None) -> None:
    accent, background = _STATUS_STYLE.get(view.status, ("#333", "#eee"))
    agreement = view.evidence_agreement
    agreement_label = (
        agreement.display_label if hasattr(agreement, "display_label") else str(agreement)
    )
    prediction_html = ""
    fake_score_html = ""
    if operating_state is not None:
        pred = operating_state.prediction
        prediction_html = (
            f"<h4>Current decision</h4>"
            f'<p class="value">{pred.current_decision_word}</p>'
        )
        fake = pred.raw_fake_score
        fake_txt = "unavailable" if fake is None else f"{fake:.3f}"
        fake_score_html = (
            f"<h4>Raw model fake score</h4>"
            f'<p class="value">{fake_txt}</p>'
        )
    st.markdown(
        f"""
        <div class="assessment-panel">
          {prediction_html}
          {fake_score_html}
          <h4>Evidence agreement</h4>
          <p class="value">{agreement_label}</p>
          <h4>Prototype evidence status</h4>
          <p class="value">{view.status.value}</p>
        </div>
        <div class="status-banner" style="--accent:{accent};--bg:{background}">
          <h2>Prototype evidence status: {view.status.value}</h2>
          <p>{view.rationale}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if operating_state is not None:
        st.caption(operating_state.evidence_separation_note)
    else:
        st.caption(
            "Detection threshold changes the real/manipulated decision. "
            "Evidence agreement describes whether the available detectors agree."
        )
    st.caption(view.threshold_note or THRESHOLD_NOTE)
    st.caption(view.evidence_band_note or EVIDENCE_BAND_NOTE)


def _render_card(card, *, highlight: bool = False, predicted_label: Optional[str] = None, fake_score: Optional[float] = None) -> None:
    title = card.title
    if highlight:
        st.markdown(f"#### {title} · active")
    else:
        st.markdown(f"#### {title}")
    if card.error:
        st.error(card.error)
        return
    if predicted_label is not None:
        word = "MANIPULATED" if predicted_label == "fake" else (
            "REAL" if predicted_label == "real" else "—"
        )
        st.markdown(f"**Current decision:** `{word}`")
        shown_fake = fake_score if fake_score is not None else card.fake_score
        if shown_fake is None:
            st.caption("Raw model fake score: unavailable")
        else:
            st.caption(f"Raw model fake score: {shown_fake:.3f}")
    else:
        st.caption("Not the active configuration.")
    st.caption("Source: X2-DFD / LLaVA")
    _score_bar(LABEL_MODEL_REAL, card.real_score)
    _score_bar(LABEL_MODEL_FAKE, card.fake_score)
    if card.expert_scores:
        st.markdown("**Specialist detector score(s)**")
        st.caption("Detector scores (not model token scores).")
        for name, score in card.expert_scores.items():
            _score_bar(detector_score_label(name), score)
    else:
        st.caption("No specialist detector score for this configuration.")
    if card.explanation:
        st.caption(f"Model text: {card.explanation}")


def _init_operating_session(view: DashboardView) -> None:
    """Initialise session keys for operating settings once per matrix view."""

    matrix_key = str(view.matrix_dir)
    if st.session_state.get("_ops_matrix_key") != matrix_key:
        st.session_state._ops_matrix_key = matrix_key
        st.session_state.ops_run_name = PRIMARY_ASSESSMENT_RUN
        st.session_state.ops_threshold = 0.50
        st.session_state.ops_threshold_source = THRESHOLD_SOURCE_REFERENCE
        st.session_state.ops_preset = None


def _render_operating_settings(view: DashboardView):
    """Focused Detection settings panel (no inference)."""

    _init_operating_session(view)

    st.subheader("Detection settings")
    st.caption(
        "User-adjustable operating settings for already-saved raw scores. "
        "Changing these controls does not rerun inference."
    )

    protocol_default = st.session_state.get("ops_validation_path", "") or ""
    env_hint = f"Env override: `{ENV_VALIDATION_PROTOCOL}`"
    protocol_path = st.text_input(
        "Validation protocol path (optional)",
        value=protocol_default,
        help=(
            "Path to validation_protocol.json from the threshold-sweep protocol. "
            f"{env_hint}. Missing/malformed files are ignored safely."
        ),
        key="ops_validation_path_input",
    )
    st.session_state.ops_validation_path = protocol_path.strip()
    validation = load_validation_protocol(
        protocol_path.strip() or None,
    )
    if validation.error:
        st.caption(f"Validation metadata: {validation.error}")
    elif validation.available:
        st.caption(
            f"Validation metadata loaded"
            + (f" · {validation.dataset_name}" if validation.dataset_name else "")
            + f" · `{validation.path}`"
        )
    else:
        st.caption(
            "No validation metadata loaded. Presets stay unavailable until a "
            f"validation_protocol.json path is set (or {ENV_VALIDATION_PROTOCOL})."
        )

    available = []
    unavailable = []
    for card in view.cards:
        if card.error or card.fake_score is None:
            unavailable.append(card.run_name)
        else:
            available.append(card.run_name)
    # Keep RUN_ORDER for any missing cards.
    for name in RUN_ORDER:
        if name not in available and name not in unavailable:
            unavailable.append(name)

    if not available:
        st.warning("No usable expert-configuration scores are available for threshold control.")
        state = build_operating_settings_state(
            cards=view.cards,
            active_run_name=PRIMARY_ASSESSMENT_RUN,
            decision_threshold=float(st.session_state.get("ops_threshold", 0.50)),
            threshold_source=st.session_state.get(
                "ops_threshold_source", THRESHOLD_SOURCE_REFERENCE
            ),
            validation=validation,
        )
        return state

    current_run = st.session_state.get("ops_run_name", PRIMARY_ASSESSMENT_RUN)
    if current_run not in available:
        current_run = available[0]
        st.session_state.ops_run_name = current_run

    labels = [config_display_label(name) for name in available]
    label_to_run = dict(zip(labels, available))
    choice = st.selectbox(
        "Active expert configuration",
        labels,
        index=labels.index(config_display_label(current_run)),
        help="Switches which saved config cell is displayed. Does not run inference.",
    )
    selected_run = label_to_run[choice]
    if selected_run != st.session_state.ops_run_name:
        st.session_state.ops_run_name = selected_run
        state_probe = build_operating_settings_state(
            cards=view.cards,
            active_run_name=selected_run,
            threshold_source=THRESHOLD_SOURCE_VALIDATION,
            validation=validation,
        )
        balanced = state_probe.presets.get(PRESET_BALANCED)
        if state_probe.presets_available and balanced is not None:
            st.session_state.ops_threshold = float(balanced)
            st.session_state.ops_threshold_source = THRESHOLD_SOURCE_VALIDATION
            st.session_state.ops_preset = PRESET_BALANCED
        else:
            st.session_state.ops_threshold = 0.50
            st.session_state.ops_threshold_source = THRESHOLD_SOURCE_REFERENCE
            st.session_state.ops_preset = None

    if unavailable:
        st.caption(
            "Unavailable configurations (missing or failed scores): "
            + ", ".join(config_display_label(n) for n in unavailable)
        )

    state = build_operating_settings_state(
        cards=view.cards,
        active_run_name=st.session_state.ops_run_name,
        decision_threshold=float(st.session_state.get("ops_threshold", 0.50)),
        threshold_source=st.session_state.get(
            "ops_threshold_source", THRESHOLD_SOURCE_REFERENCE
        ),
        validation=validation,
    )

    st.markdown(f"**Threshold source:** {state.threshold_source_label}")
    st.caption(state.raw_score_note)
    st.caption(state.tradeoff_note)

    preset_cols = st.columns(len(PRESET_NAMES))
    if state.presets_available:
        for col, name in zip(preset_cols, PRESET_NAMES):
            info = state.preset_details.get(name)
            thr = None if info is None else info.threshold
            with col:
                if thr is None:
                    st.button(name, disabled=True, key=f"ops_preset_missing_{name}")
                else:
                    selected = (
                        st.session_state.get("ops_preset") == name
                        or (
                            state.matched_preset == name
                            and abs(float(st.session_state.get("ops_threshold", 0.5)) - thr) <= 1e-9
                        )
                    )
                    label = f"{name} — {thr:.2f}"
                    if selected:
                        label = f"● {label}"
                    if st.button(
                        label,
                        key=f"ops_preset_{name}",
                        help=info.user_copy if info is not None else "",
                    ):
                        st.session_state.ops_threshold = float(thr)
                        st.session_state.ops_threshold_source = THRESHOLD_SOURCE_VALIDATION
                        st.session_state.ops_preset = name
                        st.rerun()
                    if info is not None:
                        st.caption(info.user_copy)
                        st.caption(
                            f"Fake sensitivity: {format_rate_percent(info.validation_sensitivity_fake)}"
                        )
                        st.caption(
                            f"Real specificity: {format_rate_percent(info.validation_specificity_real)}"
                        )
        if state.presets_collapsed and state.collapse_note:
            st.info(state.collapse_note)
        if state.unexpected_order and state.order_note:
            st.caption(state.order_note)
    else:
        for col, name in zip(preset_cols, PRESET_NAMES):
            with col:
                st.button(name, disabled=True, key=f"ops_preset_disabled_{name}")
        if state.presets_unavailable_reason:
            st.caption(state.presets_unavailable_reason)

    new_threshold = st.slider(
        "Decision threshold (raw fake_score)",
        min_value=0.0,
        max_value=1.0,
        value=float(st.session_state.get("ops_threshold", 0.50)),
        step=0.01,
        help="predict fake if fake_score >= threshold. Raw scores are unchanged.",
    )
    if abs(new_threshold - float(st.session_state.get("ops_threshold", 0.50))) > 1e-12:
        st.session_state.ops_threshold = float(new_threshold)
        matched_now = None
        if state.presets_available:
            for name, info in state.preset_details.items():
                if info.threshold is not None and abs(info.threshold - new_threshold) <= 1e-9:
                    matched_now = name
                    break
        st.session_state.ops_threshold_source = THRESHOLD_SOURCE_CUSTOM
        st.session_state.ops_preset = matched_now
        st.rerun()

    # Rebuild after possible slider/preset updates already applied via session.
    state = build_operating_settings_state(
        cards=view.cards,
        active_run_name=st.session_state.ops_run_name,
        decision_threshold=float(st.session_state.ops_threshold),
        threshold_source=st.session_state.ops_threshold_source,
        validation=validation,
    )
    st.markdown(f"**Active decision threshold:** `{state.decision_threshold:.2f}`")
    st.caption(state.lower_threshold_copy)
    st.caption(state.higher_threshold_copy)

    st.markdown("**Threshold provenance**")
    for line in state.provenance_lines:
        st.caption(line)

    if state.presets_available:
        with st.expander("Preset technical details", expanded=False):
            dataset = (
                state.validation.dataset_name
                if state.validation is not None
                else None
            )
            for name in PRESET_NAMES:
                info = state.preset_details.get(name)
                if info is None:
                    continue
                st.markdown(f"**{name}**")
                st.caption(f"Raw threshold: {info.threshold if info.threshold is not None else 'unavailable'}")
                if dataset:
                    st.caption(f"Validation dataset: {dataset}")
                st.caption(f"Validation criterion / target: {info.target or info.provenance}")
                st.caption(
                    f"Validation fake sensitivity: {format_rate_percent(info.validation_sensitivity_fake)}"
                )
                st.caption(
                    f"Validation real specificity: {format_rate_percent(info.validation_specificity_real)}"
                )
                if info.held_out_balanced_accuracy is not None:
                    st.caption(
                        f"Held-out balanced accuracy (evaluation only, not used to choose the preset): "
                        f"{info.held_out_balanced_accuracy:.3f}"
                    )
            if state.validation and state.validation.path:
                st.caption(f"Metadata source: {state.validation.path}")
            st.caption(state.raw_score_note)

    # Keep view.decision_operating_settings aligned with interactive state.
    view.decision_operating_settings = state.decision_operating_settings
    return state


def _render_conflicts(view: DashboardView) -> None:
    st.subheader("Evidence conflict detail")
    if not view.evidence_conflicts:
        st.info(
            "No specialist-versus-model conflict detail for this run. "
            "See Evidence agreement above; official prototype status is unchanged."
        )
        return
    body = "".join(f"<p>{note}</p>" for note in view.evidence_conflicts)
    st.markdown(
        f"""
        <div class="conflict-box">
          <h4>Observed specialist / model tension</h4>
          {body}
          <p style="margin:0.5rem 0 0 0;font-size:0.9rem;color:#555">
            This note is descriptive only. Evidence agreement is a separate axis from
            the prototype evidence status
            (<strong>{view.status.value}</strong>) returned by the existing evaluator.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_provenance(view: DashboardView) -> None:
    items = view.evidence_provenance or list(EVIDENCE_PROVENANCE)
    lines = "".join(f"<li>{item}</li>" for item in items)
    st.markdown(
        f"""
        <div class="disclaimer">
          <strong>Evidence provenance</strong>
          <ul style="margin:0.35rem 0 0.35rem 1.1rem;padding:0">{lines}</ul>
          <p style="margin:0;font-size:0.92rem;color:#444">{view.calibration_note or CALIBRATION_NOTE}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_technical(view: DashboardView, *, operating_state=None) -> None:
    with st.expander("Technical details", expanded=False):
        st.markdown(f"**Quantisation:** {view.quantisation}")
        st.caption(view.nf4_note)
        st.caption(view.threshold_note or THRESHOLD_NOTE)
        settings = (
            operating_state.decision_operating_settings
            if operating_state is not None
            else (view.decision_operating_settings or {})
        )
        if settings or operating_state is not None:
            st.markdown("**Decision operating settings** (user-adjustable; not calibrated)")
            if operating_state is not None:
                st.caption(f"Active config: {operating_state.active_config_label}")
                st.caption(
                    f"Decision threshold: {operating_state.decision_threshold:.2f} "
                    f"({operating_state.threshold_source_label})"
                )
                st.caption(operating_state.prediction.display_text)
                st.caption(
                    f"Raw model fake score: "
                    f"{'unavailable' if operating_state.prediction.raw_fake_score is None else f'{operating_state.prediction.raw_fake_score:.3f}'}"
                )
                st.caption(operating_state.prediction.saved_decision_provenance)
                st.caption(operating_state.raw_score_note)
                st.caption(operating_state.evidence_separation_note)
                if operating_state.matched_preset:
                    info = operating_state.preset_details.get(operating_state.matched_preset)
                    st.markdown(f"**Matching preset:** {operating_state.matched_preset}")
                    if info is not None:
                        st.caption(f"Raw threshold: {info.threshold}")
                        if operating_state.validation and operating_state.validation.dataset_name:
                            st.caption(f"Validation dataset: {operating_state.validation.dataset_name}")
                        st.caption(f"Validation criterion / target: {info.target or info.provenance}")
                        st.caption(
                            f"Validation fake sensitivity: {format_rate_percent(info.validation_sensitivity_fake)}"
                        )
                        st.caption(
                            f"Validation real specificity: {format_rate_percent(info.validation_specificity_real)}"
                        )
                        if info.held_out_balanced_accuracy is not None:
                            st.caption(
                                f"Held-out balanced accuracy (evaluation only): "
                                f"{info.held_out_balanced_accuracy:.3f}"
                            )
                if operating_state.validation and operating_state.validation.path:
                    st.caption(f"Metadata source: {operating_state.validation.path}")
            else:
                st.caption(
                    f"Active decision threshold on raw fake_score: "
                    f"{settings.get('decision_threshold', '—')}"
                )
                st.caption(
                    f"Active expert configuration: "
                    f"{settings.get('expert_configuration', '—')}"
                )
            if settings.get("wording_lower_threshold"):
                st.caption(settings["wording_lower_threshold"])
            if settings.get("wording_higher_threshold"):
                st.caption(settings["wording_higher_threshold"])
            if settings.get("wording_not_confidence"):
                st.caption(settings["wording_not_confidence"])
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
    _render_provenance(view)

    for message in view.errors:
        st.error(message)
    for message in view.warnings:
        st.warning(message)

    if view.errors and not view.cards:
        return

    operating_state = _render_operating_settings(view)

    left, right = st.columns([0.92, 2.08], gap="large")
    with left:
        st.subheader("Analysed image")
        if view.image_path is not None:
            st.image(str(view.image_path), width=300)
            st.caption(view.image_path.name)
        else:
            st.error("Analysed image is not available on disk.")
        _render_status(view, operating_state=operating_state)
        _render_conflicts(view)

    with right:
        st.subheader("Four expert configurations")
        active_run = (
            operating_state.active_run_name if operating_state else PRIMARY_ASSESSMENT_RUN
        )
        active_pred = (
            operating_state.prediction.predicted_label if operating_state else None
        )
        row1 = st.columns(2, gap="medium")
        row2 = st.columns(2, gap="medium")
        for column, card in zip(list(row1) + list(row2), view.cards):
            with column:
                with st.container(border=True):
                    _render_card(
                        card,
                        highlight=card.run_name == active_run,
                        predicted_label=active_pred if card.run_name == active_run else None,
                        fake_score=(
                            operating_state.prediction.raw_fake_score
                            if operating_state and card.run_name == active_run
                            else None
                        ),
                    )

    _render_technical(view, operating_state=operating_state)


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
