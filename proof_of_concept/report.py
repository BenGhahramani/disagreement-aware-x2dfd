"""Render an :class:`ImageComparison` (or several) to a Markdown report."""
from __future__ import annotations

from typing import Iterable, Optional

from .schema import ImageComparison, RunRecord


_DISCLAIMER = (
    "> **Proof-of-concept notice.** This report is produced by a "
    "*disagreement-aware comparison layer* built on top of the existing X2DFD "
    "pipeline. It is **not a new deepfake detector** and does not modify any "
    "X2DFD model code. It re-uses the same image, same model, same experts — "
    "the only thing it adds is running each image under several expert settings "
    "and applying a small rule-based evaluator over the resulting JSONs to "
    "surface agreement, disagreement, and low-confidence cases."
)

_THESIS_FOOTER = (
    "## How this supports the thesis\n\n"
    "The thesis investigates **explainable** deepfake detection. A single "
    "verdict + rationale from one MLLM run is hard to trust on its own. By "
    "running the same image under multiple expert configurations and "
    "comparing the results, this layer gives a supervisor / end user three "
    "extra pieces of information that an isolated X2DFD call cannot:\n\n"
    "1. **Robustness** — does the verdict survive removing or changing experts?\n"
    "2. **Source attribution** — when verdicts differ, *which* expert flipped it?\n"
    "3. **Calibration cue** — is the confidence high enough to act on, or "
    "should the case be flagged for human review?\n\n"
    "The Markdown report above is intentionally simple so each rule can be "
    "audited; a richer dashboard view is a follow-up, not a POC requirement."
)

_RUN_ORDER: tuple[str, ...] = ("none", "blending", "diffusion", "blending_diffusion")


def _fmt(v: Optional[float], *, prec: int = 3) -> str:
    if v is None:
        return "—"
    return f"{v:.{prec}f}"


def _truncate(text: str, n: int = 40) -> str:
    if len(text) <= n:
        return text
    return text[: n - 1] + "…"


def _sanitise_cell(text: str) -> str:
    """Make ``text`` safe to embed in a single Markdown table cell.

    Replaces pipe characters and any line breaks so a runner-emitted error
    message cannot accidentally close the column or split the row.
    """
    return text.replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def _row(r: RunRecord) -> str:
    experts = ",".join(r.experts_used) if r.experts_used else "—"
    pred = r.prediction if r.prediction else "—"
    if r.error:
        err = _sanitise_cell(_truncate(r.error, 40))
    else:
        err = "—"
    return (
        f"| `{r.run_name}` | {experts} | {pred} | "
        f"{_fmt(r.real_score)} | {_fmt(r.fake_score)} | {_fmt(r.confidence)} | "
        f"{_fmt(r.runtime_seconds, prec=2)} | {err} |"
    )


def _scenario_block(name: str, comp: ImageComparison) -> str:
    header = f"## Scenario: `{name}`\n\n**Image:** `{comp.image}`\n"

    table_lines = [
        "| run | experts | prediction | real | fake | confidence | runtime (s) | error |",
        "|-----|---------|------------|------|------|------------|-------------|-------|",
    ]
    for k in _RUN_ORDER:
        if k in comp.runs:
            table_lines.append(_row(comp.runs[k]))
    table = "\n".join(table_lines)

    status_line = (
        f"\n\n**Evidence status:** `{comp.status.value}`\n\n"
        f"**Rationale:** {comp.rationale}\n"
    )

    excerpt_lines = ["", "**Per-run answer excerpts:**", ""]
    for k in _RUN_ORDER:
        r = comp.runs.get(k)
        if r is None:
            continue
        text = r.explanation or "(no answer)"
        if len(text) > 200:
            text = text[:200] + "…"
        excerpt_lines.append(f"- `{k}`: {text}")
    excerpts = "\n".join(excerpt_lines)

    return "\n".join([header, table, status_line, excerpts, ""])


def render(comparisons: Iterable[tuple[str, ImageComparison]]) -> str:
    """Render a Markdown document covering all provided scenarios."""
    parts: list[str] = [
        "# Disagreement-aware X2DFD comparison — POC report",
        "",
        _DISCLAIMER,
        "",
    ]
    for name, comp in comparisons:
        parts.append(_scenario_block(name, comp))
    parts.append(_THESIS_FOOTER)
    parts.append("")
    return "\n".join(parts)
