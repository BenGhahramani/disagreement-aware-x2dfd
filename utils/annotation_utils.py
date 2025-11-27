"""Small helpers for annotation formatting and merging.

This module centralizes how we:
- Build the binary classification question for the human turn
- Prefix the model explanation with a simple real/fake statement
"""

from __future__ import annotations

from typing import Literal, Optional, List, Tuple


BinaryLabel = Literal["real", "fake"]


def build_binary_question() -> str:
    """Return the human question used for binary classification."""
    return "Is this image real or fake?"


def build_blending_prompt(alias: str, score: Optional[float]) -> str:
    """Return the fixed human prompt with optional blending score.

    Template:
      "<image>\nIs this image real or fake? And by observation of {alias} expert, the blending score is {score|N/A}."

    - alias: readable name for the provider (e.g., "Blending")
    - score: None to render as N/A; otherwise formatted to 3 decimals
    """
    score_str = "N/A" if score is None else f"{float(score):.3f}"
    return (
        f"<image>\nIs this image real or fake? And by observation of {alias} expert, "
        f"the blending score is {score_str}."
    )


def normalize_label(label: str) -> BinaryLabel:
    """Normalize arbitrary label strings to 'real' or 'fake' (default fake)."""
    lowered = (label or "").strip().lower()
    if lowered == "real":
        return "real"  # type: ignore[return-value]
    return "fake"  # type: ignore[return-value]


def compose_labeled_response(label: str, explanation: str) -> str:
    """Prefix the model explanation with a binary statement.

    - If label == 'real' -> "This image is real. {explanation}"
    - Else -> "This image is fake. {explanation}"
    """
    norm = normalize_label(label)
    prefix = "This image is real." if norm == "real" else "This image is fake."
    if explanation:
        return f"{prefix} {explanation}"
    return prefix


# ---- Multi-expert prompt helpers (non-breaking additions) ----

def format_score(score: Optional[float]) -> str:
    """Format an optional float score to 3 decimals or 'N/A'."""
    if score is None:
        return "N/A"
    try:
        return f"{float(score):.3f}"
    except Exception:
        return "N/A"


def expert_sentence(alias: str, score: Optional[float]) -> str:
    """One expert-tail sentence appended to the binary question.

    Example: " And by observation of Blending expert, the Blending score is 0.812."
    """
    return (
        f" And by observation of {alias} expert, the {alias} score is {format_score(score)}."
    )


def build_multi_expert_prompt(experts: List[Tuple[str, Optional[float]]]) -> str:
    """Compose the base binary question plus a sentence per expert.

    experts: list of (alias, score)
    """
    base = "<image>\nIs this image real or fake?"
    tail = "".join(expert_sentence(a, s) for a, s in experts)
    return base + tail
