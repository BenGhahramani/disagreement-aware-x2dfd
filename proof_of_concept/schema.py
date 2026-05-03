"""Typed records used across the POC comparison layer.

These are intentionally small and self-contained so the POC stays portable and
does not depend on anything from `src/`, `train/`, or `utils/`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Status(str, Enum):
    """Four-bucket evidence status for a comparison across runs."""

    STABLE = "Stable"
    CONTESTED = "Contested"
    UNCERTAIN = "Uncertain"
    FAILED = "Failed/insufficient"


@dataclass
class RunRecord:
    """Normalised view of a single X2DFD inference run for one image."""

    run_name: str
    experts_used: list[str]
    prediction: Optional[str]            # "real" | "fake" | None
    real_score: Optional[float]
    fake_score: Optional[float]
    confidence: Optional[float]          # max(real, fake) when both available
    explanation: str                     # free-text answer from the model
    runtime_seconds: Optional[float]
    error: Optional[str]


@dataclass
class ImageComparison:
    """Aggregate comparison for one image across multiple runs."""

    image: str
    runs: dict[str, RunRecord] = field(default_factory=dict)
    status: Status = Status.FAILED
    rationale: str = ""
