"""Saved demonstration examples for the dashboard Saved example tab.

Loads pre-computed expert-matrix outputs only — never runs inference.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

from dashboard.view_model import (
    DEFAULT_MATRIX_DIR,
    DEFAULT_SUMMARY,
    DashboardView,
    PROJECT_ROOT,
    build_dashboard_view,
)


@dataclass(frozen=True)
class SavedExampleSpec:
    """One labelled saved run the supervisor can switch between."""

    example_id: str
    selector_label: str
    matrix_dir: Path
    summary_path: Path
    ground_truth_label: str
    provenance_summary: str


FFPP_LIVE_RUN = (
    PROJECT_ROOT / "eval" / "outputs" / "live_analysis" / "20260808T085812Z_ffpp_ex_deepfakes"
)

SAVED_EXAMPLES: Tuple[SavedExampleSpec, ...] = (
    SavedExampleSpec(
        example_id="nasa",
        selector_label="Known authentic — NASA portrait",
        matrix_dir=DEFAULT_MATRIX_DIR,
        summary_path=DEFAULT_SUMMARY,
        ground_truth_label="Known authentic",
        provenance_summary=(
            "Official NASA portrait of astronaut Jonny Kim (Johnson Space Center, "
            "photo jsc2024e052605_alt). Stage 3 saved outputs: "
            "`eval/outputs/expert_matrix/demo_one_crop/`."
        ),
    ),
    SavedExampleSpec(
        example_id="ffpp_deepfakes",
        selector_label="Known manipulated — FaceForensics++ Deepfakes",
        matrix_dir=FFPP_LIVE_RUN / "matrix",
        summary_path=FFPP_LIVE_RUN / "summary.json",
        ground_truth_label="Known manipulated",
        provenance_summary=(
            "FaceForensics++ official repository Deepfakes example "
            "(`images/ex_deepfakes.png`, faceswap autoencoder + Poisson edit). "
            "Saved live-analysis outputs: "
            "`eval/outputs/live_analysis/20260808T085812Z_ffpp_ex_deepfakes/`."
        ),
    ),
)

_DEFAULT_EXAMPLE_ID = SAVED_EXAMPLES[0].example_id


def saved_example_ids() -> Tuple[str, ...]:
    return tuple(spec.example_id for spec in SAVED_EXAMPLES)


def saved_example_labels() -> Tuple[str, ...]:
    return tuple(spec.selector_label for spec in SAVED_EXAMPLES)


def saved_examples_by_id() -> Dict[str, SavedExampleSpec]:
    return {spec.example_id: spec for spec in SAVED_EXAMPLES}


def resolve_saved_example(
    example_id: Optional[str] = None,
    *,
    examples: Sequence[SavedExampleSpec] = SAVED_EXAMPLES,
) -> SavedExampleSpec:
    """Return the spec for ``example_id``, or the default NASA example."""

    if example_id is None:
        return examples[0]
    lookup = {spec.example_id: spec for spec in examples}
    if example_id not in lookup:
        raise KeyError(f"unknown saved example: {example_id!r}")
    return lookup[example_id]


def load_saved_example(
    example_id: Optional[str] = None,
    *,
    examples: Sequence[SavedExampleSpec] = SAVED_EXAMPLES,
) -> DashboardView:
    """Build a :class:`DashboardView` from saved matrix outputs (no inference)."""

    spec = resolve_saved_example(example_id, examples=examples)
    return build_dashboard_view(spec.matrix_dir, summary_path=spec.summary_path)


def default_saved_example_id() -> str:
    return _DEFAULT_EXAMPLE_ID
