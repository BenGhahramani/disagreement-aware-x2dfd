"""Rule-based evidence-status evaluator for the POC comparison layer.

The evaluator looks at four X2DFD runs of the same image, one per expert
setting (``none`` / ``blending`` / ``diffusion`` / ``blending_diffusion``),
and assigns one of four statuses with a short rationale.

Why a small rule layer (and not a learned ensemble)?
    For a thesis proof-of-concept the value is **auditability**: a supervisor
    must be able to read the rules in one screen and challenge each threshold.
    Replacing this with a learned aggregator is a follow-up question, not a
    POC-week question.

Status definitions:
    Stable               All four runs agree on the verdict, and the combined
                         (blending+diffusion) run's confidence is at least
                         ``CONF_HIGH``. The verdict is robust to which experts
                         are present.

    Contested            At least two runs disagree on real/fake. The rationale
                         additionally notes *which* comparison flipped:
                           - bare LLaVA vs combined experts ("expert override"),
                           - blending alone vs diffusion alone ("expert split").

    Uncertain            All runs agree on the same verdict, but the combined
                         run's confidence is below ``CONF_HIGH``. Treat as
                         borderline; a supervisor should look at the image.

    Failed/insufficient  The combined run errored, has no scores, or two or
                         more of the four runs failed/missing. The disagreement
                         comparison cannot be honestly made.
"""
from __future__ import annotations

from typing import Optional

from .schema import RunRecord, Status


# Thresholds are module-level constants so they can be overridden by a caller
# (e.g. a dashboard) without editing logic. Rationale strings reference
# ``CONF_HIGH`` so changing it stays self-documenting in the report.
#
# CONF_HIGH: minimum combined-experts confidence to call the verdict "Stable".
# CONF_LOW : reserved for a future "Stable (weak)" bucket between [LOW, HIGH);
#            currently the rules collapse that band into "Uncertain" to keep
#            the four-bucket UI simple.
CONF_HIGH: float = 0.70
CONF_LOW: float = 0.55


def _has_error(r: Optional[RunRecord]) -> bool:
    """A run is unusable if it is missing entirely or carries an error string."""
    return r is None or r.error is not None


def _non_none_predictions(runs: dict[str, RunRecord]) -> list[tuple[str, str]]:
    """Collect ``(run_name, prediction)`` for runs that produced a verdict."""
    return [(name, r.prediction) for name, r in runs.items() if r.prediction is not None]


def evaluate(runs: dict[str, RunRecord]) -> tuple[Status, str]:
    """Classify a comparison across runs and return ``(status, rationale)``.

    Decision order (first match wins, by design — see comments at each rule):
        1. Failed/insufficient
        2. Contested
        3. Stable
        4. Uncertain (default fall-through when all runs agree)
    """
    both = runs.get("blending_diffusion")  # the production-style verdict
    none = runs.get("none")                # bare LLaVA, no expert tail
    blend = runs.get("blending")           # blending expert only
    diff = runs.get("diffusion")           # diffusion expert only

    n_errors = sum(1 for r in runs.values() if _has_error(r))

    # ------------------------------------------------------------------
    # Rule 1: Failed/insufficient
    #
    # We refuse to compute a comparison status if the headline run
    # (blending_diffusion) cannot give a verdict, OR if half or more of the
    # individual runs are unusable. This protects against silently calling
    # something "Stable" when most of the evidence is actually missing.
    # ------------------------------------------------------------------
    if both is None or _has_error(both) or both.confidence is None or n_errors >= 2:
        missing = [
            name for name, r in runs.items()
            if r is None or _has_error(r) or r.confidence is None
        ]
        rationale = (
            f"Insufficient signal: {n_errors}/{len(runs)} runs failed and/or "
            f"required scores are missing for {missing}."
        )
        return Status.FAILED, rationale

    preds = _non_none_predictions(runs)
    distinct = sorted({p for _, p in preds})

    # ------------------------------------------------------------------
    # Rule 2: Contested
    #
    # If two valid runs reach different real/fake verdicts, we surface that
    # immediately — even if the combined run is highly confident. A confident
    # but contested verdict is exactly the case the disagreement layer exists
    # to highlight to the supervisor / end user.
    #
    # The rationale also flags *which* comparison flipped, so a downstream
    # dashboard can render a one-line "why" without re-deriving it.
    # ------------------------------------------------------------------
    if len(distinct) >= 2:
        per_run = ", ".join(f"{name}={pred}" for name, pred in preds)
        flippers: list[str] = []
        if (
            none is not None and none.prediction
            and none.prediction != both.prediction
        ):
            flippers.append("adding experts flips bare-LLaVA verdict")
        if (
            blend is not None and diff is not None
            and blend.prediction and diff.prediction
            and blend.prediction != diff.prediction
        ):
            flippers.append("blending and diffusion experts disagree")
        flip_note = f" Sources of disagreement: {'; '.join(flippers)}." if flippers else ""
        return Status.CONTESTED, f"Disagreement across runs ({per_run}).{flip_note}"

    # All non-None predictions agree from here on.
    agreed = distinct[0] if distinct else "?"

    # ------------------------------------------------------------------
    # Rule 3: Stable
    #
    # All runs agree AND the combined-experts confidence clears CONF_HIGH.
    # We anchor on the combined run because that is the X2DFD configuration
    # the upstream paper recommends; the other three are diagnostic.
    # ------------------------------------------------------------------
    if both.confidence >= CONF_HIGH:
        return Status.STABLE, (
            f"All four runs agree on '{agreed}'; combined-experts confidence "
            f"{both.confidence:.2f} >= {CONF_HIGH:.2f}."
        )

    # ------------------------------------------------------------------
    # Rule 4: Uncertain (default for "agreement but low confidence")
    #
    # Predictions match across runs but the combined-experts probability sits
    # below CONF_HIGH. Useful as a "human review needed" flag in a dashboard.
    # ------------------------------------------------------------------
    return Status.UNCERTAIN, (
        f"All four runs agree on '{agreed}' but combined-experts confidence "
        f"{both.confidence:.2f} is below {CONF_HIGH:.2f}; treat as borderline."
    )
