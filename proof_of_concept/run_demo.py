"""CLI entry point for the disagreement-aware POC.

Examples (run from repo root)::

    python -m proof_of_concept.run_demo --all
    python -m proof_of_concept.run_demo --scenario stable
    python -m proof_of_concept.run_demo --scenario-dir /path/to/real_outputs
    python -m proof_of_concept.run_demo --all \\
        --output proof_of_concept/outputs/report.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .evaluator import evaluate
from .normaliser import load_scenario, primary_image
from .report import render
from .schema import ImageComparison


_FIXTURES_ROOT = Path(__file__).resolve().parent / "fixtures"
_DEFAULT_SCENARIOS = ["stable", "contested", "uncertain", "failed"]


def _build_comparison(directory: Path) -> ImageComparison:
    runs = load_scenario(directory)
    image = primary_image(runs, directory)
    status, rationale = evaluate(runs)
    return ImageComparison(image=image, runs=runs, status=status, rationale=rationale)


def _select(args: argparse.Namespace) -> list[tuple[str, Path]]:
    if args.scenario_dir is not None:
        d: Path = args.scenario_dir
        return [(d.name or "custom", d)]
    if args.scenario:
        return [(args.scenario, _FIXTURES_ROOT / args.scenario)]
    out: list[tuple[str, Path]] = []
    for s in _DEFAULT_SCENARIOS:
        d = _FIXTURES_ROOT / s
        if d.exists():
            out.append((s, d))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Disagreement-aware X2DFD comparison POC",
    )
    ap.add_argument(
        "--scenario",
        choices=_DEFAULT_SCENARIOS,
        help="Render a single built-in fixture scenario",
    )
    ap.add_argument(
        "--all",
        action="store_true",
        help="Render all built-in fixture scenarios (default when nothing else specified)",
    )
    ap.add_argument(
        "--scenario-dir",
        type=Path,
        default=None,
        help="Render a custom directory containing demo_*.json files (real or mock)",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write Markdown to this file (default: stdout)",
    )
    args = ap.parse_args()

    selections = _select(args)
    comparisons: list[tuple[str, ImageComparison]] = []
    for name, d in selections:
        if not d.exists():
            print(f"[POC] scenario directory not found: {d}", file=sys.stderr)
            continue
        comparisons.append((name, _build_comparison(d)))

    if not comparisons:
        print("[POC] nothing to render", file=sys.stderr)
        return 1

    md = render(comparisons)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(md, encoding="utf-8")
        print(f"[POC] wrote {args.output}")
    else:
        sys.stdout.write(md)
        if not md.endswith("\n"):
            sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
