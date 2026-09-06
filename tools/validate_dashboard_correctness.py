#!/usr/bin/env python3
"""Run dashboard correctness / robustness checks (CPU only, no inference).

Writes a machine-readable summary under ``eval/outputs/`` by default (gitignored).

Examples::

    python -m tools.validate_dashboard_correctness
    python -m tools.validate_dashboard_correctness --output eval/outputs/dashboard_validation_summary.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence

from eval.dashboard_correctness import PROJECT_ROOT, run_all_checks

LOGGER = logging.getLogger("x2dfd.validate_dashboard_correctness")

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_USAGE = 2

DEFAULT_OUTPUT = (
    PROJECT_ROOT / "eval" / "outputs" / "dashboard_validation_summary.json"
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m tools.validate_dashboard_correctness",
        description=(
            "Validate dashboard score fidelity, semantics, provenance, "
            "agreement/status logic, saved/live consistency, and malformed-output "
            "handling. Does not load models or use a GPU."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"JSON summary path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--scratch-dir",
        type=Path,
        default=None,
        help="scratch directory for malformed-output fixtures",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    scratch = args.scratch_dir
    if scratch is None:
        scratch = args.output.parent / "_dashboard_correctness_scratch"
    report = run_all_checks(scratch_dir=scratch)
    payload = report.as_dict()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    LOGGER.info(
        "dashboard validation: total=%s passed=%s failed=%s skipped=%s",
        payload["total_checks"],
        payload["passed"],
        payload["failed"],
        payload["skipped"],
    )
    LOGGER.info("wrote %s", args.output.resolve())
    if payload["failed"]:
        for check in payload["checks"]:
            if not check["passed"] and not check.get("skipped"):
                LOGGER.error("FAIL %s/%s: %s", check["category"], check["name"], check["detail"])
        return EXIT_FAIL
    return EXIT_PASS


if __name__ == "__main__":
    raise SystemExit(main())
