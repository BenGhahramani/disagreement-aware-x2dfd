#!/usr/bin/env python3
"""Generate thesis evidence bundle from saved labelled evaluations (CPU only).

Does not run inference, retrain, or rewrite raw scores.

Examples::

    python -m tools.analyse_thesis_evidence \\
      --deepfakeface-aggregate path/to/deepfakeface/aggregate.json \\
      --deepfakeface-validation-protocol path/to/dff/validation_protocol.json \\
      --celebdf-aggregate path/to/celebdf/aggregate.json \\
      --celebdf-validation-protocol path/to/celebdf/validation_protocol.json \\
      --output-dir eval/outputs/thesis_evidence

    # Single dataset
    python -m tools.analyse_thesis_evidence \\
      --dataset-name DeepFakeFace \\
      --labelled-aggregate path/to/aggregate.json \\
      --validation-protocol path/to/validation_protocol.json \\
      --output-dir eval/outputs/thesis_evidence/deepfakeface
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from eval.ground_truth_audit import GroundTruthAuditError
from eval.thesis_evidence import generate_thesis_evidence

LOGGER = logging.getLogger("x2dfd.analyse_thesis_evidence")

EXIT_PASS = 0
EXIT_FAIL = 1

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "eval" / "outputs" / "thesis_evidence"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m tools.analyse_thesis_evidence")
    p.add_argument(
        "--labelled-aggregate",
        type=Path,
        default=None,
        help="single labelled aggregate.json (use with --dataset-name)",
    )
    p.add_argument(
        "--dataset-name",
        default="dataset",
        help="display name for --labelled-aggregate",
    )
    p.add_argument(
        "--validation-protocol",
        type=Path,
        default=None,
        help="optional validation_protocol.json for the single aggregate",
    )
    p.add_argument(
        "--deepfakeface-aggregate",
        type=Path,
        default=None,
        help="DeepFakeFace labelled aggregate.json",
    )
    p.add_argument(
        "--deepfakeface-validation-protocol",
        type=Path,
        default=None,
        help="DeepFakeFace validation_protocol.json",
    )
    p.add_argument(
        "--celebdf-aggregate",
        type=Path,
        default=None,
        help="Celeb-DF-v2 labelled aggregate.json",
    )
    p.add_argument(
        "--celebdf-validation-protocol",
        type=Path,
        default=None,
        help="Celeb-DF-v2 validation_protocol.json",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"thesis evidence root (default: {DEFAULT_OUTPUT})",
    )
    p.add_argument(
        "--no-figures",
        action="store_true",
        help="skip matplotlib figure generation",
    )
    p.add_argument(
        "--no-ecdf",
        action="store_true",
        help="skip optional ECDF figures",
    )
    return p.parse_args(argv)


def _build_dataset_specs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    if args.deepfakeface_aggregate is not None:
        specs.append(
            {
                "name": "DeepFakeFace",
                "aggregate": args.deepfakeface_aggregate,
                "validation_protocol": args.deepfakeface_validation_protocol,
            }
        )
    if args.celebdf_aggregate is not None:
        specs.append(
            {
                "name": "Celeb-DF-v2",
                "aggregate": args.celebdf_aggregate,
                "validation_protocol": args.celebdf_validation_protocol,
            }
        )
    if args.labelled_aggregate is not None:
        specs.append(
            {
                "name": args.dataset_name,
                "aggregate": args.labelled_aggregate,
                "validation_protocol": args.validation_protocol,
            }
        )
    return specs


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    specs = _build_dataset_specs(args)
    if not specs:
        LOGGER.error(
            "provide --labelled-aggregate and/or "
            "--deepfakeface-aggregate / --celebdf-aggregate"
        )
        return EXIT_FAIL
    try:
        result = generate_thesis_evidence(
            specs,
            args.output_dir,
            write_figures=not args.no_figures,
            write_ecdf=not args.no_ecdf,
        )
    except (GroundTruthAuditError, OSError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return EXIT_FAIL

    LOGGER.info(
        "thesis evidence written to %s (%d audit rows, %d cases)",
        result["output_dir"],
        result["n_audit_rows"],
        result["n_cases"],
    )
    for note in result.get("unavailable_information") or []:
        LOGGER.warning("unavailable: %s", note)
    LOGGER.info("manifest: %s", result["written"].get("thesis_evidence_manifest.json"))
    return EXIT_PASS


if __name__ == "__main__":
    sys.exit(main())
