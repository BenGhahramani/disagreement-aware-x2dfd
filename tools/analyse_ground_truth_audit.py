#!/usr/bin/env python3
"""Ground-truth audit CLI over saved labelled aggregates (CPU only).

Does not run inference. Writes audit tables and representative cases.

Examples::

    python -m tools.analyse_ground_truth_audit \\
      --labelled-aggregate path/to/aggregate.json \\
      --run-name blending \\
      --validation-protocol path/to/validation_protocol.json \\
      --output-dir eval/outputs/ground_truth_audit/deepfakeface_blending

    python -m tools.analyse_ground_truth_audit \\
      --labelled-aggregate path/to/aggregate.json \\
      --all-runs \\
      --threshold 0.78 \\
      --output-dir eval/outputs/ground_truth_audit/celebdf
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence

from eval.experiment_configs import PRIMARY_ASSESSMENT_RUN, RUN_ORDER
from eval.ground_truth_audit import (
    GroundTruthAuditError,
    build_audit_for_runs,
    build_audit_rows,
    load_aggregate_json,
    load_validation_protocol_payload,
    write_ground_truth_audit_outputs,
)

LOGGER = logging.getLogger("x2dfd.analyse_ground_truth_audit")

EXIT_PASS = 0
EXIT_USAGE = 2
EXIT_FAIL = 1

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "eval" / "outputs" / "ground_truth_audit"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m tools.analyse_ground_truth_audit")
    p.add_argument(
        "--labelled-aggregate",
        type=Path,
        required=True,
        help="labelled-evaluation aggregate.json",
    )
    p.add_argument(
        "--run-name",
        default=PRIMARY_ASSESSMENT_RUN,
        choices=list(RUN_ORDER),
        help=f"single configuration (default: {PRIMARY_ASSESSMENT_RUN})",
    )
    p.add_argument(
        "--all-runs",
        action="store_true",
        help="audit all four expert configurations",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="explicit active decision threshold (overrides validation Balanced)",
    )
    p.add_argument(
        "--validation-protocol",
        type=Path,
        default=None,
        help="optional validation_protocol.json for Balanced threshold metadata",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"output directory (default: {DEFAULT_OUTPUT})",
    )
    p.add_argument(
        "--no-representative",
        action="store_true",
        help="skip representative case artefacts",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    try:
        payload = load_aggregate_json(args.labelled_aggregate)
        protocol, protocol_error = load_validation_protocol_payload(args.validation_protocol)
        if protocol_error:
            LOGGER.warning("%s", protocol_error)

        if args.all_runs:
            thresholds = None
            if args.threshold is not None:
                thresholds = {name: float(args.threshold) for name in RUN_ORDER}
            rows, meta = build_audit_for_runs(
                payload,
                run_names=RUN_ORDER,
                thresholds_by_run=thresholds,
                protocol=protocol,
            )
        else:
            rows, run_meta = build_audit_rows(
                payload,
                run_name=args.run_name,
                active_threshold=args.threshold,
                protocol=protocol,
            )
            meta = {"per_run": {args.run_name: run_meta}, "n_rows": len(rows)}

        written = write_ground_truth_audit_outputs(
            rows,
            args.output_dir,
            meta={
                "aggregate": str(args.labelled_aggregate.resolve()),
                "validation_protocol": (
                    str(args.validation_protocol.resolve())
                    if args.validation_protocol
                    else None
                ),
                "validation_protocol_error": protocol_error,
                **meta,
            },
            write_representative=not args.no_representative,
        )
    except (GroundTruthAuditError, OSError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return EXIT_FAIL

    LOGGER.info("wrote %d artefact(s) to %s", len(written), args.output_dir)
    for key, path in written.items():
        LOGGER.info("  %s -> %s", key, path)
    return EXIT_PASS


if __name__ == "__main__":
    sys.exit(main())
