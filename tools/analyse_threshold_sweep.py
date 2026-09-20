#!/usr/bin/env python3
"""Run threshold-sweep analysis on saved evaluation scores (CPU only).

Does not run inference. Operates on raw ``fake_score`` fields already stored
in labelled aggregates or FF++ score files.

Examples::

    # Full-set descriptive sweep (one config)
    python -m tools.analyse_threshold_sweep \\
      --labelled-aggregate path/to/aggregate.json \\
      --run-name blending_diffusion \\
      --output-dir eval/outputs/threshold_sweep/deepfakeface

    # Validation→select / held-out→evaluate protocol (all four configs)
    python -m tools.analyse_threshold_sweep \\
      --labelled-aggregate path/to/aggregate.json \\
      --validation-protocol \\
      --output-dir eval/outputs/threshold_sweep/deepfakeface_valtest

    # FF++ frame scores
    python -m tools.analyse_threshold_sweep \\
      --ffpp-prep-dir datasets/evaluation/ffpp_c23_source \\
      --ffpp-scores path/to/frame_scores.jsonl \\
      --ffpp-level frame \\
      --output-dir eval/outputs/threshold_sweep/ffpp_frame
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional, Sequence

from eval.experiment_configs import PRIMARY_ASSESSMENT_RUN, RUN_ORDER
from eval.threshold_sweep import (
    DEFAULT_SPLIT_SEED,
    DEFAULT_VAL_FRACTION,
    REFERENCE_THRESHOLD,
    SELECTION_CRITERIA,
    ThresholdSweepError,
    load_ffpp_frame_rows,
    load_samples_by_run_from_labelled_aggregate,
    load_samples_from_ffpp_frames,
    load_samples_from_labelled_aggregate,
    run_threshold_sweep,
    run_validation_test_protocol,
    write_threshold_outputs,
    write_validation_protocol_outputs,
)

LOGGER = logging.getLogger("x2dfd.analyse_threshold_sweep")

EXIT_PASS = 0
EXIT_USAGE = 2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "eval" / "outputs" / "threshold_sweep"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m tools.analyse_threshold_sweep")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--labelled-aggregate",
        type=Path,
        help="labelled-evaluation aggregate.json",
    )
    src.add_argument(
        "--ffpp-merged-frames",
        type=Path,
        help="FF++ analysis merged_frames.json",
    )
    src.add_argument(
        "--ffpp-prep-dir",
        type=Path,
        help="FF++ preparation dir (requires --ffpp-scores)",
    )
    p.add_argument("--ffpp-scores", type=Path, default=None, help="frame_scores.jsonl")
    p.add_argument(
        "--ffpp-level",
        choices=("frame", "video"),
        default="frame",
        help="FF++ sample unit (default: frame)",
    )
    p.add_argument(
        "--run-name",
        default=PRIMARY_ASSESSMENT_RUN,
        choices=list(RUN_ORDER),
        help=f"labelled aggregate config cell for full-set sweep "
        f"(default: {PRIMARY_ASSESSMENT_RUN})",
    )
    p.add_argument(
        "--validation-protocol",
        action="store_true",
        help=(
            "Stratified val/test protocol: select thresholds on validation only, "
            "evaluate frozen thresholds on held-out test for all four configs"
        ),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SPLIT_SEED,
        help=f"split seed (default: {DEFAULT_SPLIT_SEED})",
    )
    p.add_argument(
        "--val-fraction",
        type=float,
        default=DEFAULT_VAL_FRACTION,
        help=f"validation fraction per class (default: {DEFAULT_VAL_FRACTION})",
    )
    p.add_argument(
        "--criteria",
        nargs="+",
        choices=list(SELECTION_CRITERIA),
        default=list(SELECTION_CRITERIA),
        help="validation selection criteria",
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--step", type=float, default=0.01)
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    try:
        if args.validation_protocol:
            if args.labelled_aggregate is None:
                raise ThresholdSweepError(
                    "--validation-protocol requires --labelled-aggregate"
                )
            payload = json.loads(args.labelled_aggregate.read_text(encoding="utf-8"))
            samples_by_run, meta = load_samples_by_run_from_labelled_aggregate(payload)
            meta["input_path"] = str(args.labelled_aggregate.resolve())
            result = run_validation_test_protocol(
                samples_by_run,
                seed=args.seed,
                val_fraction=args.val_fraction,
                criteria=args.criteria,
                step=args.step,
            )
            written = write_validation_protocol_outputs(
                result, args.output_dir, meta=meta
            )
            print(
                json.dumps(
                    {
                        "mode": "validation_protocol",
                        "output_dir": str(Path(args.output_dir).resolve()),
                        "seed": args.seed,
                        "val_fraction": args.val_fraction,
                        "n_comparison_rows": len(result["comparison_table"]),
                        "split": {
                            "n_val": result["split"]["n_val"],
                            "n_test": result["split"]["n_test"],
                        },
                        "score_semantics": result["score_semantics"],
                        "written": written,
                    },
                    indent=2,
                )
            )
            return EXIT_PASS

        if args.labelled_aggregate is not None:
            payload = json.loads(args.labelled_aggregate.read_text(encoding="utf-8"))
            samples, meta = load_samples_from_labelled_aggregate(
                payload, run_name=args.run_name
            )
            meta["input_path"] = str(args.labelled_aggregate.resolve())
        elif args.ffpp_merged_frames is not None:
            frames = load_ffpp_frame_rows(merged_frames_path=args.ffpp_merged_frames)
            samples, meta = load_samples_from_ffpp_frames(frames, level=args.ffpp_level)
            meta["input_path"] = str(args.ffpp_merged_frames.resolve())
        else:
            if args.ffpp_scores is None:
                raise ThresholdSweepError("--ffpp-scores is required with --ffpp-prep-dir")
            frames = load_ffpp_frame_rows(
                prep_dir=args.ffpp_prep_dir, scores_path=args.ffpp_scores
            )
            samples, meta = load_samples_from_ffpp_frames(frames, level=args.ffpp_level)
            meta["prep_dir"] = str(args.ffpp_prep_dir.resolve())
            meta["scores_path"] = str(args.ffpp_scores.resolve())

        result = run_threshold_sweep(samples, step=args.step)
        written = write_threshold_outputs(
            result,
            args.output_dir,
            meta=meta,
            write_plots=not args.no_plots,
        )
    except (OSError, json.JSONDecodeError, ThresholdSweepError) as exc:
        LOGGER.error("%s", exc)
        return EXIT_USAGE

    ref = next(
        (
            r
            for r in result["thresholds"]
            if abs(r["threshold"] - REFERENCE_THRESHOLD) < 1e-12
        ),
        None,
    )
    print(
        json.dumps(
            {
                "mode": "full_set_sweep",
                "output_dir": str(Path(args.output_dir).resolve()),
                "n_samples": result["n_samples"],
                "roc_auc": result["roc"].get("roc_auc"),
                "reference_0_50": ref,
                "operating_points": {
                    k: (v.get("threshold") if isinstance(v, dict) else v)
                    for k, v in result["operating_points"].items()
                    if k != "note"
                },
                "written": written,
            },
            indent=2,
        )
    )
    return EXIT_PASS


if __name__ == "__main__":
    raise SystemExit(main())
