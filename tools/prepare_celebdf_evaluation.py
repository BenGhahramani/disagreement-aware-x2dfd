#!/usr/bin/env python3
"""Prepare a Celeb-DF-v2 image-level evaluation subset for labelled evaluation.

Reads the official Celeb-DF-v2 test list, deterministically samples 60 real +
60 fake videos, and extracts one full-frame PNG per video. Does not run model
inference and does not face-crop (``already_cropped: false``).

Example::

    python -m tools.prepare_celebdf_evaluation \\
      --dataset-root "C:/Users/Ben/Desktop/UNI/REIT" \\
      --output-dir datasets/evaluation/celebdf_v2_final

    python -m tools.prepare_celebdf_evaluation \\
      --dataset-root "C:/Users/Ben/Desktop/UNI/REIT" \\
      --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence

from eval.celebdf_v2_prepare import (
    DEFAULT_N_FAKE,
    DEFAULT_N_REAL,
    DEFAULT_SEED,
    CelebDFPrepareError,
    run_preparation,
)

LOGGER = logging.getLogger("x2dfd.prepare_celebdf_evaluation")

EXIT_PASS = 0
EXIT_USAGE = 2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "datasets" / "evaluation" / "celebdf_v2_final"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m tools.prepare_celebdf_evaluation",
        description=(
            "Build a deterministic Celeb-DF-v2 official-test image subset "
            "(full frames, not face-cropped) for tools.run_labelled_evaluation."
        ),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help=(
            "Extracted Celeb-DF-v2 root containing Celeb-real/, YouTube-real/, "
            "Celeb-synthesis/, and List_of_testing_videos.txt"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output directory (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--test-list",
        type=Path,
        default=None,
        help="Override path to List_of_testing_videos.txt",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Sampling seed")
    parser.add_argument("--n-real", type=int, default=DEFAULT_N_REAL)
    parser.add_argument("--n-fake", type=int, default=DEFAULT_N_FAKE)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report selection without writing frames or the final manifest",
    )
    parser.add_argument(
        "--skip-video-hash",
        action="store_true",
        help="Skip SHA-256 of source videos (faster; frame hashes still computed)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    if args.n_real < 1 or args.n_fake < 1:
        LOGGER.error("--n-real and --n-fake must be >= 1")
        return EXIT_USAGE
    try:
        result = run_preparation(
            dataset_root=args.dataset_root,
            output_dir=args.output_dir,
            seed=args.seed,
            n_real=args.n_real,
            n_fake=args.n_fake,
            dry_run=args.dry_run,
            hash_videos=not args.skip_video_hash,
            test_list_path=args.test_list,
        )
    except CelebDFPrepareError as exc:
        LOGGER.error("%s", exc)
        return EXIT_USAGE
    except OSError as exc:
        LOGGER.error("%s", exc)
        return EXIT_USAGE

    print(json.dumps(result, indent=2))
    if args.dry_run:
        LOGGER.info(
            "dry-run only: wrote %s; no frames or final manifest",
            result["written"].get("provenance"),
        )
    else:
        LOGGER.info("wrote outputs under %s", result["output_dir"])
    return EXIT_PASS


if __name__ == "__main__":
    raise SystemExit(main())
