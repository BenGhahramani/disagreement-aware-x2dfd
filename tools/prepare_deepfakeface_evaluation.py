#!/usr/bin/env python3
"""Prepare a DeepFakeFace image-level evaluation subset for labelled evaluation.

Deterministically samples 120 wiki (real) + 40 insight + 40 inpainting +
40 text2img (fake) native images from the DeepFakeFace ZIP archives. Does not
run model inference and does not face-crop (``already_cropped: false``).

Example::

    python -m tools.prepare_deepfakeface_evaluation \\
      --dataset-root "C:/Users/Ben/Desktop/UNI/REIT/DeepFakeFace" \\
      --output-dir datasets/evaluation/deepfakeface_final
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence

from eval.deepfakeface_prepare import (
    DEFAULT_SEED,
    DeepFakeFacePrepareError,
    run_preparation,
)

LOGGER = logging.getLogger("x2dfd.prepare_deepfakeface_evaluation")

EXIT_PASS = 0
EXIT_USAGE = 2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "datasets" / "evaluation" / "deepfakeface_final"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m tools.prepare_deepfakeface_evaluation",
        description=(
            "Build a deterministic DeepFakeFace image subset "
            "(native bytes, not face-cropped) for tools.run_labelled_evaluation."
        ),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Directory containing wiki.zip, insight.zip, inpainting.zip, text2img.zip",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output directory (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Sampling seed")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report selection without writing images or the final manifest",
    )
    parser.add_argument(
        "--hash-zips",
        action="store_true",
        help="SHA-256 the multi-GB source ZIP archives (slow; off by default)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    try:
        result = run_preparation(
            dataset_root=args.dataset_root,
            output_dir=args.output_dir,
            seed=args.seed,
            dry_run=args.dry_run,
            hash_zips=args.hash_zips,
        )
    except DeepFakeFacePrepareError as exc:
        LOGGER.error("%s", exc)
        return EXIT_USAGE
    except OSError as exc:
        LOGGER.error("%s", exc)
        return EXIT_USAGE

    print(json.dumps(result, indent=2))
    if args.dry_run:
        LOGGER.info(
            "dry-run only: wrote %s; no images or final manifest",
            result["written"].get("provenance"),
        )
    else:
        LOGGER.info("wrote outputs under %s", result["output_dir"])
    return EXIT_PASS


if __name__ == "__main__":
    raise SystemExit(main())
