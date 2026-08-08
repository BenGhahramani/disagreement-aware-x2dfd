#!/usr/bin/env python3
"""Produce a deterministic, square face crop from a portrait photograph.

X2DFD's detectors were trained on DeepfakeBench-style face crops, not full
portraits, so a raw photo under-represents the face and feeds the detectors
mostly background. This makes a crop with a fixed, documented recipe so the same
input always yields byte-comparable output:

1. Downscale a copy to ``--detect-width`` on the long side (INTER_AREA) purely
   for detection, so runtime does not depend on the source resolution.
2. Detect faces with OpenCV's Haar frontal-face cascade at fixed
   ``scaleFactor`` / ``minNeighbors``. Haar is deterministic - no RNG, no
   learned weights beyond the shipped cascade file.
3. Keep the largest detection and map it back to full-resolution coordinates.
4. Expand it by ``--margin`` around its centre, square it off, and clamp to the
   image bounds.
5. Resize to ``--size`` (INTER_AREA) and write JPEG at ``--quality``.

The source image is never modified.

Exit codes:
    0   OK      - a crop was written
    1   FAIL    - no face detected
    2   USAGE   - bad arguments or unreadable image

Example::

    python -m tools.make_face_crop --image datasets/raw/images/poc/real_face_01.jpg \
        --output datasets/raw/images/poc/real_face_01_crop.jpg
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

LOGGER = logging.getLogger("x2dfd.make_face_crop")

DEFAULT_SIZE = 256
DEFAULT_MARGIN = 1.3
DEFAULT_DETECT_WIDTH = 1024
DEFAULT_SCALE_FACTOR = 1.1
DEFAULT_MIN_NEIGHBOURS = 5
DEFAULT_QUALITY = 95
CASCADE_NAME = "haarcascade_frontalface_default.xml"

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_USAGE = 2

Box = Tuple[int, int, int, int]  # x, y, w, h


class FaceCropError(Exception):
    """The crop could not be produced (missing file, decode failure, I/O)."""


class NoFaceFoundError(FaceCropError):
    """Haar cascade found no face in the source image."""


@dataclass(frozen=True)
class CropPlan:
    """The square region chosen for cropping, in source-image pixels."""

    detection: Box
    square: Box

    def as_dict(self) -> dict:
        return {"detection_xywh": list(self.detection), "crop_xywh": list(self.square)}


def largest_box(boxes: Sequence[Box]) -> Optional[Box]:
    if not boxes:
        return None
    return max(boxes, key=lambda b: b[2] * b[3])


def scale_box(box: Box, factor: float) -> Box:
    x, y, w, h = box
    return (int(round(x * factor)), int(round(y * factor)), int(round(w * factor)), int(round(h * factor)))


def square_with_margin(box: Box, margin: float, width: int, height: int) -> Box:
    """Expand a detection into a square, clamped to the image.

    The square is centred on the detection and sized ``margin`` times its longer
    side, then shifted (not shrunk) if it would leave the image, so the face
    stays centred wherever possible.
    """

    x, y, w, h = box
    centre_x = x + w / 2.0
    centre_y = y + h / 2.0
    side = int(round(max(w, h) * margin))
    side = max(1, min(side, width, height))

    left = int(round(centre_x - side / 2.0))
    top = int(round(centre_y - side / 2.0))
    left = max(0, min(left, width - side))
    top = max(0, min(top, height - side))
    return (left, top, side, side)


def detect_faces(
    image_bgr,
    *,
    detect_width: int,
    scale_factor: float,
    min_neighbours: int,
) -> Tuple[List[Box], float]:
    """Run the Haar cascade on a downscaled copy; return boxes and that scale."""

    import cv2

    height, width = image_bgr.shape[:2]
    longest = max(height, width)
    ratio = detect_width / float(longest) if longest > detect_width else 1.0
    if ratio < 1.0:
        small = cv2.resize(
            image_bgr,
            (int(round(width * ratio)), int(round(height * ratio))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        small = image_bgr

    grey = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    grey = cv2.equalizeHist(grey)

    cascade_path = str(Path(cv2.data.haarcascades) / CASCADE_NAME)
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        raise RuntimeError(f"could not load Haar cascade: {cascade_path}")

    detections = cascade.detectMultiScale(
        grey,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbours,
        minSize=(48, 48),
    )
    boxes = [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in detections]
    return boxes, ratio


def plan_crop(
    image_bgr,
    *,
    detect_width: int,
    scale_factor: float,
    min_neighbours: int,
    margin: float,
) -> Optional[CropPlan]:
    height, width = image_bgr.shape[:2]
    boxes, ratio = detect_faces(
        image_bgr,
        detect_width=detect_width,
        scale_factor=scale_factor,
        min_neighbours=min_neighbours,
    )
    best = largest_box(boxes)
    if best is None:
        return None
    full = scale_box(best, 1.0 / ratio) if ratio < 1.0 else best
    return CropPlan(detection=full, square=square_with_margin(full, margin, width, height))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m tools.make_face_crop",
        description="Deterministically crop the largest face from an image.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Exit codes: 0 crop written, 1 no face found, 2 usage error.",
    )
    parser.add_argument("--image", type=Path, required=True, help="source image (never modified)")
    parser.add_argument("--output", type=Path, required=True, help="destination JPEG")
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE, help=f"output edge in px (default: {DEFAULT_SIZE})")
    parser.add_argument(
        "--margin",
        type=float,
        default=DEFAULT_MARGIN,
        help=f"square side as a multiple of the detection (default: {DEFAULT_MARGIN})",
    )
    parser.add_argument(
        "--detect-width",
        type=int,
        default=DEFAULT_DETECT_WIDTH,
        help=f"long edge used for detection (default: {DEFAULT_DETECT_WIDTH})",
    )
    parser.add_argument("--scale-factor", type=float, default=DEFAULT_SCALE_FACTOR)
    parser.add_argument("--min-neighbours", type=int, default=DEFAULT_MIN_NEIGHBOURS)
    parser.add_argument("--quality", type=int, default=DEFAULT_QUALITY, help="JPEG quality")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output file")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if args.size <= 0 or args.margin <= 0 or args.detect_width <= 0:
        parser.error("--size, --margin and --detect-width must be positive")
    return args


def write_face_crop(
    image_path: Path,
    output_path: Path,
    *,
    size: int = DEFAULT_SIZE,
    margin: float = DEFAULT_MARGIN,
    detect_width: int = DEFAULT_DETECT_WIDTH,
    scale_factor: float = DEFAULT_SCALE_FACTOR,
    min_neighbours: int = DEFAULT_MIN_NEIGHBOURS,
    quality: int = DEFAULT_QUALITY,
    overwrite: bool = False,
) -> CropPlan:
    """Write the Stage 3 deterministic 256×256 JPEG face crop.

    Raises:
        NoFaceFoundError: Haar found no face.
        FaceCropError: missing file, undecodable image, or write failure.
    """

    import cv2

    image_path = Path(image_path)
    output_path = Path(output_path)
    if not image_path.is_file():
        raise FaceCropError(f"image not found: {image_path}")
    if output_path.exists() and not overwrite:
        raise FaceCropError(f"output already exists (pass overwrite=True): {output_path}")

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FaceCropError(f"cannot decode image: {image_path}")
    height, width = image.shape[:2]
    LOGGER.info("source: %dx%d px", width, height)

    try:
        plan = plan_crop(
            image,
            detect_width=detect_width,
            scale_factor=scale_factor,
            min_neighbours=min_neighbours,
            margin=margin,
        )
    except RuntimeError as exc:
        raise FaceCropError(str(exc)) from exc

    if plan is None:
        raise NoFaceFoundError(f"no face detected in {image_path}")

    x, y, side, _ = plan.square
    cropped = image[y : y + side, x : x + side]
    resized = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), resized, [int(cv2.IMWRITE_JPEG_QUALITY), quality]):
        raise FaceCropError(f"failed to write {output_path}")
    return plan


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    try:
        plan = write_face_crop(
            args.image,
            args.output,
            size=args.size,
            margin=args.margin,
            detect_width=args.detect_width,
            scale_factor=args.scale_factor,
            min_neighbours=args.min_neighbours,
            quality=args.quality,
            overwrite=args.overwrite,
        )
    except NoFaceFoundError as exc:
        LOGGER.error("%s", exc)
        return EXIT_FAIL
    except FaceCropError as exc:
        LOGGER.error("%s", exc)
        return EXIT_USAGE

    print(f"source        : {args.image}")
    print(f"detection xywh: {plan.detection}")
    print(f"crop xywh     : {plan.square}")
    print(f"output        : {args.output} ({args.size}x{args.size}, JPEG q{args.quality})")
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
