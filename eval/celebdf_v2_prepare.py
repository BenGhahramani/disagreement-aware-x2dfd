"""Celeb-DF-v2 image-level evaluation subset preparation (no inference).

Reads the official ``List_of_testing_videos.txt``, deterministically samples
videos, and extracts one full-frame PNG per selected video for the labelled
evaluation pipeline. Face cropping remains the responsibility of
``tools.run_labelled_evaluation``.
"""
from __future__ import annotations

import csv
import json
import logging
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from eval.reproducibility import get_git_provenance, sha256_file, utc_timestamp

LOGGER = logging.getLogger("x2dfd.celebdf_v2_prepare")

SCRIPT_VERSION = "1.1.1"
DEFAULT_SEED = 20260305
DEFAULT_N_REAL = 60
DEFAULT_N_FAKE = 60
DEFAULT_TEST_LIST_NAME = "List_of_testing_videos.txt"
DATASET_NAME = "Celeb-DF-v2"
OFFICIAL_SPLIT = "official_test"

# Official label codes in List_of_testing_videos.txt (Celeb-DF authors).
LABEL_REAL = 1
LABEL_FAKE = 0

REAL_DIRS = frozenset({"Celeb-real", "YouTube-real"})
FAKE_DIRS = frozenset({"Celeb-synthesis"})

VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}

# Label-blind usable-frame policy (fractions of video duration, evaluated in order).
CANDIDATE_FRACTIONS: Tuple[float, ...] = (0.50, 0.40, 0.60, 0.30, 0.70)

# Variance of Laplacian on the full-frame grayscale image. Frames below this
# threshold are treated as severely blurred (OpenCV focus-measure convention).
#
# Calibrated on Celeb-DF-v2 official-test videos: compressed talking-head frames
# often sit well below the still-photo heuristic of 100. With threshold 100 the
# full fake test pool (~340) yielded only ~56 usable videos, so 60+60 was
# impossible. 25 rejects clearly soft/blurred frames while remaining achievable
# on this corpus (documented constant — not tuned per label or model score).
MIN_LAPLACIAN_VARIANCE = 25.0

# Largest Haar face must sit at least this fraction of min(H, W) inside each edge.
FACE_EDGE_MARGIN_FRAC = 0.02

SELECTION_ALGORITHM = (
    "1) Parse official List_of_testing_videos.txt lines as '<label> <relpath>'. "
    "2) Keep only entries whose video file exists under the dataset root. "
    "3) Classify by official label (1=real, 0=fake), cross-checked against the "
    "top-level directory (Celeb-real|YouTube-real vs Celeb-synthesis). "
    "4) Sort eligible videos within each class by POSIX relative path. "
    "5) Draw without replacement via random.Random(seed).sample(sorted_list, k) "
    "to form an initial candidate set (label used only for class quotas). "
    "6) For each selected video, evaluate full frames at fixed duration fractions "
    f"{list(CANDIDATE_FRACTIONS)} in that order (label-blind). "
    "A candidate is usable only if it decodes, Haar finds ≥1 face "
    "(tools.make_face_crop defaults), the largest face is sufficiently inside "
    f"image bounds (edge margin {FACE_EDGE_MARGIN_FRAC:.0%} of min side), and "
    f"full-frame Laplacian variance ≥ {MIN_LAPLACIAN_VARIANCE}. "
    "Save the FULL FRAME (no evaluation crop). "
    "7) If no candidate passes, mark the video unsuitable and deterministically "
    "replace it with the next unused video from the same class (sorted by path). "
    "Never silently emit fewer than the requested class counts. "
    "Frame choice never uses ground-truth beyond class quota, nor X2DFD / "
    "blending / diffusion scores."
)


class CelebDFPrepareError(ValueError):
    """Dataset layout, test list, sampling, or frame extraction failed."""


@dataclass(frozen=True)
class CelebDFTestVideo:
    """One official-test-split video."""

    relative_path: str  # POSIX path relative to dataset root
    ground_truth: str  # real | fake
    label_code: int
    absolute_path: Path

    @property
    def stem_id(self) -> str:
        """Stable id fragment from relative path (no extension)."""

        stem = self.relative_path.rsplit(".", 1)[0]
        return re.sub(r"[^A-Za-z0-9._-]+", "_", stem.replace("/", "__"))


@dataclass
class CandidateEvaluation:
    """One timestamp-fraction trial for usable-frame selection."""

    fraction: float
    frame_index: Optional[int]
    decoded: bool
    usable: bool
    rejection_reason: Optional[str]
    blur_laplacian_var: Optional[float] = None
    face_count: Optional[int] = None
    face_detection_xywh: Optional[List[int]] = None
    face_inside_bounds: Optional[bool] = None
    width: Optional[int] = None
    height: Optional[int] = None
    timestamp_s: Optional[float] = None


@dataclass
class FrameExtraction:
    """Chosen full frame after label-blind usability screening."""

    frame_index: int
    total_frames: int
    fps: Optional[float]
    timestamp_s: Optional[float]
    width: int
    height: int
    candidate_fraction: float
    candidates: List[CandidateEvaluation] = field(default_factory=list)
    blur_laplacian_var: Optional[float] = None
    face_detection_xywh: Optional[List[int]] = None

    @property
    def used_fallback(self) -> bool:
        """True when a non-primary (not 50%) candidate was chosen."""

        return abs(self.candidate_fraction - CANDIDATE_FRACTIONS[0]) > 1e-12


@dataclass
class PreparedSample:
    image_id: str
    video: CelebDFTestVideo
    frame_rel_path: str
    frame_abs_path: Optional[Path]
    extraction: Optional[FrameExtraction]
    video_sha256: Optional[str]
    frame_sha256: Optional[str]
    notes: str
    replaced_from: Optional[str] = None
    unsuitable_attempts: List[Dict[str, Any]] = field(default_factory=list)


class UnusableVideoError(CelebDFPrepareError):
    """No candidate fraction yielded a usable frame for this video."""

    def __init__(self, message: str, *, candidates: Optional[List[CandidateEvaluation]] = None):
        super().__init__(message)
        self.candidates = candidates or []



def locate_test_list(dataset_root: Path) -> Path:
    """Find ``List_of_testing_videos.txt`` under the dataset root."""

    root = Path(dataset_root)
    candidates = [
        root / DEFAULT_TEST_LIST_NAME,
        root / "Celeb-DF" / DEFAULT_TEST_LIST_NAME,
        root / "Celeb-DF-v2" / DEFAULT_TEST_LIST_NAME,
    ]
    for path in candidates:
        if path.is_file():
            return path.resolve()
    # Shallow search (depth 2) without executing anything.
    for child in sorted(root.iterdir()) if root.is_dir() else []:
        if child.is_file() and child.name == DEFAULT_TEST_LIST_NAME:
            return child.resolve()
        if child.is_dir():
            nested = child / DEFAULT_TEST_LIST_NAME
            if nested.is_file():
                return nested.resolve()
    raise CelebDFPrepareError(
        f"official test list {DEFAULT_TEST_LIST_NAME!r} not found under {root}"
    )


def _normalise_relpath(raw: str) -> str:
    text = raw.strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return text


def classify_from_relpath(relpath: str) -> Optional[str]:
    """Return 'real' / 'fake' from the official top-level directory, else None."""

    top = _normalise_relpath(relpath).split("/", 1)[0]
    if top in REAL_DIRS:
        return "real"
    if top in FAKE_DIRS:
        return "fake"
    return None


def parse_test_list(
    test_list_path: Path,
    dataset_root: Path,
    *,
    require_files: bool = True,
) -> List[CelebDFTestVideo]:
    """Parse the official Celeb-DF-v2 test list into video records."""

    try:
        lines = test_list_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise CelebDFPrepareError(f"cannot read test list {test_list_path}: {exc}") from exc

    root = Path(dataset_root)
    videos: List[CelebDFTestVideo] = []
    seen: Dict[str, int] = {}
    for line_no, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise CelebDFPrepareError(
                f"{test_list_path.name}:{line_no}: expected '<label> <relpath>', got {raw!r}"
            )
        label_raw, rel_raw = parts
        try:
            label_code = int(label_raw)
        except ValueError as exc:
            raise CelebDFPrepareError(
                f"{test_list_path.name}:{line_no}: label must be integer, got {label_raw!r}"
            ) from exc
        if label_code not in {LABEL_REAL, LABEL_FAKE}:
            raise CelebDFPrepareError(
                f"{test_list_path.name}:{line_no}: label must be 0 or 1, got {label_code}"
            )
        rel = _normalise_relpath(rel_raw)
        if Path(rel).suffix.lower() not in VIDEO_SUFFIXES:
            raise CelebDFPrepareError(
                f"{test_list_path.name}:{line_no}: unsupported video suffix in {rel!r}"
            )
        gt_from_label = "real" if label_code == LABEL_REAL else "fake"
        gt_from_dir = classify_from_relpath(rel)
        if gt_from_dir is None:
            raise CelebDFPrepareError(
                f"{test_list_path.name}:{line_no}: path not under known Celeb-DF dirs: {rel}"
            )
        if gt_from_dir != gt_from_label:
            raise CelebDFPrepareError(
                f"{test_list_path.name}:{line_no}: label {label_code} conflicts with "
                f"directory class {gt_from_dir!r} for {rel}"
            )
        if rel in seen:
            raise CelebDFPrepareError(
                f"duplicate video in test list: {rel} "
                f"(lines {seen[rel]} and {line_no})"
            )
        seen[rel] = line_no
        abs_path = (root / rel).resolve()
        if require_files and not abs_path.is_file():
            raise CelebDFPrepareError(f"test-list video missing on disk: {abs_path}")
        videos.append(
            CelebDFTestVideo(
                relative_path=rel,
                ground_truth=gt_from_label,
                label_code=label_code,
                absolute_path=abs_path,
            )
        )
    if not videos:
        raise CelebDFPrepareError(f"no videos parsed from {test_list_path}")
    return videos


def select_balanced_videos(
    videos: Sequence[CelebDFTestVideo],
    *,
    n_real: int = DEFAULT_N_REAL,
    n_fake: int = DEFAULT_N_FAKE,
    seed: int = DEFAULT_SEED,
) -> Tuple[List[CelebDFTestVideo], Dict[str, Any]]:
    """Deterministically sample ``n_real`` + ``n_fake`` videos (no duplicates)."""

    reals = sorted(
        (v for v in videos if v.ground_truth == "real"),
        key=lambda v: v.relative_path,
    )
    fakes = sorted(
        (v for v in videos if v.ground_truth == "fake"),
        key=lambda v: v.relative_path,
    )
    if len(reals) < n_real:
        raise CelebDFPrepareError(
            f"need {n_real} real test videos, only {len(reals)} eligible"
        )
    if len(fakes) < n_fake:
        raise CelebDFPrepareError(
            f"need {n_fake} fake test videos, only {len(fakes)} eligible"
        )

    rng = random.Random(seed)
    chosen_real = rng.sample(reals, n_real)
    chosen_fake = rng.sample(fakes, n_fake)
    # Keep class blocks sorted for stable manifest ordering within each class.
    chosen_real = sorted(chosen_real, key=lambda v: v.relative_path)
    chosen_fake = sorted(chosen_fake, key=lambda v: v.relative_path)
    selected = chosen_real + chosen_fake

    ids = [v.relative_path for v in selected]
    if len(ids) != len(set(ids)):
        raise CelebDFPrepareError("duplicate video selected (internal error)")

    meta = {
        "seed": seed,
        "n_real_requested": n_real,
        "n_fake_requested": n_fake,
        "n_real_eligible": len(reals),
        "n_fake_eligible": len(fakes),
        "n_total_eligible": len(videos),
        "selected_real": [v.relative_path for v in chosen_real],
        "selected_fake": [v.relative_path for v in chosen_fake],
        "selection_algorithm": SELECTION_ALGORITHM,
    }
    return selected, meta


def candidate_frame_indices(total_frames: int) -> List[Tuple[float, int]]:
    """Map each candidate fraction to a frame index (order preserved)."""

    if total_frames <= 0:
        raise CelebDFPrepareError("video has no frames")
    last = total_frames - 1
    pairs: List[Tuple[float, int]] = []
    seen_idx: set[int] = set()
    for frac in CANDIDATE_FRACTIONS:
        index = int(round(last * frac))
        index = max(0, min(last, index))
        if index in seen_idx:
            # Extremely short videos may collide; keep first occurrence only.
            continue
        seen_idx.add(index)
        pairs.append((frac, index))
    return pairs


def face_sufficiently_inside(
    box: Tuple[int, int, int, int],
    width: int,
    height: int,
    *,
    margin_frac: float = FACE_EDGE_MARGIN_FRAC,
) -> bool:
    """True when the detection box is inset from every image edge."""

    x, y, w, h = box
    if w <= 0 or h <= 0 or width <= 0 or height <= 0:
        return False
    margin = margin_frac * float(min(width, height))
    return (
        x >= margin
        and y >= margin
        and (x + w) <= (width - margin)
        and (y + h) <= (height - margin)
    )


def laplacian_variance(frame_bgr: Any) -> float:
    """Deterministic blur metric: variance of Laplacian on grayscale."""

    import cv2  # type: ignore

    grey = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(grey, cv2.CV_64F).var())


def evaluate_frame_usability(
    frame_bgr: Any,
    *,
    min_laplacian_var: float = MIN_LAPLACIAN_VARIANCE,
    edge_margin_frac: float = FACE_EDGE_MARGIN_FRAC,
    plan_crop_fn: Optional[Callable[..., Any]] = None,
) -> Dict[str, Any]:
    """Label-blind usability checks for one decoded BGR frame.

    Does not crop for evaluation; only uses Haar planning to decide whether a
    face exists and is sufficiently inside the frame.
    """

    from tools.make_face_crop import (
        DEFAULT_DETECT_WIDTH,
        DEFAULT_MARGIN,
        DEFAULT_MIN_NEIGHBOURS,
        DEFAULT_SCALE_FACTOR,
        plan_crop,
    )

    height, width = int(frame_bgr.shape[0]), int(frame_bgr.shape[1])
    crop_fn = plan_crop_fn or plan_crop
    plan = crop_fn(
        frame_bgr,
        detect_width=DEFAULT_DETECT_WIDTH,
        scale_factor=DEFAULT_SCALE_FACTOR,
        min_neighbours=DEFAULT_MIN_NEIGHBOURS,
        margin=DEFAULT_MARGIN,
    )
    if plan is None:
        return {
            "usable": False,
            "rejection_reason": "no_face",
            "blur_laplacian_var": None,
            "face_count": 0,
            "face_detection_xywh": None,
            "face_inside_bounds": None,
            "width": width,
            "height": height,
        }

    detection = tuple(int(v) for v in plan.detection)
    inside = face_sufficiently_inside(
        detection, width, height, margin_frac=edge_margin_frac
    )
    if not inside:
        return {
            "usable": False,
            "rejection_reason": "face_near_border",
            "blur_laplacian_var": None,
            "face_count": 1,
            "face_detection_xywh": list(detection),
            "face_inside_bounds": False,
            "width": width,
            "height": height,
        }

    blur = laplacian_variance(frame_bgr)
    if blur < min_laplacian_var:
        return {
            "usable": False,
            "rejection_reason": "severe_blur",
            "blur_laplacian_var": blur,
            "face_count": 1,
            "face_detection_xywh": list(detection),
            "face_inside_bounds": True,
            "width": width,
            "height": height,
        }

    return {
        "usable": True,
        "rejection_reason": None,
        "blur_laplacian_var": blur,
        "face_count": 1,
        "face_detection_xywh": list(detection),
        "face_inside_bounds": True,
        "width": width,
        "height": height,
    }


def _opencv_version() -> Optional[str]:
    try:
        import cv2  # type: ignore

        return str(getattr(cv2, "__version__", "unknown"))
    except ImportError:
        return None


def probe_video_meta(video_path: Path) -> Tuple[int, Optional[float]]:
    """Return (frame_count, fps) using OpenCV."""

    try:
        import cv2  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only when cv2 missing
        raise CelebDFPrepareError("OpenCV (cv2) is required for frame extraction") from exc

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise CelebDFPrepareError(f"cannot open video: {video_path}")
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps_raw = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        fps = fps_raw if fps_raw > 1e-6 else None
        return total, fps
    finally:
        capture.release()


def _count_frames_sequential(video_path: Path) -> int:
    import cv2  # type: ignore

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return 0
    count = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            count += 1
    finally:
        capture.release()
    return count


def _read_frame_at(capture: Any, index: int) -> Tuple[bool, Any]:
    import cv2  # type: ignore

    capture.set(cv2.CAP_PROP_POS_FRAMES, float(index))
    ok, frame = capture.read()
    if ok and frame is not None and getattr(frame, "size", 0) > 0:
        return True, frame
    return False, None


def select_usable_frame(
    video_path: Path,
    *,
    evaluate_fn: Callable[[Any], Dict[str, Any]] = evaluate_frame_usability,
) -> Tuple[Any, FrameExtraction]:
    """Pick the first usable full frame among ``CANDIDATE_FRACTIONS``.

    Raises ``UnusableVideoError`` when every candidate fails.
    """

    try:
        import cv2  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise CelebDFPrepareError("OpenCV (cv2) is required for frame extraction") from exc

    total, fps = probe_video_meta(video_path)
    if total <= 0:
        total = _count_frames_sequential(video_path)
    if total <= 0:
        raise UnusableVideoError(f"no decodable frames in {video_path}", candidates=[])

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise UnusableVideoError(f"cannot open video: {video_path}", candidates=[])

    candidates: List[CandidateEvaluation] = []
    try:
        for frac, index in candidate_frame_indices(total):
            timestamp = (index / fps) if fps else None
            ok, frame = _read_frame_at(capture, index)
            if not ok:
                candidates.append(
                    CandidateEvaluation(
                        fraction=frac,
                        frame_index=index,
                        decoded=False,
                        usable=False,
                        rejection_reason="decode_failed",
                        timestamp_s=timestamp,
                    )
                )
                continue
            height, width = int(frame.shape[0]), int(frame.shape[1])
            verdict = evaluate_fn(frame)
            trial = CandidateEvaluation(
                fraction=frac,
                frame_index=index,
                decoded=True,
                usable=bool(verdict["usable"]),
                rejection_reason=verdict.get("rejection_reason"),
                blur_laplacian_var=verdict.get("blur_laplacian_var"),
                face_count=verdict.get("face_count"),
                face_detection_xywh=verdict.get("face_detection_xywh"),
                face_inside_bounds=verdict.get("face_inside_bounds"),
                width=width,
                height=height,
                timestamp_s=timestamp,
            )
            candidates.append(trial)
            if trial.usable:
                return frame, FrameExtraction(
                    frame_index=index,
                    total_frames=total,
                    fps=fps,
                    timestamp_s=timestamp,
                    width=width,
                    height=height,
                    candidate_fraction=frac,
                    candidates=candidates,
                    blur_laplacian_var=trial.blur_laplacian_var,
                    face_detection_xywh=trial.face_detection_xywh,
                )
    finally:
        capture.release()

    raise UnusableVideoError(
        f"no usable frame among candidates {list(CANDIDATE_FRACTIONS)} for {video_path}",
        candidates=candidates,
    )


def write_frame_png(frame_bgr: Any, dest: Path) -> None:
    """Write a lossless PNG (no compression parameter randomisation)."""

    try:
        import cv2  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise CelebDFPrepareError("OpenCV (cv2) is required to write PNG frames") from exc

    dest.parent.mkdir(parents=True, exist_ok=True)
    # IMWRITE_PNG_COMPRESSION=0 → deterministic, uncompressed payload when possible.
    ok = cv2.imwrite(
        str(dest),
        frame_bgr,
        [int(getattr(cv2, "IMWRITE_PNG_COMPRESSION", 16)), 0],
    )
    if not ok or not dest.is_file():
        raise CelebDFPrepareError(f"failed to write PNG: {dest}")


def make_image_id(video: CelebDFTestVideo) -> str:
    return f"celebdf_v2_{video.ground_truth}_{video.stem_id}"


def frame_output_relpath(video: CelebDFTestVideo) -> str:
    return f"frames/{video.ground_truth}/{video.stem_id}.png"


def build_notes(video: CelebDFTestVideo, extraction: Optional[FrameExtraction]) -> str:
    parts = [
        f"Official Celeb-DF-v2 {OFFICIAL_SPLIT} video {video.relative_path}.",
        "Full-frame PNG (not face-cropped); labelled evaluation runner performs cropping.",
        "Label-blind usable-frame policy at fractions "
        f"{list(CANDIDATE_FRACTIONS)} (decode + Haar face + in-bounds + Laplacian).",
    ]
    if extraction is not None:
        parts.append(
            f"chosen fraction={extraction.candidate_fraction:.2f} "
            f"frame_index={extraction.frame_index}/{extraction.total_frames}."
        )
    if video.ground_truth == "fake":
        parts.append(
            "Fake videos are Celeb-synthesis DeepFakes; no finer manipulation "
            "subtype is published in the official release, so manipulation is null."
        )
    return " ".join(parts)


def sample_to_manifest_entry(sample: PreparedSample) -> Dict[str, Any]:
    """Build a labelled-manifest-compatible entry with provenance extras."""

    extraction = sample.extraction
    entry: Dict[str, Any] = {
        "id": sample.image_id,
        "path": sample.frame_rel_path.replace("\\", "/"),
        "ground_truth": sample.video.ground_truth,
        "dataset": DATASET_NAME,
        "manipulation": None,  # Celeb-DF-v2 does not publish FF++-style subtypes
        "source": sample.video.relative_path,
        "already_cropped": False,
        "notes": sample.notes,
        # Provenance extras (ignored by LabelledImage parser, retained in JSON)
        "split": OFFICIAL_SPLIT,
        "source_video": sample.video.relative_path,
        "source_video_filename": Path(sample.video.relative_path).name,
        "frame_index": extraction.frame_index if extraction else None,
        "total_frames": extraction.total_frames if extraction else None,
        "fps": extraction.fps if extraction else None,
        "frame_timestamp_s": extraction.timestamp_s if extraction else None,
        "frame_width": extraction.width if extraction else None,
        "frame_height": extraction.height if extraction else None,
        "candidate_fraction": extraction.candidate_fraction if extraction else None,
        "frame_fallback_used": extraction.used_fallback if extraction else None,
        "blur_laplacian_var": extraction.blur_laplacian_var if extraction else None,
        "face_detection_xywh": extraction.face_detection_xywh if extraction else None,
        "video_sha256": sample.video_sha256,
        "frame_sha256": sample.frame_sha256,
        "replaced_from": sample.replaced_from,
    }
    return entry


def _class_pools(
    videos: Sequence[CelebDFTestVideo],
) -> Tuple[List[CelebDFTestVideo], List[CelebDFTestVideo]]:
    reals = sorted(
        (v for v in videos if v.ground_truth == "real"),
        key=lambda v: v.relative_path,
    )
    fakes = sorted(
        (v for v in videos if v.ground_truth == "fake"),
        key=lambda v: v.relative_path,
    )
    return reals, fakes


def fill_class_with_usable_frames(
    *,
    ground_truth: str,
    primary: Sequence[CelebDFTestVideo],
    reserve: Sequence[CelebDFTestVideo],
    n_required: int,
    output_dir: Path,
    hash_videos: bool,
    dry_run: bool,
    select_fn: Callable[[Path], Tuple[Any, FrameExtraction]],
) -> Tuple[List[PreparedSample], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Accept usable videos from ``primary``, then ``reserve``, until ``n_required``.

    Returns ``(samples, replacements, unsuitable_records)``.
    """

    samples: List[PreparedSample] = []
    replacements: List[Dict[str, Any]] = []
    primary_paths = {v.relative_path for v in primary}
    queue = list(primary) + list(reserve)
    seen_paths: set[str] = set()
    unsuitable: List[Dict[str, Any]] = []
    pending_rejections: List[str] = []

    for video in queue:
        if len(samples) >= n_required:
            break
        if video.relative_path in seen_paths:
            continue
        if video.ground_truth != ground_truth:
            raise CelebDFPrepareError(
                f"internal class mismatch: expected {ground_truth}, got {video.ground_truth}"
            )
        seen_paths.add(video.relative_path)
        image_id = make_image_id(video)
        frame_rel = frame_output_relpath(video)
        frame_abs = (output_dir / frame_rel).resolve()

        try:
            if dry_run:
                # Still run selection so replacements mirror a real run; skip writes.
                _frame, extraction = select_fn(video.absolute_path)
                video_sha = None
                frame_sha = None
                frame_path: Optional[Path] = None
            else:
                frame_bgr, extraction = select_fn(video.absolute_path)
                write_frame_png(frame_bgr, frame_abs)
                video_sha = sha256_file(video.absolute_path) if hash_videos else None
                frame_sha = sha256_file(frame_abs)
                frame_path = frame_abs
        except UnusableVideoError as exc:
            record = {
                "source_video": video.relative_path,
                "ground_truth": ground_truth,
                "reason": str(exc),
                "candidates": [asdict(c) for c in exc.candidates],
            }
            unsuitable.append(record)
            if video.relative_path in primary_paths:
                pending_rejections.append(video.relative_path)
            continue

        replaced_from = None
        if video.relative_path not in primary_paths:
            replaced_from = (
                pending_rejections.pop(0)
                if pending_rejections
                else "unusable_primary_exhausted_tracking"
            )
            replacements.append(
                {
                    "rejected_video": replaced_from,
                    "replacement_video": video.relative_path,
                    "ground_truth": ground_truth,
                }
            )

        samples.append(
            PreparedSample(
                image_id=image_id,
                video=video,
                frame_rel_path=frame_rel.replace("\\", "/"),
                frame_abs_path=frame_path,
                extraction=extraction,
                video_sha256=video_sha,
                frame_sha256=frame_sha,
                notes=build_notes(video, extraction),
                replaced_from=replaced_from,
                unsuitable_attempts=list(unsuitable),
            )
        )

    if len(samples) < n_required:
        raise CelebDFPrepareError(
            f"could not obtain {n_required} usable {ground_truth} videos "
            f"(got {len(samples)}; unsuitable={len(unsuitable)})"
        )
    return samples, replacements, unsuitable


def prepare_samples(
    selected: Sequence[CelebDFTestVideo],
    *,
    output_dir: Path,
    all_videos: Sequence[CelebDFTestVideo],
    n_real: int,
    n_fake: int,
    hash_videos: bool = True,
    dry_run: bool = False,
    select_fn: Callable[[Path], Tuple[Any, FrameExtraction]] = select_usable_frame,
) -> Tuple[List[PreparedSample], Dict[str, Any]]:
    """Extract usable full frames with deterministic same-class replacement."""

    reals, fakes = _class_pools(all_videos)
    primary_real = [v for v in selected if v.ground_truth == "real"]
    primary_fake = [v for v in selected if v.ground_truth == "fake"]
    primary_real_set = {v.relative_path for v in primary_real}
    primary_fake_set = {v.relative_path for v in primary_fake}
    reserve_real = [v for v in reals if v.relative_path not in primary_real_set]
    reserve_fake = [v for v in fakes if v.relative_path not in primary_fake_set]

    real_samples, real_repl, real_bad = fill_class_with_usable_frames(
        ground_truth="real",
        primary=primary_real,
        reserve=reserve_real,
        n_required=n_real,
        output_dir=output_dir,
        hash_videos=hash_videos,
        dry_run=dry_run,
        select_fn=select_fn,
    )
    fake_samples, fake_repl, fake_bad = fill_class_with_usable_frames(
        ground_truth="fake",
        primary=primary_fake,
        reserve=reserve_fake,
        n_required=n_fake,
        output_dir=output_dir,
        hash_videos=hash_videos,
        dry_run=dry_run,
        select_fn=select_fn,
    )
    samples = real_samples + fake_samples
    ids = [s.image_id for s in samples]
    if len(ids) != len(set(ids)):
        raise CelebDFPrepareError("duplicate image id after replacement")
    vids = [s.video.relative_path for s in samples]
    if len(vids) != len(set(vids)):
        raise CelebDFPrepareError("duplicate source video after replacement")

    extra = {
        "replacements": real_repl + fake_repl,
        "unsuitable_videos": real_bad + fake_bad,
        "usability_policy": {
            "candidate_fractions": list(CANDIDATE_FRACTIONS),
            "min_laplacian_variance": MIN_LAPLACIAN_VARIANCE,
            "face_edge_margin_frac": FACE_EDGE_MARGIN_FRAC,
            "face_detector": "tools.make_face_crop.plan_crop (Haar frontal)",
            "saves_full_frame": True,
            "label_blind": True,
        },
    }
    return samples, extra


def validate_prepared(
    samples: Sequence[PreparedSample],
    *,
    n_real: int,
    n_fake: int,
    dry_run: bool = False,
) -> None:
    """Enforce class balance, unique IDs, and frame existence."""

    reals = [s for s in samples if s.video.ground_truth == "real"]
    fakes = [s for s in samples if s.video.ground_truth == "fake"]
    if len(reals) != n_real or len(fakes) != n_fake:
        raise CelebDFPrepareError(
            f"expected {n_real} real + {n_fake} fake, got {len(reals)} real + {len(fakes)} fake"
        )
    ids = [s.image_id for s in samples]
    if len(ids) != len(set(ids)):
        raise CelebDFPrepareError("duplicate image IDs in prepared set")
    vids = [s.video.relative_path for s in samples]
    if len(vids) != len(set(vids)):
        raise CelebDFPrepareError("duplicate source videos in prepared set")
    for sample in samples:
        if sample.video.ground_truth not in {"real", "fake"}:
            raise CelebDFPrepareError(f"invalid label for {sample.image_id}")
        if dry_run:
            continue
        if sample.frame_abs_path is None or not sample.frame_abs_path.is_file():
            raise CelebDFPrepareError(f"missing frame file for {sample.image_id}")
        if not sample.frame_sha256:
            raise CelebDFPrepareError(f"missing frame hash for {sample.image_id}")


def write_outputs(
    samples: Sequence[PreparedSample],
    *,
    output_dir: Path,
    selection_meta: Dict[str, Any],
    test_list_path: Path,
    dataset_root: Path,
    dry_run: bool,
) -> Dict[str, Path]:
    """Write manifest, provenance, and optional CSV under ``output_dir``."""

    output_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, Path] = {}

    try:
        test_list_sha = sha256_file(test_list_path)
    except OSError:
        test_list_sha = None

    entries = [sample_to_manifest_entry(s) for s in samples]
    manifest = {
        "description": (
            "Celeb-DF-v2 official-test image-level subset: one deterministic "
            "full frame per selected video for the X2DFD labelled evaluation "
            "pipeline. Not a video-level detector benchmark."
        ),
        "dataset": DATASET_NAME,
        "split": OFFICIAL_SPLIT,
        "seed": selection_meta["seed"],
        "n_real": selection_meta["n_real_requested"],
        "n_fake": selection_meta["n_fake_requested"],
        "images": entries,
    }
    if not dry_run:
        manifest_path = output_dir / "celebdf_v2_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        written["manifest"] = manifest_path

        csv_path = output_dir / "celebdf_v2_summary.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "id",
                    "ground_truth",
                    "path",
                    "source_video",
                    "frame_index",
                    "total_frames",
                    "fps",
                    "frame_sha256",
                    "video_sha256",
                ],
            )
            writer.writeheader()
            for entry in entries:
                writer.writerow(
                    {
                        "id": entry["id"],
                        "ground_truth": entry["ground_truth"],
                        "path": entry["path"],
                        "source_video": entry["source_video"],
                        "frame_index": entry["frame_index"],
                        "total_frames": entry["total_frames"],
                        "fps": entry["fps"],
                        "frame_sha256": entry["frame_sha256"],
                        "video_sha256": entry["video_sha256"],
                    }
                )
        written["csv"] = csv_path

    git = get_git_provenance()
    non_primary = [
        {
            "id": s.image_id,
            "source_video": s.video.relative_path,
            "candidate_fraction": s.extraction.candidate_fraction if s.extraction else None,
            "frame_index": s.extraction.frame_index if s.extraction else None,
        }
        for s in samples
        if s.extraction and s.extraction.used_fallback
    ]
    provenance = {
        "generated_at": utc_timestamp(),
        "script": "tools.prepare_celebdf_evaluation",
        "script_version": SCRIPT_VERSION,
        "git_commit": git.get("git_commit"),
        "git_dirty": git.get("git_dirty"),
        "dry_run": dry_run,
        "dataset_root": str(Path(dataset_root).resolve()),
        "test_list_path": str(Path(test_list_path).resolve()),
        "test_list_sha256": test_list_sha,
        "opencv_version": _opencv_version(),
        "selection": selection_meta,
        "usability_policy": selection_meta.get("usability_policy"),
        "replacements": selection_meta.get("replacements", []),
        "unsuitable_videos": selection_meta.get("unsuitable_videos", []),
        "selected_video_ids": [s.video.relative_path for s in samples],
        "selected_image_ids": [s.image_id for s in samples],
        "non_primary_fraction_selections": non_primary,
        "samples": [
            {
                "id": s.image_id,
                "source_video": s.video.relative_path,
                "ground_truth": s.video.ground_truth,
                "frame_path": s.frame_rel_path,
                "video_sha256": s.video_sha256,
                "frame_sha256": s.frame_sha256,
                "replaced_from": s.replaced_from,
                "extraction": asdict(s.extraction) if s.extraction else None,
                "candidate_evaluations": (
                    [asdict(c) for c in s.extraction.candidates] if s.extraction else None
                ),
            }
            for s in samples
        ],
        "notes": (
            "Celeb-DF-v2 is video-based; X2DFD is evaluated at the image level. "
            "One usable full frame per video (label-blind candidate fractions "
            f"{list(CANDIDATE_FRACTIONS)}; Haar face + in-bounds + Laplacian≥"
            f"{MIN_LAPLACIAN_VARIANCE}). already_cropped=false so the labelled "
            "runner applies the same face-crop stage used elsewhere. Frame choice "
            "never uses X2DFD / blending / diffusion scores."
        ),
    }
    prov_path = output_dir / "sampling_provenance.json"
    if dry_run:
        # Still useful to inspect selection without writing frames/manifest.
        dry_path = output_dir / "sampling_provenance.dry_run.json"
        dry_path.parent.mkdir(parents=True, exist_ok=True)
        dry_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
        written["provenance"] = dry_path
    else:
        prov_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
        written["provenance"] = prov_path

    return written


def run_preparation(
    *,
    dataset_root: Path,
    output_dir: Path,
    seed: int = DEFAULT_SEED,
    n_real: int = DEFAULT_N_REAL,
    n_fake: int = DEFAULT_N_FAKE,
    dry_run: bool = False,
    hash_videos: bool = True,
    test_list_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """End-to-end Celeb-DF-v2 subset preparation."""

    root = Path(dataset_root).resolve()
    if not root.is_dir():
        raise CelebDFPrepareError(f"dataset root is not a directory: {root}")
    test_list = Path(test_list_path).resolve() if test_list_path else locate_test_list(root)
    videos = parse_test_list(test_list, root, require_files=True)
    selected, selection_meta = select_balanced_videos(
        videos, n_real=n_real, n_fake=n_fake, seed=seed
    )
    LOGGER.info(
        "selected %s real + %s fake from %s eligible test videos (seed=%s)",
        n_real,
        n_fake,
        len(videos),
        seed,
    )
    out = Path(output_dir)
    samples, usability_extra = prepare_samples(
        selected,
        output_dir=out,
        all_videos=videos,
        n_real=n_real,
        n_fake=n_fake,
        hash_videos=hash_videos,
        dry_run=dry_run,
    )
    selection_meta = {**selection_meta, **usability_extra}
    validate_prepared(samples, n_real=n_real, n_fake=n_fake, dry_run=dry_run)
    written = write_outputs(
        samples,
        output_dir=out,
        selection_meta=selection_meta,
        test_list_path=test_list,
        dataset_root=root,
        dry_run=dry_run,
    )
    return {
        "n_selected": len(samples),
        "n_real": n_real,
        "n_fake": n_fake,
        "seed": seed,
        "dry_run": dry_run,
        "test_list": str(test_list),
        "output_dir": str(out.resolve()),
        "written": {key: str(path) for key, path in written.items()},
        "selected_video_ids": [s.video.relative_path for s in samples],
        "replacements": usability_extra.get("replacements", []),
    }
