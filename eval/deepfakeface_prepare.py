"""DeepFakeFace image-level evaluation subset preparation (no inference).

Deterministically samples native images from the four DeepFakeFace ZIP archives
(wiki / insight / inpainting / text2img), extracts only the selected members,
and writes a labelled evaluation manifest. Face cropping remains the
responsibility of ``tools.run_labelled_evaluation`` (``already_cropped: false``).
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import random
import re
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from eval.reproducibility import get_git_provenance, sha256_file, utc_timestamp

LOGGER = logging.getLogger("x2dfd.deepfakeface_prepare")

SCRIPT_VERSION = "1.0.1"
DEFAULT_SEED = 20260305
DATASET_NAME = "DeepFakeFace"

# Category quotas (120 real + 120 fake).
CATEGORY_SPECS: Tuple[Dict[str, Any], ...] = (
    {
        "category": "wiki",
        "zip_name": "wiki.zip",
        "ground_truth": "real",
        "manipulation": None,
        "n": 120,
    },
    {
        "category": "insight",
        "zip_name": "insight.zip",
        "ground_truth": "fake",
        "manipulation": "insight",
        "n": 40,
    },
    {
        "category": "inpainting",
        "zip_name": "inpainting.zip",
        "ground_truth": "fake",
        "manipulation": "inpainting",
        "n": 40,
    },
    {
        "category": "text2img",
        "zip_name": "text2img.zip",
        "ground_truth": "fake",
        "manipulation": "text2img",
        "n": 40,
    },
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
# Labelled evaluation accepts jpg/jpeg/png only; sniff content for output suffix.
JPEG_MAGIC = b"\xff\xd8\xff"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
FACE_EDGE_MARGIN_FRAC = 0.02

SELECTION_ALGORITHM = (
    "1) For each category in fixed order "
    "[wiki, insight, inpainting, text2img], open the matching ZIP under the "
    "dataset root and collect image member paths (sorted POSIX). "
    "2) Draw without replacement via random.Random(seed).sample(sorted_list, k) "
    "for the category quota (one shared RNG advanced in category order). "
    "3) Remaining members (sorted) form a deterministic replacement pool. "
    "4) A candidate is usable only if ZIP bytes decode, Haar finds ≥1 face "
    "(tools.make_face_crop.plan_crop defaults), and the largest face is "
    f"sufficiently inside image bounds (edge margin {FACE_EDGE_MARGIN_FRAC:.0%} "
    "of min side). No Laplacian / model-score gate. "
    "5) Extract selected members byte-for-byte; output suffix follows content magic (JPEG/PNG) so mislabeled ZIP members remain valid for labelled evaluation. "
    "6) already_cropped=false; labelled evaluation performs the final crop. "
    "Usability never uses X2DFD / blending / diffusion scores."
)


class DeepFakeFacePrepareError(ValueError):
    """Dataset layout, sampling, or extraction failed."""


@dataclass(frozen=True)
class ZipMemberCandidate:
    category: str
    zip_name: str
    member_name: str  # POSIX path inside the ZIP
    ground_truth: str
    manipulation: Optional[str]

    @property
    def stem_id(self) -> str:
        stem = Path(self.member_name).stem
        return re.sub(r"[^A-Za-z0-9._-]+", "_", stem)


@dataclass
class UsabilityVerdict:
    usable: bool
    rejection_reason: Optional[str]
    width: Optional[int] = None
    height: Optional[int] = None
    face_count: Optional[int] = None
    face_detection_xywh: Optional[List[int]] = None
    face_inside_bounds: Optional[bool] = None


@dataclass
class PreparedSample:
    image_id: str
    candidate: ZipMemberCandidate
    image_rel_path: str
    image_abs_path: Optional[Path]
    member_sha256: Optional[str]
    image_sha256: Optional[str]
    usability: Optional[UsabilityVerdict]
    notes: str
    replaced_from: Optional[str] = None
    unsuitable_attempts: List[Dict[str, Any]] = field(default_factory=list)


class UnusableImageError(DeepFakeFacePrepareError):
    def __init__(self, message: str, *, verdict: Optional[UsabilityVerdict] = None):
        super().__init__(message)
        self.verdict = verdict


def locate_zip(dataset_root: Path, zip_name: str) -> Path:
    root = Path(dataset_root)
    direct = root / zip_name
    if direct.is_file():
        return direct.resolve()
    nested = root / "DeepFakeFace" / zip_name
    if nested.is_file():
        return nested.resolve()
    raise DeepFakeFacePrepareError(f"{zip_name} not found under {root}")


def list_image_members(zip_path: Path) -> List[str]:
    """Return sorted image member paths inside a ZIP (files only)."""

    names: List[str] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = info.filename.replace("\\", "/")
            if Path(name).suffix.lower() in IMAGE_SUFFIXES:
                names.append(name)
    names.sort()
    if not names:
        raise DeepFakeFacePrepareError(f"no image members in {zip_path}")
    return names


def face_sufficiently_inside(
    detection_xywh: Sequence[int],
    width: int,
    height: int,
    *,
    margin_frac: float = FACE_EDGE_MARGIN_FRAC,
) -> bool:
    x, y, w, h = (int(v) for v in detection_xywh)
    if w <= 0 or h <= 0 or width <= 0 or height <= 0:
        return False
    margin = margin_frac * float(min(width, height))
    return (
        x >= margin
        and y >= margin
        and (x + w) <= (width - margin)
        and (y + h) <= (height - margin)
    )


def evaluate_image_bytes(
    raw: bytes,
    *,
    plan_crop_fn: Optional[Callable[..., Any]] = None,
    edge_margin_frac: float = FACE_EDGE_MARGIN_FRAC,
) -> UsabilityVerdict:
    """Decode image bytes and apply the face-crop planner gate (no final crop)."""

    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise DeepFakeFacePrepareError("OpenCV/numpy required for usability checks") from exc

    from tools.make_face_crop import (
        DEFAULT_DETECT_WIDTH,
        DEFAULT_MARGIN,
        DEFAULT_MIN_NEIGHBOURS,
        DEFAULT_SCALE_FACTOR,
        plan_crop,
    )

    if sniff_output_suffix(raw) is None:
        return UsabilityVerdict(
            usable=False,
            rejection_reason="unsupported_image_encoding",
        )

    arr = np.frombuffer(raw, dtype=np.uint8)
    image_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image_bgr is None or getattr(image_bgr, "size", 0) == 0:
        return UsabilityVerdict(usable=False, rejection_reason="decode_failed")

    height, width = int(image_bgr.shape[0]), int(image_bgr.shape[1])
    if width <= 0 or height <= 0:
        return UsabilityVerdict(
            usable=False,
            rejection_reason="invalid_dimensions",
            width=width,
            height=height,
        )

    crop_fn = plan_crop_fn or plan_crop
    plan = crop_fn(
        image_bgr,
        detect_width=DEFAULT_DETECT_WIDTH,
        scale_factor=DEFAULT_SCALE_FACTOR,
        min_neighbours=DEFAULT_MIN_NEIGHBOURS,
        margin=DEFAULT_MARGIN,
    )
    if plan is None:
        return UsabilityVerdict(
            usable=False,
            rejection_reason="no_face",
            width=width,
            height=height,
            face_count=0,
        )

    detection = tuple(int(v) for v in plan.detection)
    inside = face_sufficiently_inside(
        detection, width, height, margin_frac=edge_margin_frac
    )
    if not inside:
        return UsabilityVerdict(
            usable=False,
            rejection_reason="face_near_border",
            width=width,
            height=height,
            face_count=1,
            face_detection_xywh=list(detection),
            face_inside_bounds=False,
        )

    return UsabilityVerdict(
        usable=True,
        rejection_reason=None,
        width=width,
        height=height,
        face_count=1,
        face_detection_xywh=list(detection),
        face_inside_bounds=True,
    )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_image_id(candidate: ZipMemberCandidate) -> str:
    return f"dff_{candidate.category}_{candidate.stem_id}"


def sniff_output_suffix(raw: bytes) -> Optional[str]:
    """Return labelled-manifest-compatible suffix from content magic, or None."""
    if raw.startswith(JPEG_MAGIC):
        return ".jpg"
    if raw.startswith(PNG_MAGIC):
        return ".png"
    return None


def image_output_relpath(candidate: ZipMemberCandidate, *, raw: bytes) -> str:
    """Build output path using content magic (ZIP names may lie about format)."""
    suffix = sniff_output_suffix(raw)
    if suffix is None:
        raise DeepFakeFacePrepareError(
            f"unsupported image encoding for member {candidate.member_name!r}"
        )
    gt = candidate.ground_truth
    return f"images/{gt}/{candidate.category}__{candidate.stem_id}{suffix}"


def build_notes(candidate: ZipMemberCandidate, *, replaced_from: Optional[str]) -> str:
    parts = [
        f"DeepFakeFace native image from {candidate.zip_name} member {candidate.member_name}.",
        "Extracted byte-for-byte (encoding preserved); not face-cropped.",
        "Usability gate: decode + Haar face via tools.make_face_crop.plan_crop + in-bounds.",
        "already_cropped=false so the labelled evaluation runner performs cropping.",
    ]
    if replaced_from:
        parts.append(f"Replacement for unsuitable member {replaced_from}.")
    return " ".join(parts)


def read_zip_member(zip_path: Path, member_name: str) -> bytes:
    with zipfile.ZipFile(zip_path, "r") as zf:
        try:
            return zf.read(member_name)
        except KeyError as exc:
            raise DeepFakeFacePrepareError(
                f"member {member_name!r} missing from {zip_path}"
            ) from exc


def extract_member_bytes(raw: bytes, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(raw)
    if not dest.is_file() or dest.stat().st_size != len(raw):
        raise DeepFakeFacePrepareError(f"failed to write extracted image: {dest}")


def select_primary_and_reserve(
    members: Sequence[str],
    *,
    n: int,
    rng: random.Random,
) -> Tuple[List[str], List[str]]:
    if n < 1:
        raise DeepFakeFacePrepareError("n must be >= 1")
    if len(members) < n:
        raise DeepFakeFacePrepareError(
            f"need at least {n} members, found {len(members)}"
        )
    primary = rng.sample(list(members), n)
    primary_set = set(primary)
    reserve = [m for m in members if m not in primary_set]
    return primary, reserve


def fill_category(
    *,
    category: str,
    zip_path: Path,
    zip_name: str,
    ground_truth: str,
    manipulation: Optional[str],
    primary: Sequence[str],
    reserve: Sequence[str],
    n_required: int,
    output_dir: Path,
    dry_run: bool,
    evaluate_fn: Callable[[bytes], UsabilityVerdict],
) -> Tuple[List[PreparedSample], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Accept usable members from primary then reserve until quota is met."""

    samples: List[PreparedSample] = []
    replacements: List[Dict[str, Any]] = []
    unsuitable: List[Dict[str, Any]] = []
    reserve_iter = iter(reserve)
    pending = list(primary)

    while len(samples) < n_required:
        if not pending:
            raise DeepFakeFacePrepareError(
                f"could not obtain {n_required} usable {category} images "
                f"from {zip_name} (exhausted primary + reserve)"
            )
        member = pending.pop(0)
        candidate = ZipMemberCandidate(
            category=category,
            zip_name=zip_name,
            member_name=member,
            ground_truth=ground_truth,
            manipulation=manipulation,
        )
        try:
            raw = read_zip_member(zip_path, member)
        except DeepFakeFacePrepareError as exc:
            unsuitable.append(
                {
                    "category": category,
                    "zip_name": zip_name,
                    "member": member,
                    "rejection_reason": "zip_read_failed",
                    "detail": str(exc),
                }
            )
            try:
                pending.append(next(reserve_iter))
            except StopIteration:
                pass
            continue

        verdict = evaluate_fn(raw)
        if not verdict.usable:
            unsuitable.append(
                {
                    "category": category,
                    "zip_name": zip_name,
                    "member": member,
                    "rejection_reason": verdict.rejection_reason,
                    "width": verdict.width,
                    "height": verdict.height,
                }
            )
            try:
                replacement_member = next(reserve_iter)
            except StopIteration:
                continue
            replacements.append(
                {
                    "category": category,
                    "rejected_member": member,
                    "replacement_member": replacement_member,
                    "rejection_reason": verdict.rejection_reason,
                }
            )
            pending.append(replacement_member)
            continue

        image_id = make_image_id(candidate)
        rel = image_output_relpath(candidate, raw=raw)
        abs_path: Optional[Path] = None
        member_hash = sha256_bytes(raw)
        image_hash: Optional[str] = None
        if not dry_run:
            abs_path = (output_dir / rel).resolve()
            extract_member_bytes(raw, abs_path)
            image_hash = sha256_file(abs_path)
            if image_hash != member_hash:
                raise DeepFakeFacePrepareError(
                    f"extracted bytes hash mismatch for {member}"
                )

        replaced_from = None
        for item in replacements:
            if item["replacement_member"] == member and item["category"] == category:
                replaced_from = item["rejected_member"]
        samples.append(
            PreparedSample(
                image_id=image_id,
                candidate=candidate,
                image_rel_path=rel,
                image_abs_path=abs_path,
                member_sha256=member_hash,
                image_sha256=image_hash if not dry_run else member_hash,
                usability=verdict,
                notes=build_notes(candidate, replaced_from=replaced_from),
                replaced_from=replaced_from,
            )
        )

    return samples, replacements, unsuitable


def _zip_identity(zip_path: Path, *, hash_zip: bool) -> Dict[str, Any]:
    """Identify a source ZIP without always hashing multi-GB archives."""

    st = zip_path.stat()
    info: Dict[str, Any] = {
        "zip_path": str(zip_path.resolve()),
        "zip_size_bytes": int(st.st_size),
        "zip_mtime_ns": int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))),
        "zip_sha256": None,
    }
    if hash_zip:
        info["zip_sha256"] = sha256_file(zip_path)
    return info


def prepare_samples(
    dataset_root: Path,
    output_dir: Path,
    *,
    seed: int = DEFAULT_SEED,
    dry_run: bool = False,
    hash_zips: bool = False,
    evaluate_fn: Optional[Callable[[bytes], UsabilityVerdict]] = None,
) -> Tuple[List[PreparedSample], Dict[str, Any]]:
    """Sample, screen, and extract the DeepFakeFace evaluation subset."""

    root = Path(dataset_root)
    if not root.is_dir():
        raise DeepFakeFacePrepareError(f"dataset root not found: {root}")

    rng = random.Random(seed)
    eval_fn = evaluate_fn or evaluate_image_bytes
    all_samples: List[PreparedSample] = []
    all_replacements: List[Dict[str, Any]] = []
    all_unsuitable: List[Dict[str, Any]] = []
    category_meta: List[Dict[str, Any]] = []

    for spec in CATEGORY_SPECS:
        category = str(spec["category"])
        zip_name = str(spec["zip_name"])
        n_required = int(spec["n"])
        zip_path = locate_zip(root, zip_name)
        members = list_image_members(zip_path)
        primary, reserve = select_primary_and_reserve(members, n=n_required, rng=rng)
        samples, replacements, unsuitable = fill_category(
            category=category,
            zip_path=zip_path,
            zip_name=zip_name,
            ground_truth=str(spec["ground_truth"]),
            manipulation=spec["manipulation"],
            primary=primary,
            reserve=reserve,
            n_required=n_required,
            output_dir=Path(output_dir),
            dry_run=dry_run,
            evaluate_fn=eval_fn,
        )
        all_samples.extend(samples)
        all_replacements.extend(replacements)
        all_unsuitable.extend(unsuitable)
        category_meta.append(
            {
                "category": category,
                "zip_name": zip_name,
                **_zip_identity(zip_path, hash_zip=hash_zips),
                "n_members": len(members),
                "n_requested": n_required,
                "n_accepted": len(samples),
                "n_replacements": len(replacements),
                "n_unsuitable_tried": len(unsuitable),
                "ground_truth": spec["ground_truth"],
                "manipulation": spec["manipulation"],
                "primary_members": list(primary),
            }
        )
        LOGGER.info(
            "%s: accepted %s/%s (replacements=%s unsuitable=%s)",
            category,
            len(samples),
            n_required,
            len(replacements),
            len(unsuitable),
        )

    selection_meta: Dict[str, Any] = {
        "seed": seed,
        "selection_algorithm": SELECTION_ALGORITHM,
        "script_version": SCRIPT_VERSION,
        "categories": category_meta,
        "replacements": all_replacements,
        "unsuitable_images": all_unsuitable,
        "usability_policy": {
            "face_edge_margin_frac": FACE_EDGE_MARGIN_FRAC,
            "face_detector": "tools.make_face_crop.plan_crop (Haar frontal)",
            "saves_native_bytes": True,
            "label_blind": True,
            "model_scores_used": False,
        },
        "n_real_requested": 120,
        "n_fake_requested": 120,
    }
    return all_samples, selection_meta


def validate_prepared(
    samples: Sequence[PreparedSample],
    *,
    dry_run: bool = False,
) -> None:
    reals = [s for s in samples if s.candidate.ground_truth == "real"]
    fakes = [s for s in samples if s.candidate.ground_truth == "fake"]
    if len(reals) != 120 or len(fakes) != 120:
        raise DeepFakeFacePrepareError(
            f"expected 120 real + 120 fake, got {len(reals)} real + {len(fakes)} fake"
        )
    by_cat: Dict[str, int] = {}
    for s in samples:
        by_cat[s.candidate.category] = by_cat.get(s.candidate.category, 0) + 1
    expected = {spec["category"]: spec["n"] for spec in CATEGORY_SPECS}
    if by_cat != expected:
        raise DeepFakeFacePrepareError(f"category counts {by_cat} != {expected}")
    ids = [s.image_id for s in samples]
    if len(ids) != len(set(ids)):
        raise DeepFakeFacePrepareError("duplicate image IDs in prepared set")
    members = [(s.candidate.zip_name, s.candidate.member_name) for s in samples]
    if len(members) != len(set(members)):
        raise DeepFakeFacePrepareError("duplicate ZIP members in prepared set")
    for sample in samples:
        if dry_run:
            continue
        if sample.image_abs_path is None or not sample.image_abs_path.is_file():
            raise DeepFakeFacePrepareError(f"missing image file for {sample.image_id}")
        if not sample.image_sha256:
            raise DeepFakeFacePrepareError(f"missing image hash for {sample.image_id}")


def sample_to_manifest_entry(sample: PreparedSample) -> Dict[str, Any]:
    c = sample.candidate
    u = sample.usability
    return {
        "id": sample.image_id,
        "path": sample.image_rel_path,
        "ground_truth": c.ground_truth,
        "dataset": DATASET_NAME,
        "manipulation": c.manipulation,
        "source": f"{c.zip_name}:{c.member_name}",
        "already_cropped": False,
        "notes": sample.notes,
        "category": c.category,
        "zip_name": c.zip_name,
        "zip_member": c.member_name,
        "member_sha256": sample.member_sha256,
        "image_sha256": sample.image_sha256,
        "frame_width": u.width if u else None,
        "frame_height": u.height if u else None,
        "face_detection_xywh": u.face_detection_xywh if u else None,
        "replaced_from": sample.replaced_from,
    }


def write_outputs(
    samples: Sequence[PreparedSample],
    *,
    output_dir: Path,
    selection_meta: Dict[str, Any],
    dataset_root: Path,
    dry_run: bool,
) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, Path] = {}
    entries = [sample_to_manifest_entry(s) for s in samples]
    manifest = {
        "description": (
            "DeepFakeFace image-level subset for the X2DFD labelled evaluation "
            "pipeline: 120 wiki (real) + 40 insight + 40 inpainting + 40 "
            "text2img (fake). Native extracted images; not face-cropped."
        ),
        "dataset": DATASET_NAME,
        "seed": selection_meta["seed"],
        "n_real": 120,
        "n_fake": 120,
        "category_counts": {
            spec["category"]: spec["n"] for spec in CATEGORY_SPECS
        },
        "images": entries,
    }
    if not dry_run:
        manifest_path = output_dir / "deepfakeface_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        written["manifest"] = manifest_path

        csv_path = output_dir / "deepfakeface_summary.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "id",
                    "ground_truth",
                    "category",
                    "path",
                    "zip_name",
                    "zip_member",
                    "image_sha256",
                    "member_sha256",
                ],
            )
            writer.writeheader()
            for entry in entries:
                writer.writerow({k: entry.get(k) for k in writer.fieldnames})
        written["csv"] = csv_path

    git = get_git_provenance()
    provenance = {
        "generated_at": utc_timestamp(),
        "script": "tools.prepare_deepfakeface_evaluation",
        "script_version": SCRIPT_VERSION,
        "git_commit": git.get("git_commit"),
        "git_dirty": git.get("git_dirty"),
        "dry_run": dry_run,
        "dataset_root": str(Path(dataset_root).resolve()),
        "selection": selection_meta,
        "usability_policy": selection_meta.get("usability_policy"),
        "replacements": selection_meta.get("replacements", []),
        "unsuitable_images": selection_meta.get("unsuitable_images", []),
        "selected_image_ids": [s.image_id for s in samples],
        "samples": [
            {
                "id": s.image_id,
                "category": s.candidate.category,
                "zip_name": s.candidate.zip_name,
                "zip_member": s.candidate.member_name,
                "ground_truth": s.candidate.ground_truth,
                "manipulation": s.candidate.manipulation,
                "image_path": s.image_rel_path,
                "member_sha256": s.member_sha256,
                "image_sha256": s.image_sha256,
                "replaced_from": s.replaced_from,
                "usability": asdict(s.usability) if s.usability else None,
            }
            for s in samples
        ],
        "notes": (
            "DeepFakeFace preparation extracts native ZIP member bytes only. "
            "Face-crop planner is used solely as a usability gate; "
            "already_cropped=false. Selection never uses X2DFD / blending / "
            "diffusion scores."
        ),
    }
    prov_path = output_dir / "sampling_provenance.json"
    prov_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    written["provenance"] = prov_path
    return written


def run_preparation(
    *,
    dataset_root: Path,
    output_dir: Path,
    seed: int = DEFAULT_SEED,
    dry_run: bool = False,
    hash_zips: bool = False,
    evaluate_fn: Optional[Callable[[bytes], UsabilityVerdict]] = None,
) -> Dict[str, Any]:
    samples, selection_meta = prepare_samples(
        dataset_root,
        output_dir,
        seed=seed,
        dry_run=dry_run,
        hash_zips=hash_zips,
        evaluate_fn=evaluate_fn,
    )
    validate_prepared(samples, dry_run=dry_run)
    written = write_outputs(
        samples,
        output_dir=output_dir,
        selection_meta=selection_meta,
        dataset_root=dataset_root,
        dry_run=dry_run,
    )
    n_real = sum(1 for s in samples if s.candidate.ground_truth == "real")
    n_fake = sum(1 for s in samples if s.candidate.ground_truth == "fake")
    return {
        "output_dir": str(Path(output_dir).resolve()),
        "n_images": len(samples),
        "n_real": n_real,
        "n_fake": n_fake,
        "n_replacements": len(selection_meta.get("replacements", [])),
        "n_unsuitable": len(selection_meta.get("unsuitable_images", [])),
        "seed": seed,
        "dry_run": dry_run,
        "written": {k: str(v) for k, v in written.items()},
        "category_counts": {
            spec["category"]: sum(
                1 for s in samples if s.candidate.category == spec["category"]
            )
            for spec in CATEGORY_SPECS
        },
    }
