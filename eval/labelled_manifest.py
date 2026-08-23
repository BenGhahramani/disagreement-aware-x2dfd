"""Labelled evaluation image manifest: schema, loading, and validation.

Used by ``tools.run_labelled_evaluation``. Does not run inference.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
GROUND_TRUTH_VALUES = frozenset({"real", "fake"})
JPEG_MAGIC = b"\xff\xd8\xff"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


class LabelledManifestError(ValueError):
    """The labelled evaluation manifest is missing, malformed, or inconsistent."""


@dataclass(frozen=True)
class LabelledImage:
    """One labelled source image for batch evaluation."""

    id: str
    path: Path
    ground_truth: str
    dataset: Optional[str] = None
    manipulation: Optional[str] = None
    source: Optional[str] = None
    already_cropped: bool = False
    notes: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "path": str(self.path),
            "ground_truth": self.ground_truth,
            "dataset": self.dataset,
            "manipulation": self.manipulation,
            "source": self.source,
            "already_cropped": self.already_cropped,
            "notes": self.notes,
        }


def _require_mapping(payload: Any, *, context: str) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise LabelledManifestError(f"{context} must be a JSON object")
    return payload


def _optional_str(raw: Any, *, field_name: str) -> Optional[str]:
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise LabelledManifestError(f"{field_name} must be a string when present")
    text = raw.strip()
    return text or None


def _coerce_bool(raw: Any, *, field_name: str, default: bool = False) -> bool:
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    raise LabelledManifestError(f"{field_name} must be a boolean when present")


def _validate_image_file(path: Path) -> Path:
    if not path.exists():
        raise LabelledManifestError(f"image not found: {path}")
    if not path.is_file():
        raise LabelledManifestError(f"image path is not a file: {path}")
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_IMAGE_SUFFIXES:
        raise LabelledManifestError(
            f"unsupported image type {suffix!r} for {path}; "
            f"expected one of {sorted(SUPPORTED_IMAGE_SUFFIXES)}"
        )
    try:
        header = path.read_bytes()[:16]
    except OSError as exc:
        raise LabelledManifestError(f"cannot read image {path}: {exc}") from exc
    if suffix in {".jpg", ".jpeg"} and not header.startswith(JPEG_MAGIC):
        raise LabelledManifestError(f"file is not a JPEG: {path}")
    if suffix == ".png" and not header.startswith(PNG_MAGIC):
        raise LabelledManifestError(f"file is not a PNG: {path}")
    return path.resolve()


def parse_labelled_entry(
    raw: Any,
    *,
    index: int,
    base_dir: Path,
) -> LabelledImage:
    """Parse and validate one manifest entry."""

    item = _require_mapping(raw, context=f"images[{index}]")
    image_id = item.get("id")
    if not isinstance(image_id, str) or not image_id.strip():
        raise LabelledManifestError(f"images[{index}].id must be a non-empty string")
    image_id = image_id.strip()

    path_raw = item.get("path")
    if not isinstance(path_raw, str) or not path_raw.strip():
        raise LabelledManifestError(f"images[{index}].path must be a non-empty string")
    candidate = Path(path_raw)
    if not candidate.is_absolute():
        candidate = (base_dir / candidate).resolve()
    else:
        candidate = candidate.resolve()
    path = _validate_image_file(candidate)

    ground_truth = item.get("ground_truth")
    if not isinstance(ground_truth, str):
        raise LabelledManifestError(
            f"images[{index}].ground_truth must be a string ('real' or 'fake')"
        )
    ground_truth = ground_truth.strip().lower()
    if ground_truth not in GROUND_TRUTH_VALUES:
        raise LabelledManifestError(
            f"images[{index}].ground_truth must be 'real' or 'fake', "
            f"got {ground_truth!r}"
        )

    return LabelledImage(
        id=image_id,
        path=path,
        ground_truth=ground_truth,
        dataset=_optional_str(item.get("dataset"), field_name=f"images[{index}].dataset"),
        manipulation=_optional_str(
            item.get("manipulation"), field_name=f"images[{index}].manipulation"
        ),
        source=_optional_str(item.get("source"), field_name=f"images[{index}].source"),
        already_cropped=_coerce_bool(
            item.get("already_cropped"),
            field_name=f"images[{index}].already_cropped",
        ),
        notes=_optional_str(item.get("notes"), field_name=f"images[{index}].notes"),
    )


def load_labelled_manifest(
    path: Path | str,
    *,
    base_dir: Path | str | None = None,
) -> List[LabelledImage]:
    """Load a labelled evaluation manifest.

    Accepts either a top-level JSON list of image objects, or an object with an
    ``images`` array. Relative ``path`` values resolve against ``base_dir``
    (default: the manifest file's parent directory).
    """

    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise LabelledManifestError(f"manifest not found: {manifest_path}")

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise LabelledManifestError(f"cannot read manifest {manifest_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise LabelledManifestError(f"manifest is not valid JSON: {exc}") from exc

    if isinstance(payload, list):
        raw_images = payload
    elif isinstance(payload, dict):
        raw_images = payload.get("images")
        if not isinstance(raw_images, list):
            raise LabelledManifestError("manifest object must contain an 'images' array")
    else:
        raise LabelledManifestError("manifest must be a JSON list or object with 'images'")

    if not raw_images:
        raise LabelledManifestError("manifest contains no images")

    root = Path(base_dir) if base_dir is not None else manifest_path.parent
    entries: List[LabelledImage] = []
    seen: Dict[str, int] = {}
    for index, raw in enumerate(raw_images):
        entry = parse_labelled_entry(raw, index=index, base_dir=root)
        if entry.id in seen:
            raise LabelledManifestError(
                f"duplicate image id {entry.id!r} at images[{index}] "
                f"(first seen at images[{seen[entry.id]}])"
            )
        seen[entry.id] = index
        entries.append(entry)
    return entries


def write_labelled_manifest(path: Path | str, images: Sequence[LabelledImage]) -> Path:
    """Write a canonical labelled manifest (primarily for tests / scaffolding)."""

    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = {"images": [image.as_dict() for image in images]}
    dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return dest.resolve()
