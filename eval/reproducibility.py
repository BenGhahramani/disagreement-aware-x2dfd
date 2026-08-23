"""Reproducibility metadata for labelled batch evaluation.

Hashes, git revision, model/checkpoint fingerprints, and threshold snapshots
used for provenance-aware resume. Does not run inference.

Material identity (invalidates raw-output reuse when changed):
- source / crop SHA-256
- inference config path **and** content SHA-256
- load_4bit
- specialists / LoRA / base-model artefact fingerprints
- run_name / experts_arg / already_cropped

Interpretation-only (rebuild summary; do **not** invalidate raw inference):
- prototype Stable/Uncertain cutoff (CONF_HIGH = 0.70)
- evidence-agreement specialist bands (EXPERT_LO = 0.30, EXPERT_HI = 0.70)

LLaVA base models are multi-GB. This module fingerprints the base model via
its identifier plus deterministic hashes of index / configuration / shard
metadata files rather than hashing every weight shard. Specialist and LoRA
checkpoints are hashed in full when present.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dashboard.view_model import EXPERT_HI, EXPERT_LO
from proof_of_concept.evaluator import CONF_HIGH

try:
    import yaml
except ImportError:  # pragma: no cover - yaml is a project dependency
    yaml = None  # type: ignore[assignment]


# Metadata files used to fingerprint a Hugging Face-style LLaVA base directory
# without hashing multi-GB weight shards.
_BASE_MODEL_METADATA_NAMES: Tuple[str, ...] = (
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "preprocessor_config.json",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path | str) -> str:
    """Return the SHA-256 hex digest of a file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_canonical(payload: Any) -> str:
    """SHA-256 of a JSON-serialisable object with sorted keys."""

    return sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def get_git_commit(project_root: Path | str | None = None) -> Optional[str]:
    """Return ``git rev-parse HEAD`` when available, else ``None``."""

    root = Path(project_root) if project_root is not None else Path.cwd()
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    commit = (result.stdout or "").strip()
    return commit or None


def get_git_dirty(project_root: Path | str | None = None) -> Optional[bool]:
    """Return whether the working tree has uncommitted changes.

    ``None`` when git is unavailable. Does not block batch execution.
    """

    root = Path(project_root) if project_root is not None else Path.cwd()
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return bool((result.stdout or "").strip())


def get_git_provenance(project_root: Path | str | None = None) -> Dict[str, Any]:
    return {
        "git_commit": get_git_commit(project_root),
        "git_dirty": get_git_dirty(project_root),
    }


def interpretation_thresholds() -> Dict[str, float]:
    """Prototype thresholds used only for interpretation — not inference.

    Includes:
    - ``prototype_stable_uncertain`` — evaluator Stable / Uncertain cutoff
    - ``evidence_expert_lo`` / ``evidence_expert_hi`` — evidence-agreement bands
    """

    return {
        "prototype_stable_uncertain": CONF_HIGH,
        "evidence_expert_lo": EXPERT_LO,
        "evidence_expert_hi": EXPERT_HI,
    }


def _resolve_under(root: Path, relative: str | Path) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute():
        return candidate.resolve()
    return (root / candidate).resolve()


def fingerprint_file(path: Path, *, role: str) -> Dict[str, Any]:
    """Full-file SHA-256 of a single checkpoint / weight file."""

    record: Dict[str, Any] = {
        "role": role,
        "path": str(path),
        "fingerprint_kind": "full_file_sha256",
        "exists": path.is_file(),
        "sha256": None,
    }
    if path.is_file():
        try:
            record["sha256"] = sha256_file(path)
        except OSError as exc:
            record["fingerprint_kind"] = "unreadable"
            record["error"] = str(exc)
    else:
        record["fingerprint_kind"] = "missing"
    return record


def fingerprint_lora_adapter(path: Path) -> Dict[str, Any]:
    """Hash adapter_config.json plus adapter weight files when present."""

    record: Dict[str, Any] = {
        "role": "lora_adapter",
        "path": str(path),
        "fingerprint_kind": "adapter_directory",
        "exists": path.exists(),
        "files": {},
        "sha256": None,
    }
    if not path.exists():
        record["fingerprint_kind"] = "missing"
        return record

    names = (
        "adapter_config.json",
        "adapter_model.safetensors",
        "adapter_model.bin",
        "pytorch_model.bin",
    )
    file_hashes: Dict[str, Optional[str]] = {}
    for name in names:
        candidate = path / name if path.is_dir() else path
        if path.is_dir() and candidate.is_file():
            try:
                file_hashes[name] = sha256_file(candidate)
            except OSError:
                file_hashes[name] = None
        elif path.is_file() and path.name == name:
            try:
                file_hashes[name] = sha256_file(path)
            except OSError:
                file_hashes[name] = None
    if not file_hashes and path.is_file():
        try:
            file_hashes[path.name] = sha256_file(path)
        except OSError:
            file_hashes[path.name] = None
    record["files"] = file_hashes
    record["sha256"] = sha256_canonical(file_hashes) if file_hashes else None
    if not file_hashes:
        record["fingerprint_kind"] = "missing_adapter_files"
    return record


def fingerprint_llava_base(path: Path) -> Dict[str, Any]:
    """Fingerprint a multi-GB LLaVA base via config / index metadata only.

    Weight shards (``*.safetensors``, ``pytorch_model-*.bin``) are **not**
    hashed. The model identifier (path) plus hashes of index and configuration
    files provide a practical, deterministic identity for resume checks.
    """

    record: Dict[str, Any] = {
        "role": "llava_base",
        "path": str(path),
        "fingerprint_kind": "base_model_metadata",
        "note": (
            "LLaVA base weights are not fully hashed; identity is the model "
            "path plus SHA-256 of config/index/tokenizer metadata files."
        ),
        "exists": path.exists(),
        "files": {},
        "sha256": None,
    }
    if not path.is_dir():
        record["fingerprint_kind"] = "missing"
        return record

    file_hashes: Dict[str, Optional[str]] = {}
    for name in _BASE_MODEL_METADATA_NAMES:
        candidate = path / name
        if candidate.is_file():
            try:
                file_hashes[name] = sha256_file(candidate)
            except OSError:
                file_hashes[name] = None
    record["files"] = file_hashes
    record["sha256"] = sha256_canonical(
        {"path": str(path), "files": file_hashes}
    )
    return record


def fingerprint_diffusion_checkpoint(
    weights_dir: Path,
    model_name: str,
) -> Dict[str, Any]:
    """Hash ``{weights_dir}/{model}/config.yaml`` and the configured weights file."""

    model_dir = weights_dir / model_name
    record: Dict[str, Any] = {
        "role": "diffusion_checkpoint",
        "path": str(model_dir),
        "model": model_name,
        "weights_dir": str(weights_dir),
        "fingerprint_kind": "diffusion_model_dir",
        "exists": model_dir.is_dir(),
        "files": {},
        "sha256": None,
    }
    if not model_dir.is_dir():
        record["fingerprint_kind"] = "missing"
        return record

    file_hashes: Dict[str, Optional[str]] = {}
    config_path = model_dir / "config.yaml"
    if config_path.is_file():
        try:
            file_hashes["config.yaml"] = sha256_file(config_path)
        except OSError:
            file_hashes["config.yaml"] = None
        weights_file = "checkpoint.pth"
        if yaml is not None:
            try:
                cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
                if isinstance(cfg, dict) and cfg.get("weights_file"):
                    weights_file = str(cfg["weights_file"])
            except (OSError, yaml.YAMLError):
                pass
        weight_path = model_dir / weights_file
        if weight_path.is_file():
            try:
                file_hashes[weights_file] = sha256_file(weight_path)
            except OSError:
                file_hashes[weights_file] = None
        else:
            file_hashes[weights_file] = None
    record["files"] = file_hashes
    record["sha256"] = sha256_canonical(file_hashes) if file_hashes else None
    return record


def collect_model_artefacts(
    inference_config_path: Path,
    *,
    project_root: Path,
) -> Dict[str, Any]:
    """Resolve and fingerprint model artefacts once per batch.

    Reads the inference YAML for LLaVA base/adapter and specialist weight paths.
    Missing weights are recorded as ``missing`` so a later appearance still
    changes the fingerprint.
    """

    config_path = inference_config_path.resolve()
    try:
        config_sha = sha256_file(config_path)
    except OSError:
        config_sha = None

    payload: Dict[str, Any] = {}
    if yaml is not None and config_path.is_file():
        try:
            loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                payload = loaded
        except (OSError, yaml.YAMLError):
            payload = {}

    model_section = payload.get("model") if isinstance(payload.get("model"), dict) else {}
    base_rel = model_section.get("base") or "weights/base/llava-v1.5-7b"
    adapter_rel = model_section.get("adapter") or ""

    blending_rel = "weights/blending_models/best_gf.pth"
    diffusion_dir_rel = "weights/"
    diffusion_model = "ours-sync"
    supplies = payload.get("weak_supplies")
    if isinstance(supplies, list):
        for entry in supplies:
            if not isinstance(entry, dict):
                continue
            provider = str(entry.get("provider") or "").lower()
            if provider == "blending" and entry.get("weights_path"):
                blending_rel = str(entry["weights_path"])
            if provider in {"diffusion_detector", "diffdet", "diffusion", "aligner"}:
                if entry.get("weights_dir"):
                    diffusion_dir_rel = str(entry["weights_dir"])
                if entry.get("model"):
                    diffusion_model = str(entry["model"])

    artefacts = [
        fingerprint_llava_base(_resolve_under(project_root, base_rel)),
        fingerprint_lora_adapter(_resolve_under(project_root, adapter_rel))
        if adapter_rel
        else {
            "role": "lora_adapter",
            "path": "",
            "fingerprint_kind": "missing",
            "exists": False,
            "sha256": None,
        },
        fingerprint_file(
            _resolve_under(project_root, blending_rel),
            role="blending_checkpoint",
        ),
        fingerprint_diffusion_checkpoint(
            _resolve_under(project_root, diffusion_dir_rel),
            diffusion_model,
        ),
    ]
    return {
        "inference_config_path": str(config_path),
        "inference_config_sha256": config_sha,
        "artefacts": artefacts,
        "artefacts_fingerprint": sha256_canonical(artefacts),
    }


@dataclass(frozen=True)
class MaterialSettings:
    """Settings that must match to reuse raw inference outputs."""

    source_sha256: str
    crop_sha256: str
    inference_config_path: str
    inference_config_sha256: str
    load_4bit: bool
    run_name: str
    experts_arg: str
    already_cropped: bool
    model_artefacts_fingerprint: str

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConfigProvenance:
    """Provenance recorded when a configuration finishes successfully."""

    run_name: str
    experts_arg: str
    material: MaterialSettings
    interpretation_thresholds: Dict[str, float] = field(
        default_factory=interpretation_thresholds
    )
    git_commit: Optional[str] = None
    git_dirty: Optional[bool] = None
    timestamp: str = field(default_factory=utc_timestamp)
    output_path: Optional[str] = None
    model_artefacts: Optional[Dict[str, Any]] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "run_name": self.run_name,
            "experts_arg": self.experts_arg,
            "material": self.material.as_dict(),
            "interpretation_thresholds": dict(self.interpretation_thresholds),
            "git_commit": self.git_commit,
            "git_dirty": self.git_dirty,
            "timestamp": self.timestamp,
            "output_path": self.output_path,
            "model_artefacts": self.model_artefacts,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ConfigProvenance":
        material_raw = payload.get("material") or {}
        if not isinstance(material_raw, dict):
            raise ValueError("config provenance material must be an object")
        material = MaterialSettings(
            source_sha256=str(material_raw["source_sha256"]),
            crop_sha256=str(material_raw["crop_sha256"]),
            inference_config_path=str(material_raw["inference_config_path"]),
            inference_config_sha256=str(
                material_raw.get("inference_config_sha256")
                or material_raw.get("inference_config_hash")
                or ""
            ),
            load_4bit=bool(material_raw["load_4bit"]),
            run_name=str(material_raw["run_name"]),
            experts_arg=str(material_raw["experts_arg"]),
            already_cropped=bool(material_raw["already_cropped"]),
            model_artefacts_fingerprint=str(
                material_raw.get("model_artefacts_fingerprint") or ""
            ),
        )
        thresholds = payload.get("interpretation_thresholds") or {}
        if not isinstance(thresholds, dict):
            thresholds = {}
        artefacts = payload.get("model_artefacts")
        return cls(
            run_name=str(payload.get("run_name", material.run_name)),
            experts_arg=str(payload.get("experts_arg", material.experts_arg)),
            material=material,
            interpretation_thresholds={
                str(k): float(v) for k, v in thresholds.items()
            },
            git_commit=payload.get("git_commit"),
            git_dirty=payload.get("git_dirty"),
            timestamp=str(payload.get("timestamp") or utc_timestamp()),
            output_path=payload.get("output_path"),
            model_artefacts=artefacts if isinstance(artefacts, dict) else None,
        )


@dataclass
class ImageProvenance:
    """Image-level reproducibility metadata for one batch item."""

    image_id: str
    source_path: str
    source_sha256: str
    crop_sha256: str
    ground_truth: str
    dataset: Optional[str] = None
    manipulation: Optional[str] = None
    source: Optional[str] = None
    already_cropped: bool = False
    notes: Optional[str] = None
    inference_config_path: str = ""
    inference_config_sha256: Optional[str] = None
    load_4bit: bool = True
    interpretation_thresholds: Dict[str, float] = field(
        default_factory=interpretation_thresholds
    )
    git_commit: Optional[str] = None
    git_dirty: Optional[bool] = None
    model_artefacts_fingerprint: Optional[str] = None
    timestamp: str = field(default_factory=utc_timestamp)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "image_id": self.image_id,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "crop_sha256": self.crop_sha256,
            "ground_truth": self.ground_truth,
            "dataset": self.dataset,
            "manipulation": self.manipulation,
            "source": self.source,
            "already_cropped": self.already_cropped,
            "notes": self.notes,
            "inference_config_path": self.inference_config_path,
            "inference_config_sha256": self.inference_config_sha256,
            "load_4bit": self.load_4bit,
            "interpretation_thresholds": dict(self.interpretation_thresholds),
            "git_commit": self.git_commit,
            "git_dirty": self.git_dirty,
            "model_artefacts_fingerprint": self.model_artefacts_fingerprint,
            "timestamp": self.timestamp,
        }


def material_settings_match(stored: MaterialSettings, current: MaterialSettings) -> bool:
    return stored.as_dict() == current.as_dict()


def interpretation_changed(
    stored: Optional[Dict[str, float]],
    current: Optional[Dict[str, float]] = None,
) -> bool:
    """True when interpretation thresholds differ from those recorded."""

    current = current if current is not None else interpretation_thresholds()
    if not stored:
        return True
    return {str(k): float(v) for k, v in stored.items()} != {
        str(k): float(v) for k, v in current.items()
    }


def load_config_provenance(path: Path) -> Optional[ConfigProvenance]:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    try:
        return ConfigProvenance.from_dict(payload)
    except (KeyError, TypeError, ValueError):
        return None


def write_config_provenance(path: Path, provenance: ConfigProvenance) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(provenance.as_dict(), indent=2) + "\n", encoding="utf-8")
    return path.resolve()
