#!/usr/bin/env python3
"""Verify that this machine can run X2DFD one-image inference.

The checker is read-only apart from creating (and cleaning up inside) the
output directory it is asked to validate. It never downloads weights and never
loads a model; it only inspects the interpreter, installed packages, GPU and
the paths referenced by an X2DFD eval config.

Exit codes:
    0   PASS  - every required check succeeded
    1   FAIL  - at least one required check failed (or a warning with --strict)
    2   USAGE - the config could not be read / arguments were invalid

Examples::

    python -m tools.check_environment
    python -m tools.check_environment --config eval/configs/infer_config.yaml
    python -m tools.check_environment --json --output eval/outputs/env_check.json
    python -m tools.check_environment --skip-weights --skip-datasets
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

LOGGER = logging.getLogger("x2dfd.check_environment")

PROJECT_ROOT_DEFAULT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_RELATIVE = Path("eval") / "configs" / "infer_config.yaml"

MIN_PYTHON: Tuple[int, int] = (3, 10)
RECOMMENDED_PYTHON: Tuple[int, int] = (3, 10)

# LLaVA-1.5-7B in fp16 needs roughly this much VRAM without quantisation.
RECOMMENDED_VRAM_GIB: float = 14.0

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_USAGE = 2


class Status(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    SKIP = "SKIP"


@dataclass(frozen=True)
class CheckResult:
    """One line of the environment report."""

    name: str
    status: Status
    detail: str
    required: bool = True

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "detail": self.detail,
            "required": self.required,
        }


@dataclass(frozen=True)
class ImportSpec:
    """A python module the pipeline imports, and why."""

    module: str
    purpose: str
    required: bool = True


@dataclass(frozen=True)
class PathRequirement:
    """A file or directory the pipeline reads at runtime."""

    label: str
    path: Path
    kind: str = "file"  # "file" | "dir"
    required: bool = True


@dataclass
class Options:
    """Resolved CLI options."""

    project_root: Path
    config_path: Path
    check_weights: bool = True
    check_datasets: bool = True
    strict: bool = False
    dataset_sample_images: int = 1
    extra_config_paths: List[Path] = field(default_factory=list)


# Imports that the inherited pipeline needs for LoRA inference with both experts.
CORE_IMPORTS: Tuple[ImportSpec, ...] = (
    ImportSpec("torch", "tensor runtime for LLaVA and both detectors"),
    ImportSpec("torchvision", "image transforms used by the blending detector"),
    ImportSpec("transformers", "LLaVA backbone loading"),
    ImportSpec("accelerate", "device mapping used by LLaVA model builder"),
    ImportSpec("peft", "LoRA adapter loading"),
    ImportSpec("llava", "upstream LLaVA package used by utils/lora_inference.py"),
    ImportSpec("timm", "SwinV2 backbone for src/blending/detector.py"),
    ImportSpec("cv2", "image decoding in src/blending/detector.py"),
    ImportSpec("numpy", "array maths across the pipeline"),
    ImportSpec("PIL", "image loading in utils/lora_inference.py"),
    ImportSpec("yaml", "eval/train config parsing"),
    ImportSpec("tqdm", "progress bars in runner and providers"),
    ImportSpec("einops", "tensor reshaping in LLaVA"),
    ImportSpec("sentencepiece", "LLaVA tokenizer"),
    ImportSpec("safetensors", "weight loading"),
)

OPTIONAL_IMPORTS: Tuple[ImportSpec, ...] = (
    ImportSpec("bitsandbytes", "optional 4/8-bit loading (helps on <14 GiB VRAM)", required=False),
    ImportSpec("deepspeed", "optional, training only", required=False),
    ImportSpec("pytest", "running this repository's tests", required=False),
)


def default_module_finder(module: str) -> bool:
    """Return True when ``module`` can be located without importing it."""
    import importlib.util

    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError, AttributeError) as exc:
        LOGGER.debug("find_spec(%s) raised %s", module, exc)
        return False


def default_torch_importer() -> Any:
    import torch

    return torch


def check_python(
    version_info: Sequence[int] = sys.version_info,
    *,
    minimum: Tuple[int, int] = MIN_PYTHON,
    recommended: Tuple[int, int] = RECOMMENDED_PYTHON,
) -> CheckResult:
    """Check the running interpreter against the documented Python version."""
    major, minor = int(version_info[0]), int(version_info[1])
    printable = f"{major}.{minor}.{int(version_info[2]) if len(version_info) > 2 else 0}"
    if (major, minor) < minimum:
        return CheckResult(
            "python.version",
            Status.FAIL,
            f"Python {printable} is below the minimum {minimum[0]}.{minimum[1]}",
        )
    if (major, minor) != recommended:
        return CheckResult(
            "python.version",
            Status.WARN,
            (
                f"Python {printable} differs from the documented "
                f"{recommended[0]}.{recommended[1]} (install.sh creates a 3.10 conda env; "
                "the pinned LLaVA/transformers wheels are built against 3.10)"
            ),
        )
    return CheckResult("python.version", Status.PASS, f"Python {printable}")


def check_imports(
    specs: Iterable[ImportSpec],
    finder: Callable[[str], bool] = default_module_finder,
) -> List[CheckResult]:
    """Report availability of each module in ``specs``."""
    results: List[CheckResult] = []
    for spec in specs:
        available = finder(spec.module)
        if available:
            results.append(
                CheckResult(f"import.{spec.module}", Status.PASS, "importable", spec.required)
            )
        else:
            results.append(
                CheckResult(
                    f"import.{spec.module}",
                    Status.FAIL if spec.required else Status.WARN,
                    f"not installed - needed for: {spec.purpose}",
                    spec.required,
                )
            )
    return results


def check_torch(
    torch_module: Optional[Any] = None,
    *,
    importer: Callable[[], Any] = default_torch_importer,
    recommended_vram_gib: float = RECOMMENDED_VRAM_GIB,
) -> List[CheckResult]:
    """Report torch version, CUDA availability, GPU name and VRAM."""
    if torch_module is None:
        try:
            torch_module = importer()
        except ImportError as exc:
            return [CheckResult("torch.import", Status.FAIL, f"cannot import torch: {exc}")]

    results: List[CheckResult] = [
        CheckResult(
            "torch.version",
            Status.PASS,
            f"torch {getattr(torch_module, '__version__', 'unknown')} "
            f"(built for CUDA {getattr(getattr(torch_module, 'version', None), 'cuda', None)})",
        )
    ]

    try:
        cuda_available = bool(torch_module.cuda.is_available())
    except (AttributeError, RuntimeError, OSError) as exc:
        return results + [
            CheckResult("torch.cuda", Status.FAIL, f"CUDA probe failed: {type(exc).__name__}: {exc}")
        ]

    if not cuda_available:
        results.append(
            CheckResult(
                "torch.cuda",
                Status.FAIL,
                "torch.cuda.is_available() is False - LLaVA-1.5-7B inference on CPU is not practical",
            )
        )
        return results

    results.append(CheckResult("torch.cuda", Status.PASS, "CUDA available"))

    try:
        device_count = int(torch_module.cuda.device_count())
    except (AttributeError, RuntimeError) as exc:
        results.append(
            CheckResult("torch.cuda.devices", Status.WARN, f"device_count failed: {exc}", required=False)
        )
        return results

    for index in range(device_count):
        try:
            name = str(torch_module.cuda.get_device_name(index))
            props = torch_module.cuda.get_device_properties(index)
            total_gib = float(props.total_memory) / (1024**3)
        except (AttributeError, RuntimeError, OSError) as exc:
            results.append(
                CheckResult(
                    f"gpu.{index}",
                    Status.WARN,
                    f"could not query device {index}: {type(exc).__name__}: {exc}",
                    required=False,
                )
            )
            continue

        detail = f"{name} with {total_gib:.1f} GiB VRAM"
        if total_gib + 0.05 < recommended_vram_gib:
            results.append(
                CheckResult(
                    f"gpu.{index}",
                    Status.WARN,
                    (
                        f"{detail} - below the ~{recommended_vram_gib:.0f} GiB needed for "
                        "fp16 LLaVA-1.5-7B; expect to need 4-bit loading or CPU offload"
                    ),
                    required=False,
                )
            )
        else:
            results.append(CheckResult(f"gpu.{index}", Status.PASS, detail, required=False))
    return results


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load a YAML eval config. Raises ``ConfigError`` on any problem."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - yaml is a hard dependency of the repo
        raise ConfigError(f"PyYAML is required to read {config_path}: {exc}") from exc

    try:
        raw = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read config {config_path}: {exc}") from exc

    try:
        parsed = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {config_path}: {exc}") from exc

    if parsed is None:
        return {}
    if not isinstance(parsed, dict):
        raise ConfigError(f"config {config_path} must be a mapping, got {type(parsed).__name__}")
    return parsed


class ConfigError(RuntimeError):
    """Raised when the eval config cannot be used."""


def resolve_path(raw: str, project_root: Path) -> Path:
    """Expand env vars and resolve ``raw`` relative to the project root.

    Mirrors ``eval/infer/runner.py`` path handling so the checker validates the
    same locations the runner will use.
    """
    expanded = os.path.expandvars(str(raw))
    candidate = Path(expanded)
    if candidate.is_absolute():
        return candidate
    return (project_root / candidate).resolve()


def collect_weight_requirements(config: Dict[str, Any], project_root: Path) -> List[PathRequirement]:
    """Derive the weight files/directories referenced by an eval config."""
    requirements: List[PathRequirement] = []

    model_cfg = config.get("model") if isinstance(config.get("model"), dict) else {}
    adapter = model_cfg.get("adapter")
    base = model_cfg.get("base")

    if isinstance(adapter, str) and adapter.strip():
        requirements.append(
            PathRequirement("weights.model.adapter", resolve_path(adapter, project_root), "dir")
        )
    if isinstance(base, str) and base.strip():
        # The runner only needs the base model when the adapter is a LoRA dir,
        # which is the documented default configuration.
        requirements.append(
            PathRequirement("weights.model.base", resolve_path(base, project_root), "dir")
        )

    supplies = config.get("weak_supplies")
    entries: List[Dict[str, Any]] = list(supplies) if isinstance(supplies, list) else []
    single = config.get("weak_supply")
    if isinstance(single, dict):
        entries.append(single)

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        provider = str(entry.get("provider") or "").strip().lower()
        alias = str(entry.get("alias") or provider or "expert").strip()
        if provider == "blending":
            weights_path = entry.get("weights_path")
            if isinstance(weights_path, str) and weights_path.strip():
                requirements.append(
                    PathRequirement(
                        f"weights.expert.{alias}",
                        resolve_path(weights_path, project_root),
                        "file",
                    )
                )
        elif provider in {"aligner", "diffusion", "diffusion_detector", "diffdet", "forensics"}:
            weights_dir = entry.get("weights_dir")
            model_name = entry.get("model")
            if isinstance(weights_dir, str) and weights_dir.strip() and isinstance(model_name, str):
                model_dir = resolve_path(weights_dir, project_root) / model_name
                requirements.append(
                    PathRequirement(f"weights.expert.{alias}.dir", model_dir, "dir")
                )
                requirements.append(
                    PathRequirement(
                        f"weights.expert.{alias}.config",
                        model_dir / "config.yaml",
                        "file",
                    )
                )
    return requirements


def check_paths(requirements: Iterable[PathRequirement]) -> List[CheckResult]:
    """Check existence (and type) of each required path."""
    results: List[CheckResult] = []
    for requirement in requirements:
        path = requirement.path
        if requirement.kind == "dir":
            exists = path.is_dir()
            expectation = "directory"
        else:
            exists = path.is_file()
            expectation = "file"
        if exists:
            results.append(CheckResult(requirement.label, Status.PASS, str(path), requirement.required))
        else:
            results.append(
                CheckResult(
                    requirement.label,
                    Status.FAIL if requirement.required else Status.WARN,
                    f"missing {expectation}: {path}",
                    requirement.required,
                )
            )
    return results


def check_writable_dir(path: Path, *, label: str = "output.writable") -> CheckResult:
    """Ensure ``path`` exists and accepts writes; the probe file is removed."""
    probe = path / ".x2dfd_write_probe"
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe.write_text("ok", encoding="utf-8")
    except OSError as exc:
        return CheckResult(label, Status.FAIL, f"not writable: {path} ({type(exc).__name__}: {exc})")
    finally:
        try:
            probe.unlink()
        except OSError:
            LOGGER.debug("could not remove write probe %s", probe)
    return CheckResult(label, Status.PASS, f"writable: {path}")


def check_dataset_inputs(
    config: Dict[str, Any],
    project_root: Path,
    *,
    sample_images: int = 1,
) -> List[CheckResult]:
    """Check that configured dataset JSONs exist and their images resolve."""
    infer_cfg = config.get("infer") if isinstance(config.get("infer"), dict) else {}
    inputs = infer_cfg.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        return [
            CheckResult(
                "datasets.inputs",
                Status.WARN,
                "no infer.inputs configured; a dataset JSON must be passed with --json",
                required=False,
            )
        ]

    results: List[CheckResult] = []
    for raw in inputs:
        if not isinstance(raw, str):
            continue
        json_path = resolve_path(raw, project_root)
        label = f"datasets.{Path(raw).name}"
        if not json_path.is_file():
            results.append(CheckResult(label, Status.FAIL, f"missing dataset JSON: {json_path}"))
            continue
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            results.append(
                CheckResult(label, Status.FAIL, f"unreadable dataset JSON {json_path}: {exc}")
            )
            continue

        images = payload.get("images") if isinstance(payload, dict) else None
        if not isinstance(images, list) or not images:
            results.append(CheckResult(label, Status.FAIL, f"no 'images' list in {json_path}"))
            continue

        description = payload.get("Description") if isinstance(payload, dict) else None
        missing: List[str] = []
        checked = 0
        for item in images[: max(1, sample_images)]:
            if not isinstance(item, dict):
                continue
            rel = item.get("image_path") or item.get("path")
            if not isinstance(rel, str):
                continue
            candidate = Path(os.path.expandvars(rel))
            if not candidate.is_absolute():
                if not isinstance(description, str) or not description.strip():
                    missing.append(f"{rel} (no top-level 'Description' root in JSON)")
                    checked += 1
                    continue
                candidate = Path(os.path.expandvars(description)) / rel
            checked += 1
            if not candidate.exists():
                missing.append(str(candidate))

        if checked == 0:
            results.append(CheckResult(label, Status.FAIL, f"no usable image entries in {json_path}"))
        elif missing:
            results.append(
                CheckResult(
                    label,
                    Status.FAIL,
                    f"image files not found on this machine (checked {checked}): " + "; ".join(missing),
                )
            )
        else:
            results.append(
                CheckResult(label, Status.PASS, f"{json_path.name}: {checked} sampled image(s) exist")
            )
    return results


def run_checks(
    options: Options,
    *,
    finder: Callable[[str], bool] = default_module_finder,
    torch_module: Optional[Any] = None,
    torch_importer: Callable[[], Any] = default_torch_importer,
) -> List[CheckResult]:
    """Execute every enabled check and return the results in report order."""
    results: List[CheckResult] = [check_python()]
    results.extend(check_imports(CORE_IMPORTS, finder=finder))
    results.extend(check_imports(OPTIONAL_IMPORTS, finder=finder))
    results.extend(check_torch(torch_module, importer=torch_importer))

    config_paths = [options.config_path, *options.extra_config_paths]
    results.extend(
        check_paths(
            [
                PathRequirement(f"config.{p.name}", p, "file")
                for p in config_paths
            ]
        )
    )

    try:
        config = load_config(options.config_path)
    except ConfigError as exc:
        results.append(CheckResult("config.parse", Status.FAIL, str(exc)))
        return results

    results.append(CheckResult("config.parse", Status.PASS, f"parsed {options.config_path}"))

    if options.check_weights:
        requirements = collect_weight_requirements(config, options.project_root)
        if requirements:
            results.extend(check_paths(requirements))
        else:
            results.append(
                CheckResult(
                    "weights.requirements",
                    Status.WARN,
                    "config declares no model/expert weights",
                    required=False,
                )
            )
    else:
        results.append(CheckResult("weights", Status.SKIP, "skipped via --skip-weights", required=False))

    paths_cfg = config.get("paths") if isinstance(config.get("paths"), dict) else {}
    results_dir_raw = paths_cfg.get("results_dir") or str(Path("eval") / "outputs")
    results_dir = resolve_path(str(results_dir_raw), options.project_root)
    results.append(check_writable_dir(results_dir, label="output.results_dir"))

    if options.check_datasets:
        results.extend(
            check_dataset_inputs(
                config, options.project_root, sample_images=options.dataset_sample_images
            )
        )
    else:
        results.append(
            CheckResult("datasets", Status.SKIP, "skipped via --skip-datasets", required=False)
        )

    return results


def summarise(results: Sequence[CheckResult], *, strict: bool = False) -> Tuple[bool, Dict[str, int]]:
    """Return ``(passed, counts)`` for a sequence of results."""
    counts: Dict[str, int] = {status.value: 0 for status in Status}
    for result in results:
        counts[result.status.value] += 1
    hard_failures = any(r.status is Status.FAIL and r.required for r in results)
    soft_failures = any(
        r.status is Status.WARN or (r.status is Status.FAIL and not r.required) for r in results
    )
    passed = not hard_failures and not (strict and soft_failures)
    return passed, counts


def render_text(results: Sequence[CheckResult], *, strict: bool = False) -> str:
    """Human-readable report ending in a single PASS/FAIL line."""
    width = max((len(r.name) for r in results), default=10)
    lines = ["X2DFD environment check", "=" * 60]
    for result in results:
        flag = "" if result.required else " (optional)"
        lines.append(f"[{result.status.value:<4}] {result.name:<{width}}  {result.detail}{flag}")
    passed, counts = summarise(results, strict=strict)
    lines.append("=" * 60)
    lines.append(
        "PASS={PASS} WARN={WARN} FAIL={FAIL} SKIP={SKIP}".format(**counts)
    )
    verdict = "PASS" if passed else "FAIL"
    reason = ""
    if not passed:
        blocking = [r.name for r in results if r.status is Status.FAIL and r.required]
        if not blocking and strict:
            blocking = [r.name for r in results if r.status is Status.WARN]
            reason = " (warnings are fatal under --strict)"
        reason = f" - blocking: {', '.join(blocking)}{reason}"
    lines.append(f"OVERALL: {verdict}{reason}")
    return "\n".join(lines)


def build_report(results: Sequence[CheckResult], *, strict: bool = False) -> Dict[str, Any]:
    """Machine-readable report body."""
    passed, counts = summarise(results, strict=strict)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "overall": "PASS" if passed else "FAIL",
        "exit_code": EXIT_PASS if passed else EXIT_FAIL,
        "strict": strict,
        "counts": counts,
        "checks": [r.as_dict() for r in results],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.check_environment",
        description="Verify Python, packages, GPU, weights and dataset paths for X2DFD inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT_DEFAULT,
        help="Repository root used to resolve relative config paths",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Eval config to validate (default: <project-root>/eval/configs/infer_config.yaml)",
    )
    parser.add_argument(
        "--also-config",
        type=Path,
        action="append",
        default=[],
        help="Additional config file that must exist (repeatable)",
    )
    parser.add_argument("--skip-weights", action="store_true", help="Do not check weight paths")
    parser.add_argument("--skip-datasets", action="store_true", help="Do not check dataset JSONs/images")
    parser.add_argument(
        "--dataset-sample-images",
        type=int,
        default=1,
        help="How many image entries to probe per dataset JSON",
    )
    parser.add_argument("--strict", action="store_true", help="Treat warnings as failures")
    parser.add_argument("--json", action="store_true", help="Print the JSON report instead of text")
    parser.add_argument("--output", type=Path, default=None, help="Also write the JSON report here")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    project_root: Path = args.project_root.resolve()
    if not project_root.is_dir():
        parser.error(f"--project-root is not a directory: {project_root}")

    config_path: Path = (args.config or (project_root / DEFAULT_CONFIG_RELATIVE)).resolve()
    if args.dataset_sample_images < 1:
        parser.error("--dataset-sample-images must be >= 1")

    options = Options(
        project_root=project_root,
        config_path=config_path,
        check_weights=not args.skip_weights,
        check_datasets=not args.skip_datasets,
        strict=args.strict,
        dataset_sample_images=args.dataset_sample_images,
        extra_config_paths=[p.resolve() for p in args.also_config],
    )

    results = run_checks(options)
    report = build_report(results, strict=options.strict)

    if args.output is not None:
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        except OSError as exc:
            LOGGER.error("could not write report to %s: %s", args.output, exc)
            return EXIT_USAGE

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(render_text(results, strict=options.strict))

    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
