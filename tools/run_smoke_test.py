#!/usr/bin/env python3
"""Run one image through the inherited X2DFD runner and prove the result is real.

This is a wrapper, not a second inference implementation: it shells out to
``eval.infer.runner`` exactly as a human would, then checks that the run
actually produced a usable prediction. "Exit code 0" is not treated as success
on its own - the output JSON must exist, parse, and carry a non-error ``gpt``
answer together with real and fake scores.

Stage 2 runs this with ``--experts none`` so the language model is exercised on
its own, before any detector is brought into the picture.

Exit codes:
    0   PASS  - inference ran and the output validated
    1   FAIL  - inference failed, timed out, hit CUDA OOM, or output was unusable
    2   USAGE - bad arguments or an invalid manifest

Examples::

    python -m tools.run_smoke_test --manifest datasets/raw/data/poc/demo_one.json
    python -m tools.run_smoke_test --manifest ... --load-4bit --timeout 3600
    python -m tools.run_smoke_test --manifest ... --experts none --json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

LOGGER = logging.getLogger("x2dfd.run_smoke_test")

PROJECT_ROOT_DEFAULT = Path(__file__).resolve().parents[1]

DEFAULT_MANIFEST = Path("datasets") / "raw" / "data" / "poc" / "demo_one.json"
DEFAULT_CONFIG = Path("eval") / "configs" / "infer_config.yaml"
DEFAULT_SUMMARY = Path("eval") / "outputs" / "smoke_test_summary.json"
DEFAULT_OUTPUT_DIR = Path("eval") / "outputs" / "smoke_test"
DEFAULT_TIMEOUT_S = 3600.0

SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# The runner prints this once it has written the conversation JSON.
SAVED_MARKER = re.compile(r"Saved conversation-style results:\s*(?P<path>.+)$", re.MULTILINE)

# Substrings that mean "the GPU ran out of memory", not "the model said no".
OOM_PATTERNS = (
    "cuda out of memory",
    "torch.cuda.outofmemoryerror",
    "outofmemoryerror",
    "cublas_status_alloc_failed",
    "cudnn_status_alloc_failed",
)

# lora_inference records failures inside the answer turn rather than raising.
INFERENCE_ERROR_PREFIX = "inference error:"

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_USAGE = 2


class ManifestError(ValueError):
    """The manifest could not be used to launch inference."""


class OutputError(ValueError):
    """The runner's output was missing, malformed or incomplete."""


@dataclass(frozen=True)
class ManifestInfo:
    path: Path
    root: Optional[str]
    images: List[Path]


@dataclass(frozen=True)
class ProcessResult:
    exit_code: Optional[int]
    stdout: str
    stderr: str
    timed_out: bool
    duration_s: float

    @property
    def combined_output(self) -> str:
        return f"{self.stdout}\n{self.stderr}"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "exit_code": self.exit_code,
            "timed_out": self.timed_out,
            "duration_s": round(self.duration_s, 3),
            "stdout": self.stdout,
            "stderr": self.stderr,
        }


@dataclass(frozen=True)
class Prediction:
    image: Optional[str]
    answer: str
    real_score: float
    fake_score: float
    label: Optional[str]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "image": self.image,
            "answer": self.answer,
            "real_score": self.real_score,
            "fake_score": self.fake_score,
            "label": self.label,
        }


@dataclass
class Summary:
    status: str
    manifest: str
    command: List[str]
    process: Dict[str, Any]
    output_path: Optional[str]
    predictions: List[Dict[str, Any]] = field(default_factory=list)
    failures: List[str] = field(default_factory=list)
    cuda_oom: bool = False
    peak_vram_mib: Optional[int] = None
    experts: str = "none"
    quantisation: str = "fp16"
    started_at: str = ""
    finished_at: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "manifest": self.manifest,
            "command": self.command,
            "experts": self.experts,
            "quantisation": self.quantisation,
            "process": self.process,
            "output_path": self.output_path,
            "predictions": self.predictions,
            "failures": self.failures,
            "cuda_oom": self.cuda_oom,
            "peak_vram_mib": self.peak_vram_mib,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


# --------------------------------------------------------------------------
# manifest
# --------------------------------------------------------------------------


def load_manifest(path: Path) -> ManifestInfo:
    """Read a dataset manifest and confirm every image it names is usable.

    Mirrors the runner's rule: a top-level ``Description`` is the only prefix
    applied to relative ``image_path`` entries.
    """

    if not path.is_file():
        raise ManifestError(f"manifest not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ManifestError(f"manifest is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ManifestError("manifest must be a JSON object")

    entries = payload.get("images")
    if not isinstance(entries, list) or not entries:
        raise ManifestError("manifest needs a non-empty 'images' list")

    root = payload.get("Description")
    if root is not None and not isinstance(root, str):
        raise ManifestError("manifest 'Description' must be a string when present")

    images: List[Path] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ManifestError(f"images[{index}] must be an object")
        raw = entry.get("image_path") or entry.get("path")
        if not isinstance(raw, str) or not raw.strip():
            raise ManifestError(f"images[{index}] has no usable 'image_path'")
        candidate = Path(raw)
        if not candidate.is_absolute():
            if not root:
                raise ManifestError(
                    f"images[{index}] is relative but the manifest has no 'Description' root"
                )
            candidate = Path(root) / raw
        if candidate.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            raise ManifestError(
                f"images[{index}] has unsupported extension {candidate.suffix!r}; "
                f"expected one of {sorted(SUPPORTED_IMAGE_SUFFIXES)}"
            )
        if not candidate.is_file():
            raise ManifestError(f"images[{index}] does not exist: {candidate}")
        images.append(candidate)

    return ManifestInfo(path=path, root=root, images=images)


# --------------------------------------------------------------------------
# command construction and execution
# --------------------------------------------------------------------------


def build_command(
    *,
    manifest: Path,
    output: Path,
    config: Path,
    experts: str,
    python_executable: str = sys.executable,
    extra_args: Optional[Sequence[str]] = None,
) -> List[str]:
    """Build the runner invocation. The output path is always explicit."""

    command = [
        python_executable,
        "-m",
        "eval.infer.runner",
        "--config",
        str(config),
        "--json",
        str(manifest),
        "--output",
        str(output),
        "--experts",
        experts,
    ]
    if extra_args:
        command.extend(extra_args)
    return command


def build_environment(
    *,
    base_env: Optional[Dict[str, str]] = None,
    load_4bit: bool = False,
    load_8bit: bool = False,
) -> Dict[str, str]:
    """Environment for the child process, opting into LLaVA's own quantisation."""

    env = dict(os.environ if base_env is None else base_env)
    env["USE_PROGRESS_BAR"] = "0"  # keep captured stdout parseable
    env["PYTHONIOENCODING"] = "utf-8"
    if load_4bit:
        env["X2DFD_LOAD_4BIT"] = "1"
    if load_8bit:
        env["X2DFD_LOAD_8BIT"] = "1"
    return env


def run_process(
    command: Sequence[str],
    *,
    timeout_s: float,
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
    runner: Callable[..., Any] = subprocess.run,
    clock: Callable[[], float] = time.perf_counter,
) -> ProcessResult:
    """Run the inference subprocess, capturing everything it emits."""

    started = clock()
    try:
        completed = runner(
            list(command),
            cwd=str(cwd) if cwd else None,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return ProcessResult(
            exit_code=None,
            stdout=_as_text(exc.stdout),
            stderr=_as_text(exc.stderr),
            timed_out=True,
            duration_s=clock() - started,
        )
    return ProcessResult(
        exit_code=completed.returncode,
        stdout=_as_text(completed.stdout),
        stderr=_as_text(completed.stderr),
        timed_out=False,
        duration_s=clock() - started,
    )


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def detect_cuda_oom(text: str) -> bool:
    """True when the captured output shows the GPU ran out of memory."""

    lowered = text.lower()
    return any(pattern in lowered for pattern in OOM_PATTERNS)


# --------------------------------------------------------------------------
# output location and validation
# --------------------------------------------------------------------------


def locate_output(expected: Path, stdout: str) -> Optional[Path]:
    """Find the JSON the runner wrote, trusting the filesystem over the log."""

    if expected.is_file():
        return expected
    match = SAVED_MARKER.search(stdout or "")
    if match:
        reported = Path(match.group("path").strip())
        if reported.is_file():
            return reported
    return None


def _turn_value(conversations: Sequence[Any], sender: str) -> Optional[str]:
    for turn in conversations:
        if isinstance(turn, dict) and turn.get("from") == sender:
            value = turn.get("value")
            if isinstance(value, str):
                return value
    return None


def _label_from_answer(answer: str) -> Optional[str]:
    lowered = answer.lower()
    has_fake = "fake" in lowered
    has_real = "real" in lowered
    if has_fake and not has_real:
        return "fake"
    if has_real and not has_fake:
        return "real"
    return None


def validate_output(path: Path) -> List[Prediction]:
    """Parse the runner's conversation JSON and require a usable prediction.

    Raises:
        OutputError: unreadable JSON, wrong shape, an error answer, or missing
            real/fake scores.
    """

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise OutputError(f"cannot read output {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise OutputError(f"output {path} is not valid JSON: {exc}") from exc

    if not isinstance(payload, list) or not payload:
        raise OutputError("output must be a non-empty JSON list of items")

    predictions: List[Prediction] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise OutputError(f"item {index} is not an object")
        conversations = item.get("conversations")
        if not isinstance(conversations, list) or not conversations:
            raise OutputError(f"item {index} has no conversations")

        answer = _turn_value(conversations, "gpt")
        if answer is None or not answer.strip():
            raise OutputError(f"item {index} has no 'gpt' prediction")
        if answer.strip().lower().startswith(INFERENCE_ERROR_PREFIX):
            raise OutputError(f"item {index} recorded an inference error: {answer.strip()}")

        real_raw = _turn_value(conversations, "real score")
        fake_raw = _turn_value(conversations, "fake score")
        missing = [
            name
            for name, value in (("real score", real_raw), ("fake score", fake_raw))
            if value is None
        ]
        if missing:
            raise OutputError(f"item {index} is missing {' and '.join(missing)} turn(s)")

        try:
            real_score = float(real_raw)  # type: ignore[arg-type]
            fake_score = float(fake_raw)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise OutputError(f"item {index} has non-numeric scores: {exc}") from exc

        image = item.get("image")
        predictions.append(
            Prediction(
                image=image if isinstance(image, str) else None,
                answer=answer.strip(),
                real_score=real_score,
                fake_score=fake_score,
                label=_label_from_answer(answer),
            )
        )
    return predictions


# --------------------------------------------------------------------------
# VRAM sampling (best effort)
# --------------------------------------------------------------------------


class VramSampler:
    """Poll nvidia-smi in the background to record peak used VRAM.

    Entirely optional: if nvidia-smi is unavailable the peak stays None rather
    than failing the smoke test.
    """

    def __init__(self, interval_s: float = 2.0, enabled: bool = True) -> None:
        self._interval_s = interval_s
        self._enabled = enabled and shutil.which("nvidia-smi") is not None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.peak_mib: Optional[int] = None

    def _sample_once(self) -> Optional[int]:
        try:
            completed = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        values: List[int] = []
        for line in (completed.stdout or "").splitlines():
            line = line.strip()
            if line.isdigit():
                values.append(int(line))
        return max(values) if values else None

    def _loop(self) -> None:
        while not self._stop.is_set():
            value = self._sample_once()
            if value is not None and (self.peak_mib is None or value > self.peak_mib):
                self.peak_mib = value
            self._stop.wait(self._interval_s)

    def __enter__(self) -> "VramSampler":
        if self._enabled:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)


# --------------------------------------------------------------------------
# summary
# --------------------------------------------------------------------------


def write_summary(path: Path, summary: Summary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary.as_dict(), indent=2) + "\n", encoding="utf-8")


def render_summary(summary: Summary) -> str:
    lines = [
        "X2DFD single-image smoke test",
        "=" * 60,
        f"status     : {summary.status}",
        f"manifest   : {summary.manifest}",
        f"experts    : {summary.experts}",
        f"loading    : {summary.quantisation}",
        f"exit code  : {summary.process.get('exit_code')}",
        f"runtime    : {summary.process.get('duration_s')} s",
    ]
    if summary.peak_vram_mib is not None:
        lines.append(f"peak VRAM  : {summary.peak_vram_mib} MiB")
    if summary.output_path:
        lines.append(f"output     : {summary.output_path}")
    for prediction in summary.predictions:
        lines.append(
            "prediction : %s | real=%.4f fake=%.4f | %r"
            % (
                prediction.get("label"),
                prediction.get("real_score", float("nan")),
                prediction.get("fake_score", float("nan")),
                prediction.get("answer"),
            )
        )
    for failure in summary.failures:
        lines.append(f"FAILURE    : {failure}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Options:
    manifest: Path
    config: Path
    output: Optional[Path]
    summary: Path
    experts: str
    timeout_s: float
    load_4bit: bool
    load_8bit: bool
    project_root: Path
    sample_vram: bool
    as_json: bool


def parse_args(argv: Optional[Sequence[str]] = None) -> Options:
    parser = argparse.ArgumentParser(
        prog="python -m tools.run_smoke_test",
        description="Run one image through the X2DFD runner and validate the result.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Exit codes: 0 pass, 1 inference or validation failure, 2 usage error.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"dataset manifest JSON (default: {DEFAULT_MANIFEST})",
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG, help=f"eval config (default: {DEFAULT_CONFIG})"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"where the runner should write results (default: {DEFAULT_OUTPUT_DIR}/<manifest>_result.json)",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY,
        help=f"machine-readable summary destination (default: {DEFAULT_SUMMARY})",
    )
    parser.add_argument(
        "--experts",
        default="none",
        help="experts to enable, passed straight to the runner (default: none)",
    )
    parser.add_argument(
        "--timeout",
        dest="timeout_s",
        type=float,
        default=DEFAULT_TIMEOUT_S,
        help=f"seconds before the inference subprocess is killed (default: {DEFAULT_TIMEOUT_S:.0f})",
    )
    parser.add_argument(
        "--load-4bit",
        action="store_true",
        help="use LLaVA's built-in 4-bit loading (needed on a 10 GiB card)",
    )
    parser.add_argument(
        "--load-8bit", action="store_true", help="use LLaVA's built-in 8-bit loading"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT_DEFAULT,
        help="repository root used as the subprocess working directory",
    )
    parser.add_argument(
        "--no-vram-sampling", action="store_true", help="skip nvidia-smi peak VRAM polling"
    )
    parser.add_argument("--json", dest="as_json", action="store_true", help="print the summary as JSON")
    parser.add_argument("-v", "--verbose", action="store_true", help="enable debug logging")

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if args.load_4bit and args.load_8bit:
        parser.error("--load-4bit and --load-8bit are mutually exclusive")
    if args.timeout_s <= 0:
        parser.error("--timeout must be positive")

    return Options(
        manifest=args.manifest,
        config=args.config,
        output=args.output,
        summary=args.summary,
        experts=args.experts,
        timeout_s=args.timeout_s,
        load_4bit=args.load_4bit,
        load_8bit=args.load_8bit,
        project_root=args.project_root,
        sample_vram=not args.no_vram_sampling,
        as_json=args.as_json,
    )


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else (root / path)


def main(argv: Optional[Sequence[str]] = None, *, runner: Optional[Callable[..., Any]] = None) -> int:
    options = parse_args(argv)
    root = options.project_root

    manifest_path = _resolve(root, options.manifest)
    try:
        manifest = load_manifest(manifest_path)
    except ManifestError as exc:
        LOGGER.error("%s", exc)
        return EXIT_USAGE

    LOGGER.info("manifest OK: %d image(s)", len(manifest.images))

    if options.output is not None:
        output_path = _resolve(root, options.output)
    else:
        output_path = _resolve(root, DEFAULT_OUTPUT_DIR) / f"{manifest_path.stem}_result.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # A stale file from an earlier run must not be mistaken for a fresh result.
    if output_path.exists():
        output_path.unlink()

    command = build_command(
        manifest=manifest_path,
        output=output_path,
        config=_resolve(root, options.config),
        experts=options.experts,
    )
    environment = build_environment(load_4bit=options.load_4bit, load_8bit=options.load_8bit)

    quantisation = "4-bit" if options.load_4bit else ("8-bit" if options.load_8bit else "fp16")
    summary = Summary(
        status="fail",
        manifest=str(manifest_path),
        command=command,
        process={},
        output_path=None,
        experts=options.experts,
        quantisation=quantisation,
        started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )

    LOGGER.info("running: %s", " ".join(command))
    with VramSampler(enabled=options.sample_vram) as sampler:
        result = run_process(
            command,
            timeout_s=options.timeout_s,
            cwd=root,
            env=environment,
            runner=runner or subprocess.run,
        )
    summary.peak_vram_mib = sampler.peak_mib
    summary.process = result.as_dict()
    summary.finished_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    summary.cuda_oom = detect_cuda_oom(result.combined_output)

    failures: List[str] = []
    if result.timed_out:
        failures.append(f"inference exceeded the {options.timeout_s:.0f}s timeout")
    if summary.cuda_oom:
        failures.append("CUDA out of memory reported by the inference subprocess")
    if not result.timed_out and result.exit_code != 0:
        failures.append(f"inference exited with code {result.exit_code}")

    located = locate_output(output_path, result.stdout)
    if located is None:
        failures.append(f"no output JSON produced at {output_path}")
    else:
        summary.output_path = str(located)
        try:
            predictions = validate_output(located)
            summary.predictions = [p.as_dict() for p in predictions]
        except OutputError as exc:
            failures.append(str(exc))

    summary.failures = failures
    summary.status = "pass" if not failures else "fail"

    summary_path = _resolve(root, options.summary)
    try:
        write_summary(summary_path, summary)
    except OSError as exc:
        LOGGER.error("cannot write summary %s: %s", summary_path, exc)
        return EXIT_FAIL

    if options.as_json:
        print(json.dumps(summary.as_dict(), indent=2))
    else:
        print(render_summary(summary))
        print(f"summary    : {summary_path}")

    if failures:
        for failure in failures:
            LOGGER.error("%s", failure)
        return EXIT_FAIL
    return EXIT_PASS


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
