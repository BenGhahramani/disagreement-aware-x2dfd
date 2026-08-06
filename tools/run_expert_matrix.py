#!/usr/bin/env python3
"""Run one image through X2DFD once per expert configuration and compare.

Stage 2 proved a single ``--experts none`` run works. This runs the same image
under each expert setting - none, blending, diffusion, both - so the effect of
the experts can actually be measured rather than assumed.

Two design points worth stating:

*Separate subprocesses.* Each configuration is a fresh ``eval.infer.runner``
process. A 7B LLaVA plus a detector does not comfortably share 10 GiB, and
process exit is the only reliable way to hand the VRAM back. It also means one
configuration crashing cannot poison the next.

*Filenames the POC already understands.* Results are written as
``demo_<run_name>.json`` in one directory, which is exactly what
``proof_of_concept.normaliser.load_scenario`` expects, alongside the
``runtimes.json`` sidecar it looks for. No POC code needs to change to consume
a real run.

All inference mechanics (manifest validation, subprocess capture, OOM
detection, output validation, VRAM sampling) are reused from
``tools.run_smoke_test`` so there is one implementation of each.

Exit codes:
    0   PASS  - every configuration ran and validated
    1   FAIL  - at least one configuration failed
    2   USAGE - bad arguments or an invalid manifest

Examples::

    python -m tools.run_expert_matrix --manifest datasets/raw/data/poc/demo_one_crop.json --load-4bit
    python -m tools.run_expert_matrix --manifest ... --configs none blending --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from tools.run_smoke_test import (
    ManifestError,
    OutputError,
    Prediction,
    ProcessResult,
    VramSampler,
    build_command,
    build_environment,
    detect_cuda_oom,
    load_manifest,
    locate_output,
    run_process,
    validate_output,
)

LOGGER = logging.getLogger("x2dfd.run_expert_matrix")

PROJECT_ROOT_DEFAULT = Path(__file__).resolve().parents[1]

DEFAULT_MANIFEST = Path("datasets") / "raw" / "data" / "poc" / "demo_one_crop.json"
DEFAULT_CONFIG = Path("eval") / "configs" / "infer_config.yaml"
DEFAULT_OUTPUT_DIR = Path("eval") / "outputs" / "expert_matrix"
DEFAULT_SUMMARY = Path("eval") / "outputs" / "expert_matrix_summary.json"
DEFAULT_TIMEOUT_S = 3600.0
DEFAULT_CONFIGS: Tuple[str, ...] = ("none", "blending", "diffusion", "blending,diffusion")

# Canonical expert names, in the order they appear in a run name. Values are the
# token the runner matches (against a weak_supplies provider *or* alias, see
# eval/infer/runner.py::_filter_experts); keys are every spelling we accept.
EXPERT_ALIASES: Dict[str, str] = {
    "blending": "blending",
    "blend": "blending",
    "ble": "blending",
    "diffusion": "diffusion",
    "diffusion_detector": "diffusion",
    "diffdet": "diffusion",
    "aligner": "diffusion",
    "diff": "diffusion",
}
EXPERT_ORDER: Tuple[str, ...] = ("blending", "diffusion")

# The prompt tail the runner builds, e.g. "And the blending score is 0.8120,
# and the diffusion score is N/A." - see eval/infer/runner.py::_format_multi_scores.
PROMPT_SCORE_RE = re.compile(
    r"the\s+(?P<alias>[A-Za-z0-9_]+)\s+score\s+is\s+(?P<score>[0-9]*\.?[0-9]+|N/A)",
    re.IGNORECASE,
)

# Deliberately substring-based: real output says "UserWarning", "FutureWarning",
# "warnings.warn", none of which start on a word boundary at "warn".
WARNING_RE = re.compile(r"^.*warning.*$", re.IGNORECASE | re.MULTILINE)
TRACEBACK_RE = re.compile(r"^Traceback \(most recent call last\):.*?(?=\n\S|\Z)", re.DOTALL | re.MULTILINE)

MAX_CAPTURED_WARNINGS = 12
LOG_TAIL_LINES = 40

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_USAGE = 2


class ConfigError(ValueError):
    """An expert configuration string could not be understood."""


@dataclass(frozen=True)
class ExpertConfig:
    """A normalised expert configuration.

    ``run_name`` is the POC-facing identity (``none``, ``blending``,
    ``diffusion``, ``blending_diffusion``); ``experts_arg`` is what gets passed
    to the runner's ``--experts`` flag.
    """

    run_name: str
    experts: Tuple[str, ...]

    @property
    def experts_arg(self) -> str:
        return ",".join(self.experts) if self.experts else "none"

    def output_path(self, directory: Path) -> Path:
        return directory / f"demo_{self.run_name}.json"


@dataclass
class RunOutcome:
    """Everything recorded about one configuration's run."""

    run_name: str
    experts: List[str]
    experts_arg: str
    command: List[str]
    output_path: Optional[str]
    status: str = "fail"
    validation: str = "not run"
    exit_code: Optional[int] = None
    timed_out: bool = False
    cuda_oom: bool = False
    runtime_s: Optional[float] = None
    peak_vram_mib: Optional[int] = None
    prediction_text: Optional[str] = None
    label: Optional[str] = None
    real_score: Optional[float] = None
    fake_score: Optional[float] = None
    expert_scores: Dict[str, Optional[float]] = field(default_factory=dict)
    prompt: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    failures: List[str] = field(default_factory=list)
    stderr_tail: str = ""
    log_path: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "run_name": self.run_name,
            "experts": self.experts,
            "experts_arg": self.experts_arg,
            "status": self.status,
            "validation": self.validation,
            "prediction_text": self.prediction_text,
            "label": self.label,
            "real_score": self.real_score,
            "fake_score": self.fake_score,
            "expert_scores": self.expert_scores,
            "exit_code": self.exit_code,
            "timed_out": self.timed_out,
            "cuda_oom": self.cuda_oom,
            "runtime_s": None if self.runtime_s is None else round(self.runtime_s, 3),
            "peak_vram_mib": self.peak_vram_mib,
            "output_path": self.output_path,
            "log_path": self.log_path,
            "prompt": self.prompt,
            "warnings": self.warnings,
            "failures": self.failures,
            "stderr_tail": self.stderr_tail,
            "command": self.command,
        }


# --------------------------------------------------------------------------
# configuration normalisation
# --------------------------------------------------------------------------


def normalise_config(spec: str) -> ExpertConfig:
    """Turn a user-supplied config string into a canonical :class:`ExpertConfig`.

    Accepts any casing, spacing and ordering, plus the runner's provider
    spellings, so ``"Diffusion , blending"`` and ``"blending,diffusion_detector"``
    both become the single run ``blending_diffusion``.

    Raises:
        ConfigError: empty input or an unknown expert name.
    """

    if not isinstance(spec, str) or not spec.strip():
        raise ConfigError("expert configuration must be a non-empty string")

    tokens = [t.strip().lower() for t in spec.split(",")]
    tokens = [t for t in tokens if t]
    if not tokens:
        raise ConfigError(f"no expert names in {spec!r}")

    if any(t == "none" for t in tokens):
        if len(tokens) > 1:
            raise ConfigError(f"'none' cannot be combined with other experts: {spec!r}")
        return ExpertConfig(run_name="none", experts=())

    canonical: List[str] = []
    for token in tokens:
        mapped = EXPERT_ALIASES.get(token)
        if mapped is None:
            raise ConfigError(
                f"unknown expert {token!r}; expected one of "
                f"{sorted(set(EXPERT_ALIASES)) + ['none']}"
            )
        if mapped not in canonical:
            canonical.append(mapped)

    ordered = tuple(name for name in EXPERT_ORDER if name in canonical)
    return ExpertConfig(run_name="_".join(ordered), experts=ordered)


def normalise_configs(specs: Sequence[str]) -> List[ExpertConfig]:
    """Normalise several configs, rejecting duplicates so outputs never collide."""

    out: List[ExpertConfig] = []
    seen: Dict[str, str] = {}
    for spec in specs:
        config = normalise_config(spec)
        if config.run_name in seen:
            raise ConfigError(
                f"{spec!r} and {seen[config.run_name]!r} both normalise to "
                f"{config.run_name!r}; each configuration may only appear once"
            )
        seen[config.run_name] = spec
        out.append(config)
    return out


# --------------------------------------------------------------------------
# output inspection
# --------------------------------------------------------------------------


def extract_prompt(path: Path) -> Optional[str]:
    """Return the human turn from a runner output file, if it is readable."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, list) or not payload or not isinstance(payload[0], dict):
        return None
    for turn in payload[0].get("conversations") or []:
        if isinstance(turn, dict) and turn.get("from") == "human":
            value = turn.get("value")
            if isinstance(value, str):
                return value
    return None


def parse_expert_scores(prompt: Optional[str]) -> Dict[str, Optional[float]]:
    """Pull ``{'blending': 0.812, 'diffusion': None}`` out of the prompt tail.

    ``None`` means the runner rendered ``N/A`` - the expert was requested but
    produced no score.
    """

    scores: Dict[str, Optional[float]] = {}
    for match in PROMPT_SCORE_RE.finditer(prompt or ""):
        alias = match.group("alias").lower()
        alias = EXPERT_ALIASES.get(alias, alias)
        raw = match.group("score")
        if raw.upper() == "N/A":
            scores[alias] = None
        else:
            try:
                scores[alias] = float(raw)
            except ValueError:
                scores[alias] = None
    return scores


def collect_warnings(text: str, *, limit: int = MAX_CAPTURED_WARNINGS) -> List[str]:
    """Deduplicated warning lines from captured output, newest ordering preserved."""

    seen: List[str] = []
    for match in WARNING_RE.finditer(text or ""):
        candidate = match.group(0).strip()
        if candidate and candidate not in seen:
            seen.append(candidate)
        if len(seen) >= limit:
            break
    return seen


def tail(text: str, lines: int = LOG_TAIL_LINES) -> str:
    stripped = (text or "").strip()
    if not stripped:
        return ""
    return "\n".join(stripped.splitlines()[-lines:])


# --------------------------------------------------------------------------
# a single configuration
# --------------------------------------------------------------------------


def execute_config(
    config: ExpertConfig,
    *,
    manifest_path: Path,
    config_path: Path,
    output_dir: Path,
    log_dir: Optional[Path],
    project_root: Path,
    timeout_s: float,
    load_4bit: bool,
    load_8bit: bool,
    sample_vram: bool,
    process_runner: Callable[..., Any] = subprocess.run,
) -> RunOutcome:
    """Run one configuration end to end and record what happened.

    Never raises for an inference problem: a failed run is a populated
    :class:`RunOutcome` with ``status='fail'``, so the matrix always completes.
    """

    output_path = config.output_path(output_dir)
    command = build_command(
        manifest=manifest_path,
        output=output_path,
        config=config_path,
        experts=config.experts_arg,
    )
    outcome = RunOutcome(
        run_name=config.run_name,
        experts=list(config.experts),
        experts_arg=config.experts_arg,
        command=command,
        output_path=None,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # A stale file from an earlier matrix must not be read as a fresh result.
    if output_path.exists():
        output_path.unlink()

    environment = build_environment(load_4bit=load_4bit, load_8bit=load_8bit)

    LOGGER.info("[%s] %s", config.run_name, " ".join(command))
    with VramSampler(enabled=sample_vram) as sampler:
        result: ProcessResult = run_process(
            command,
            timeout_s=timeout_s,
            cwd=project_root,
            env=environment,
            runner=process_runner,
        )
    outcome.peak_vram_mib = sampler.peak_mib
    outcome.runtime_s = result.duration_s
    outcome.exit_code = result.exit_code
    outcome.timed_out = result.timed_out
    outcome.cuda_oom = detect_cuda_oom(result.combined_output)
    outcome.warnings = collect_warnings(result.combined_output)
    outcome.stderr_tail = tail(result.stderr)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{config.run_name}.log"
        try:
            log_path.write_text(
                f"$ {' '.join(command)}\n\n--- stdout ---\n{result.stdout}\n"
                f"--- stderr ---\n{result.stderr}\n",
                encoding="utf-8",
                errors="replace",
            )
            outcome.log_path = str(log_path)
        except OSError as exc:
            LOGGER.warning("could not write log for %s: %s", config.run_name, exc)

    failures: List[str] = []
    if result.timed_out:
        failures.append(f"exceeded the {timeout_s:.0f}s timeout")
    if outcome.cuda_oom:
        failures.append("CUDA out of memory reported by the inference subprocess")
    if not result.timed_out and result.exit_code != 0:
        failures.append(f"runner exited with code {result.exit_code}")
        for match in TRACEBACK_RE.finditer(result.combined_output):
            failures.append(tail(match.group(0), lines=3))
            break

    located = locate_output(output_path, result.stdout)
    if located is None:
        failures.append(f"no output JSON produced at {output_path}")
        outcome.validation = "missing output"
    else:
        outcome.output_path = str(located)
        outcome.prompt = extract_prompt(located)
        outcome.expert_scores = parse_expert_scores(outcome.prompt)
        try:
            predictions: List[Prediction] = validate_output(located)
        except OutputError as exc:
            failures.append(str(exc))
            outcome.validation = "invalid"
        else:
            first = predictions[0]
            outcome.prediction_text = first.answer
            outcome.label = first.label
            outcome.real_score = first.real_score
            outcome.fake_score = first.fake_score
            outcome.validation = "valid"
            if first.label is None:
                failures.append(f"answer {first.answer!r} is not clearly real or fake")

        # An expert that silently scored nothing makes the run a different
        # experiment from the one that was asked for, so say so explicitly.
        for expert in config.experts:
            if outcome.expert_scores.get(expert, "absent") is None:
                failures.append(f"{expert} expert produced no score (rendered as N/A)")
            elif expert not in outcome.expert_scores:
                failures.append(f"{expert} expert score is absent from the prompt")

    outcome.failures = failures
    outcome.status = "pass" if not failures else "fail"
    return outcome


# --------------------------------------------------------------------------
# summary
# --------------------------------------------------------------------------


def build_summary(
    outcomes: Sequence[RunOutcome],
    *,
    manifest: Path,
    images: Sequence[Path],
    config_path: Path,
    output_dir: Path,
    quantisation: str,
    started_at: str,
    finished_at: str,
) -> Dict[str, Any]:
    runtimes = [o.runtime_s for o in outcomes if o.runtime_s is not None]
    vrams = [o.peak_vram_mib for o in outcomes if o.peak_vram_mib is not None]
    passed = [o for o in outcomes if o.status == "pass"]
    labels = sorted({o.label for o in outcomes if o.label is not None})

    return {
        "status": "pass" if len(passed) == len(outcomes) and outcomes else "fail",
        "manifest": str(manifest),
        "images": [str(p) for p in images],
        "config": str(config_path),
        "output_dir": str(output_dir),
        "quantisation": quantisation,
        "started_at": started_at,
        "finished_at": finished_at,
        "totals": {
            "configurations": len(outcomes),
            "passed": len(passed),
            "failed": len(outcomes) - len(passed),
            "total_runtime_s": round(sum(runtimes), 3) if runtimes else None,
            "max_peak_vram_mib": max(vrams) if vrams else None,
            "distinct_labels": labels,
            "labels_agree": len(labels) <= 1,
        },
        "runs": [o.as_dict() for o in outcomes],
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_runtimes_sidecar(directory: Path, outcomes: Sequence[RunOutcome]) -> Path:
    """Write the ``runtimes.json`` the POC normaliser looks for next to results."""

    payload = {o.run_name: None if o.runtime_s is None else round(o.runtime_s, 3) for o in outcomes}
    path = directory / "runtimes.json"
    write_json(path, payload)
    return path


def render_summary(summary: Dict[str, Any]) -> str:
    lines = [
        "X2DFD expert matrix",
        "=" * 78,
        f"manifest : {summary['manifest']}",
        f"images   : {', '.join(summary['images'])}",
        f"loading  : {summary['quantisation']}",
        "",
        f"{'run':<20} {'status':<7} {'label':<6} {'real':>7} {'fake':>7} {'time s':>8} {'VRAM MiB':>9}  experts",
        "-" * 78,
    ]
    for run in summary["runs"]:
        scores = run.get("expert_scores") or {}
        expert_text = ", ".join(
            f"{k}={'N/A' if v is None else f'{v:.4f}'}" for k, v in sorted(scores.items())
        ) or "-"
        lines.append(
            "{run:<20} {status:<7} {label:<6} {real:>7} {fake:>7} {time:>8} {vram:>9}  {experts}".format(
                run=run["run_name"],
                status=run["status"],
                label=run["label"] or "-",
                real="-" if run["real_score"] is None else f"{run['real_score']:.4f}",
                fake="-" if run["fake_score"] is None else f"{run['fake_score']:.4f}",
                time="-" if run["runtime_s"] is None else f"{run['runtime_s']:.1f}",
                vram="-" if run["peak_vram_mib"] is None else str(run["peak_vram_mib"]),
                experts=expert_text,
            )
        )
    totals = summary["totals"]
    lines.extend(
        [
            "-" * 78,
            f"passed {totals['passed']}/{totals['configurations']}"
            f" | total {totals['total_runtime_s']} s"
            f" | labels {totals['distinct_labels'] or '-'}"
            f" | agree: {totals['labels_agree']}",
        ]
    )
    for run in summary["runs"]:
        for failure in run["failures"]:
            lines.append(f"FAILURE [{run['run_name']}]: {failure}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Options:
    manifest: Path
    config: Path
    output_dir: Path
    summary: Path
    configs: List[str]
    timeout_s: float
    load_4bit: bool
    load_8bit: bool
    project_root: Path
    sample_vram: bool
    keep_logs: bool
    dry_run: bool
    as_json: bool


def parse_args(argv: Optional[Sequence[str]] = None) -> Options:
    parser = argparse.ArgumentParser(
        prog="python -m tools.run_expert_matrix",
        description="Run one image under several X2DFD expert configurations and compare them.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Each configuration is a comma-joined expert list, so quote or space-separate them:\n"
            "  --configs none blending diffusion blending,diffusion\n\n"
            "Exit codes: 0 all configurations passed, 1 at least one failed, 2 usage error."
        ),
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
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"directory for per-run results (default: {DEFAULT_OUTPUT_DIR}/<manifest stem>)",
    )
    parser.add_argument(
        "--summary", type=Path, default=DEFAULT_SUMMARY, help=f"summary JSON (default: {DEFAULT_SUMMARY})"
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=list(DEFAULT_CONFIGS),
        metavar="CONFIG",
        help="expert configurations to run (default: %s)" % " ".join(DEFAULT_CONFIGS),
    )
    parser.add_argument(
        "--timeout",
        dest="timeout_s",
        type=float,
        default=DEFAULT_TIMEOUT_S,
        help=f"per-configuration timeout in seconds (default: {DEFAULT_TIMEOUT_S:.0f})",
    )
    parser.add_argument("--load-4bit", action="store_true", help="4-bit loading (needed on a 10 GiB card)")
    parser.add_argument("--load-8bit", action="store_true", help="8-bit loading")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT_DEFAULT,
        help="repository root used as the subprocess working directory",
    )
    parser.add_argument("--no-vram-sampling", action="store_true", help="skip nvidia-smi peak VRAM polling")
    parser.add_argument("--no-logs", action="store_true", help="do not write per-run stdout/stderr logs")
    parser.add_argument(
        "--dry-run", action="store_true", help="print the commands that would run and exit"
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
        output_dir=args.output_dir,
        summary=args.summary,
        configs=list(args.configs),
        timeout_s=args.timeout_s,
        load_4bit=args.load_4bit,
        load_8bit=args.load_8bit,
        project_root=args.project_root,
        sample_vram=not args.no_vram_sampling,
        keep_logs=not args.no_logs,
        dry_run=args.dry_run,
        as_json=args.as_json,
    )


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else (root / path)


def main(argv: Optional[Sequence[str]] = None, *, process_runner: Optional[Callable[..., Any]] = None) -> int:
    options = parse_args(argv)
    root = options.project_root

    try:
        configs = normalise_configs(options.configs)
    except ConfigError as exc:
        LOGGER.error("%s", exc)
        return EXIT_USAGE

    manifest_path = _resolve(root, options.manifest)
    try:
        manifest = load_manifest(manifest_path)
    except ManifestError as exc:
        LOGGER.error("%s", exc)
        return EXIT_USAGE

    config_path = _resolve(root, options.config)
    output_dir = _resolve(root, options.output_dir) / manifest_path.stem
    log_dir = (output_dir / "logs") if options.keep_logs else None

    LOGGER.info(
        "manifest OK: %d image(s); %d configuration(s): %s",
        len(manifest.images),
        len(configs),
        ", ".join(c.run_name for c in configs),
    )

    if options.dry_run:
        for config in configs:
            command = build_command(
                manifest=manifest_path,
                output=config.output_path(output_dir),
                config=config_path,
                experts=config.experts_arg,
            )
            print(f"[{config.run_name}] {' '.join(command)}")
        return EXIT_PASS

    quantisation = "4-bit" if options.load_4bit else ("8-bit" if options.load_8bit else "fp16")
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    outcomes: List[RunOutcome] = []
    for index, config in enumerate(configs, start=1):
        LOGGER.info("(%d/%d) running configuration %r", index, len(configs), config.run_name)
        outcome = execute_config(
            config,
            manifest_path=manifest_path,
            config_path=config_path,
            output_dir=output_dir,
            log_dir=log_dir,
            project_root=root,
            timeout_s=options.timeout_s,
            load_4bit=options.load_4bit,
            load_8bit=options.load_8bit,
            sample_vram=options.sample_vram,
            process_runner=process_runner or subprocess.run,
        )
        outcomes.append(outcome)
        LOGGER.info(
            "(%d/%d) %s -> %s in %.1fs",
            index,
            len(configs),
            config.run_name,
            outcome.status,
            outcome.runtime_s or 0.0,
        )

    summary = build_summary(
        outcomes,
        manifest=manifest_path,
        images=manifest.images,
        config_path=config_path,
        output_dir=output_dir,
        quantisation=quantisation,
        started_at=started_at,
        finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )

    summary_path = _resolve(root, options.summary)
    try:
        write_json(summary_path, summary)
        sidecar = write_runtimes_sidecar(output_dir, outcomes)
    except OSError as exc:
        LOGGER.error("cannot write summary: %s", exc)
        return EXIT_FAIL

    if options.as_json:
        print(json.dumps(summary, indent=2))
    else:
        print(render_summary(summary))
        print(f"\nsummary  : {summary_path}")
        print(f"runtimes : {sidecar}")

    return EXIT_PASS if summary["status"] == "pass" else EXIT_FAIL


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
