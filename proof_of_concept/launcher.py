"""Run the four X2DFD inference settings for one image and (optionally) render the POC report.

This wraps ``eval/infer/runner.py`` four times — once per expert setting —
and:

* names each output ``demo_<run>.json`` so the POC normaliser picks it up,
* wall-clock-times each run with :func:`time.perf_counter` and writes a
  ``runtimes.json`` sidecar in the same directory,
* (optionally) renders the Markdown report immediately afterwards by calling
  :mod:`proof_of_concept.run_demo`.

The launcher does NOT modify any X2DFD code. If a runner call fails it records
the wall-clock time and continues; the POC parser will mark missing/errored
runs as ``Failed/insufficient`` automatically.

Examples (from repo root)::

    # Single image, defaults to a timestamped output dir, then render the report
    python -m proof_of_concept.launcher --image /abs/path/to/image.png --render

    # Existing dataset JSON, custom output dir, custom X2DFD config
    python -m proof_of_concept.launcher \\
        --json datasets/raw/data/poc/demo_one.json \\
        --config eval/configs/infer_config.yaml \\
        --output-dir eval/outputs/poc/real_run_001 \\
        --render

    # Print the four commands without invoking X2DFD (useful for smoke tests)
    python -m proof_of_concept.launcher --json some.json --dry-run
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Optional


_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CONFIG = _REPO_ROOT / "eval" / "configs" / "infer_config.yaml"

# Mapping: (poc run_name, X2DFD --experts flag value, output filename)
_RUN_PLAN: list[tuple[str, str, str]] = [
    ("none", "none", "demo_none.json"),
    ("blending", "blending", "demo_blending.json"),
    ("diffusion", "diffusion", "demo_diffusion.json"),
    ("blending_diffusion", "blending,diffusion", "demo_blending_diffusion.json"),
]


def _make_one_image_json(image_path: Path, target_dir: Path) -> Path:
    """Write a minimal X2DFD dataset JSON describing a single image.

    The X2DFD runner requires ``Description`` (root) + relative ``image_path``,
    or fully absolute image_paths. We choose the first form for clarity.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        # Forward slashes work cross-platform inside the JSON
        "Description": str(image_path.parent.resolve()).replace("\\", "/"),
        "images": [{"image_path": image_path.name}],
    }
    out_path = target_dir / "_launcher_one_image.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def _build_runner_cmd(
    config_path: Path,
    json_path: Path,
    experts_flag: str,
    output_path: Path,
) -> list[str]:
    return [
        sys.executable, "-m", "eval.infer.runner",
        "--config", str(config_path),
        "--json", str(json_path),
        "--experts", experts_flag,
        "--output", str(output_path),
    ]


def _run_one(cmd: list[str]) -> tuple[float, int, str]:
    """Execute one X2DFD invocation. Returns ``(elapsed_seconds, rc, tail)``."""
    start = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed = time.perf_counter() - start
    combined = (proc.stdout or "") + (proc.stderr or "")
    tail_lines = [ln for ln in combined.strip().splitlines() if ln][-3:]
    return elapsed, proc.returncode, "\n".join(tail_lines)


def _render_report(scenario_dir: Path, report_path: Path) -> int:
    cmd = [
        sys.executable, "-m", "proof_of_concept.run_demo",
        "--scenario-dir", str(scenario_dir),
        "--output", str(report_path),
    ]
    proc = subprocess.run(cmd, cwd=str(_REPO_ROOT), check=False)
    return proc.returncode


def _resolve_inputs(args: argparse.Namespace) -> Optional[Path]:
    """Resolve --image / --json into a single dataset JSON path. Returns None on error."""
    if args.image is not None:
        image_path = args.image.resolve()
        if not image_path.exists():
            print(f"[POC launcher] image not found: {image_path}", file=sys.stderr)
            return None
        synth_dir = Path(tempfile.mkdtemp(prefix="poc_launcher_"))
        json_path = _make_one_image_json(image_path, synth_dir)
        print(f"[POC launcher] generated one-image JSON: {json_path}")
        return json_path
    json_path = args.json.resolve()
    if not json_path.exists():
        print(f"[POC launcher] dataset JSON not found: {json_path}", file=sys.stderr)
        return None
    return json_path


def _resolve_output_dir(arg_dir: Optional[Path]) -> Path:
    if arg_dir is None:
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        return _REPO_ROOT / "eval" / "outputs" / "poc" / f"launcher_{ts}"
    return arg_dir if arg_dir.is_absolute() else _REPO_ROOT / arg_dir


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run X2DFD four times for one image and render the POC report",
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--image", type=Path, default=None,
        help="Absolute path to an image; a one-image dataset JSON is generated",
    )
    src.add_argument(
        "--json", type=Path, default=None,
        help="Existing X2DFD dataset JSON (Description + images list)",
    )

    ap.add_argument(
        "--config", type=Path, default=_DEFAULT_CONFIG,
        help=f"X2DFD eval config (default: {_DEFAULT_CONFIG.relative_to(_REPO_ROOT)})",
    )
    ap.add_argument(
        "--output-dir", type=Path, default=None,
        help="Directory for the four demo_<run>.json files (default: eval/outputs/poc/launcher_<timestamp>/)",
    )
    ap.add_argument(
        "--render", action="store_true",
        help="After all runs, render a Markdown report (report.md) in the output directory",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Print the four runner commands and exit without invoking X2DFD",
    )
    args = ap.parse_args()

    json_path = _resolve_inputs(args)
    if json_path is None:
        return 2

    config_path = args.config if args.config.is_absolute() else _REPO_ROOT / args.config
    if not config_path.exists():
        print(f"[POC launcher] config not found: {config_path}", file=sys.stderr)
        return 2

    output_dir = _resolve_output_dir(args.output_dir)

    if args.dry_run:
        print(f"[POC launcher] DRY RUN -- would write into: {output_dir}")
        for run_name, experts_flag, fname in _RUN_PLAN:
            cmd = _build_runner_cmd(config_path, json_path, experts_flag, output_dir / fname)
            print(f"  # {run_name}")
            print("  " + " ".join(cmd))
        print(f"  # then write sidecar: {output_dir / 'runtimes.json'}")
        if args.render:
            print(f"  # then render: python -m proof_of_concept.run_demo "
                  f"--scenario-dir {output_dir} --output {output_dir / 'report.md'}")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[POC launcher] output dir: {output_dir}")
    print(f"[POC launcher] config:     {config_path}")
    print(f"[POC launcher] dataset:    {json_path}")

    runtimes: dict[str, float] = {}
    fails: list[str] = []
    for run_name, experts_flag, fname in _RUN_PLAN:
        out_path = output_dir / fname
        print(f"[POC launcher] running {run_name} (--experts {experts_flag}) ...", flush=True)
        cmd = _build_runner_cmd(config_path, json_path, experts_flag, out_path)
        elapsed, rc, tail = _run_one(cmd)
        runtimes[run_name] = round(elapsed, 2)
        verdict = "OK" if rc == 0 else f"FAIL (rc={rc})"
        print(f"[POC launcher]   {run_name}: {verdict} in {elapsed:.2f}s -> {out_path.name}")
        if rc != 0:
            fails.append(run_name)
            if tail:
                print(f"[POC launcher]   tail: {tail}", file=sys.stderr)

    sidecar = output_dir / "runtimes.json"
    sidecar.write_text(json.dumps(runtimes, indent=2), encoding="utf-8")
    print(f"[POC launcher] wrote {sidecar}")

    if fails:
        print(
            f"[POC launcher] {len(fails)}/{len(_RUN_PLAN)} runner call(s) failed: {fails}. "
            "The POC report will mark these as missing/errored runs.",
            file=sys.stderr,
        )

    if args.render:
        report_path = output_dir / "report.md"
        rc = _render_report(output_dir, report_path)
        if rc == 0:
            print(f"[POC launcher] report rendered: {report_path}")
        else:
            print(f"[POC launcher] report rendering failed (rc={rc})", file=sys.stderr)
            return 1

    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
