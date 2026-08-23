#!/usr/bin/env python3
"""Environment check tuned for UQ Bunya compute nodes.

Thin wrapper around ``tools.check_environment`` with Bunya-oriented defaults:
the Bunya infer config, optional strict bitsandbytes requirement for 4-bit
smoke tests, and JSON output suitable for Slurm logs.

Run only on a GPU compute node (interactive ``salloc`` or batch job), never on
Bunya login nodes.

Examples::

    python -m tools.check_bunya_environment
    python -m tools.check_bunya_environment --require-bitsandbytes --json
    python -m tools.check_bunya_environment --project-root /scratch/$USER/x2dfd/repo
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from tools import check_environment as ce

DEFAULT_CONFIG_RELATIVE = Path("eval") / "configs" / "infer_config.bunya.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.check_bunya_environment",
        description=(
            "Verify Python, CUDA, packages, and weight paths for Bunya inference. "
            "Defaults to eval/configs/infer_config.bunya.yaml."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=ce.PROJECT_ROOT_DEFAULT,
        help="Repository root used to resolve relative config and weight paths",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Eval config to validate (default: <project-root>/eval/configs/infer_config.bunya.yaml)",
    )
    parser.add_argument(
        "--require-bitsandbytes",
        action="store_true",
        help="Treat bitsandbytes as required (use when running --load-4bit smoke tests)",
    )
    parser.add_argument("--skip-weights", action="store_true", help="Do not check weight paths")
    parser.add_argument("--skip-datasets", action="store_true", help="Do not check dataset JSONs/images")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as failures")
    parser.add_argument("--json", action="store_true", help="Print the JSON report instead of text")
    parser.add_argument("--output", type=Path, default=None, help="Also write the JSON report here")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser


def run_checks(
    *,
    project_root: Path,
    config_path: Path,
    require_bitsandbytes: bool = False,
    skip_weights: bool = False,
    skip_datasets: bool = False,
    strict: bool = False,
    finder=ce.default_module_finder,
    torch_module=None,
    torch_importer=ce.default_torch_importer,
) -> list[ce.CheckResult]:
    """Run the standard checker, optionally upgrading bitsandbytes to required."""

    options = ce.Options(
        project_root=project_root,
        config_path=config_path,
        check_weights=not skip_weights,
        check_datasets=not skip_datasets,
        strict=strict,
    )
    results = ce.run_checks(
        options,
        finder=finder,
        torch_module=torch_module,
        torch_importer=torch_importer,
    )
    if not require_bitsandbytes:
        return results

    upgraded: list[ce.CheckResult] = []
    for result in results:
        if result.name == "import.bitsandbytes" and result.status is ce.Status.WARN:
            upgraded.append(
                ce.CheckResult(
                    result.name,
                    ce.Status.FAIL,
                    result.detail + " (required for Bunya 4-bit smoke tests)",
                    required=True,
                )
            )
        else:
            upgraded.append(result)
    return upgraded


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    import logging

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    project_root = args.project_root.resolve()
    if not project_root.is_dir():
        parser.error(f"--project-root is not a directory: {project_root}")

    config_path = (args.config or (project_root / DEFAULT_CONFIG_RELATIVE)).resolve()

    results = run_checks(
        project_root=project_root,
        config_path=config_path,
        require_bitsandbytes=args.require_bitsandbytes,
        skip_weights=args.skip_weights,
        skip_datasets=args.skip_datasets,
        strict=args.strict,
    )
    report = ce.build_report(results, strict=args.strict)
    report["profile"] = "bunya"
    report["config_path"] = str(config_path)

    if args.output is not None:
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                __import__("json").dumps(report, indent=2) + "\n",
                encoding="utf-8",
            )
        except OSError as exc:
            print(f"could not write report to {args.output}: {exc}", file=sys.stderr)
            return ce.EXIT_USAGE

    if args.json:
        print(__import__("json").dumps(report, indent=2))
    else:
        print(ce.render_text(results, strict=args.strict))

    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
