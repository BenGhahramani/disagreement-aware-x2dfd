#!/usr/bin/env python3
"""Down-convert a PEFT adapter config so an older installed PEFT can parse it.

An ``adapter_config.json`` written by a newer PEFT carries fields the installed
``LoraConfig`` has never heard of, and ``PeftConfig.from_pretrained`` fails with
``TypeError: LoraConfig.__init__() got an unexpected keyword argument ...``.

This tool removes those fields, but only when they are demonstrably inert. A
field is inert when its value is null, false, an empty container, or when it is
switched off by another flag in the same file (see ``GATED_FIELDS``). Anything
else is treated as potentially architecture-changing: the tool refuses to write
a converted config and exits non-zero so the mismatch gets a human decision
rather than a silently altered adapter.

Supported fields are copied through byte-for-byte; no value is ever rewritten.

Exit codes:
    0   OK      - a compatible config was written (or the input needed no change)
    1   BLOCKED - unsupported fields carry meaningful values; nothing was written
    2   USAGE   - bad arguments, unreadable config, or PEFT is not importable

Examples::

    python -m tools.make_peft_compatible_config weights/checkpoints/ckpt/my-lora
    python -m tools.make_peft_compatible_config path/to/adapter_config.json --output /tmp/compat.json
    python -m tools.make_peft_compatible_config path/to/adapter_dir --in-place
    python -m tools.make_peft_compatible_config path/to/adapter_dir --json
"""
from __future__ import annotations

import argparse
import inspect
import json
import logging
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

LOGGER = logging.getLogger("x2dfd.make_peft_compatible_config")

CONFIG_FILENAME = "adapter_config.json"
DEFAULT_OUTPUT_SUFFIX = ".compatible.json"
DEFAULT_BACKUP_SUFFIX = ".original.json"

EXIT_OK = 0
EXIT_BLOCKED = 1
EXIT_USAGE = 2

# Fields that only take effect when another flag is enabled. Mapping is
# ``field -> gate field``: when the gate is falsy (or absent), the field cannot
# influence the adapter no matter what it holds.
#
# ``qalora_group_size`` defaults to 16 in PEFT and its own help text says
# "Only used when `use_qalora=True`", so a serialised 16 alongside
# ``use_qalora: false`` is noise from the newer writer, not a setting.
GATED_FIELDS: Dict[str, str] = {
    "qalora_group_size": "use_qalora",
}


@dataclass(frozen=True)
class FieldVerdict:
    """How one unsupported key was classified."""

    key: str
    value: Any
    operative: bool
    reason: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "value": self.value,
            "operative": self.operative,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class Analysis:
    """The result of comparing a config against an accepted field set."""

    supported: List[str]
    unsupported: List[FieldVerdict]

    @property
    def removable(self) -> List[FieldVerdict]:
        return [v for v in self.unsupported if not v.operative]

    @property
    def blocking(self) -> List[FieldVerdict]:
        return [v for v in self.unsupported if v.operative]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "supported": list(self.supported),
            "unsupported": [v.as_dict() for v in self.unsupported],
            "removable": [v.key for v in self.removable],
            "blocking": [v.key for v in self.blocking],
        }


@dataclass(frozen=True)
class Options:
    config: Path
    output: Optional[Path]
    backup: Optional[Path]
    in_place: bool
    force: bool
    as_json: bool


def accepted_fields(config_cls: type) -> Set[str]:
    """Return the keyword names the given config class actually accepts."""

    signature = inspect.signature(config_cls.__init__)
    return {
        name
        for name, parameter in signature.parameters.items()
        if name != "self"
        and parameter.kind
        not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    }


def load_lora_config_class() -> type:
    """Import LoraConfig from the installed PEFT.

    Raises:
        ImportError: when PEFT is missing, so the caller can exit with USAGE.
    """

    from peft import LoraConfig  # imported lazily so --help works without PEFT

    return LoraConfig


def classify_value(key: str, value: Any, config: Mapping[str, Any]) -> Tuple[bool, str]:
    """Decide whether an unsupported value could affect the adapter.

    Returns ``(operative, reason)``. ``operative`` True means "do not drop this".
    """

    if value is None:
        return False, "null"
    if value is False:
        return False, "false (feature disabled)"
    if isinstance(value, (dict, list, tuple, set, str)) and len(value) == 0:
        return False, f"empty {type(value).__name__}"

    gate = GATED_FIELDS.get(key)
    if gate is not None:
        gate_value = config.get(gate)
        if not gate_value:
            return False, f"inert: gated by {gate}={gate_value!r}"
        return True, f"gate {gate}={gate_value!r} is enabled"

    return True, "non-default value that may change the adapter"


def analyse(config: Mapping[str, Any], accepted: Set[str]) -> Analysis:
    """Split a config's keys into supported and (classified) unsupported ones."""

    supported: List[str] = []
    unsupported: List[FieldVerdict] = []
    for key in config:
        if key in accepted:
            supported.append(key)
            continue
        operative, reason = classify_value(key, config[key], config)
        unsupported.append(
            FieldVerdict(key=key, value=config[key], operative=operative, reason=reason)
        )
    return Analysis(supported=supported, unsupported=unsupported)


def strip_fields(config: Mapping[str, Any], drop: Sequence[str]) -> Dict[str, Any]:
    """Copy a config without the named keys, preserving order and values."""

    dropped = set(drop)
    return {key: value for key, value in config.items() if key not in dropped}


def resolve_config_path(target: Path) -> Path:
    """Accept either an adapter directory or the JSON file itself."""

    if target.is_dir():
        return target / CONFIG_FILENAME
    return target


def read_config(path: Path) -> Dict[str, Any]:
    """Load an adapter config, raising ValueError with a readable message."""

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object, got {type(data).__name__}")
    return data


def write_config(path: Path, config: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def default_output_for(config_path: Path) -> Path:
    return config_path.with_name(config_path.stem + DEFAULT_OUTPUT_SUFFIX)


def default_backup_for(config_path: Path) -> Path:
    return config_path.with_name(config_path.stem + DEFAULT_BACKUP_SUFFIX)


def make_backup(config_path: Path, backup_path: Path, *, force: bool) -> str:
    """Snapshot the original before an in-place rewrite.

    An existing backup is kept when it already matches the source, so repeated
    runs stay idempotent instead of overwriting a pristine copy with a converted
    one.
    """

    if backup_path.exists():
        same = backup_path.read_bytes() == config_path.read_bytes()
        if same:
            return f"backup already current: {backup_path}"
        if not force:
            raise ValueError(
                f"backup {backup_path} exists and differs from {config_path}; "
                "refusing to overwrite it (pass --force to replace)"
            )
        LOGGER.warning("replacing differing backup at %s", backup_path)
    shutil.copy2(config_path, backup_path)
    return f"backup written: {backup_path}"


def render_report(analysis: Analysis, config_path: Path, accepted_count: int) -> str:
    lines = [
        f"adapter config: {config_path}",
        f"installed LoraConfig accepts {accepted_count} fields",
        f"supported keys  : {len(analysis.supported)}",
        f"unsupported keys: {len(analysis.unsupported)}",
    ]
    if analysis.unsupported:
        lines.append("")
        lines.append(f"{'KEY':<28} {'VERDICT':<12} {'VALUE':<10} REASON")
        lines.append("-" * 88)
        for verdict in analysis.unsupported:
            lines.append(
                "%-28s %-12s %-10s %s"
                % (
                    verdict.key,
                    "KEEP" if verdict.operative else "remove",
                    repr(verdict.value),
                    verdict.reason,
                )
            )
    return "\n".join(lines)


def parse_args(argv: Optional[Sequence[str]] = None) -> Options:
    parser = argparse.ArgumentParser(
        prog="python -m tools.make_peft_compatible_config",
        description=(
            "Remove provably inert fields from a PEFT adapter config so the "
            "installed (older) PEFT can parse it."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exit codes: 0 converted or already compatible, "
            "1 blocked by meaningful unsupported fields, 2 usage error."
        ),
    )
    parser.add_argument(
        "config",
        type=Path,
        help=f"adapter directory containing {CONFIG_FILENAME}, or the JSON file itself",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"where to write the compatible copy (default: <name>{DEFAULT_OUTPUT_SUFFIX})",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="overwrite the input config after taking a backup (explicit opt-in)",
    )
    parser.add_argument(
        "--backup",
        type=Path,
        default=None,
        help=f"backup path used by --in-place (default: <name>{DEFAULT_BACKUP_SUFFIX})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="allow replacing an existing backup that differs from the input",
    )
    parser.add_argument("--json", dest="as_json", action="store_true", help="emit a JSON report")
    parser.add_argument("-v", "--verbose", action="store_true", help="enable debug logging")

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if args.in_place and args.output is not None:
        parser.error("--in-place and --output are mutually exclusive")
    return Options(
        config=args.config,
        output=args.output,
        backup=args.backup,
        in_place=args.in_place,
        force=args.force,
        as_json=args.as_json,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    options = parse_args(argv)

    config_path = resolve_config_path(options.config)
    if not config_path.is_file():
        LOGGER.error("adapter config not found: %s", config_path)
        return EXIT_USAGE

    try:
        config = read_config(config_path)
    except ValueError as exc:
        LOGGER.error("%s", exc)
        return EXIT_USAGE

    try:
        config_cls = load_lora_config_class()
    except ImportError as exc:
        LOGGER.error("cannot import LoraConfig from peft: %s", exc)
        return EXIT_USAGE

    accepted = accepted_fields(config_cls)
    analysis = analyse(config, accepted)

    report: Dict[str, Any] = {
        "config": str(config_path),
        "accepted_field_count": len(accepted),
        **analysis.as_dict(),
        "written": None,
        "backup": None,
    }

    if not options.as_json:
        print(render_report(analysis, config_path, len(accepted)))

    if analysis.blocking:
        for verdict in analysis.blocking:
            LOGGER.error(
                "unsupported field %r=%r looks meaningful (%s); not writing a converted config",
                verdict.key,
                verdict.value,
                verdict.reason,
            )
        report["status"] = "blocked"
        if options.as_json:
            print(json.dumps(report, indent=2))
        return EXIT_BLOCKED

    destination = (
        config_path if options.in_place else (options.output or default_output_for(config_path))
    )

    if options.in_place:
        backup_path = options.backup or default_backup_for(config_path)
        try:
            message = make_backup(config_path, backup_path, force=options.force)
        except (OSError, ValueError) as exc:
            LOGGER.error("%s", exc)
            return EXIT_USAGE
        report["backup"] = str(backup_path)
        if not options.as_json:
            print(message)

    converted = strip_fields(config, [v.key for v in analysis.removable])
    try:
        write_config(destination, converted)
    except OSError as exc:
        LOGGER.error("cannot write %s: %s", destination, exc)
        return EXIT_USAGE

    report["written"] = str(destination)
    report["status"] = "converted" if analysis.removable else "already-compatible"

    if options.as_json:
        print(json.dumps(report, indent=2))
    else:
        if analysis.removable:
            print("")
            print(f"removed {len(analysis.removable)} inert field(s):")
            for verdict in analysis.removable:
                print(f"  - {verdict.key} = {verdict.value!r}  ({verdict.reason})")
        else:
            print("")
            print("no unsupported fields; config copied unchanged")
        print(f"kept {len(analysis.supported)} supported field(s) verbatim")
        print(f"written: {destination}")

    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
