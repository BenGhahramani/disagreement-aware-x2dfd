"""Canonical 2x2 experimental design for X2-DFD expert configurations.

Exactly four inference **conditions** (cells): baseline, blending-only,
diffusion-only, and combined. Individual cells are not themselves main
effects or interactions.

The complete 2x2 factorial enables analysis of:

- the blending main effect;
- the diffusion main effect;
- the blending × diffusion interaction.

Those quantities are estimated from contrasts across cells of the full
design, not from any single configuration.

Pipeline CLI names (``--experts`` tokens and POC ``run_name`` values) are
preserved unchanged. This module is the single shared source of truth for the
four-cell design. It does not run inference and does not alter evaluator
behaviour.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class ExperimentConfig:
    """One cell of the blending × diffusion 2x2 design."""

    run_name: str
    experts_arg: str
    blending: bool
    diffusion: bool
    role: str
    display_name: str

    @property
    def flag_pair(self) -> Tuple[bool, bool]:
        return (self.blending, self.diffusion)


# Order is fixed: none → blending → diffusion → blending,diffusion.
EXPERIMENT_CONFIGS: Tuple[ExperimentConfig, ...] = (
    ExperimentConfig(
        run_name="none",
        experts_arg="none",
        blending=False,
        diffusion=False,
        role="baseline condition",
        display_name="No expert",
    ),
    ExperimentConfig(
        run_name="blending",
        experts_arg="blending",
        blending=True,
        diffusion=False,
        role="blending-only condition",
        display_name="Blending",
    ),
    ExperimentConfig(
        run_name="diffusion",
        experts_arg="diffusion",
        blending=False,
        diffusion=True,
        role="diffusion-only condition",
        display_name="Diffusion",
    ),
    ExperimentConfig(
        run_name="blending_diffusion",
        experts_arg="blending,diffusion",
        blending=True,
        diffusion=True,
        role="combined condition",
        display_name="Blending + Diffusion",
    ),
)

# Analyses enabled by the complete 2x2 (contrasts across cells — not cell roles).
FACTORIAL_ANALYSES: Tuple[str, ...] = (
    "the blending main effect",
    "the diffusion main effect",
    "the blending × diffusion interaction",
)

EXPECTED_2X2_FLAGS: frozenset[Tuple[bool, bool]] = frozenset(
    {
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    }
)

RUN_ORDER: Tuple[str, ...] = tuple(cfg.run_name for cfg in EXPERIMENT_CONFIGS)
RUN_TITLES: Dict[str, str] = {cfg.run_name: cfg.display_name for cfg in EXPERIMENT_CONFIGS}
ROLES: Dict[str, str] = {cfg.run_name: cfg.role for cfg in EXPERIMENT_CONFIGS}

# Strings passed to ``--experts`` / ``normalise_configs`` (comma form for both).
DEFAULT_CONFIGS: Tuple[str, ...] = tuple(cfg.experts_arg for cfg in EXPERIMENT_CONFIGS)

PRIMARY_ASSESSMENT_RUN: str = "blending_diffusion"


class ExperimentConfigError(ValueError):
    """The 2x2 experiment registry is incomplete or inconsistent."""


def get_experiment_config(run_name: str) -> ExperimentConfig:
    """Return the registry entry for ``run_name`` or raise ``KeyError``."""

    for cfg in EXPERIMENT_CONFIGS:
        if cfg.run_name == run_name:
            return cfg
    raise KeyError(f"unknown experiment run_name: {run_name!r}")


def validate_experiment_configs(
    configs: Optional[Sequence[ExperimentConfig]] = None,
) -> None:
    """Ensure ``configs`` is exactly the complete blending × diffusion 2x2 set.

    Checks:
    - exactly four entries;
    - flag pairs are ``{(False,False), (True,False), (False,True), (True,True)}``
      with each combination appearing once;
    - canonical ``run_name`` / ``experts_arg`` / order match the registry.

    Raises:
        ExperimentConfigError: if the set is incomplete, duplicated, or reordered.
    """

    entries = tuple(configs) if configs is not None else EXPERIMENT_CONFIGS
    if len(entries) != 4:
        raise ExperimentConfigError(
            f"expected exactly four experiment configs, got {len(entries)}"
        )

    flag_pairs = [cfg.flag_pair for cfg in entries]
    if len(set(flag_pairs)) != 4:
        raise ExperimentConfigError(
            f"duplicate or incomplete 2x2 flag pairs: {flag_pairs}"
        )
    if set(flag_pairs) != EXPECTED_2X2_FLAGS:
        raise ExperimentConfigError(
            f"flag pairs {set(flag_pairs)} != expected {set(EXPECTED_2X2_FLAGS)}"
        )

    expected = EXPERIMENT_CONFIGS
    for got, want in zip(entries, expected):
        if (
            got.run_name != want.run_name
            or got.experts_arg != want.experts_arg
            or got.blending != want.blending
            or got.diffusion != want.diffusion
            or got.role != want.role
            or got.display_name != want.display_name
        ):
            raise ExperimentConfigError(
                f"config mismatch at {want.run_name!r}: got {got!r}, expected {want!r}"
            )

    if tuple(cfg.run_name for cfg in entries) != RUN_ORDER:
        raise ExperimentConfigError(
            f"run_name order {tuple(cfg.run_name for cfg in entries)} != {RUN_ORDER}"
        )


def as_run_title_map(configs: Iterable[ExperimentConfig] = EXPERIMENT_CONFIGS) -> Mapping[str, str]:
    """Display names keyed by canonical ``run_name``."""

    return {cfg.run_name: cfg.display_name for cfg in configs}


# Fail fast if the module is edited into an incomplete 2x2.
validate_experiment_configs()
