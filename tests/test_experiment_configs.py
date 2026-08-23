"""Unit tests for the canonical blending × diffusion 2x2 experiment registry."""
from __future__ import annotations

from pathlib import Path

import pytest

from eval.experiment_configs import (
    DEFAULT_CONFIGS,
    EXPECTED_2X2_FLAGS,
    EXPERIMENT_CONFIGS,
    FACTORIAL_ANALYSES,
    PRIMARY_ASSESSMENT_RUN,
    ROLES,
    RUN_ORDER,
    RUN_TITLES,
    ExperimentConfig,
    ExperimentConfigError,
    get_experiment_config,
    validate_experiment_configs,
)
from tools.run_expert_matrix import (
    DEFAULT_CONFIGS as MATRIX_DEFAULT_CONFIGS,
    normalise_configs,
)

pytestmark = pytest.mark.unit


def test_exactly_four_configs() -> None:
    assert len(EXPERIMENT_CONFIGS) == 4
    validate_experiment_configs()


def test_all_binary_combinations_exactly_once() -> None:
    pairs = [cfg.flag_pair for cfg in EXPERIMENT_CONFIGS]
    assert len(pairs) == 4
    assert set(pairs) == EXPECTED_2X2_FLAGS
    assert len(set(pairs)) == 4


def test_canonical_names_and_order_unchanged() -> None:
    assert RUN_ORDER == ("none", "blending", "diffusion", "blending_diffusion")
    assert DEFAULT_CONFIGS == ("none", "blending", "diffusion", "blending,diffusion")
    assert [cfg.run_name for cfg in EXPERIMENT_CONFIGS] == list(RUN_ORDER)
    assert [cfg.experts_arg for cfg in EXPERIMENT_CONFIGS] == list(DEFAULT_CONFIGS)
    assert PRIMARY_ASSESSMENT_RUN == "blending_diffusion"
    assert get_experiment_config("blending_diffusion").experts_arg == "blending,diffusion"


def test_display_labels_and_roles() -> None:
    assert RUN_TITLES == {
        "none": "No expert",
        "blending": "Blending",
        "diffusion": "Diffusion",
        "blending_diffusion": "Blending + Diffusion",
    }
    assert ROLES == {
        "none": "baseline condition",
        "blending": "blending-only condition",
        "diffusion": "diffusion-only condition",
        "blending_diffusion": "combined condition",
    }
    for cfg in EXPERIMENT_CONFIGS:
        assert cfg.display_name == RUN_TITLES[cfg.run_name]
        assert cfg.role == ROLES[cfg.run_name]
        assert "main effect" not in cfg.role.lower()
        assert "main-effect" not in cfg.role.lower()
        assert "interaction" not in cfg.role.lower()


def test_factorial_analyses_documented_separately_from_cell_roles() -> None:
    assert FACTORIAL_ANALYSES == (
        "the blending main effect",
        "the diffusion main effect",
        "the blending × diffusion interaction",
    )
    # Main effects / interaction are design-level, not per-cell role labels.
    for phrase in FACTORIAL_ANALYSES:
        assert phrase not in ROLES.values()


def test_combined_cell_is_not_labelled_as_the_interaction() -> None:
    assert ROLES["blending_diffusion"] == "combined condition"
    assert "interaction" not in ROLES["blending_diffusion"].lower()


def test_no_fifth_configuration() -> None:
    with pytest.raises(ExperimentConfigError, match="exactly four"):
        validate_experiment_configs(
            (
                *EXPERIMENT_CONFIGS,
                ExperimentConfig(
                    run_name="extra",
                    experts_arg="extra",
                    blending=True,
                    diffusion=True,
                    role="invalid",
                    display_name="Extra",
                ),
            )
        )


def test_matrix_and_live_defaults_share_registry() -> None:
    assert MATRIX_DEFAULT_CONFIGS == DEFAULT_CONFIGS
    assert tuple(MATRIX_DEFAULT_CONFIGS) == (
        "none",
        "blending",
        "diffusion",
        "blending,diffusion",
    )


def test_live_analysis_commands_receive_same_expert_arguments() -> None:
    """Default matrix configs still normalise to the historical --experts tokens."""

    configs = normalise_configs(DEFAULT_CONFIGS)
    assert [cfg.experts_arg for cfg in configs] == [
        "none",
        "blending",
        "diffusion",
        "blending,diffusion",
    ]
    assert [cfg.run_name for cfg in configs] == [
        "none",
        "blending",
        "diffusion",
        "blending_diffusion",
    ]
    assert [cfg.experts for cfg in configs] == [
        (),
        ("blending",),
        ("diffusion",),
        ("blending", "diffusion"),
    ]


def test_tools_do_not_depend_on_dashboard_for_registry() -> None:
    import tools.run_expert_matrix as mx

    source = Path(mx.__file__).read_text(encoding="utf-8")
    assert "dashboard.experiment_configs" not in source
    assert "eval.experiment_configs" in source


def test_view_model_reexports_canonical_order() -> None:
    from dashboard import view_model as vm

    assert vm.RUN_ORDER == RUN_ORDER
    assert vm.RUN_TITLES == RUN_TITLES
    assert vm.PRIMARY_ASSESSMENT_RUN == PRIMARY_ASSESSMENT_RUN
