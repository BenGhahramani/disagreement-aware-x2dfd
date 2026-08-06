"""Unit tests for tools/make_peft_compatible_config.py.

No adapter weights, no GPU and no real PEFT import: the accepted-field set is
injected, and every file lives inside a pytest tmp_path.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Set

import pytest

from tools import make_peft_compatible_config as mpc

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

# The fields PEFT 0.10.0's LoraConfig actually accepts.
OLD_PEFT_FIELDS: Set[str] = {
    "peft_type",
    "auto_mapping",
    "base_model_name_or_path",
    "revision",
    "task_type",
    "inference_mode",
    "r",
    "target_modules",
    "lora_alpha",
    "lora_dropout",
    "fan_in_fan_out",
    "bias",
    "use_rslora",
    "modules_to_save",
    "init_lora_weights",
    "layers_to_transform",
    "layers_pattern",
    "rank_pattern",
    "alpha_pattern",
    "megatron_config",
    "megatron_core",
    "loftq_config",
    "use_dora",
    "layer_replication",
}


def compatible_config() -> Dict[str, Any]:
    """A config an old PEFT can already parse."""

    return {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": "weights/base/llava-v1.5-7b",
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "bias": "none",
        "inference_mode": True,
        "target_modules": ["q_proj", "k_proj", "v_proj"],
        "rank_pattern": {},
        "alpha_pattern": {},
    }


def write_config(directory: Path, config: Dict[str, Any]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / mpc.CONFIG_FILENAME
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return path


class StubLoraConfig:
    """Stand-in whose __init__ signature mirrors an old LoraConfig."""

    def __init__(
        self,
        peft_type=None,
        task_type=None,
        base_model_name_or_path=None,
        r=8,
        lora_alpha=8,
        **_ignored,
    ):  # pragma: no cover - never called, only introspected
        pass


# --------------------------------------------------------------------------
# accepted_fields
# --------------------------------------------------------------------------


def test_accepted_fields_reads_signature_dynamically() -> None:
    accepted = mpc.accepted_fields(StubLoraConfig)

    assert "r" in accepted
    assert "lora_alpha" in accepted
    assert "self" not in accepted
    # **kwargs must not be reported as an accepted field name
    assert "_ignored" not in accepted


def test_accepted_fields_matches_installed_peft() -> None:
    peft = pytest.importorskip("peft")

    accepted = mpc.accepted_fields(peft.LoraConfig)

    assert {"r", "lora_alpha", "target_modules"} <= accepted


# --------------------------------------------------------------------------
# classification
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected_reason",
    [
        (None, "null"),
        (False, "false (feature disabled)"),
        ({}, "empty dict"),
        ([], "empty list"),
        ("", "empty str"),
    ],
)
def test_classify_value_treats_empty_states_as_inert(value: Any, expected_reason: str) -> None:
    operative, reason = mpc.classify_value("some_new_field", value, {})

    assert operative is False
    assert reason == expected_reason


@pytest.mark.parametrize("value", [16, 0.5, "corda", ["a"], {"k": "v"}, True])
def test_classify_value_treats_real_values_as_meaningful(value: Any) -> None:
    operative, _ = mpc.classify_value("some_new_field", value, {})

    assert operative is True


def test_classify_value_gated_field_is_inert_when_gate_off() -> None:
    config = {"use_qalora": False, "qalora_group_size": 16}

    operative, reason = mpc.classify_value("qalora_group_size", 16, config)

    assert operative is False
    assert "use_qalora" in reason


def test_classify_value_gated_field_is_meaningful_when_gate_on() -> None:
    config = {"use_qalora": True, "qalora_group_size": 16}

    operative, reason = mpc.classify_value("qalora_group_size", 16, config)

    assert operative is True
    assert "enabled" in reason


# --------------------------------------------------------------------------
# analyse
# --------------------------------------------------------------------------


def test_analyse_already_compatible_config_has_nothing_to_remove() -> None:
    analysis = mpc.analyse(compatible_config(), OLD_PEFT_FIELDS)

    assert analysis.unsupported == []
    assert analysis.removable == []
    assert analysis.blocking == []
    assert set(analysis.supported) == set(compatible_config())


def test_analyse_flags_unsupported_null_and_false_fields_as_removable() -> None:
    config = {**compatible_config(), "corda_config": None, "use_qalora": False}

    analysis = mpc.analyse(config, OLD_PEFT_FIELDS)

    assert {v.key for v in analysis.removable} == {"corda_config", "use_qalora"}
    assert analysis.blocking == []


def test_analyse_blocks_on_meaningful_unsupported_field() -> None:
    config = {**compatible_config(), "exclude_modules": ["vision_tower"]}

    analysis = mpc.analyse(config, OLD_PEFT_FIELDS)

    assert [v.key for v in analysis.blocking] == ["exclude_modules"]


# --------------------------------------------------------------------------
# CLI: already-compatible input
# --------------------------------------------------------------------------


def test_main_already_compatible_writes_identical_copy(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(mpc, "load_lora_config_class", lambda: StubLoraConfig)
    monkeypatch.setattr(mpc, "accepted_fields", lambda cls: OLD_PEFT_FIELDS)
    path = write_config(tmp_path / "adapter", compatible_config())

    exit_code = mpc.main([str(path)])

    assert exit_code == mpc.EXIT_OK
    output = path.with_name("adapter_config.compatible.json")
    assert json.loads(output.read_text()) == compatible_config()


# --------------------------------------------------------------------------
# CLI: removal of inert fields
# --------------------------------------------------------------------------


def new_peft_config() -> Dict[str, Any]:
    """A config shaped like the real PEFT 0.17 output."""

    return {
        **compatible_config(),
        "corda_config": None,
        "eva_config": None,
        "exclude_modules": None,
        "lora_bias": False,
        "qalora_group_size": 16,
        "target_parameters": None,
        "trainable_token_indices": None,
        "use_qalora": False,
    }


def test_main_removes_unsupported_null_fields(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(mpc, "load_lora_config_class", lambda: StubLoraConfig)
    monkeypatch.setattr(mpc, "accepted_fields", lambda cls: OLD_PEFT_FIELDS)
    config = {**compatible_config(), "corda_config": None, "eva_config": None}
    path = write_config(tmp_path / "adapter", config)

    exit_code = mpc.main([str(path)])

    assert exit_code == mpc.EXIT_OK
    written = json.loads(path.with_name("adapter_config.compatible.json").read_text())
    assert "corda_config" not in written
    assert "eva_config" not in written
    printed = capsys.readouterr().out
    assert "corda_config = None" in printed
    assert "eva_config = None" in printed


def test_main_removes_unsupported_false_fields(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(mpc, "load_lora_config_class", lambda: StubLoraConfig)
    monkeypatch.setattr(mpc, "accepted_fields", lambda cls: OLD_PEFT_FIELDS)
    config = {**compatible_config(), "lora_bias": False, "use_qalora": False}
    path = write_config(tmp_path / "adapter", config)

    exit_code = mpc.main([str(path)])

    assert exit_code == mpc.EXIT_OK
    written = json.loads(path.with_name("adapter_config.compatible.json").read_text())
    assert "lora_bias" not in written
    assert "use_qalora" not in written
    assert "lora_bias = False" in capsys.readouterr().out


def test_main_removes_full_new_peft_field_set(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(mpc, "load_lora_config_class", lambda: StubLoraConfig)
    monkeypatch.setattr(mpc, "accepted_fields", lambda cls: OLD_PEFT_FIELDS)
    path = write_config(tmp_path / "adapter", new_peft_config())

    exit_code = mpc.main([str(path)])

    assert exit_code == mpc.EXIT_OK
    written = json.loads(path.with_name("adapter_config.compatible.json").read_text())
    assert set(written) == set(compatible_config())


# --------------------------------------------------------------------------
# CLI: refusal on meaningful values
# --------------------------------------------------------------------------


def test_main_blocks_on_meaningful_unsupported_value(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(mpc, "load_lora_config_class", lambda: StubLoraConfig)
    monkeypatch.setattr(mpc, "accepted_fields", lambda cls: OLD_PEFT_FIELDS)
    config = {**compatible_config(), "trainable_token_indices": [1, 2, 3]}
    path = write_config(tmp_path / "adapter", config)

    exit_code = mpc.main([str(path)])

    assert exit_code == mpc.EXIT_BLOCKED
    assert not path.with_name("adapter_config.compatible.json").exists()


def test_main_blocks_when_gate_is_enabled(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(mpc, "load_lora_config_class", lambda: StubLoraConfig)
    monkeypatch.setattr(mpc, "accepted_fields", lambda cls: OLD_PEFT_FIELDS)
    config = {**compatible_config(), "use_qalora": True, "qalora_group_size": 32}
    path = write_config(tmp_path / "adapter", config)

    exit_code = mpc.main([str(path)])

    assert exit_code == mpc.EXIT_BLOCKED
    assert not path.with_name("adapter_config.compatible.json").exists()


# --------------------------------------------------------------------------
# CLI: preservation guarantees
# --------------------------------------------------------------------------


def test_supported_fields_are_preserved_exactly(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(mpc, "load_lora_config_class", lambda: StubLoraConfig)
    monkeypatch.setattr(mpc, "accepted_fields", lambda cls: OLD_PEFT_FIELDS)
    path = write_config(tmp_path / "adapter", new_peft_config())

    mpc.main([str(path)])

    written = json.loads(path.with_name("adapter_config.compatible.json").read_text())
    for key, value in compatible_config().items():
        assert written[key] == value
    assert list(written) == list(compatible_config())


def test_original_is_not_overwritten_by_default(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(mpc, "load_lora_config_class", lambda: StubLoraConfig)
    monkeypatch.setattr(mpc, "accepted_fields", lambda cls: OLD_PEFT_FIELDS)
    path = write_config(tmp_path / "adapter", new_peft_config())
    before = path.read_bytes()

    mpc.main([str(path)])

    assert path.read_bytes() == before
    assert json.loads(path.read_text()) == new_peft_config()


def test_directory_argument_resolves_to_adapter_config(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(mpc, "load_lora_config_class", lambda: StubLoraConfig)
    monkeypatch.setattr(mpc, "accepted_fields", lambda cls: OLD_PEFT_FIELDS)
    directory = tmp_path / "adapter"
    write_config(directory, new_peft_config())

    exit_code = mpc.main([str(directory)])

    assert exit_code == mpc.EXIT_OK
    assert (directory / "adapter_config.compatible.json").is_file()


def test_output_flag_controls_destination(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(mpc, "load_lora_config_class", lambda: StubLoraConfig)
    monkeypatch.setattr(mpc, "accepted_fields", lambda cls: OLD_PEFT_FIELDS)
    path = write_config(tmp_path / "adapter", new_peft_config())
    destination = tmp_path / "elsewhere" / "compat.json"

    exit_code = mpc.main([str(path), "--output", str(destination)])

    assert exit_code == mpc.EXIT_OK
    assert destination.is_file()


# --------------------------------------------------------------------------
# CLI: in-place mode and backups
# --------------------------------------------------------------------------


def test_in_place_backs_up_then_rewrites(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(mpc, "load_lora_config_class", lambda: StubLoraConfig)
    monkeypatch.setattr(mpc, "accepted_fields", lambda cls: OLD_PEFT_FIELDS)
    path = write_config(tmp_path / "adapter", new_peft_config())

    exit_code = mpc.main([str(path), "--in-place"])

    assert exit_code == mpc.EXIT_OK
    backup = path.with_name("adapter_config.original.json")
    assert json.loads(backup.read_text()) == new_peft_config()
    assert set(json.loads(path.read_text())) == set(compatible_config())


def test_in_place_keeps_a_pristine_existing_backup(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(mpc, "load_lora_config_class", lambda: StubLoraConfig)
    monkeypatch.setattr(mpc, "accepted_fields", lambda cls: OLD_PEFT_FIELDS)
    path = write_config(tmp_path / "adapter", new_peft_config())
    backup = path.with_name("adapter_config.original.json")
    backup.write_bytes(path.read_bytes())

    exit_code = mpc.main([str(path), "--in-place"])

    assert exit_code == mpc.EXIT_OK
    assert json.loads(backup.read_text()) == new_peft_config()


def test_in_place_refuses_to_clobber_a_differing_backup(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(mpc, "load_lora_config_class", lambda: StubLoraConfig)
    monkeypatch.setattr(mpc, "accepted_fields", lambda cls: OLD_PEFT_FIELDS)
    path = write_config(tmp_path / "adapter", new_peft_config())
    backup = path.with_name("adapter_config.original.json")
    backup.write_text('{"something": "precious"}', encoding="utf-8")
    unchanged = path.read_bytes()

    exit_code = mpc.main([str(path), "--in-place"])

    assert exit_code == mpc.EXIT_USAGE
    assert json.loads(backup.read_text()) == {"something": "precious"}
    assert path.read_bytes() == unchanged


# --------------------------------------------------------------------------
# CLI: input errors
# --------------------------------------------------------------------------


def test_missing_config_returns_usage_error(tmp_path: Path) -> None:
    assert mpc.main([str(tmp_path / "nope.json")]) == mpc.EXIT_USAGE


def test_malformed_json_returns_usage_error(tmp_path: Path) -> None:
    path = tmp_path / mpc.CONFIG_FILENAME
    path.write_text("{not json", encoding="utf-8")

    assert mpc.main([str(path)]) == mpc.EXIT_USAGE


def test_non_object_json_returns_usage_error(tmp_path: Path) -> None:
    path = tmp_path / mpc.CONFIG_FILENAME
    path.write_text("[1, 2, 3]", encoding="utf-8")

    assert mpc.main([str(path)]) == mpc.EXIT_USAGE


def test_json_report_mode_emits_parseable_output(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(mpc, "load_lora_config_class", lambda: StubLoraConfig)
    monkeypatch.setattr(mpc, "accepted_fields", lambda cls: OLD_PEFT_FIELDS)
    path = write_config(tmp_path / "adapter", new_peft_config())

    exit_code = mpc.main([str(path), "--json"])

    assert exit_code == mpc.EXIT_OK
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "converted"
    assert len(report["removable"]) == 8
    assert report["blocking"] == []
