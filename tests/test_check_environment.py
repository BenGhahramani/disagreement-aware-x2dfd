"""Unit tests for tools/check_environment.py.

These tests never touch real weights, real datasets or a GPU: dependency state
is mocked and every path lives inside a pytest tmp_path.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from tools import check_environment as ce

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def fake_torch(
    *,
    version: str = "2.1.0+cu121",
    cuda_available: bool = True,
    device_count: int = 1,
    name: str = "NVIDIA GeForce RTX 3080",
    total_memory_gib: float = 10.0,
) -> SimpleNamespace:
    """Build a stand-in torch module exposing only what the checker touches."""

    def get_device_properties(index: int) -> SimpleNamespace:
        return SimpleNamespace(total_memory=int(total_memory_gib * (1024**3)))

    return SimpleNamespace(
        __version__=version,
        version=SimpleNamespace(cuda="12.1"),
        cuda=SimpleNamespace(
            is_available=lambda: cuda_available,
            device_count=lambda: device_count,
            get_device_name=lambda index: name,
            get_device_properties=get_device_properties,
        ),
    )


def finder_for(available: set[str]):
    def _finder(module: str) -> bool:
        return module in available

    return _finder


def status_of(results: List[ce.CheckResult], name: str) -> ce.Status:
    for result in results:
        if result.name == name:
            return result.status
    raise AssertionError(f"no check named {name} in {[r.name for r in results]}")


def write_config(path: Path, payload: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# python version
# --------------------------------------------------------------------------


def test_python_version_pass_on_recommended() -> None:
    result = ce.check_python((3, 10, 13))
    assert result.status is ce.Status.PASS


def test_python_version_warns_on_newer_minor() -> None:
    result = ce.check_python((3, 11, 9))
    assert result.status is ce.Status.WARN
    assert "3.10" in result.detail


def test_python_version_fails_below_minimum() -> None:
    result = ce.check_python((3, 8, 10))
    assert result.status is ce.Status.FAIL


# --------------------------------------------------------------------------
# imports
# --------------------------------------------------------------------------


def test_required_import_missing_is_fail() -> None:
    specs = [ce.ImportSpec("torch", "runtime"), ce.ImportSpec("peft", "lora")]
    results = ce.check_imports(specs, finder=finder_for({"torch"}))
    assert status_of(results, "import.torch") is ce.Status.PASS
    assert status_of(results, "import.peft") is ce.Status.FAIL
    assert "lora" in [r for r in results if r.name == "import.peft"][0].detail


def test_optional_import_missing_is_warn_only() -> None:
    specs = [ce.ImportSpec("bitsandbytes", "quantisation", required=False)]
    results = ce.check_imports(specs, finder=finder_for(set()))
    assert status_of(results, "import.bitsandbytes") is ce.Status.WARN
    assert results[0].required is False


def test_protobuf_is_a_required_import() -> None:
    """Regression: a missing protobuf only surfaced at tokenizer load time.

    LlamaTokenizer(use_fast=False) raises ImportError deep inside the runner,
    where lora_inference swallows it into the answer text, so the checker has to
    catch it up front.
    """
    modules = {spec.module for spec in ce.CORE_IMPORTS}
    assert "google.protobuf" in modules

    results = ce.check_imports(list(ce.CORE_IMPORTS), finder=finder_for(set()))
    assert status_of(results, "import.google.protobuf") is ce.Status.FAIL


# --------------------------------------------------------------------------
# torch / CUDA / VRAM
# --------------------------------------------------------------------------


def test_torch_import_failure_reports_fail() -> None:
    def boom() -> Any:
        raise ImportError("No module named 'torch'")

    results = ce.check_torch(None, importer=boom)
    assert status_of(results, "torch.import") is ce.Status.FAIL


def test_torch_without_cuda_fails() -> None:
    results = ce.check_torch(fake_torch(cuda_available=False))
    assert status_of(results, "torch.cuda") is ce.Status.FAIL


def test_gpu_below_recommended_vram_warns_but_does_not_block() -> None:
    results = ce.check_torch(fake_torch(total_memory_gib=10.0), recommended_vram_gib=14.0)
    gpu = [r for r in results if r.name == "gpu.0"][0]
    assert gpu.status is ce.Status.WARN
    assert gpu.required is False
    passed, _ = ce.summarise(results)
    assert passed is True


def test_gpu_with_enough_vram_passes() -> None:
    results = ce.check_torch(fake_torch(total_memory_gib=24.0), recommended_vram_gib=14.0)
    assert status_of(results, "gpu.0") is ce.Status.PASS


def test_cuda_probe_error_is_reported_not_swallowed() -> None:
    broken = fake_torch()
    def raise_runtime() -> bool:
        raise RuntimeError("driver mismatch")

    broken.cuda.is_available = raise_runtime
    results = ce.check_torch(broken)
    detail = [r for r in results if r.name == "torch.cuda"][0].detail
    assert "driver mismatch" in detail


# --------------------------------------------------------------------------
# config parsing and weight requirements
# --------------------------------------------------------------------------


def test_load_config_rejects_non_mapping(tmp_path: Path) -> None:
    cfg = write_config(tmp_path / "c.yaml", "- just\n- a list\n")
    with pytest.raises(ce.ConfigError):
        ce.load_config(cfg)


def test_load_config_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ce.ConfigError):
        ce.load_config(tmp_path / "nope.yaml")


def test_collect_weight_requirements_covers_model_and_experts(tmp_path: Path) -> None:
    config: Dict[str, Any] = {
        "model": {"base": "weights/base/llava", "adapter": "weights/ckpt/lora"},
        "weak_supplies": [
            {"provider": "blending", "alias": "Blending", "weights_path": "weights/blending/best.pth"},
            {"provider": "diffusion_detector", "alias": "Diffusion", "weights_dir": "weights/", "model": "ours-sync"},
        ],
    }
    reqs = ce.collect_weight_requirements(config, tmp_path)
    labels = {r.label for r in reqs}
    assert {
        "weights.model.base",
        "weights.model.adapter",
        "weights.expert.Blending",
        "weights.expert.Diffusion.dir",
        "weights.expert.Diffusion.config",
    } <= labels
    diffusion_cfg = [r for r in reqs if r.label == "weights.expert.Diffusion.config"][0]
    assert diffusion_cfg.path == tmp_path / "weights" / "ours-sync" / "config.yaml"


def test_resolve_path_expands_env_vars(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("X2DFD_TEST_ROOT", str(tmp_path))
    resolved = ce.resolve_path("${X2DFD_TEST_ROOT}/weights/base", tmp_path)
    assert resolved == Path(str(tmp_path)) / "weights" / "base"


def test_check_paths_distinguishes_file_and_dir(tmp_path: Path) -> None:
    a_file = tmp_path / "w.pth"
    a_file.write_bytes(b"0")
    a_dir = tmp_path / "adapter"
    a_dir.mkdir()
    results = ce.check_paths(
        [
            ce.PathRequirement("weights.file", a_file, "file"),
            ce.PathRequirement("weights.dir", a_dir, "dir"),
            ce.PathRequirement("weights.absent", tmp_path / "gone", "dir"),
            ce.PathRequirement("weights.file_as_dir", a_file, "dir"),
        ]
    )
    assert status_of(results, "weights.file") is ce.Status.PASS
    assert status_of(results, "weights.dir") is ce.Status.PASS
    assert status_of(results, "weights.absent") is ce.Status.FAIL
    assert status_of(results, "weights.file_as_dir") is ce.Status.FAIL


# --------------------------------------------------------------------------
# writable output dir
# --------------------------------------------------------------------------


def test_writable_dir_creates_and_cleans_probe(tmp_path: Path) -> None:
    target = tmp_path / "eval" / "outputs"
    result = ce.check_writable_dir(target)
    assert result.status is ce.Status.PASS
    assert target.is_dir()
    assert not (target / ".x2dfd_write_probe").exists()


def test_writable_dir_fails_when_parent_is_a_file(tmp_path: Path) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("not a dir", encoding="utf-8")
    result = ce.check_writable_dir(blocker / "outputs")
    assert result.status is ce.Status.FAIL


# --------------------------------------------------------------------------
# dataset inputs
# --------------------------------------------------------------------------


def make_dataset_json(tmp_path: Path, *, image_exists: bool) -> Path:
    root = tmp_path / "images"
    root.mkdir(parents=True, exist_ok=True)
    if image_exists:
        (root / "a.png").write_bytes(b"\x89PNG")
    payload = {"Description": str(root), "images": [{"image_path": "a.png"}]}
    json_path = tmp_path / "data.json"
    json_path.write_text(json.dumps(payload), encoding="utf-8")
    return json_path


def test_dataset_check_passes_when_image_present(tmp_path: Path) -> None:
    json_path = make_dataset_json(tmp_path, image_exists=True)
    config = {"infer": {"inputs": [str(json_path)]}}
    results = ce.check_dataset_inputs(config, tmp_path)
    assert results[0].status is ce.Status.PASS


def test_dataset_check_fails_when_image_missing(tmp_path: Path) -> None:
    json_path = make_dataset_json(tmp_path, image_exists=False)
    config = {"infer": {"inputs": [str(json_path)]}}
    results = ce.check_dataset_inputs(config, tmp_path)
    assert results[0].status is ce.Status.FAIL
    assert "not found" in results[0].detail


def test_dataset_check_fails_on_missing_json(tmp_path: Path) -> None:
    config = {"infer": {"inputs": ["datasets/does_not_exist.json"]}}
    results = ce.check_dataset_inputs(config, tmp_path)
    assert results[0].status is ce.Status.FAIL


def test_dataset_check_fails_on_malformed_json(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    config = {"infer": {"inputs": [str(bad)]}}
    results = ce.check_dataset_inputs(config, tmp_path)
    assert results[0].status is ce.Status.FAIL


def test_dataset_check_warns_when_no_inputs_configured(tmp_path: Path) -> None:
    results = ce.check_dataset_inputs({}, tmp_path)
    assert results[0].status is ce.Status.WARN
    assert results[0].required is False


# --------------------------------------------------------------------------
# summarise / render
# --------------------------------------------------------------------------


def test_summarise_passes_with_only_optional_warnings() -> None:
    results = [
        ce.CheckResult("a", ce.Status.PASS, ""),
        ce.CheckResult("b", ce.Status.WARN, "", required=False),
    ]
    passed, counts = ce.summarise(results)
    assert passed is True
    assert counts["WARN"] == 1


def test_summarise_fails_on_required_failure() -> None:
    results = [ce.CheckResult("a", ce.Status.FAIL, "missing")]
    passed, _ = ce.summarise(results)
    assert passed is False


def test_strict_mode_turns_warnings_into_failure() -> None:
    results = [ce.CheckResult("a", ce.Status.WARN, "", required=False)]
    assert ce.summarise(results, strict=False)[0] is True
    assert ce.summarise(results, strict=True)[0] is False


def test_render_text_lists_blocking_checks() -> None:
    results = [
        ce.CheckResult("import.peft", ce.Status.FAIL, "not installed"),
        ce.CheckResult("python.version", ce.Status.PASS, "3.10.13"),
    ]
    text = ce.render_text(results)
    assert "OVERALL: FAIL" in text
    assert "import.peft" in text


# --------------------------------------------------------------------------
# run_checks + main
# --------------------------------------------------------------------------


def minimal_config_yaml() -> str:
    return (
        "paths:\n"
        "  results_dir: eval/outputs\n"
        "model:\n"
        "  base: weights/base/llava-v1.5-7b\n"
        "  adapter: weights/ckpt/lora\n"
        "weak_supplies:\n"
        "  - provider: blending\n"
        "    alias: Blending\n"
        "    weights_path: weights/blending_models/best_gf.pth\n"
        "infer:\n"
        "  inputs: []\n"
    )


def test_run_checks_reports_missing_weights_without_touching_real_paths(tmp_path: Path) -> None:
    config_path = write_config(tmp_path / "eval" / "configs" / "infer_config.yaml", minimal_config_yaml())
    options = ce.Options(project_root=tmp_path, config_path=config_path)
    results = ce.run_checks(
        options,
        finder=finder_for({spec.module for spec in ce.CORE_IMPORTS}),
        torch_module=fake_torch(total_memory_gib=24.0),
    )
    assert status_of(results, "config.parse") is ce.Status.PASS
    assert status_of(results, "weights.model.adapter") is ce.Status.FAIL
    assert status_of(results, "output.results_dir") is ce.Status.PASS
    passed, _ = ce.summarise(results)
    assert passed is False


def test_run_checks_all_green_when_everything_present(tmp_path: Path) -> None:
    config_path = write_config(tmp_path / "cfg.yaml", minimal_config_yaml())
    (tmp_path / "weights" / "base" / "llava-v1.5-7b").mkdir(parents=True)
    (tmp_path / "weights" / "ckpt" / "lora").mkdir(parents=True)
    blending = tmp_path / "weights" / "blending_models"
    blending.mkdir(parents=True)
    (blending / "best_gf.pth").write_bytes(b"0")

    options = ce.Options(project_root=tmp_path, config_path=config_path)
    results = ce.run_checks(
        options,
        finder=finder_for(
            {spec.module for spec in ce.CORE_IMPORTS} | {spec.module for spec in ce.OPTIONAL_IMPORTS}
        ),
        torch_module=fake_torch(total_memory_gib=24.0),
    )
    passed, counts = ce.summarise(results)
    assert passed is True, [r for r in results if r.status is ce.Status.FAIL]
    assert counts["FAIL"] == 0


def test_run_checks_honours_skip_flags(tmp_path: Path) -> None:
    config_path = write_config(tmp_path / "cfg.yaml", minimal_config_yaml())
    options = ce.Options(
        project_root=tmp_path,
        config_path=config_path,
        check_weights=False,
        check_datasets=False,
    )
    results = ce.run_checks(
        options,
        finder=finder_for({spec.module for spec in ce.CORE_IMPORTS}),
        torch_module=fake_torch(total_memory_gib=24.0),
    )
    assert status_of(results, "weights") is ce.Status.SKIP
    assert status_of(results, "datasets") is ce.Status.SKIP


def test_run_checks_reports_unparsable_config(tmp_path: Path) -> None:
    config_path = write_config(tmp_path / "cfg.yaml", "just a string")
    options = ce.Options(project_root=tmp_path, config_path=config_path)
    results = ce.run_checks(
        options,
        finder=finder_for({spec.module for spec in ce.CORE_IMPORTS}),
        torch_module=fake_torch(),
    )
    assert status_of(results, "config.parse") is ce.Status.FAIL


def test_main_exit_code_and_json_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setattr(
        ce, "run_checks", lambda options, **kwargs: [ce.CheckResult("x", ce.Status.PASS, "ok")]
    )
    out_file = tmp_path / "report.json"
    code = ce.main(
        [
            "--project-root",
            str(tmp_path),
            "--config",
            str(tmp_path / "cfg.yaml"),
            "--json",
            "--output",
            str(out_file),
        ]
    )
    assert code == ce.EXIT_PASS
    payload = json.loads(out_file.read_text(encoding="utf-8"))
    assert payload["overall"] == "PASS"
    assert json.loads(capsys.readouterr().out)["counts"]["PASS"] == 1


def test_main_returns_one_on_required_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ce,
        "run_checks",
        lambda options, **kwargs: [ce.CheckResult("weights.model.base", ce.Status.FAIL, "missing")],
    )
    code = ce.main(["--project-root", str(tmp_path)])
    assert code == ce.EXIT_FAIL


def test_main_rejects_bad_project_root(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as excinfo:
        ce.main(["--project-root", str(tmp_path / "nope")])
    assert excinfo.value.code == ce.EXIT_USAGE


def test_help_is_available() -> None:
    with pytest.raises(SystemExit) as excinfo:
        ce.main(["--help"])
    assert excinfo.value.code == 0
