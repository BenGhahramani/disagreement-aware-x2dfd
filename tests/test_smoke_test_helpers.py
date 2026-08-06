"""Unit tests for tools/run_smoke_test.py.

The inference subprocess is always mocked: no model, no GPU, no network. Every
path lives inside a pytest tmp_path.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence

import pytest

from tools import run_smoke_test as rst

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def make_image(directory: Path, name: str = "face.jpg") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(b"\xff\xd8\xff\xe0not-a-real-jpeg")
    return path


def make_manifest(tmp_path: Path, *, image: Optional[Path] = None, **overrides: Any) -> Path:
    image = image if image is not None else make_image(tmp_path / "images")
    payload: Dict[str, Any] = {
        "Description": str(image.parent),
        "images": [{"image_path": image.name, "label": "real"}],
    }
    payload.update(overrides)
    path = tmp_path / "demo_one.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def conversation_item(
    *,
    answer: Optional[str] = "This image is real.",
    real: Optional[str] = "0.9210",
    fake: Optional[str] = "0.0790",
    image: str = "C:/images/face.jpg",
) -> Dict[str, Any]:
    conversations: List[Dict[str, str]] = [
        {"from": "human", "value": "<image>\nIs this image real or fake?"}
    ]
    if answer is not None:
        conversations.append({"from": "gpt", "value": answer})
    if real is not None:
        conversations.append({"from": "real score", "value": real})
    if fake is not None:
        conversations.append({"from": "fake score", "value": fake})
    return {"id": "1", "image": image, "conversations": conversations}


class FakeRunner:
    """Stand-in for subprocess.run that can also write the output file."""

    def __init__(
        self,
        *,
        returncode: int = 0,
        stdout: str = "",
        stderr: str = "",
        writes: Optional[Dict[Path, Any]] = None,
        raises: Optional[BaseException] = None,
    ) -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.writes = writes or {}
        self.raises = raises
        self.calls: List[Sequence[str]] = []

    def __call__(self, command: Sequence[str], **kwargs: Any) -> SimpleNamespace:
        self.calls.append(list(command))
        if self.raises is not None:
            raise self.raises
        for path, payload in self.writes.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            text = payload if isinstance(payload, str) else json.dumps(payload)
            path.write_text(text, encoding="utf-8")
        return SimpleNamespace(returncode=self.returncode, stdout=self.stdout, stderr=self.stderr)


def run_main(
    tmp_path: Path,
    manifest: Path,
    runner: FakeRunner,
    *,
    extra: Optional[Sequence[str]] = None,
) -> int:
    argv = [
        "--manifest",
        str(manifest),
        "--project-root",
        str(tmp_path),
        "--summary",
        str(tmp_path / "summary.json"),
        "--output",
        str(tmp_path / "result.json"),
        "--no-vram-sampling",
    ]
    if extra:
        argv.extend(extra)
    return rst.main(argv, runner=runner)


def read_summary(tmp_path: Path) -> Dict[str, Any]:
    return json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# manifest validation
# --------------------------------------------------------------------------


def test_manifest_missing_file_is_usage_error(tmp_path: Path) -> None:
    runner = FakeRunner()

    exit_code = run_main(tmp_path, tmp_path / "nope.json", runner)

    assert exit_code == rst.EXIT_USAGE
    assert runner.calls == []


def test_manifest_malformed_json_is_usage_error(tmp_path: Path) -> None:
    manifest = tmp_path / "demo_one.json"
    manifest.write_text("{not json", encoding="utf-8")

    assert run_main(tmp_path, manifest, FakeRunner()) == rst.EXIT_USAGE


def test_manifest_without_images_is_rejected(tmp_path: Path) -> None:
    manifest = tmp_path / "demo_one.json"
    manifest.write_text(json.dumps({"Description": str(tmp_path)}), encoding="utf-8")

    assert run_main(tmp_path, manifest, FakeRunner()) == rst.EXIT_USAGE


def test_manifest_missing_image_file_is_rejected(tmp_path: Path) -> None:
    manifest = tmp_path / "demo_one.json"
    manifest.write_text(
        json.dumps({"Description": str(tmp_path), "images": [{"image_path": "ghost.jpg"}]}),
        encoding="utf-8",
    )

    assert run_main(tmp_path, manifest, FakeRunner()) == rst.EXIT_USAGE


def test_manifest_unsupported_extension_is_rejected(tmp_path: Path) -> None:
    image = make_image(tmp_path / "images", name="notes.txt")
    manifest = make_manifest(tmp_path, image=image)

    assert run_main(tmp_path, manifest, FakeRunner()) == rst.EXIT_USAGE


def test_manifest_relative_path_without_description_is_rejected(tmp_path: Path) -> None:
    manifest = tmp_path / "demo_one.json"
    manifest.write_text(json.dumps({"images": [{"image_path": "face.jpg"}]}), encoding="utf-8")

    assert run_main(tmp_path, manifest, FakeRunner()) == rst.EXIT_USAGE


def test_manifest_accepts_absolute_paths(tmp_path: Path) -> None:
    image = make_image(tmp_path / "images")
    manifest = tmp_path / "demo_one.json"
    manifest.write_text(
        json.dumps({"images": [{"image_path": str(image)}]}), encoding="utf-8"
    )

    info = rst.load_manifest(manifest)

    assert info.images == [image]


# --------------------------------------------------------------------------
# command construction
# --------------------------------------------------------------------------


def test_build_command_uses_runner_module_and_explicit_output(tmp_path: Path) -> None:
    command = rst.build_command(
        manifest=tmp_path / "m.json",
        output=tmp_path / "out.json",
        config=tmp_path / "cfg.yaml",
        experts="none",
        python_executable="py",
    )

    assert command[:4] == ["py", "-m", "eval.infer.runner", "--config"]
    assert "--experts" in command and command[command.index("--experts") + 1] == "none"
    assert command[command.index("--output") + 1] == str(tmp_path / "out.json")


def test_build_environment_sets_quantisation_flag() -> None:
    env = rst.build_environment(base_env={}, load_4bit=True)

    assert env["X2DFD_LOAD_4BIT"] == "1"
    assert "X2DFD_LOAD_8BIT" not in env
    assert env["USE_PROGRESS_BAR"] == "0"


def test_build_environment_defaults_to_no_quantisation() -> None:
    env = rst.build_environment(base_env={})

    assert "X2DFD_LOAD_4BIT" not in env
    assert "X2DFD_LOAD_8BIT" not in env


# --------------------------------------------------------------------------
# subprocess failure modes
# --------------------------------------------------------------------------


def test_subprocess_failure_is_reported(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path)
    runner = FakeRunner(returncode=1, stderr="ModuleNotFoundError: llava")

    exit_code = run_main(tmp_path, manifest, runner)

    assert exit_code == rst.EXIT_FAIL
    summary = read_summary(tmp_path)
    assert summary["status"] == "fail"
    assert any("exited with code 1" in f for f in summary["failures"])
    assert summary["process"]["stderr"] == "ModuleNotFoundError: llava"


def test_timeout_is_recorded_and_fails(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path)
    runner = FakeRunner(
        raises=subprocess.TimeoutExpired(cmd="runner", timeout=5, output="partial", stderr="")
    )

    exit_code = run_main(tmp_path, manifest, runner, extra=["--timeout", "5"])

    assert exit_code == rst.EXIT_FAIL
    summary = read_summary(tmp_path)
    assert summary["process"]["timed_out"] is True
    assert summary["process"]["exit_code"] is None
    assert any("timeout" in f for f in summary["failures"])


def test_cuda_oom_is_detected(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path)
    runner = FakeRunner(
        returncode=1,
        stderr="torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 224.00 MiB",
    )

    exit_code = run_main(tmp_path, manifest, runner)

    assert exit_code == rst.EXIT_FAIL
    summary = read_summary(tmp_path)
    assert summary["cuda_oom"] is True
    assert any("out of memory" in f.lower() for f in summary["failures"])


@pytest.mark.parametrize(
    "text, expected",
    [
        ("CUDA out of memory", True),
        ("torch.cuda.OutOfMemoryError", True),
        ("CUBLAS_STATUS_ALLOC_FAILED", True),
        ("everything fine", False),
        ("", False),
    ],
)
def test_detect_cuda_oom(text: str, expected: bool) -> None:
    assert rst.detect_cuda_oom(text) is expected


# --------------------------------------------------------------------------
# output location and validation
# --------------------------------------------------------------------------


def test_missing_output_fails(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path)
    runner = FakeRunner(returncode=0, stdout="done")

    exit_code = run_main(tmp_path, manifest, runner)

    assert exit_code == rst.EXIT_FAIL
    assert any("no output JSON" in f for f in read_summary(tmp_path)["failures"])


def test_malformed_output_json_fails(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path)
    runner = FakeRunner(returncode=0, writes={tmp_path / "result.json": "{broken"})

    exit_code = run_main(tmp_path, manifest, runner)

    assert exit_code == rst.EXIT_FAIL
    assert any("not valid JSON" in f for f in read_summary(tmp_path)["failures"])


def test_output_must_be_a_non_empty_list(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path)
    runner = FakeRunner(returncode=0, writes={tmp_path / "result.json": []})

    assert run_main(tmp_path, manifest, runner) == rst.EXIT_FAIL


def test_missing_prediction_fails(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path)
    runner = FakeRunner(
        returncode=0, writes={tmp_path / "result.json": [conversation_item(answer=None)]}
    )

    exit_code = run_main(tmp_path, manifest, runner)

    assert exit_code == rst.EXIT_FAIL
    assert any("no 'gpt' prediction" in f for f in read_summary(tmp_path)["failures"])


def test_empty_prediction_fails(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path)
    runner = FakeRunner(
        returncode=0, writes={tmp_path / "result.json": [conversation_item(answer="   ")]}
    )

    assert run_main(tmp_path, manifest, runner) == rst.EXIT_FAIL


def test_recorded_inference_error_fails_even_with_exit_zero(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path)
    item = conversation_item(answer="Inference error: RuntimeError: boom", real=None, fake=None)
    runner = FakeRunner(returncode=0, writes={tmp_path / "result.json": [item]})

    exit_code = run_main(tmp_path, manifest, runner)

    assert exit_code == rst.EXIT_FAIL
    assert any("inference error" in f.lower() for f in read_summary(tmp_path)["failures"])


def test_missing_scores_fail(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path)
    runner = FakeRunner(
        returncode=0,
        writes={tmp_path / "result.json": [conversation_item(real=None, fake=None)]},
    )

    exit_code = run_main(tmp_path, manifest, runner)

    assert exit_code == rst.EXIT_FAIL
    failures = read_summary(tmp_path)["failures"]
    assert any("real score" in f and "fake score" in f for f in failures)


def test_missing_only_fake_score_fails(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path)
    runner = FakeRunner(
        returncode=0, writes={tmp_path / "result.json": [conversation_item(fake=None)]}
    )

    exit_code = run_main(tmp_path, manifest, runner)

    assert exit_code == rst.EXIT_FAIL
    assert any("fake score" in f for f in read_summary(tmp_path)["failures"])


def test_non_numeric_scores_fail(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path)
    runner = FakeRunner(
        returncode=0, writes={tmp_path / "result.json": [conversation_item(real="high")]}
    )

    assert run_main(tmp_path, manifest, runner) == rst.EXIT_FAIL


def test_valid_output_passes_and_writes_summary(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path)
    runner = FakeRunner(returncode=0, writes={tmp_path / "result.json": [conversation_item()]})

    exit_code = run_main(tmp_path, manifest, runner)

    assert exit_code == rst.EXIT_PASS
    summary = read_summary(tmp_path)
    assert summary["status"] == "pass"
    assert summary["failures"] == []
    assert summary["output_path"] == str(tmp_path / "result.json")
    prediction = summary["predictions"][0]
    assert prediction["label"] == "real"
    assert prediction["real_score"] == pytest.approx(0.9210)
    assert prediction["fake_score"] == pytest.approx(0.0790)


def test_summary_records_runtime_and_command(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path)
    runner = FakeRunner(returncode=0, writes={tmp_path / "result.json": [conversation_item()]})

    run_main(tmp_path, manifest, runner)

    summary = read_summary(tmp_path)
    assert isinstance(summary["process"]["duration_s"], float)
    assert summary["process"]["duration_s"] >= 0
    assert "eval.infer.runner" in summary["command"]
    assert summary["experts"] == "none"
    assert summary["quantisation"] == "fp16"


def test_quantisation_recorded_when_4bit_requested(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path)
    runner = FakeRunner(returncode=0, writes={tmp_path / "result.json": [conversation_item()]})

    run_main(tmp_path, manifest, runner, extra=["--load-4bit"])

    assert read_summary(tmp_path)["quantisation"] == "4-bit"


def test_stale_output_is_removed_before_running(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path)
    stale = tmp_path / "result.json"
    stale.write_text(json.dumps([conversation_item(answer="stale answer")]), encoding="utf-8")
    runner = FakeRunner(returncode=0)  # writes nothing this time

    exit_code = run_main(tmp_path, manifest, runner)

    assert exit_code == rst.EXIT_FAIL
    assert any("no output JSON" in f for f in read_summary(tmp_path)["failures"])


def test_output_located_via_stdout_marker(tmp_path: Path) -> None:
    elsewhere = tmp_path / "runs" / "out.json"
    elsewhere.parent.mkdir(parents=True, exist_ok=True)
    elsewhere.write_text(json.dumps([conversation_item()]), encoding="utf-8")

    located = rst.locate_output(
        tmp_path / "missing.json", f"Saved conversation-style results: {elsewhere}\n"
    )

    assert located == elsewhere


def test_label_inference_from_answer() -> None:
    assert rst._label_from_answer("This image is fake.") == "fake"
    assert rst._label_from_answer("real") == "real"
    assert rst._label_from_answer("It could be real or fake") is None


# --------------------------------------------------------------------------
# quantisation switch in utils.lora_inference (regression cover)
# --------------------------------------------------------------------------


def test_quantisation_from_env_defaults_to_fp16(monkeypatch) -> None:
    lora_inference = pytest.importorskip("utils.lora_inference")
    monkeypatch.delenv("X2DFD_LOAD_4BIT", raising=False)
    monkeypatch.delenv("X2DFD_LOAD_8BIT", raising=False)

    assert lora_inference.quantisation_from_env() == (False, False)


def test_quantisation_from_env_reads_4bit(monkeypatch) -> None:
    lora_inference = pytest.importorskip("utils.lora_inference")
    monkeypatch.setenv("X2DFD_LOAD_4BIT", "1")
    monkeypatch.delenv("X2DFD_LOAD_8BIT", raising=False)

    assert lora_inference.quantisation_from_env() == (False, True)


def test_quantisation_from_env_rejects_both(monkeypatch) -> None:
    lora_inference = pytest.importorskip("utils.lora_inference")
    monkeypatch.setenv("X2DFD_LOAD_4BIT", "1")
    monkeypatch.setenv("X2DFD_LOAD_8BIT", "1")

    with pytest.raises(ValueError):
        lora_inference.quantisation_from_env()


def test_no_quantisation_kwargs_when_disabled() -> None:
    lora_inference = pytest.importorskip("utils.lora_inference")

    assert lora_inference.quantisation_kwargs(False, False) == {}


def test_4bit_keeps_projector_and_lm_head_in_fp16() -> None:
    """Regression: quantising mm_projector breaks non_lora_trainables loading.

    LLaVA's own load_4bit flag converts every Linear, so the fp16 projector in
    non_lora_trainables.bin no longer fits the packed uint8 parameter and the
    builder raises a size mismatch ([4096, 1024] vs [2097152, 1]).
    """
    lora_inference = pytest.importorskip("utils.lora_inference")
    pytest.importorskip("transformers")

    kwargs = lora_inference.quantisation_kwargs(False, True)
    config = kwargs["quantization_config"]

    assert config.load_in_4bit is True
    assert config.bnb_4bit_quant_type == "nf4"
    assert "mm_projector" in config.llm_int8_skip_modules
    assert "lm_head" in config.llm_int8_skip_modules


def test_8bit_also_keeps_projector_in_fp16() -> None:
    lora_inference = pytest.importorskip("utils.lora_inference")
    pytest.importorskip("transformers")

    config = lora_inference.quantisation_kwargs(True, False)["quantization_config"]

    assert config.load_in_8bit is True
    assert "mm_projector" in config.llm_int8_skip_modules
