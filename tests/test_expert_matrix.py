"""Unit tests for tools/run_expert_matrix.py.

Every inference subprocess is mocked: no model, no GPU, no network. The fake
runner writes whatever output JSON a scenario needs, keyed by the ``--output``
path in the command it receives, so the tests exercise the real command
construction rather than a parallel copy of it.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence

import pytest

from tools import run_expert_matrix as mx

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def make_manifest(tmp_path: Path, name: str = "demo_one.json") -> Path:
    images = tmp_path / "images"
    images.mkdir(parents=True, exist_ok=True)
    image = images / "face.jpg"
    image.write_bytes(b"\xff\xd8\xff\xe0not-a-real-jpeg")
    path = tmp_path / name
    path.write_text(
        json.dumps({"Description": str(images), "images": [{"image_path": image.name}]}),
        encoding="utf-8",
    )
    return path


def prompt_for(experts: Sequence[str], scores: Optional[Dict[str, Optional[float]]] = None) -> str:
    """Reproduce eval/infer/runner.py::_format_multi_scores for a prompt."""
    base = "<image>\nIs this image real or fake?"
    if not experts:
        return base
    scores = scores or {name: 0.5 for name in experts}
    parts = [
        "the %s score is %s" % (name, "N/A" if scores.get(name) is None else f"{scores[name]:.4f}")
        for name in experts
    ]
    if len(parts) == 1:
        return base + " And " + parts[0] + "."
    return base + " And " + ", ".join(parts[:-1]) + ", and " + parts[-1] + "."


def result_payload(
    *,
    experts: Sequence[str] = (),
    scores: Optional[Dict[str, Optional[float]]] = None,
    answer: Optional[str] = "real",
    real: Optional[str] = "0.9210",
    fake: Optional[str] = "0.0790",
) -> List[Dict[str, Any]]:
    conversations: List[Dict[str, str]] = [{"from": "human", "value": prompt_for(experts, scores)}]
    if answer is not None:
        conversations.append({"from": "gpt", "value": answer})
    if real is not None:
        conversations.append({"from": "real score", "value": real})
    if fake is not None:
        conversations.append({"from": "fake score", "value": fake})
    return [{"id": "1", "image": "C:/images/face.jpg", "conversations": conversations}]


DEFAULT_EXPERT_SCORE = 0.5


class MatrixRunner:
    """subprocess.run stand-in scripted per expert configuration.

    ``scenarios`` maps a run name (``none``, ``blending_diffusion``, ...) to a
    dict with any of: ``returncode``, ``stdout``, ``stderr``, ``payload``,
    ``raises``. Anything not specified falls back to the constructor kwargs.

    By default it writes a healthy result whose prompt carries exactly the
    experts that were requested on the command line, so a test only has to
    describe the thing it is actually testing. Pass ``payload=None`` for a run
    that should write no file at all.
    """

    def __init__(self, scenarios: Optional[Dict[str, Dict[str, Any]]] = None, **default: Any) -> None:
        self.scenarios = scenarios or {}
        self.default = default
        self.commands: List[List[str]] = []

    @staticmethod
    def _flag(command: Sequence[str], flag: str) -> Optional[str]:
        command = list(command)
        if flag not in command:
            return None
        index = command.index(flag)
        return command[index + 1] if index + 1 < len(command) else None

    def run_name(self, command: Sequence[str]) -> str:
        output = self._flag(command, "--output") or ""
        return Path(output).stem.replace("demo_", "")

    def requested_experts(self, command: Sequence[str]) -> List[str]:
        arg = self._flag(command, "--experts") or "none"
        return [] if arg == "none" else arg.split(",")

    def __call__(self, command: Sequence[str], **kwargs: Any) -> SimpleNamespace:
        self.commands.append(list(command))
        spec = {**self.default, **self.scenarios.get(self.run_name(command), {})}
        if spec.get("raises") is not None:
            raise spec["raises"]
        experts = self.requested_experts(command)
        payload = spec.get(
            "payload",
            result_payload(
                experts=experts, scores={name: DEFAULT_EXPERT_SCORE for name in experts}
            ),
        )
        if payload is not None:
            output = Path(self._flag(command, "--output") or "")
            output.parent.mkdir(parents=True, exist_ok=True)
            text = payload if isinstance(payload, str) else json.dumps(payload)
            output.write_text(text, encoding="utf-8")
        return SimpleNamespace(
            returncode=spec.get("returncode", 0),
            stdout=spec.get("stdout", ""),
            stderr=spec.get("stderr", ""),
        )


def base_argv(tmp_path: Path, manifest: Path, *configs: str) -> List[str]:
    argv = [
        "--manifest",
        str(manifest),
        "--project-root",
        str(tmp_path),
        "--output-dir",
        str(tmp_path / "out"),
        "--summary",
        str(tmp_path / "summary.json"),
        "--no-vram-sampling",
    ]
    if configs:
        argv += ["--configs", *configs]
    return argv


def read_summary(tmp_path: Path) -> Dict[str, Any]:
    return json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))


def runs_by_name(summary: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {run["run_name"]: run for run in summary["runs"]}


# --------------------------------------------------------------------------
# configuration normalisation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spec, run_name, experts",
    [
        ("none", "none", ()),
        ("NONE", "none", ()),
        ("  none  ", "none", ()),
        ("blending", "blending", ("blending",)),
        ("Blending", "blending", ("blending",)),
        ("diffusion", "diffusion", ("diffusion",)),
        ("diffusion_detector", "diffusion", ("diffusion",)),
        ("aligner", "diffusion", ("diffusion",)),
        ("blending,diffusion", "blending_diffusion", ("blending", "diffusion")),
        ("diffusion,blending", "blending_diffusion", ("blending", "diffusion")),
        (" DIFFUSION , blend ", "blending_diffusion", ("blending", "diffusion")),
        ("blending,blending", "blending", ("blending",)),
        ("blending,,diffusion", "blending_diffusion", ("blending", "diffusion")),
    ],
)
def test_normalise_config_canonicalises(spec: str, run_name: str, experts: tuple) -> None:
    config = mx.normalise_config(spec)
    assert config.run_name == run_name
    assert config.experts == experts


def test_normalise_config_builds_runner_argument() -> None:
    assert mx.normalise_config("none").experts_arg == "none"
    assert mx.normalise_config("diffusion,blending").experts_arg == "blending,diffusion"


@pytest.mark.parametrize("spec", ["", "   ", ",", "wavelet", "none,blending"])
def test_normalise_config_rejects_bad_input(spec: str) -> None:
    with pytest.raises(mx.ConfigError):
        mx.normalise_config(spec)


def test_normalise_configs_rejects_duplicates_that_would_collide() -> None:
    with pytest.raises(mx.ConfigError, match="blending_diffusion"):
        mx.normalise_configs(["blending,diffusion", "diffusion,blending"])


def test_normalise_configs_preserves_requested_order() -> None:
    configs = mx.normalise_configs(["diffusion", "none", "blending"])
    assert [c.run_name for c in configs] == ["diffusion", "none", "blending"]


# --------------------------------------------------------------------------
# command construction and output paths
# --------------------------------------------------------------------------


def test_output_paths_are_distinct_and_poc_shaped(tmp_path: Path) -> None:
    configs = mx.normalise_configs(list(mx.DEFAULT_CONFIGS))
    paths = [c.output_path(tmp_path) for c in configs]
    assert [p.name for p in paths] == [
        "demo_none.json",
        "demo_blending.json",
        "demo_diffusion.json",
        "demo_blending_diffusion.json",
    ]
    assert len({str(p) for p in paths}) == len(paths)


def test_commands_carry_the_normalised_experts_and_own_output(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path)
    fake = MatrixRunner()

    mx.main(base_argv(tmp_path, manifest, "none", "diffusion,blending"), process_runner=fake)

    assert len(fake.commands) == 2
    first, second = fake.commands
    assert first[1:3] == ["-m", "eval.infer.runner"]
    assert first[first.index("--experts") + 1] == "none"
    assert second[second.index("--experts") + 1] == "blending,diffusion"
    assert first[first.index("--json") + 1] == str(manifest)
    outputs = {cmd[cmd.index("--output") + 1] for cmd in fake.commands}
    assert len(outputs) == 2


def test_dry_run_prints_commands_without_running(tmp_path: Path, capsys: Any) -> None:
    manifest = make_manifest(tmp_path)
    fake = MatrixRunner()

    code = mx.main(base_argv(tmp_path, manifest) + ["--dry-run"], process_runner=fake)

    assert code == mx.EXIT_PASS
    assert fake.commands == []
    out = capsys.readouterr().out
    assert "[blending_diffusion]" in out
    assert not (tmp_path / "summary.json").exists()


def test_each_configuration_runs_in_its_own_subprocess(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path)
    fake = MatrixRunner()

    mx.main(base_argv(tmp_path, manifest), process_runner=fake)

    assert len(fake.commands) == len(mx.DEFAULT_CONFIGS)


# --------------------------------------------------------------------------
# usage errors
# --------------------------------------------------------------------------


def test_missing_manifest_is_a_usage_error(tmp_path: Path) -> None:
    fake = MatrixRunner()
    code = mx.main(base_argv(tmp_path, tmp_path / "absent.json"), process_runner=fake)
    assert code == mx.EXIT_USAGE
    assert fake.commands == []


def test_unknown_expert_is_a_usage_error(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path)
    fake = MatrixRunner()
    code = mx.main(base_argv(tmp_path, manifest, "wavelet"), process_runner=fake)
    assert code == mx.EXIT_USAGE
    assert fake.commands == []


# --------------------------------------------------------------------------
# valid outputs
# --------------------------------------------------------------------------


def test_all_success_records_scores_and_experts(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path)
    fake = MatrixRunner(
        {
            "none": {"payload": result_payload()},
            "blending": {"payload": result_payload(experts=["blending"], scores={"blending": 0.12})},
            "diffusion": {"payload": result_payload(experts=["diffusion"], scores={"diffusion": 0.34})},
            "blending_diffusion": {
                "payload": result_payload(
                    experts=["blending", "diffusion"], scores={"blending": 0.12, "diffusion": 0.34}
                )
            },
        }
    )

    code = mx.main(base_argv(tmp_path, manifest), process_runner=fake)

    assert code == mx.EXIT_PASS
    summary = read_summary(tmp_path)
    assert summary["status"] == "pass"
    assert summary["totals"] == {
        **summary["totals"],
        "configurations": 4,
        "passed": 4,
        "failed": 0,
        "distinct_labels": ["real"],
        "labels_agree": True,
    }
    runs = runs_by_name(summary)
    assert runs["none"]["expert_scores"] == {}
    assert runs["blending"]["expert_scores"] == {"blending": 0.12}
    assert runs["blending_diffusion"]["expert_scores"] == {"blending": 0.12, "diffusion": 0.34}
    for run in runs.values():
        assert run["validation"] == "valid"
        assert run["label"] == "real"
        assert run["real_score"] == pytest.approx(0.9210)
        assert run["fake_score"] == pytest.approx(0.0790)


def test_summary_records_runtime_and_writes_runtimes_sidecar(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path)
    fake = MatrixRunner()

    mx.main(base_argv(tmp_path, manifest, "none", "blending"), process_runner=fake)

    summary = read_summary(tmp_path)
    for run in summary["runs"]:
        assert run["runtime_s"] is not None and run["runtime_s"] >= 0
    assert summary["totals"]["total_runtime_s"] is not None
    sidecar = json.loads((tmp_path / "out" / "demo_one" / "runtimes.json").read_text(encoding="utf-8"))
    assert sorted(sidecar) == ["blending", "none"]


def test_disagreement_is_visible_in_totals(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path)
    fake = MatrixRunner(
        {
            "none": {"payload": result_payload(answer="real", real="0.80", fake="0.20")},
            "blending": {
                "payload": result_payload(
                    experts=["blending"], scores={"blending": 0.9}, answer="fake", real="0.10", fake="0.90"
                )
            },
        }
    )

    mx.main(base_argv(tmp_path, manifest, "none", "blending"), process_runner=fake)

    totals = read_summary(tmp_path)["totals"]
    assert totals["distinct_labels"] == ["fake", "real"]
    assert totals["labels_agree"] is False


# --------------------------------------------------------------------------
# failure modes
# --------------------------------------------------------------------------


def test_missing_output_fails_that_run(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path)
    fake = MatrixRunner({"none": {"payload": None}})

    code = mx.main(base_argv(tmp_path, manifest, "none", "blending"), process_runner=fake)

    assert code == mx.EXIT_FAIL
    runs = runs_by_name(read_summary(tmp_path))
    assert runs["none"]["validation"] == "missing output"
    assert any("no output JSON" in f for f in runs["none"]["failures"])
    assert runs["blending"]["status"] == "pass"


def test_malformed_json_output_fails_validation(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path)
    fake = MatrixRunner({"none": {"payload": "{not json"}})

    code = mx.main(base_argv(tmp_path, manifest, "none"), process_runner=fake)

    assert code == mx.EXIT_FAIL
    run = runs_by_name(read_summary(tmp_path))["none"]
    assert run["validation"] == "invalid"
    assert any("not valid JSON" in f for f in run["failures"])


def test_missing_scores_fail_validation(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path)
    fake = MatrixRunner({"none": {"payload": result_payload(real=None, fake=None)}})

    code = mx.main(base_argv(tmp_path, manifest, "none"), process_runner=fake)

    assert code == mx.EXIT_FAIL
    run = runs_by_name(read_summary(tmp_path))["none"]
    assert run["validation"] == "invalid"
    assert any("real score" in f for f in run["failures"])


def test_hidden_inference_error_is_not_reported_as_success(tmp_path: Path) -> None:
    """The runner exits 0 and writes a file even when generation blew up."""
    manifest = make_manifest(tmp_path)
    fake = MatrixRunner(
        {"none": {"returncode": 0, "payload": result_payload(answer="Inference error: CUDA error")}}
    )

    code = mx.main(base_argv(tmp_path, manifest, "none"), process_runner=fake)

    assert code == mx.EXIT_FAIL
    run = runs_by_name(read_summary(tmp_path))["none"]
    assert run["exit_code"] == 0
    assert run["validation"] == "invalid"
    assert any("inference error" in f.lower() for f in run["failures"])


def test_expert_scored_na_fails_even_with_a_valid_prediction(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path)
    fake = MatrixRunner(
        {"blending": {"payload": result_payload(experts=["blending"], scores={"blending": None})}}
    )

    code = mx.main(base_argv(tmp_path, manifest, "blending"), process_runner=fake)

    assert code == mx.EXIT_FAIL
    run = runs_by_name(read_summary(tmp_path))["blending"]
    assert run["validation"] == "valid"
    assert run["label"] == "real"
    assert run["expert_scores"] == {"blending": None}
    assert any("no score" in f for f in run["failures"])


def test_nonzero_exit_is_recorded_with_a_traceback_hint(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path)
    stderr = (
        "Traceback (most recent call last):\n"
        '  File "eval/infer/runner.py", line 1, in <module>\n'
        "RuntimeError: detector weights not found\n"
    )
    fake = MatrixRunner({"blending": {"returncode": 1, "stderr": stderr}})

    code = mx.main(base_argv(tmp_path, manifest, "none", "blending"), process_runner=fake)

    assert code == mx.EXIT_FAIL
    runs = runs_by_name(read_summary(tmp_path))
    assert runs["blending"]["exit_code"] == 1
    assert any("exited with code 1" in f for f in runs["blending"]["failures"])
    assert "RuntimeError" in runs["blending"]["stderr_tail"]


def test_cuda_oom_is_detected_and_flagged(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path)
    fake = MatrixRunner(
        {
            "blending_diffusion": {
                "returncode": 1,
                "stderr": "torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB",
            }
        }
    )

    code = mx.main(base_argv(tmp_path, manifest, "none", "blending,diffusion"), process_runner=fake)

    assert code == mx.EXIT_FAIL
    runs = runs_by_name(read_summary(tmp_path))
    assert runs["blending_diffusion"]["cuda_oom"] is True
    assert any("out of memory" in f.lower() for f in runs["blending_diffusion"]["failures"])
    assert runs["none"]["cuda_oom"] is False


def test_timeout_fails_only_the_slow_configuration(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path)
    fake = MatrixRunner({"diffusion": {"raises": subprocess.TimeoutExpired(cmd="runner", timeout=5)}})

    code = mx.main(
        base_argv(tmp_path, manifest, "none", "diffusion") + ["--timeout", "5"], process_runner=fake
    )

    assert code == mx.EXIT_FAIL
    runs = runs_by_name(read_summary(tmp_path))
    assert runs["diffusion"]["timed_out"] is True
    assert runs["none"]["status"] == "pass"


def test_partial_failure_still_runs_every_configuration(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path)
    fake = MatrixRunner({"blending": {"returncode": 1, "stderr": "boom"}})

    code = mx.main(base_argv(tmp_path, manifest), process_runner=fake)

    assert code == mx.EXIT_FAIL
    assert len(fake.commands) == 4
    summary = read_summary(tmp_path)
    assert summary["totals"]["passed"] == 3
    assert summary["totals"]["failed"] == 1
    assert summary["status"] == "fail"


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------


def test_parse_expert_scores_reads_the_prompt_tail() -> None:
    prompt = prompt_for(["blending", "diffusion"], {"blending": 0.8120, "diffusion": None})
    assert mx.parse_expert_scores(prompt) == {"blending": 0.8120, "diffusion": None}


def test_parse_expert_scores_on_a_bare_prompt_is_empty() -> None:
    assert mx.parse_expert_scores(prompt_for([])) == {}
    assert mx.parse_expert_scores(None) == {}


def test_extract_prompt_survives_unreadable_files(tmp_path: Path) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text("{nope", encoding="utf-8")
    assert mx.extract_prompt(broken) is None
    assert mx.extract_prompt(tmp_path / "absent.json") is None


def test_collect_warnings_deduplicates_and_caps() -> None:
    text = "\n".join(
        ["UserWarning: same thing"] * 5
        + ["quiet line"]
        + [f"FutureWarning: unique {i}" for i in range(20)]
    )
    warnings = mx.collect_warnings(text, limit=4)
    assert warnings[0] == "UserWarning: same thing"
    assert len(warnings) == 4
    assert len(set(warnings)) == 4
    assert "quiet line" not in warnings


def test_logs_are_written_per_run(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path)
    fake = MatrixRunner(stdout="loading model", stderr="a warning")

    mx.main(base_argv(tmp_path, manifest, "none"), process_runner=fake)

    log = tmp_path / "out" / "demo_one" / "logs" / "none.log"
    assert log.is_file()
    body = log.read_text(encoding="utf-8")
    assert "loading model" in body and "a warning" in body
