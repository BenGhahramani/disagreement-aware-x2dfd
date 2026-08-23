"""Unit tests for labelled evaluation manifest + resumable batch runner.

Inference subprocesses and Haar cropping are mocked. No GPU, no weights.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pytest

from eval.experiment_configs import DEFAULT_CONFIGS, RUN_ORDER
from eval.labelled_manifest import (
    LabelledManifestError,
    load_labelled_manifest,
)
from tests.test_live_analysis import RecordingRunner, fake_cropper, write_jpeg
from tools import run_labelled_evaluation as ble
from tools.make_face_crop import CropPlan
from tools.run_expert_matrix import ExpertConfig

pytestmark = pytest.mark.unit


def _write_manifest(tmp_path: Path, images: List[Dict[str, Any]]) -> Path:
    path = tmp_path / "pilot.json"
    path.write_text(json.dumps({"images": images}, indent=2) + "\n", encoding="utf-8")
    return path


def _entry_dict(
    tmp_path: Path,
    *,
    image_id: str = "img_001",
    ground_truth: str = "fake",
    already_cropped: bool = False,
    **extra: Any,
) -> Dict[str, Any]:
    image = write_jpeg(tmp_path / "images" / f"{image_id}.jpg")
    payload: Dict[str, Any] = {
        "id": image_id,
        "path": str(image),
        "ground_truth": ground_truth,
        "already_cropped": already_cropped,
    }
    payload.update(extra)
    return payload


# --------------------------------------------------------------------------
# manifest validation
# --------------------------------------------------------------------------


def test_load_valid_manifest(tmp_path: Path) -> None:
    path = _write_manifest(
        tmp_path,
        [
            _entry_dict(
                tmp_path,
                image_id="a",
                ground_truth="real",
                dataset="pilot",
                manipulation=None,
                source="nasa",
                notes="known authentic",
            ),
            _entry_dict(
                tmp_path,
                image_id="b",
                ground_truth="fake",
                dataset="ffpp",
                manipulation="Deepfakes",
            ),
        ],
    )
    images = load_labelled_manifest(path)
    assert [img.id for img in images] == ["a", "b"]
    assert images[0].ground_truth == "real"
    assert images[0].dataset == "pilot"
    assert images[0].source == "nasa"
    assert images[1].manipulation == "Deepfakes"


def test_duplicate_ids_rejected(tmp_path: Path) -> None:
    path = _write_manifest(
        tmp_path,
        [
            _entry_dict(tmp_path, image_id="dup"),
            _entry_dict(tmp_path, image_id="dup", ground_truth="real"),
        ],
    )
    with pytest.raises(LabelledManifestError, match="duplicate image id"):
        load_labelled_manifest(path)


def test_invalid_ground_truth_rejected(tmp_path: Path) -> None:
    path = _write_manifest(
        tmp_path,
        [_entry_dict(tmp_path, ground_truth="maybe")],
    )
    with pytest.raises(LabelledManifestError, match="ground_truth"):
        load_labelled_manifest(path)


def test_missing_image_file_rejected(tmp_path: Path) -> None:
    path = _write_manifest(
        tmp_path,
        [
            {
                "id": "gone",
                "path": str(tmp_path / "missing.jpg"),
                "ground_truth": "real",
            }
        ],
    )
    with pytest.raises(LabelledManifestError, match="image not found"):
        load_labelled_manifest(path)


# --------------------------------------------------------------------------
# batch runner helpers
# --------------------------------------------------------------------------


def _write_infer_config(
    tmp_path: Path,
    *,
    blending_bytes: bytes = b"blend-v1",
    force: bool = False,
) -> Path:
    """Write a stub infer config + tiny checkpoint files under ``tmp_path``."""

    blend = tmp_path / "weights" / "blending_models" / "best_gf.pth"
    blend.parent.mkdir(parents=True, exist_ok=True)
    if force or not blend.is_file():
        blend.write_bytes(blending_bytes)

    diff_dir = tmp_path / "weights" / "ours-sync"
    diff_dir.mkdir(parents=True, exist_ok=True)
    if force or not (diff_dir / "config.yaml").is_file():
        (diff_dir / "config.yaml").write_text(
            "arch: res50\nweights_file: checkpoint.pth\n",
            encoding="utf-8",
        )
    if force or not (diff_dir / "checkpoint.pth").is_file():
        (diff_dir / "checkpoint.pth").write_bytes(b"diff-v1")

    base = tmp_path / "weights" / "base" / "llava-v1.5-7b"
    base.mkdir(parents=True, exist_ok=True)
    if force or not (base / "config.json").is_file():
        (base / "config.json").write_text('{"model_type":"llava"}\n', encoding="utf-8")
    if force or not (base / "model.safetensors.index.json").is_file():
        (base / "model.safetensors.index.json").write_text(
            '{"metadata":{},"weight_map":{}}\n',
            encoding="utf-8",
        )

    adapter = tmp_path / "weights" / "checkpoints" / "ckpt" / "lora"
    adapter.mkdir(parents=True, exist_ok=True)
    if force or not (adapter / "adapter_config.json").is_file():
        (adapter / "adapter_config.json").write_text('{"r":8}\n', encoding="utf-8")
    if force or not (adapter / "adapter_model.safetensors").is_file():
        (adapter / "adapter_model.safetensors").write_bytes(b"lora-v1")

    config_path = tmp_path / "infer_config.windows.yaml"
    if force or not config_path.is_file():
        config_path.write_text(
            "\n".join(
                [
                    "weak_supplies:",
                    "  - provider: blending",
                    "    weights_path: weights/blending_models/best_gf.pth",
                    "  - provider: diffusion_detector",
                    "    weights_dir: weights/",
                    "    model: ours-sync",
                    "model:",
                    '  base: "weights/base/llava-v1.5-7b"',
                    '  adapter: "weights/checkpoints/ckpt/lora"',
                    "",
                ]
            ),
            encoding="utf-8",
        )
    return config_path


def _run_batch(
    tmp_path: Path,
    entries: List[Dict[str, Any]],
    *,
    runner: Optional[RecordingRunner] = None,
    resume: bool = True,
    dry_run: bool = False,
    max_images: Optional[int] = None,
    load_4bit: bool = True,
    reset_infer_config: bool = False,
) -> ble.BatchResult:
    manifest = _write_manifest(tmp_path, entries)
    fake = runner or RecordingRunner()
    config_path = _write_infer_config(tmp_path, force=reset_infer_config)

    def crop_and_remember(src: Path, dest: Path, **kwargs: Any) -> CropPlan:
        plan = fake_cropper(src, dest, **kwargs)
        fake.crop_image = dest
        return plan

    return ble.run_labelled_evaluation(
        manifest_path=manifest,
        output_dir=tmp_path / "out",
        project_root=tmp_path,
        config_path=config_path,
        timeout_s=30.0,
        load_4bit=load_4bit,
        sample_vram=False,
        resume=resume,
        max_images=max_images,
        dry_run=dry_run,
        process_runner=fake,
        crop_fn=crop_and_remember,
    )


def test_canonical_four_configs_used(tmp_path: Path) -> None:
    fake = RecordingRunner()
    batch = _run_batch(tmp_path, [_entry_dict(tmp_path)], runner=fake)
    assert batch.ok
    assert len(fake.commands) == 4
    experts = [cmd[cmd.index("--experts") + 1] for cmd in fake.commands]
    assert experts == list(DEFAULT_CONFIGS)
    assert experts == ["none", "blending", "diffusion", "blending,diffusion"]
    assert [cfg.run_name for cfg in batch.results[0].configs] == list(RUN_ORDER)
    assert len(batch.results[0].configs) == 4


def test_no_fifth_configuration(tmp_path: Path) -> None:
    batch = _run_batch(tmp_path, [_entry_dict(tmp_path)])
    assert len(batch.results[0].configs) == 4
    assert set(c.run_name for c in batch.results[0].configs) == set(RUN_ORDER)


def test_completed_configs_skipped_on_resume(tmp_path: Path) -> None:
    entry = _entry_dict(tmp_path, image_id="resume_me")
    first_runner = RecordingRunner()
    batch1 = _run_batch(tmp_path, [entry], runner=first_runner, resume=True)
    assert batch1.ok
    assert len(first_runner.commands) == 4
    work = ble.image_work_dir(tmp_path / "out", "resume_me")
    for run_name in RUN_ORDER:
        assert (work / "matrix" / ble.CONFIG_PROVENANCE_DIRNAME / f"{run_name}.json").is_file()

    second_runner = RecordingRunner()
    batch2 = _run_batch(tmp_path, [entry], runner=second_runner, resume=True)
    assert batch2.ok
    assert second_runner.commands == []
    assert all(cfg.inference_skipped for cfg in batch2.results[0].configs)


def test_same_image_and_settings_resume_skips_inference(tmp_path: Path) -> None:
    entry = _entry_dict(tmp_path, image_id="stable")
    first = RecordingRunner()
    _run_batch(tmp_path, [entry], runner=first)
    second = RecordingRunner()
    batch = _run_batch(tmp_path, [entry], runner=second, resume=True)
    assert batch.ok
    assert second.commands == []


def test_same_id_changed_image_reruns(tmp_path: Path) -> None:
    entry = _entry_dict(tmp_path, image_id="mutated")
    first = RecordingRunner()
    _run_batch(tmp_path, [entry], runner=first)
    assert len(first.commands) == 4

    image_path = Path(entry["path"])
    image_path.write_bytes(image_path.read_bytes() + b"changed-bytes")

    second = RecordingRunner()
    _run_batch(tmp_path, [entry], runner=second, resume=True)
    assert len(second.commands) == 4


def test_changed_load_4bit_reruns(tmp_path: Path) -> None:
    entry = _entry_dict(tmp_path, image_id="qbit")
    first = RecordingRunner()
    _run_batch(tmp_path, [entry], runner=first, load_4bit=True)
    assert len(first.commands) == 4

    second = RecordingRunner()
    _run_batch(tmp_path, [entry], runner=second, load_4bit=False, resume=True)
    assert len(second.commands) == 4


def test_changed_inference_config_path_reruns(tmp_path: Path) -> None:
    entry = _entry_dict(tmp_path, image_id="cfg_change")
    first = RecordingRunner()
    _run_batch(tmp_path, [entry], runner=first)
    assert len(first.commands) == 4

    alt_config = tmp_path / "infer_config.alt.yaml"
    alt_config.write_text("variant: true\n", encoding="utf-8")
    (tmp_path / "infer_config.windows.yaml").unlink()

    manifest = _write_manifest(tmp_path, [entry])
    second = RecordingRunner()

    def crop_and_remember(src: Path, dest: Path, **kwargs: Any) -> CropPlan:
        plan = fake_cropper(src, dest, **kwargs)
        second.crop_image = dest
        return plan

    ble.run_labelled_evaluation(
        manifest_path=manifest,
        output_dir=tmp_path / "out",
        project_root=tmp_path,
        config_path=alt_config,
        timeout_s=30.0,
        load_4bit=True,
        sample_vram=False,
        resume=True,
        process_runner=second,
        crop_fn=crop_and_remember,
    )
    assert len(second.commands) == 4


def test_interpretation_threshold_change_reuses_raw_inference(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    entry = _entry_dict(tmp_path, image_id="interp")
    first = RecordingRunner()
    batch1 = _run_batch(tmp_path, [entry], runner=first)
    assert batch1.ok
    assert len(first.commands) == 4

    original = ble.interpretation_thresholds

    def shifted_thresholds() -> Dict[str, float]:
        values = original()
        values["prototype_stable_uncertain"] = 0.99
        return values

    monkeypatch.setattr(ble, "interpretation_thresholds", shifted_thresholds)
    monkeypatch.setattr("eval.reproducibility.interpretation_thresholds", shifted_thresholds)

    second = RecordingRunner()
    batch2 = _run_batch(tmp_path, [entry], runner=second, resume=True)
    assert batch2.ok
    assert second.commands == []
    assert all(cfg.inference_skipped for cfg in batch2.results[0].configs)
    assert all(cfg.interpretation_rebuilt for cfg in batch2.results[0].configs)
    assert any("interpretation rebuilt" in w for w in batch2.results[0].warnings)


def test_valid_output_without_provenance_is_not_reused(tmp_path: Path) -> None:
    entry = _entry_dict(tmp_path, image_id="legacy")
    first = RecordingRunner()
    _run_batch(tmp_path, [entry], runner=first)
    work = ble.image_work_dir(tmp_path / "out", "legacy")
    prov_dir = work / "matrix" / ble.CONFIG_PROVENANCE_DIRNAME
    for path in prov_dir.glob("*.json"):
        path.unlink()

    second = RecordingRunner()
    _run_batch(tmp_path, [entry], runner=second, resume=True)
    assert len(second.commands) == 4


def test_same_config_path_changed_contents_reruns(tmp_path: Path) -> None:
    entry = _entry_dict(tmp_path, image_id="cfg_bytes")
    first = RecordingRunner()
    _run_batch(tmp_path, [entry], runner=first)
    assert len(first.commands) == 4

    config_path = tmp_path / "infer_config.windows.yaml"
    config_path.write_text(config_path.read_text(encoding="utf-8") + "\n# touched\n", encoding="utf-8")

    second = RecordingRunner()
    _run_batch(tmp_path, [entry], runner=second, resume=True)
    assert len(second.commands) == 4


def test_changed_checkpoint_fingerprint_reruns(tmp_path: Path) -> None:
    entry = _entry_dict(tmp_path, image_id="ckpt")
    first = RecordingRunner()
    _run_batch(tmp_path, [entry], runner=first)
    assert len(first.commands) == 4

    blend = tmp_path / "weights" / "blending_models" / "best_gf.pth"
    blend.write_bytes(b"blend-v2-changed")

    second = RecordingRunner()
    _run_batch(tmp_path, [entry], runner=second, resume=True)
    assert len(second.commands) == 4


def test_dirty_tree_state_is_recorded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ble, "get_git_commit", lambda *_a, **_k: "abc123deadbeef")
    monkeypatch.setattr(ble, "get_git_dirty", lambda *_a, **_k: True)
    monkeypatch.setattr(
        ble,
        "get_git_provenance",
        lambda *_a, **_k: {"git_commit": "abc123deadbeef", "git_dirty": True},
    )

    entry = _entry_dict(tmp_path, image_id="dirty")
    batch = _run_batch(tmp_path, [entry])
    assert batch.results[0].git_commit == "abc123deadbeef"
    assert batch.results[0].git_dirty is True
    meta = json.loads((tmp_path / "out" / ble.BATCH_META_FILENAME).read_text(encoding="utf-8"))
    assert meta["git_commit"] == "abc123deadbeef"
    assert meta["git_dirty"] is True
    work = ble.image_work_dir(tmp_path / "out", "dirty")
    prov = json.loads((work / ble.IMAGE_PROVENANCE_FILENAME).read_text(encoding="utf-8"))
    assert prov["git_dirty"] is True


def test_interpretation_only_changes_still_reuse_raw_inference(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    entry = _entry_dict(tmp_path, image_id="interp_bands")
    first = RecordingRunner()
    batch1 = _run_batch(tmp_path, [entry], runner=first)
    assert batch1.ok
    thresholds = batch1.results[0].interpretation_thresholds
    assert thresholds["prototype_stable_uncertain"] == 0.70
    assert thresholds["evidence_expert_lo"] == 0.30
    assert thresholds["evidence_expert_hi"] == 0.70

    original = ble.interpretation_thresholds

    def shifted_thresholds() -> Dict[str, float]:
        values = original()
        values["evidence_expert_lo"] = 0.25
        values["evidence_expert_hi"] = 0.75
        values["prototype_stable_uncertain"] = 0.80
        return values

    monkeypatch.setattr(ble, "interpretation_thresholds", shifted_thresholds)
    monkeypatch.setattr("eval.reproducibility.interpretation_thresholds", shifted_thresholds)

    second = RecordingRunner()
    batch2 = _run_batch(tmp_path, [entry], runner=second, resume=True)
    assert batch2.ok
    assert second.commands == []
    assert all(cfg.inference_skipped for cfg in batch2.results[0].configs)
    assert all(cfg.interpretation_rebuilt for cfg in batch2.results[0].configs)


def test_source_and_crop_hashes_written(tmp_path: Path) -> None:
    entry = _entry_dict(tmp_path, image_id="hashes")
    batch = _run_batch(tmp_path, [entry])
    image = batch.results[0]
    assert image.source_sha256 and len(image.source_sha256) == 64
    assert image.crop_sha256 and len(image.crop_sha256) == 64
    assert image.inference_config_sha256 and len(image.inference_config_sha256) == 64
    assert image.model_artefacts_fingerprint
    work = ble.image_work_dir(tmp_path / "out", "hashes")
    prov = json.loads((work / ble.IMAGE_PROVENANCE_FILENAME).read_text(encoding="utf-8"))
    assert prov["source_sha256"] == image.source_sha256
    assert prov["crop_sha256"] == image.crop_sha256
    assert prov["inference_config_sha256"] == image.inference_config_sha256
    meta = json.loads((tmp_path / "out" / ble.BATCH_META_FILENAME).read_text(encoding="utf-8"))
    assert meta["manifest_sha256"]
    assert len(meta["manifest_sha256"]) == 64
    assert meta["inference_config_sha256"]
    assert meta["model_artefacts"]["artefacts_fingerprint"]
    assert "git_dirty" in meta
    assert meta["started_at"]
    assert meta["finished_at"]
    assert meta["canonical_configs"] == list(DEFAULT_CONFIGS)
    assert meta["interpretation_thresholds"]["prototype_stable_uncertain"] == 0.70
    assert meta["interpretation_thresholds"]["evidence_expert_lo"] == 0.30
    assert meta["interpretation_thresholds"]["evidence_expert_hi"] == 0.70
    roles = {a["role"] for a in meta["model_artefacts"]["artefacts"]}
    assert roles == {
        "llava_base",
        "lora_adapter",
        "blending_checkpoint",
        "diffusion_checkpoint",
    }
    base = next(a for a in meta["model_artefacts"]["artefacts"] if a["role"] == "llava_base")
    assert base["fingerprint_kind"] == "base_model_metadata"
    assert "not fully hashed" in base["note"].lower() or "metadata" in base["note"].lower()


def test_partial_image_resumes_only_missing_configs(tmp_path: Path) -> None:
    entry = _entry_dict(tmp_path, image_id="partial")
    first = RecordingRunner()
    batch1 = _run_batch(tmp_path, [entry], runner=first)
    assert batch1.ok

    work = ble.image_work_dir(tmp_path / "out", "partial")
    matrix = work / "matrix"
    missing = matrix / "demo_diffusion.json"
    assert missing.is_file()
    missing.unlink()

    second = RecordingRunner()
    batch2 = _run_batch(tmp_path, [entry], runner=second, resume=True)
    assert batch2.ok
    assert len(second.commands) == 1
    assert second.commands[0][second.commands[0].index("--experts") + 1] == "diffusion"
    skipped = {cfg.run_name: cfg.skipped for cfg in batch2.results[0].configs}
    assert skipped == {
        "none": True,
        "blending": True,
        "diffusion": False,
        "blending_diffusion": True,
    }


def test_failed_config_recorded_without_aborting_batch(tmp_path: Path) -> None:
    entries = [
        _entry_dict(tmp_path, image_id="good", ground_truth="fake"),
        _entry_dict(tmp_path, image_id="bad", ground_truth="real"),
    ]
    fake = RecordingRunner(
        scenarios={
            "blending": {"payload": None, "returncode": 1, "stderr": "boom"},
        }
    )
    # Scenario keys are per-command; MatrixRunner uses run_name from --output.
    # Force failure only while processing the second image by wrapping.

    class SelectiveRunner(RecordingRunner):
        def __call__(self, command: Sequence[str], **kwargs: Any):
            output = Path(self._flag(command, "--output") or "")
            if "bad" in str(output) and self.run_name(command) == "blending":
                self.commands.append(list(command))
                return type("R", (), {"returncode": 1, "stdout": "", "stderr": "boom"})()
            return super().__call__(command, **kwargs)

    runner = SelectiveRunner()
    batch = _run_batch(tmp_path, entries, runner=runner)
    assert len(batch.results) == 2
    by_id = {r.image_id: r for r in batch.results}
    assert by_id["good"].ok is True
    assert by_id["bad"].ok is False
    assert any("blending" in err for err in by_id["bad"].errors)
    assert batch.ok is False
    assert batch.aggregate_json is not None and batch.aggregate_json.is_file()


def test_aggregate_output_written_with_metadata(tmp_path: Path) -> None:
    entry = _entry_dict(
        tmp_path,
        image_id="meta_img",
        ground_truth="fake",
        dataset="ffpp",
        manipulation="Deepfakes",
        source="FaceForensics++",
        notes="pilot",
    )
    batch = _run_batch(tmp_path, [entry])
    assert batch.aggregate_json is not None
    assert batch.aggregate_csv is not None
    payload = json.loads(batch.aggregate_json.read_text(encoding="utf-8"))
    assert payload["n_images"] == 1
    assert payload["canonical_configs"] == list(DEFAULT_CONFIGS)
    image = payload["images"][0]
    assert image["image_id"] == "meta_img"
    assert image["ground_truth"] == "fake"
    assert image["dataset"] == "ffpp"
    assert image["manipulation"] == "Deepfakes"
    assert image["source"] == "FaceForensics++"
    assert image["notes"] == "pilot"
    assert image["prototype_status"] in {"Stable", "Uncertain", "Contested", "Failed/insufficient"}
    assert image["evidence_agreement"] in {
        "agreement",
        "conflict",
        "insufficient evidence",
    }
    csv_text = batch.aggregate_csv.read_text(encoding="utf-8")
    assert "meta_img" in csv_text
    assert "Deepfakes" in csv_text


def test_already_cropped_bypasses_haar(tmp_path: Path) -> None:
    (tmp_path / "infer_config.windows.yaml").write_text("dummy: true\n", encoding="utf-8")
    fake_cropper.calls = []  # type: ignore[attr-defined]
    entry = _entry_dict(tmp_path, image_id="cropped", already_cropped=True)
    batch = _run_batch(tmp_path, [entry])
    assert batch.ok
    assert fake_cropper.calls == []  # type: ignore[attr-defined]
    work = ble.image_work_dir(tmp_path / "out", "cropped")
    crop = work / "face_crop.jpg"
    assert crop.is_file()
    plan = json.loads((work / "crop_plan.json").read_text(encoding="utf-8"))
    assert plan["already_cropped"] is True
    assert batch.results[0].already_cropped is True


def test_dry_run_does_not_invoke_inference(tmp_path: Path) -> None:
    (tmp_path / "infer_config.windows.yaml").write_text("dummy: true\n", encoding="utf-8")
    fake = RecordingRunner()
    batch = _run_batch(tmp_path, [_entry_dict(tmp_path)], runner=fake, dry_run=True)
    assert batch.ok
    assert fake.commands == []
    assert all(cfg.status == "dry-run" for cfg in batch.results[0].configs)


def test_max_images_limits_batch(tmp_path: Path) -> None:
    (tmp_path / "infer_config.windows.yaml").write_text("dummy: true\n", encoding="utf-8")
    entries = [
        _entry_dict(tmp_path, image_id="one"),
        _entry_dict(tmp_path, image_id="two"),
    ]
    batch = _run_batch(tmp_path, entries, max_images=1)
    assert len(batch.results) == 1
    assert batch.results[0].image_id == "one"


def test_is_config_output_valid_rejects_corrupt(tmp_path: Path) -> None:
    matrix = tmp_path / "matrix"
    matrix.mkdir()
    path = matrix / "demo_none.json"
    path.write_text("{broken", encoding="utf-8")
    config = ExpertConfig(run_name="none", experts=())
    assert ble.is_config_output_valid(matrix, config) is False


def test_cli_usage_error_on_bad_manifest(tmp_path: Path) -> None:
    missing = tmp_path / "nope.json"
    code = ble.main(["--manifest", str(missing), "--output-dir", str(tmp_path / "out")])
    assert code == ble.EXIT_USAGE


def test_raw_runner_json_preserved(tmp_path: Path) -> None:
    (tmp_path / "infer_config.windows.yaml").write_text("dummy: true\n", encoding="utf-8")
    batch = _run_batch(tmp_path, [_entry_dict(tmp_path, image_id="keep")])
    matrix = Path(batch.results[0].matrix_dir)
    names = sorted(p.name for p in matrix.glob("demo_*.json"))
    assert names == [
        "demo_blending.json",
        "demo_blending_diffusion.json",
        "demo_diffusion.json",
        "demo_none.json",
    ]
