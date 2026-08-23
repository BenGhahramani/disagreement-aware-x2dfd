#!/usr/bin/env python3
"""Labelled, resumable batch evaluation over the canonical 2x2 expert configs.

Runs a labelled image manifest through the existing verified pipeline
(``tools.run_expert_matrix.execute_config`` + face crop / already-cropped
bypass). Does not change inference behaviour, weights, detector configuration,
evaluator thresholds, or XAI agreement thresholds.

Exit codes:
    0  PASS — every requested image finished with all four configs valid
    1  FAIL — one or more unresolved image/config failures remain
    2  USAGE — bad arguments or an invalid labelled manifest
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from dashboard.live_analysis import (
    CROP_FILENAME,
    CROP_PLAN_FILENAME,
    DEFAULT_INFER_CONFIG,
    STAGE3_CROP_KWARGS,
    write_one_image_manifest,
)
from dashboard.view_model import build_dashboard_view
from eval.experiment_configs import DEFAULT_CONFIGS, EXPERIMENT_CONFIGS, RUN_ORDER
from eval.labelled_manifest import (
    LabelledImage,
    LabelledManifestError,
    load_labelled_manifest,
)
from eval.reproducibility import (
    ConfigProvenance,
    ImageProvenance,
    MaterialSettings,
    collect_model_artefacts,
    get_git_commit,
    get_git_dirty,
    get_git_provenance,
    interpretation_changed,
    interpretation_thresholds,
    load_config_provenance,
    material_settings_match,
    sha256_file,
    utc_timestamp,
    write_config_provenance,
)
from tools.make_face_crop import CropPlan, FaceCropError, NoFaceFoundError, write_face_crop
from tools.run_expert_matrix import (
    DEFAULT_TIMEOUT_S,
    ExpertConfig,
    RunOutcome,
    build_summary,
    execute_config,
    extract_prompt,
    normalise_configs,
    parse_expert_scores,
    write_json,
    write_runtimes_sidecar,
)
from tools.run_smoke_test import OutputError, validate_output

LOGGER = logging.getLogger("x2dfd.run_labelled_evaluation")

PROJECT_ROOT_DEFAULT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = Path("eval") / "outputs" / "labelled_evaluation"

IMAGE_SUMMARY_FILENAME = "image_summary.json"
IMAGE_PROVENANCE_FILENAME = "image_provenance.json"
CONFIG_PROVENANCE_DIRNAME = "config_provenance"
AGGREGATE_JSON_FILENAME = "aggregate.json"
AGGREGATE_CSV_FILENAME = "aggregate.csv"
BATCH_META_FILENAME = "batch_meta.json"
RUNTIMES_FILENAME = "runtimes.json"

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_USAGE = 2

ProcessRunner = Callable[..., Any]
CropFn = Callable[..., CropPlan]


@dataclass
class ConfigResult:
    """One expert-configuration cell for a labelled image."""

    run_name: str
    experts_arg: str
    status: str
    skipped: bool = False
    label: Optional[str] = None
    real_score: Optional[float] = None
    fake_score: Optional[float] = None
    expert_scores: Dict[str, Optional[float]] = field(default_factory=dict)
    runtime_s: Optional[float] = None
    peak_vram_mib: Optional[int] = None
    output_path: Optional[str] = None
    failures: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    inference_skipped: bool = False
    interpretation_rebuilt: bool = False
    provenance: Optional[Dict[str, Any]] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "run_name": self.run_name,
            "experts_arg": self.experts_arg,
            "status": self.status,
            "skipped": self.skipped,
            "inference_skipped": self.inference_skipped,
            "interpretation_rebuilt": self.interpretation_rebuilt,
            "label": self.label,
            "real_score": self.real_score,
            "fake_score": self.fake_score,
            "blending_detector_score": self.expert_scores.get("blending"),
            "diffusion_detector_score": self.expert_scores.get("diffusion"),
            "expert_scores": dict(self.expert_scores),
            "runtime_s": self.runtime_s,
            "peak_vram_mib": self.peak_vram_mib,
            "output_path": self.output_path,
            "failures": list(self.failures),
            "warnings": list(self.warnings),
            "provenance": self.provenance,
        }


@dataclass
class ImageEvaluationResult:
    """Structured per-image batch result (also written to disk)."""

    image_id: str
    source_path: str
    ground_truth: str
    work_dir: str
    matrix_dir: str
    ok: bool
    skipped_entirely: bool = False
    source_sha256: Optional[str] = None
    crop_sha256: Optional[str] = None
    dataset: Optional[str] = None
    manipulation: Optional[str] = None
    source: Optional[str] = None
    already_cropped: bool = False
    notes: Optional[str] = None
    inference_config_path: Optional[str] = None
    inference_config_sha256: Optional[str] = None
    load_4bit: Optional[bool] = None
    interpretation_thresholds: Dict[str, float] = field(default_factory=interpretation_thresholds)
    git_commit: Optional[str] = None
    git_dirty: Optional[bool] = None
    model_artefacts_fingerprint: Optional[str] = None
    model_artefacts: Optional[Dict[str, Any]] = None
    timestamp: Optional[str] = None
    prototype_status: Optional[str] = None
    evidence_agreement: Optional[str] = None
    rationale: Optional[str] = None
    configs: List[ConfigResult] = field(default_factory=list)
    total_runtime_s: Optional[float] = None
    peak_vram_mib: Optional[int] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "image_id": self.image_id,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "crop_sha256": self.crop_sha256,
            "ground_truth": self.ground_truth,
            "dataset": self.dataset,
            "manipulation": self.manipulation,
            "source": self.source,
            "already_cropped": self.already_cropped,
            "notes": self.notes,
            "inference_config_path": self.inference_config_path,
            "inference_config_sha256": self.inference_config_sha256,
            "load_4bit": self.load_4bit,
            "interpretation_thresholds": dict(self.interpretation_thresholds),
            "git_commit": self.git_commit,
            "git_dirty": self.git_dirty,
            "model_artefacts_fingerprint": self.model_artefacts_fingerprint,
            "model_artefacts": self.model_artefacts,
            "timestamp": self.timestamp,
            "work_dir": self.work_dir,
            "matrix_dir": self.matrix_dir,
            "ok": self.ok,
            "skipped_entirely": self.skipped_entirely,
            "prototype_status": self.prototype_status,
            "evidence_agreement": self.evidence_agreement,
            "rationale": self.rationale,
            "configs": [cfg.as_dict() for cfg in self.configs],
            "total_runtime_s": self.total_runtime_s,
            "peak_vram_mib": self.peak_vram_mib,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


@dataclass
class BatchResult:
    """Aggregate outcome for one labelled-evaluation batch."""

    ok: bool
    output_dir: Path
    results: List[ImageEvaluationResult] = field(default_factory=list)
    aggregate_json: Optional[Path] = None
    aggregate_csv: Optional[Path] = None
    errors: List[str] = field(default_factory=list)

    @property
    def unresolved_failures(self) -> int:
        return sum(1 for result in self.results if not result.ok)


def image_work_dir(output_dir: Path, image_id: str) -> Path:
    """Stable per-image directory under the batch output root."""

    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", image_id).strip("._") or "image"
    return (Path(output_dir) / slug).resolve()


def _read_runtimes(matrix_dir: Path) -> Dict[str, float]:
    path = matrix_dir / RUNTIMES_FILENAME
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    out: Dict[str, float] = {}
    for key, value in payload.items():
        if isinstance(value, (int, float)):
            out[str(key)] = float(value)
    return out


def is_config_output_valid(matrix_dir: Path, config: ExpertConfig) -> bool:
    """True when ``demo_<run_name>.json`` exists and passes smoke-test validation."""

    path = config.output_path(matrix_dir)
    if not path.is_file():
        return False
    try:
        predictions = validate_output(path)
    except OutputError:
        return False
    if not predictions or predictions[0].label is None:
        return False
    # Requested experts must appear in the prompt (same rule as execute_config).
    prompt = extract_prompt(path)
    scores = parse_expert_scores(prompt)
    for expert in config.experts:
        if scores.get(expert, "absent") is None:
            return False
        if expert not in scores:
            return False
    return True


def config_provenance_path(matrix_dir: Path, run_name: str) -> Path:
    return matrix_dir / CONFIG_PROVENANCE_DIRNAME / f"{run_name}.json"


def build_material_settings(
    *,
    source_sha256: str,
    crop_sha256: str,
    inference_config_path: Path,
    inference_config_sha256: str,
    load_4bit: bool,
    config: ExpertConfig,
    already_cropped: bool,
    model_artefacts_fingerprint: str,
) -> MaterialSettings:
    return MaterialSettings(
        source_sha256=source_sha256,
        crop_sha256=crop_sha256,
        inference_config_path=str(inference_config_path.resolve()),
        inference_config_sha256=inference_config_sha256,
        load_4bit=load_4bit,
        run_name=config.run_name,
        experts_arg=config.experts_arg,
        already_cropped=already_cropped,
        model_artefacts_fingerprint=model_artefacts_fingerprint,
    )


def can_reuse_raw_output(
    matrix_dir: Path,
    config: ExpertConfig,
    current_material: MaterialSettings,
    stored: Optional[ConfigProvenance],
) -> Tuple[bool, bool]:
    """Return ``(reuse_raw, interpretation_only)``.

    Raw runner JSON is reused only when output validates **and** recorded
    material settings match (including inference-config content hash and model
    artefact fingerprints). ``interpretation_only`` is True when evaluator /
    evidence-agreement thresholds changed but inference inputs did not.
    """

    if stored is None:
        return False, False
    if not is_config_output_valid(matrix_dir, config):
        return False, False
    if not material_settings_match(stored.material, current_material):
        return False, False
    current_thresholds = interpretation_thresholds()
    return True, interpretation_changed(stored.interpretation_thresholds, current_thresholds)


def persist_config_provenance(
    matrix_dir: Path,
    config: ExpertConfig,
    material: MaterialSettings,
    *,
    git_commit: Optional[str],
    git_dirty: Optional[bool],
    output_path: Optional[str],
    model_artefacts: Optional[Dict[str, Any]],
) -> ConfigProvenance:
    provenance = ConfigProvenance(
        run_name=config.run_name,
        experts_arg=config.experts_arg,
        material=material,
        interpretation_thresholds=interpretation_thresholds(),
        git_commit=git_commit,
        git_dirty=git_dirty,
        timestamp=utc_timestamp(),
        output_path=output_path,
        model_artefacts=model_artefacts,
    )
    write_config_provenance(config_provenance_path(matrix_dir, config.run_name), provenance)
    return provenance


def load_outcome_from_disk(
    config: ExpertConfig,
    matrix_dir: Path,
    *,
    runtimes: Optional[Dict[str, float]] = None,
) -> RunOutcome:
    """Rebuild a :class:`RunOutcome` from a previously validated raw JSON file."""

    path = config.output_path(matrix_dir)
    outcome = RunOutcome(
        run_name=config.run_name,
        experts=list(config.experts),
        experts_arg=config.experts_arg,
        command=[],
        output_path=str(path.resolve()),
        status="pass",
        validation="valid",
    )
    if runtimes and config.run_name in runtimes:
        outcome.runtime_s = runtimes[config.run_name]
    prompt = extract_prompt(path)
    outcome.prompt = prompt
    outcome.expert_scores = parse_expert_scores(prompt)
    try:
        prediction = validate_output(path)[0]
    except (OutputError, IndexError) as exc:
        outcome.status = "fail"
        outcome.validation = "invalid"
        outcome.failures = [str(exc)]
        return outcome
    outcome.prediction_text = prediction.answer
    outcome.label = prediction.label
    outcome.real_score = prediction.real_score
    outcome.fake_score = prediction.fake_score
    return outcome


def _default_crop(source: Path, dest: Path) -> CropPlan:
    return write_face_crop(source, dest, **STAGE3_CROP_KWARGS)


def prepare_image_input(
    entry: LabelledImage,
    work_dir: Path,
    *,
    crop_fn: Optional[CropFn] = None,
) -> Tuple[Path, Dict[str, Any]]:
    """Produce ``face_crop.jpg`` via Haar crop or already-cropped copy."""

    crop_path = work_dir / CROP_FILENAME
    cropper = crop_fn or _default_crop
    if entry.already_cropped:
        crop_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(entry.path, crop_path)
        plan = {
            "already_cropped": True,
            "source": str(entry.path),
            "output": str(crop_path.resolve()),
            "note": "Haar face crop bypassed; input treated as preprocessed crop.",
        }
        write_json(work_dir / CROP_PLAN_FILENAME, plan)
        return crop_path.resolve(), plan

    plan_obj = cropper(entry.path, crop_path)
    plan = {
        "already_cropped": False,
        "source": str(entry.path),
        "output": str(crop_path.resolve()),
        "detection": list(plan_obj.detection) if plan_obj.detection else None,
        "square": list(plan_obj.square) if plan_obj.square else None,
        **{
            key: STAGE3_CROP_KWARGS[key]
            for key in ("size", "margin", "detect_width", "scale_factor", "min_neighbours", "quality")
            if key in STAGE3_CROP_KWARGS
        },
    }
    write_json(work_dir / CROP_PLAN_FILENAME, plan)
    return crop_path.resolve(), plan


def _outcome_to_config_result(
    outcome: RunOutcome,
    *,
    skipped: bool,
    interpretation_rebuilt: bool = False,
    provenance: Optional[ConfigProvenance] = None,
) -> ConfigResult:
    return ConfigResult(
        run_name=outcome.run_name,
        experts_arg=outcome.experts_arg,
        status=outcome.status,
        skipped=skipped,
        inference_skipped=skipped,
        interpretation_rebuilt=interpretation_rebuilt,
        label=outcome.label,
        real_score=outcome.real_score,
        fake_score=outcome.fake_score,
        expert_scores=dict(outcome.expert_scores),
        runtime_s=outcome.runtime_s,
        peak_vram_mib=outcome.peak_vram_mib,
        output_path=outcome.output_path,
        failures=list(outcome.failures),
        warnings=list(outcome.warnings),
        provenance=None if provenance is None else provenance.as_dict(),
    )


def evaluate_labelled_image(
    entry: LabelledImage,
    *,
    output_dir: Path,
    project_root: Path,
    config_path: Path,
    timeout_s: float,
    load_4bit: bool = True,
    sample_vram: bool = True,
    resume: bool = True,
    dry_run: bool = False,
    process_runner: Optional[ProcessRunner] = None,
    crop_fn: Optional[CropFn] = None,
    git_commit: Optional[str] = None,
    git_dirty: Optional[bool] = None,
    model_artefacts: Optional[Dict[str, Any]] = None,
) -> ImageEvaluationResult:
    """Run or resume the canonical four configs for one labelled image."""

    work_dir = image_work_dir(output_dir, entry.id)
    matrix_dir = work_dir / "matrix"
    resolved_config = config_path.resolve()
    commit = git_commit if git_commit is not None else get_git_commit(project_root)
    dirty = git_dirty if git_dirty is not None else get_git_dirty(project_root)
    artefacts = model_artefacts
    if artefacts is None:
        artefacts = collect_model_artefacts(resolved_config, project_root=project_root)
    config_sha = artefacts.get("inference_config_sha256") or ""
    artefacts_fp = str(artefacts.get("artefacts_fingerprint") or "")
    thresholds = interpretation_thresholds()
    evaluated_at = utc_timestamp()

    result = ImageEvaluationResult(
        image_id=entry.id,
        source_path=str(entry.path),
        ground_truth=entry.ground_truth,
        work_dir=str(work_dir),
        matrix_dir=str(matrix_dir),
        ok=False,
        dataset=entry.dataset,
        manipulation=entry.manipulation,
        source=entry.source,
        already_cropped=entry.already_cropped,
        notes=entry.notes,
        inference_config_path=str(resolved_config),
        inference_config_sha256=config_sha or None,
        load_4bit=load_4bit,
        interpretation_thresholds=dict(thresholds),
        git_commit=commit,
        git_dirty=dirty,
        model_artefacts_fingerprint=artefacts_fp or None,
        model_artefacts=artefacts,
        timestamp=evaluated_at,
    )

    try:
        result.source_sha256 = sha256_file(entry.path)
    except OSError as exc:
        result.errors.append(f"cannot hash source image: {exc}")
        write_json(work_dir / IMAGE_SUMMARY_FILENAME, result.as_dict())
        return result

    expert_configs = normalise_configs(list(DEFAULT_CONFIGS))
    if [cfg.run_name for cfg in expert_configs] != list(RUN_ORDER):
        result.errors.append(
            f"canonical config order mismatch: "
            f"{[cfg.run_name for cfg in expert_configs]} != {list(RUN_ORDER)}"
        )
        return result
    if len(expert_configs) != 4:
        result.errors.append(f"expected four configs, got {len(expert_configs)}")
        return result

    if dry_run:
        result.ok = True
        result.skipped_entirely = True
        result.warnings.append("dry-run: inference not executed")
        result.configs = [
            ConfigResult(
                run_name=cfg.run_name,
                experts_arg=cfg.experts_arg,
                status="dry-run",
                skipped=True,
            )
            for cfg in expert_configs
        ]
        return result

    work_dir.mkdir(parents=True, exist_ok=True)
    matrix_dir.mkdir(parents=True, exist_ok=True)

    try:
        crop_path, _plan = prepare_image_input(entry, work_dir, crop_fn=crop_fn)
    except (NoFaceFoundError, FaceCropError, OSError) as exc:
        result.errors.append(f"preprocessing failed: {exc}")
        write_json(work_dir / IMAGE_SUMMARY_FILENAME, result.as_dict())
        return result

    try:
        result.crop_sha256 = sha256_file(crop_path)
    except OSError as exc:
        result.errors.append(f"cannot hash cropped image: {exc}")
        write_json(work_dir / IMAGE_SUMMARY_FILENAME, result.as_dict())
        return result

    image_provenance = ImageProvenance(
        image_id=entry.id,
        source_path=str(entry.path),
        source_sha256=result.source_sha256,
        crop_sha256=result.crop_sha256,
        ground_truth=entry.ground_truth,
        dataset=entry.dataset,
        manipulation=entry.manipulation,
        source=entry.source,
        already_cropped=entry.already_cropped,
        notes=entry.notes,
        inference_config_path=str(resolved_config),
        inference_config_sha256=config_sha or None,
        load_4bit=load_4bit,
        interpretation_thresholds=dict(thresholds),
        git_commit=commit,
        git_dirty=dirty,
        model_artefacts_fingerprint=artefacts_fp or None,
        timestamp=evaluated_at,
    )
    write_json(work_dir / IMAGE_PROVENANCE_FILENAME, image_provenance.as_dict())

    try:
        manifest_path = write_one_image_manifest(crop_path, work_dir / "manifest.json")
    except Exception as exc:  # noqa: BLE001 — record and continue the batch
        result.errors.append(f"manifest failed: {exc}")
        write_json(work_dir / IMAGE_SUMMARY_FILENAME, result.as_dict())
        return result

    runtimes = _read_runtimes(matrix_dir) if resume else {}
    log_dir = matrix_dir / "logs"
    outcomes: List[RunOutcome] = []
    config_results: List[ConfigResult] = []
    started_at = utc_timestamp()

    for config in expert_configs:
        current_material = build_material_settings(
            source_sha256=result.source_sha256,
            crop_sha256=result.crop_sha256,
            inference_config_path=resolved_config,
            inference_config_sha256=config_sha,
            load_4bit=load_4bit,
            config=config,
            already_cropped=entry.already_cropped,
            model_artefacts_fingerprint=artefacts_fp,
        )
        stored = (
            load_config_provenance(config_provenance_path(matrix_dir, config.run_name))
            if resume
            else None
        )
        reuse_raw, interpretation_only = (
            can_reuse_raw_output(matrix_dir, config, current_material, stored)
            if resume
            else (False, False)
        )

        if reuse_raw:
            outcome = load_outcome_from_disk(config, matrix_dir, runtimes=runtimes)
            LOGGER.info("[%s] reuse raw output for %s", entry.id, config.run_name)
            if interpretation_only:
                result.warnings.append(
                    f"{config.run_name}: interpretation rebuilt from cached raw output"
                )
            outcomes.append(outcome)
            provenance = stored or persist_config_provenance(
                matrix_dir,
                config,
                current_material,
                git_commit=commit,
                git_dirty=dirty,
                output_path=outcome.output_path,
                model_artefacts=artefacts,
            )
            if interpretation_only:
                provenance = persist_config_provenance(
                    matrix_dir,
                    config,
                    current_material,
                    git_commit=commit,
                    git_dirty=dirty,
                    output_path=outcome.output_path,
                    model_artefacts=artefacts,
                )
            config_results.append(
                _outcome_to_config_result(
                    outcome,
                    skipped=True,
                    interpretation_rebuilt=interpretation_only,
                    provenance=provenance,
                )
            )
            continue

        if not resume:
            stale = config.output_path(matrix_dir)
            if stale.exists():
                try:
                    stale.unlink()
                except OSError as exc:
                    result.warnings.append(f"could not remove stale {stale.name}: {exc}")
            prov_path = config_provenance_path(matrix_dir, config.run_name)
            if prov_path.is_file():
                try:
                    prov_path.unlink()
                except OSError as exc:
                    result.warnings.append(
                        f"could not remove stale provenance {prov_path.name}: {exc}"
                    )

        LOGGER.info("[%s] run %s", entry.id, config.run_name)
        try:
            outcome = execute_config(
                config,
                manifest_path=manifest_path,
                config_path=resolved_config,
                output_dir=matrix_dir,
                log_dir=log_dir,
                project_root=project_root,
                timeout_s=timeout_s,
                load_4bit=load_4bit,
                load_8bit=False,
                sample_vram=sample_vram,
                process_runner=process_runner or subprocess.run,
            )
        except Exception as exc:  # noqa: BLE001 — never abort the batch on one cell
            outcome = RunOutcome(
                run_name=config.run_name,
                experts=list(config.experts),
                experts_arg=config.experts_arg,
                command=[],
                output_path=None,
                status="fail",
                validation="exception",
                failures=[f"execute_config raised {type(exc).__name__}: {exc}"],
            )
        outcomes.append(outcome)
        provenance: Optional[ConfigProvenance] = None
        if outcome.status == "pass":
            provenance = persist_config_provenance(
                matrix_dir,
                config,
                current_material,
                git_commit=commit,
                git_dirty=dirty,
                output_path=outcome.output_path,
                model_artefacts=artefacts,
            )
        config_results.append(
            _outcome_to_config_result(
                outcome,
                skipped=False,
                interpretation_rebuilt=False,
                provenance=provenance,
            )
        )
        if outcome.status != "pass":
            detail = "; ".join(outcome.failures) or "configuration failed"
            result.errors.append(f"{config.run_name}: {detail}")

    quantisation = "4-bit" if load_4bit else "fp16"
    summary = build_summary(
        outcomes,
        manifest=manifest_path,
        images=[crop_path],
        config_path=resolved_config,
        output_dir=matrix_dir,
        quantisation=quantisation,
        started_at=started_at,
        finished_at=utc_timestamp(),
    )
    summary["git_commit"] = commit
    summary["interpretation_thresholds"] = dict(thresholds)
    summary["source_sha256"] = result.source_sha256
    summary["crop_sha256"] = result.crop_sha256
    summary_path = work_dir / "summary.json"
    write_json(summary_path, summary)
    write_runtimes_sidecar(matrix_dir, outcomes)

    view = build_dashboard_view(matrix_dir, summary_path=summary_path)
    if crop_path.is_file():
        view.image_path = crop_path
    result.prototype_status = view.status.value
    result.evidence_agreement = view.evidence_agreement.value
    result.rationale = view.rationale
    result.warnings.extend(view.warnings)
    if view.errors:
        result.errors.extend(view.errors)

    runtimes_values = [o.runtime_s for o in outcomes if o.runtime_s is not None]
    result.total_runtime_s = sum(runtimes_values) if runtimes_values else None
    peaks = [o.peak_vram_mib for o in outcomes if o.peak_vram_mib is not None]
    result.peak_vram_mib = max(peaks) if peaks else None
    result.configs = config_results
    result.ok = all(cfg.status == "pass" for cfg in config_results) and not view.errors

    write_json(work_dir / IMAGE_SUMMARY_FILENAME, result.as_dict())
    return result


def write_aggregate(
    results: Sequence[ImageEvaluationResult],
    output_dir: Path,
) -> Tuple[Path, Path]:
    """Write aggregate JSON + CSV under ``output_dir``."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": utc_timestamp(),
        "n_images": len(results),
        "n_ok": sum(1 for r in results if r.ok),
        "n_failed": sum(1 for r in results if not r.ok),
        "canonical_configs": list(DEFAULT_CONFIGS),
        "run_order": list(RUN_ORDER),
        "interpretation_thresholds": interpretation_thresholds(),
        "images": [r.as_dict() for r in results],
    }
    json_path = output_dir / AGGREGATE_JSON_FILENAME
    write_json(json_path, payload)

    csv_path = output_dir / AGGREGATE_CSV_FILENAME
    fieldnames = [
        "image_id",
        "ground_truth",
        "dataset",
        "manipulation",
        "source",
        "already_cropped",
        "ok",
        "prototype_status",
        "evidence_agreement",
        "total_runtime_s",
        "peak_vram_mib",
        "source_path",
        "source_sha256",
        "crop_sha256",
        "errors",
    ]
    for run_name in RUN_ORDER:
        fieldnames.extend(
            [
                f"{run_name}_status",
                f"{run_name}_label",
                f"{run_name}_real_score",
                f"{run_name}_fake_score",
                f"{run_name}_blending_score",
                f"{run_name}_diffusion_score",
                f"{run_name}_skipped",
            ]
        )

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row: Dict[str, Any] = {
                "image_id": result.image_id,
                "ground_truth": result.ground_truth,
                "dataset": result.dataset or "",
                "manipulation": result.manipulation or "",
                "source": result.source or "",
                "already_cropped": result.already_cropped,
                "ok": result.ok,
                "prototype_status": result.prototype_status or "",
                "evidence_agreement": result.evidence_agreement or "",
                "total_runtime_s": result.total_runtime_s,
                "peak_vram_mib": result.peak_vram_mib,
                "source_path": result.source_path,
                "source_sha256": result.source_sha256 or "",
                "crop_sha256": result.crop_sha256 or "",
                "errors": "; ".join(result.errors),
            }
            by_name = {cfg.run_name: cfg for cfg in result.configs}
            for run_name in RUN_ORDER:
                cfg = by_name.get(run_name)
                row[f"{run_name}_status"] = cfg.status if cfg else ""
                row[f"{run_name}_label"] = (cfg.label if cfg else "") or ""
                row[f"{run_name}_real_score"] = cfg.real_score if cfg else ""
                row[f"{run_name}_fake_score"] = cfg.fake_score if cfg else ""
                row[f"{run_name}_blending_score"] = (
                    cfg.expert_scores.get("blending") if cfg else ""
                )
                row[f"{run_name}_diffusion_score"] = (
                    cfg.expert_scores.get("diffusion") if cfg else ""
                )
                row[f"{run_name}_skipped"] = cfg.skipped if cfg else ""
            writer.writerow(row)

    return json_path.resolve(), csv_path.resolve()


def run_labelled_evaluation(
    *,
    manifest_path: Path,
    output_dir: Path,
    project_root: Path | None = None,
    config_path: Path | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    load_4bit: bool = True,
    sample_vram: bool = True,
    resume: bool = True,
    max_images: Optional[int] = None,
    dry_run: bool = False,
    process_runner: Optional[ProcessRunner] = None,
    crop_fn: Optional[CropFn] = None,
) -> BatchResult:
    """Evaluate every labelled image; never abort the batch on a single failure."""

    root = Path(project_root) if project_root is not None else PROJECT_ROOT_DEFAULT
    infer_config = Path(config_path) if config_path is not None else DEFAULT_INFER_CONFIG
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    batch_started = utc_timestamp()
    manifest_resolved = Path(manifest_path).resolve()
    try:
        manifest_sha256 = sha256_file(manifest_resolved)
    except OSError:
        manifest_sha256 = None
    git_info = get_git_provenance(root)
    git_commit = git_info["git_commit"]
    git_dirty = git_info["git_dirty"]
    quantisation = "4-bit" if load_4bit else "fp16"
    thresholds = interpretation_thresholds()
    model_artefacts = collect_model_artefacts(
        infer_config.resolve(),
        project_root=root,
    )

    batch = BatchResult(ok=False, output_dir=out_root.resolve())
    try:
        images = load_labelled_manifest(manifest_path)
    except LabelledManifestError as exc:
        batch.errors.append(str(exc))
        return batch

    if max_images is not None:
        images = images[: max(0, max_images)]

    # Guardrail: never invent a fifth configuration.
    assert len(EXPERIMENT_CONFIGS) == 4
    assert list(DEFAULT_CONFIGS) == ["none", "blending", "diffusion", "blending,diffusion"]

    for entry in images:
        LOGGER.info("=== image %s (%s) ===", entry.id, entry.ground_truth)
        try:
            result = evaluate_labelled_image(
                entry,
                output_dir=out_root,
                project_root=root,
                config_path=infer_config,
                timeout_s=timeout_s,
                load_4bit=load_4bit,
                sample_vram=sample_vram,
                resume=resume,
                dry_run=dry_run,
                process_runner=process_runner,
                crop_fn=crop_fn,
                git_commit=git_commit,
                git_dirty=git_dirty,
                model_artefacts=model_artefacts,
            )
        except Exception as exc:  # noqa: BLE001 — one image must not kill the batch
            result = ImageEvaluationResult(
                image_id=entry.id,
                source_path=str(entry.path),
                ground_truth=entry.ground_truth,
                work_dir=str(image_work_dir(out_root, entry.id)),
                matrix_dir=str(image_work_dir(out_root, entry.id) / "matrix"),
                ok=False,
                dataset=entry.dataset,
                manipulation=entry.manipulation,
                source=entry.source,
                already_cropped=entry.already_cropped,
                notes=entry.notes,
                errors=[f"unhandled error: {type(exc).__name__}: {exc}"],
            )
        batch.results.append(result)

    json_path, csv_path = write_aggregate(batch.results, out_root)
    batch.aggregate_json = json_path
    batch.aggregate_csv = csv_path
    batch_finished = utc_timestamp()
    batch.ok = batch.unresolved_failures == 0 and not batch.errors
    write_json(
        out_root / BATCH_META_FILENAME,
        {
            "ok": batch.ok,
            "manifest_path": str(manifest_resolved),
            "manifest_sha256": manifest_sha256,
            "output_dir": str(out_root.resolve()),
            "n_images": len(batch.results),
            "n_failed": batch.unresolved_failures,
            "resume": resume,
            "dry_run": dry_run,
            "canonical_configs": list(DEFAULT_CONFIGS),
            "run_order": list(RUN_ORDER),
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "quantisation": quantisation,
            "load_4bit": load_4bit,
            "inference_config_path": str(infer_config.resolve()),
            "inference_config_sha256": model_artefacts.get("inference_config_sha256"),
            "model_artefacts": model_artefacts,
            "interpretation_thresholds": dict(thresholds),
            "started_at": batch_started,
            "finished_at": batch_finished,
            "aggregate_json": str(json_path),
            "aggregate_csv": str(csv_path),
            "errors": list(batch.errors),
        },
    )
    return batch


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m tools.run_labelled_evaluation",
        description=(
            "Run a labelled image dataset through the canonical four X2-DFD "
            "expert configurations with resumable per-image outputs."
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="labelled evaluation JSON manifest",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"batch output root (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=f"infer config YAML (default: {DEFAULT_INFER_CONFIG})",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT_DEFAULT,
        help="repository root used as the inference subprocess cwd",
    )
    parser.add_argument(
        "--timeout",
        dest="timeout_s",
        type=float,
        default=DEFAULT_TIMEOUT_S,
        help=f"per-configuration timeout in seconds (default: {DEFAULT_TIMEOUT_S:.0f})",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="optional pilot limit on the number of images processed",
    )
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="skip valid completed config outputs (default)",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="re-run every configuration even when outputs already exist",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate the manifest and plan work without running inference",
    )
    parser.add_argument(
        "--load-4bit",
        action="store_true",
        default=True,
        help="4-bit loading (default; needed on a 10 GiB card)",
    )
    parser.add_argument(
        "--no-4bit",
        dest="load_4bit",
        action="store_false",
        help="disable 4-bit loading",
    )
    parser.add_argument(
        "--no-vram-sampling",
        action="store_true",
        help="skip nvidia-smi peak VRAM polling",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    try:
        batch = run_labelled_evaluation(
            manifest_path=args.manifest,
            output_dir=args.output_dir,
            project_root=args.project_root,
            config_path=args.config,
            timeout_s=args.timeout_s,
            load_4bit=args.load_4bit,
            sample_vram=not args.no_vram_sampling,
            resume=args.resume,
            max_images=args.max_images,
            dry_run=args.dry_run,
        )
    except LabelledManifestError as exc:
        LOGGER.error("%s", exc)
        return EXIT_USAGE

    if batch.errors and not batch.results:
        for message in batch.errors:
            LOGGER.error("%s", message)
        return EXIT_USAGE

    LOGGER.info(
        "batch complete: %s/%s images ok; aggregate %s",
        sum(1 for r in batch.results if r.ok),
        len(batch.results),
        batch.aggregate_json,
    )
    return EXIT_PASS if batch.ok else EXIT_FAIL


if __name__ == "__main__":
    raise SystemExit(main())
