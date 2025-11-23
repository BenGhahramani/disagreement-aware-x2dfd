"""Persistence utilities (disabled writes).

This module originally created run directories and wrote JSON/CSV artefacts
for evaluation. Per request to remove persistence side effects, all write
operations are now no-ops and no directories/files are created. The public
API and return types are preserved to avoid breaking callers.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


@dataclass
class RunPaths:
    run_id: str
    results_dir: Path
    runs_root: Path
    run_dir: Path
    metrics_dir: Path
    responses_dir: Path
    responses_real_dir: Path
    responses_fake_dir: Path
    stats_dir: Path
    stats_real_dir: Path
    stats_fake_dir: Path
    questions_dir: Path
    dataset_metrics_path: Path
    per_question_balanced_path: Path
    question_ranking_all_path: Path
    question_ranking_path: Path
    concatenated_questions_path: Path
    summary_path: Path
    settings_path: Path
    run_log_path: Path
    latest_manifest_path: Path


def prepare_run_paths(results_dir: Path, top_k: int) -> RunPaths:
    """Return a timestamped QA run path layout without creating files/dirs."""
    # Compute paths but do not create anything on disk
    runs_root = results_dir / "qa_runs"
    run_id = datetime.utcnow().strftime("qa_eval_%Y%m%dT%H%M%SZ")
    run_dir = runs_root / run_id

    metrics_dir = run_dir / "metrics"
    responses_dir = run_dir / "responses"
    responses_real_dir = responses_dir / "real"
    responses_fake_dir = responses_dir / "fake"
    stats_dir = run_dir / "stats"
    stats_real_dir = stats_dir / "real"
    stats_fake_dir = stats_dir / "fake"
    questions_dir = run_dir / "questions"

    dataset_metrics_path = metrics_dir / "dataset_metrics.json"
    per_question_balanced_path = metrics_dir / "per_question_balanced.json"
    question_ranking_all_path = metrics_dir / "question_ranking_all.json"
    question_ranking_path = metrics_dir / f"question_ranking_top{top_k}.json"
    concatenated_questions_path = questions_dir / f"top{top_k}_questions.txt"
    summary_path = run_dir / "qa_summary.json"
    settings_path = run_dir / "qa_settings.json"
    run_log_path = runs_root / "run_log.csv"
    latest_manifest_path = runs_root / "latest_run.json"

    return RunPaths(
        run_id=run_id,
        results_dir=results_dir,
        runs_root=runs_root,
        run_dir=run_dir,
        metrics_dir=metrics_dir,
        responses_dir=responses_dir,
        responses_real_dir=responses_real_dir,
        responses_fake_dir=responses_fake_dir,
        stats_dir=stats_dir,
        stats_real_dir=stats_real_dir,
        stats_fake_dir=stats_fake_dir,
        questions_dir=questions_dir,
        dataset_metrics_path=dataset_metrics_path,
        per_question_balanced_path=per_question_balanced_path,
        question_ranking_all_path=question_ranking_all_path,
        question_ranking_path=question_ranking_path,
        concatenated_questions_path=concatenated_questions_path,
        summary_path=summary_path,
        settings_path=settings_path,
        run_log_path=run_log_path,
        latest_manifest_path=latest_manifest_path,
    )


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    # Disabled: no persistence
    return None


def write_text(path: Path, content: str) -> None:
    # Disabled: no persistence
    return None


def append_run_log(
    path: Path,
    timestamp: datetime,
    seed: int,
    top_k: int,
    real_acc: float,
    fake_acc: float,
    balanced_acc: float,
    run_id: str,
) -> None:
    # Disabled: no persistence
    return None
