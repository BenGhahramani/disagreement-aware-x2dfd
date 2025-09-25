"""Persistence utilities for QA evaluation outputs."""

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
    run_dir: Path
    dataset_metrics_path: Path
    per_question_balanced_path: Path
    question_ranking_all_path: Path
    question_ranking_path: Path
    concatenated_questions_path: Path
    responses_real_path: Path
    responses_fake_path: Path
    summary_path: Path
    settings_path: Path
    run_log_path: Path


def prepare_run_paths(results_dir: Path, top_k: int) -> RunPaths:
    """Create a timestamped run directory and return all relevant paths."""
    results_dir.mkdir(parents=True, exist_ok=True)
    runs_root = results_dir / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    run_id = datetime.utcnow().strftime("run_%Y%m%dT%H%M%SZ")
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset_metrics_path = run_dir / "dataset_metrics.json"
    per_question_balanced_path = run_dir / "per_question_balanced.json"
    question_ranking_all_path = run_dir / "question_ranking_all.json"
    question_ranking_path = run_dir / f"question_ranking_top{top_k}.json"
    concatenated_questions_path = run_dir / f"concatenated_questions_top{top_k}.txt"
    responses_real_path = run_dir / "responses_real.json"
    responses_fake_path = run_dir / "responses_fake.json"
    summary_path = run_dir / "summary.json"
    settings_path = run_dir / "settings.json"
    run_log_path = results_dir / "run_log.csv"

    return RunPaths(
        run_id=run_id,
        results_dir=results_dir,
        run_dir=run_dir,
        dataset_metrics_path=dataset_metrics_path,
        per_question_balanced_path=per_question_balanced_path,
        question_ranking_all_path=question_ranking_all_path,
        question_ranking_path=question_ranking_path,
        concatenated_questions_path=concatenated_questions_path,
        responses_real_path=responses_real_path,
        responses_fake_path=responses_fake_path,
        summary_path=summary_path,
        settings_path=settings_path,
        run_log_path=run_log_path,
    )


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(content)


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
    header = [
        "timestamp",
        "run_id",
        "seed",
        "top_k",
        "real_accuracy",
        "fake_accuracy",
        "balanced_accuracy",
    ]

    new_file = not path.exists()
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            first_line = handle.readline().strip()
        expected_header = ",".join(header)
        if first_line and first_line != expected_header:
            backup_name = f"{path.stem}_legacy_{timestamp.strftime('%Y%m%dT%H%M%SZ')}{path.suffix}"
            backup_path = path.with_name(backup_name)
            path.rename(backup_path)
            new_file = True

    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if new_file:
            writer.writerow(header)
        writer.writerow(
            [
                timestamp.isoformat(timespec="seconds"),
                run_id,
                seed,
                top_k,
                f"{real_acc:.6f}",
                f"{fake_acc:.6f}",
                f"{balanced_acc:.6f}",
            ]
        )
