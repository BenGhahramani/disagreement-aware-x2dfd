#!/usr/bin/env python3
"""Entry point for simulating QA accuracy and managing run artefacts."""

from __future__ import annotations

import argparse
import random
from datetime import datetime
from pathlib import Path
from typing import Dict

from qa_utils import (
    append_run_log,
    build_question_ranking,
    compute_balanced_accuracy,
    concat_questions,
    describe_image_answer,
    evaluate_dataset,
    evaluation_result_to_dict,
    load_image_paths,
    load_questions,
    prepare_run_paths,
    write_json,
    write_text,
    single_image_infer,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
LATEST_RUN_MANIFEST = RESULTS_DIR / "latest_run.json"
QUESTION_PATH = DATA_DIR / "question.json"
REAL_PATH = DATA_DIR / "real.json"
FAKE_PATH = DATA_DIR / "fake.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate QA accuracy metrics")
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top-ranked questions to keep in outputs (default: 3)",
    )
    args = parser.parse_args()
    if args.top_k is not None and args.top_k <= 0:
        parser.error("--top-k must be greater than 0")
    return args


def build_dataset_metrics(real_result, fake_result) -> Dict[str, Dict[str, object]]:
    return {
        "real": evaluation_result_to_dict(real_result),
        "fake": evaluation_result_to_dict(fake_result),
    }


def build_response_records(rng: random.Random, dataset_label: str, responses) -> Dict[str, object]:
    records = []
    for entry in responses:
        narrative = describe_image_answer(
            rng,
            dataset_label=dataset_label,
            question=entry.question,
            answer=entry.answer,
        )
        records.append(
            {
                "image_path": entry.image_path,
                "question": entry.question,
                "answer": entry.answer,
                "narrative": narrative,
            }
        )
    return {"dataset": dataset_label, "responses": records}


def main() -> None:
    args = parse_args()

    questions = load_questions(QUESTION_PATH)
    real_images = load_image_paths(REAL_PATH)
    fake_images = load_image_paths(FAKE_PATH)

    system_rng = random.SystemRandom()
    seed = system_rng.randrange(2**32)
    rng = random.Random(seed)

    fake_result = evaluate_dataset(fake_images, questions, single_image_infer)
    real_result = evaluate_dataset(real_images, questions, single_image_infer)

    balanced_acc = compute_balanced_accuracy(real_result.accuracy, fake_result.accuracy)
    top_k = args.top_k
    full_ranking = build_question_ranking(real_result.question_stats, fake_result.question_stats, top_k=None)
    ranking = full_ranking[:top_k]
    ranking_questions = [entry["question"] for entry in ranking]
    concatenated_questions = concat_questions(ranking_questions)
    per_question_balanced = {entry["question"]: entry["balanced_accuracy"] for entry in full_ranking}

    real_responses = build_response_records(rng, "real", real_result.responses)
    fake_responses = build_response_records(rng, "fake", fake_result.responses)

    run_paths = prepare_run_paths(RESULTS_DIR, top_k)

    dataset_metrics = build_dataset_metrics(real_result, fake_result)
    # Inject narratives into dataset metrics responses
    dataset_metrics["real"]["responses"] = real_responses["responses"]
    dataset_metrics["fake"]["responses"] = fake_responses["responses"]
    write_json(run_paths.dataset_metrics_path, dataset_metrics)

    write_json(run_paths.per_question_balanced_path, per_question_balanced)
    write_json(run_paths.question_ranking_all_path, {"questions": full_ranking})
    write_json(run_paths.question_ranking_path, {"questions": ranking})
    write_text(run_paths.concatenated_questions_path, concatenated_questions)
    write_json(run_paths.responses_real_path, real_responses)
    write_json(run_paths.responses_fake_path, fake_responses)

    summary_payload = {
        "run_id": run_paths.run_id,
        "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
        "seed": seed,
        "top_k": top_k,
        "balanced_accuracy": balanced_acc,
        "datasets": {
            "real_accuracy": real_result.accuracy,
            "fake_accuracy": fake_result.accuracy,
        },
        "artefacts": {
            "dataset_metrics": str(run_paths.dataset_metrics_path),
            "per_question_balanced": str(run_paths.per_question_balanced_path),
            "question_ranking_all": str(run_paths.question_ranking_all_path),
            "question_ranking_top_k": str(run_paths.question_ranking_path),
            "concatenated_questions": str(run_paths.concatenated_questions_path),
            "responses_real": str(run_paths.responses_real_path),
            "responses_fake": str(run_paths.responses_fake_path),
        },
    }
    write_json(run_paths.summary_path, summary_payload)

    write_json(
        run_paths.settings_path,
        {
            "question_file": str(QUESTION_PATH),
            "real_file": str(REAL_PATH),
            "fake_file": str(FAKE_PATH),
            "seed": seed,
            "top_k": top_k,
            "run_id": run_paths.run_id,
        },
    )

    append_run_log(
        run_paths.run_log_path,
        timestamp=datetime.utcnow(),
        seed=seed,
        top_k=top_k,
        real_acc=real_result.accuracy,
        fake_acc=fake_result.accuracy,
        balanced_acc=balanced_acc,
        run_id=run_paths.run_id,
    )

    write_json(
        LATEST_RUN_MANIFEST,
        {
            "run_id": run_paths.run_id,
            "timestamp": summary_payload["timestamp"],
            "top_k": top_k,
            "balanced_accuracy": balanced_acc,
            "summary": str(run_paths.summary_path),
            "dataset_metrics": str(run_paths.dataset_metrics_path),
            "question_ranking_all": str(run_paths.question_ranking_all_path),
            "question_ranking_top_k": str(run_paths.question_ranking_path),
            "per_question_balanced": str(run_paths.per_question_balanced_path),
            "concatenated_questions": str(run_paths.concatenated_questions_path),
            "responses_real": str(run_paths.responses_real_path),
            "responses_fake": str(run_paths.responses_fake_path),
        },
    )

    print(f"Run ID: {run_paths.run_id}")
    print(f"Fake ACC: {fake_result.accuracy:.4f} (Fake_count={fake_result.yes_count}, Total={fake_result.total_count})")
    print(f"Real ACC: {real_result.accuracy:.4f} (Fake_count={real_result.yes_count}, Total={real_result.total_count})")
    print(f"Balanced ACC: {balanced_acc:.4f}")
    print(f"Top-{top_k} questions by balanced accuracy: {ranking_questions}")
    print(f"Responses saved for real images: {run_paths.responses_real_path}")
    print(f"Responses saved for fake images: {run_paths.responses_fake_path}")
    print(f"Summary saved to: {run_paths.summary_path}")
    print(f"Run log updated: {run_paths.run_log_path}")


if __name__ == "__main__":
    main()
