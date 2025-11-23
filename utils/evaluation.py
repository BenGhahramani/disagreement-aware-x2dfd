"""Evaluation logic for QA accuracy simulation (moved from qa_utils)."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Sequence

AnswerFn = Callable[[str, str], str]


@dataclass
class QAResponse:
    image_path: str
    question: str
    answer: str
    score: float | None = None


@dataclass
class QuestionStat:
    question: str
    yes_count: int
    no_count: int
    total_count: int
    accuracy: float  # For real: no/total, for fake: yes/total


@dataclass
class EvaluationResult:
    yes_count: int
    no_count: int
    total_count: int
    accuracy: float
    image_count: int
    question_count: int
    question_stats: List[QuestionStat]
    responses: List[QAResponse]


def evaluate_dataset(
    image_paths: Sequence[str],
    questions: Sequence[str],
    answer_fn: AnswerFn,
) -> EvaluationResult:
    """Evaluate answers across all image/question combinations."""
    yes_count = 0
    no_count = 0
    total_count = 0

    yes_by_question: Dict[str, int] = defaultdict(int)
    no_by_question: Dict[str, int] = defaultdict(int)
    total_by_question: Dict[str, int] = defaultdict(int)
    responses: List[QAResponse] = []

    for image_path in image_paths:
        for question in questions:
            answer = answer_fn(image_path, question)
            responses.append(QAResponse(image_path=image_path, question=question, answer=answer))

            # Accept "Yes" (case-insensitive, allow trailing punctuation) as positive
            answer_clean = answer.strip().lower().rstrip(".!?,")
            if answer_clean == "yes":
                yes_count += 1
                yes_by_question[question] += 1
            elif answer_clean == "no":
                no_count += 1
                no_by_question[question] += 1
            total_count += 1
            total_by_question[question] += 1

    def build_question_stat(question: str) -> QuestionStat:
        question_yes = yes_by_question[question]
        question_no = no_by_question[question]
        question_total = total_by_question[question]
        # accuracy is not meaningful here, will be set in build_question_ranking
        return QuestionStat(
            question=question,
            yes_count=question_yes,
            no_count=question_no,
            total_count=question_total,
            accuracy=0.0,
        )

    question_stats = [build_question_stat(question) for question in questions]
    accuracy = yes_count / total_count if total_count else 0.0
    return EvaluationResult(
        yes_count=yes_count,
        no_count=no_count,
        total_count=total_count,
        accuracy=accuracy,
        image_count=len(image_paths),
        question_count=len(questions),
        question_stats=question_stats,
        responses=responses,
    )


def compute_balanced_accuracy(real_acc: float, fake_acc: float) -> float:
    """Compute balanced accuracy as the average of real and fake accuracies."""
    return (real_acc + fake_acc) / 2.0


def build_question_ranking(
    real_stats: Iterable[QuestionStat],
    fake_stats: Iterable[QuestionStat],
    top_k: int | None = None,
) -> List[Dict[str, object]]:
    """
    Combine real/fake question stats, compute accuracy as:
      - real_accuracy: no_count / total_count
      - fake_accuracy: yes_count / total_count
    Then average for balanced_accuracy and sort descending.
    """
    real_lookup = {stat.question: stat for stat in real_stats}
    fake_lookup = {stat.question: stat for stat in fake_stats}

    questions = set(real_lookup) | set(fake_lookup)
    combined: List[Dict[str, object]] = []

    for question in questions:
        real_stat = real_lookup.get(question, QuestionStat(question, 0, 0, 0, 0.0))
        fake_stat = fake_lookup.get(question, QuestionStat(question, 0, 0, 0, 0.0))

        # For real: accuracy = no_count / total_count
        real_acc = (
            real_stat.no_count / real_stat.total_count
            if real_stat.total_count > 0 else 0.0
        )
        # For fake: accuracy = yes_count / total_count
        fake_acc = (
            fake_stat.yes_count / fake_stat.total_count
            if fake_stat.total_count > 0 else 0.0
        )
        balanced = (real_acc + fake_acc) / 2.0
        combined.append(
            {
                "question": question,
                "real_accuracy": real_acc,
                "fake_accuracy": fake_acc,
                "balanced_accuracy": balanced,
            }
        )

    combined.sort(key=lambda item: item["balanced_accuracy"], reverse=True)
    if top_k is not None:
        combined = combined[:top_k]
    return combined


def evaluation_result_to_dict(result: EvaluationResult) -> Dict[str, object]:
    """Convert an EvaluationResult into a JSON-serialisable dictionary."""
    return {
        "yes_count": result.yes_count,
        "no_count": result.no_count,
        "total_count": result.total_count,
        "accuracy": result.accuracy,
        "image_count": result.image_count,
        "question_count": result.question_count,
        "question_stats": [
            {
                "question": stat.question,
                "yes_count": stat.yes_count,
                "no_count": stat.no_count,
                "total_count": stat.total_count,
                "accuracy": stat.accuracy,  # This is 0.0, not used in ranking
            }
            for stat in result.question_stats
        ],
        "responses": [
            {
                "image_path": response.image_path,
                "question": response.question,
                "answer": response.answer,
                **({"score": response.score} if response.score is not None else {}),
            }
            for response in result.responses
        ],
    }


def concat_questions(questions: Sequence[str]) -> str:
    """Concatenate questions into a single sentence."""
    return " ".join(questions)
