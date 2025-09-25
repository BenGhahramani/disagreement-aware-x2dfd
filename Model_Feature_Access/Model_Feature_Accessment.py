import argparse
import json
import random
import sys
from typing import Any, Dict, List, Tuple


class FeatureAssessmentError(Exception):
    """Raised when the feature assessment process cannot be completed."""


def prepare_questions(raw_questions: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw_questions, list):
        raise FeatureAssessmentError("Input data must be a list of questions.")

    prepared: List[Dict[str, Any]] = []
    for index, item in enumerate(raw_questions):
        if isinstance(item, str):
            prepared.append({"id": f"question_{index + 1}", "question": item})
        elif isinstance(item, dict):
            if "question" not in item:
                raise FeatureAssessmentError(
                    f"Question object at index {index} must include a 'question' field."
                )
            question = dict(item)
            question.setdefault("id", f"question_{index + 1}")
            prepared.append(question)
        else:
            raise FeatureAssessmentError(
                f"Question entry at index {index} must be a string or JSON object."
            )
    return prepared


def cal_feature_score(questions: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], float]]:
    scored_pairs: List[Tuple[Dict[str, Any], float]] = []
    for question in questions:
        score = round(random.uniform(0, 1), 4)
        scored_pairs.append((question, score))
    return scored_pairs


def rank_questions_by_score(scored_pairs: List[Tuple[Dict[str, Any], float]]) -> List[Dict[str, Any]]:
    ranked: List[Dict[str, Any]] = []
    for question, score in sorted(scored_pairs, key=lambda pair: pair[1], reverse=True):
        ranked_question = dict(question)
        ranked_question["score"] = score
        ranked.append(ranked_question)
    return ranked


def load_questions(path: str | None) -> Any:
    if path:
        with open(path, "r", encoding="utf-8") as handle:
            raw = handle.read()
    else:
        raw = sys.stdin.read().strip()
        if not raw:
            raise FeatureAssessmentError(
                "No JSON payload supplied. Provide a file via --file or pipe data through stdin."
            )
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise FeatureAssessmentError(f"Invalid questions JSON: {exc.msg}.") from exc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assign random feature scores to questions and rank them."
    )
    parser.add_argument(
        "-f",
        "--file",
        help="Path to a JSON file that contains an array of questions (strings or objects).",
    )
    args = parser.parse_args()

    raw_questions = load_questions(args.file)
    prepared_questions = prepare_questions(raw_questions)
    scored_pairs = cal_feature_score(prepared_questions)
    ranked_questions = rank_questions_by_score(scored_pairs)

    output = {
        "question_scores": [
            {**question, "score": score} for question, score in scored_pairs
        ],
        "ranked_questions": ranked_questions,
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    try:
        main()
    except FeatureAssessmentError as exc:
        print(f"Error: {exc}")
    except Exception as exc:  # pragma: no cover - unexpected failures
        print(f"Unexpected error: {exc}")
