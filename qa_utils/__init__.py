"""Utility modules supporting the QA accuracy evaluation pipeline."""

from .loader import load_image_paths, load_questions
from .evaluation import (
    QAResponse,
    QuestionStat,
    EvaluationResult,
    evaluate_dataset,
    compute_balanced_accuracy,
    build_question_ranking,
    concat_questions,
    evaluation_result_to_dict,
)
from .inference import (
    single_image_infer,
    describe_image_answer,
)
from .persistence import (
    RunPaths,
    prepare_run_paths,
    write_json,
    write_text,
    append_run_log,
)

__all__ = [
    "load_image_paths",
    "load_questions",
    "QAResponse",
    "QuestionStat",
    "EvaluationResult",
    "evaluate_dataset",
    "compute_balanced_accuracy",
    "build_question_ranking",
    "concat_questions",
    "evaluation_result_to_dict",
    "single_image_infer",
    "describe_image_answer",
    "RunPaths",
    "prepare_run_paths",
    "write_json",
    "write_text",
    "append_run_log",
]
