"""Inference stubs for generating answers and narrative responses."""

from __future__ import annotations

import random


def infer_answer(rng: random.Random, image_path: str, question: str) -> str:
    """Placeholder inference that randomly returns 'real' or 'fake'."""
    return rng.choice(["real", "fake"])



def describe_image_answer(
    rng: random.Random,
    dataset_label: str,
    question: str,
    answer: str,
) -> str:
    """Compose a narrative response for a given dataset label and answer."""
    prefix_map = {
        "real": "This image is real.",
        "fake": "This image is fake.",
        "unlabeled": "This image is unlabeled.",
    }
    prefix = prefix_map.get(dataset_label, prefix_map["unlabeled"])

    filler = rng.choice(
        [
            "Answer generated for demonstration purposes.",
            "Placeholder response pending model integration.",
            "This sentence illustrates the intended narrative output.",
        ]
    )

    return f"{prefix} {filler} Question: {question} Answer: {answer}."
