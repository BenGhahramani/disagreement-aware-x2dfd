"""Data loading helpers for QA evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List


def load_questions(path: Path) -> List[str]:
    """Load questions from JSON file and ensure the data is a list of strings."""
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, list):
        raise ValueError(f"Expected a list of questions in {path}")

    questions: List[str] = []
    for idx, item in enumerate(data):
        if not isinstance(item, str):
            raise ValueError(f"Question at index {idx} in {path} is not a string")
        questions.append(item)

    return questions


def load_image_paths(path: Path) -> List[str]:
    """Load image paths from JSON file with schema matching real.json/fake.json."""
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    images = data.get("images", [])
    if not isinstance(images, list):
        raise ValueError(f"Expected 'images' to be a list in {path}")

    paths: List[str] = []
    for idx, entry in enumerate(images):
        if not isinstance(entry, dict):
            raise ValueError(f"Image entry at index {idx} in {path} is not an object")
        image_path = entry.get("image_path")
        if not isinstance(image_path, str):
            raise ValueError(f"Missing or invalid 'image_path' at index {idx} in {path}")
        paths.append(image_path)

    return paths
