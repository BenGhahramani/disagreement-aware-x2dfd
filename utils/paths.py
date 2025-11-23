"""Centralized path helpers for the refactored layout."""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS_ROOT = PROJECT_ROOT / "datasets"
WEIGHTS_ROOT = PROJECT_ROOT / "weights"
OUTPUT_ROOT = PROJECT_ROOT / "eval" / "outputs"
TRAIN_OUTPUT_ROOT = PROJECT_ROOT / "train" / "outputs"
CONFIG_ROOT_EVAL = PROJECT_ROOT / "eval" / "configs"
CONFIG_ROOT_TRAIN = PROJECT_ROOT / "train" / "configs"
PROMPTS_ROOT = DATASETS_ROOT / "prompts"

# Allow environment overrides for CI/experiments
PROJECT_ROOT = Path(os.getenv("X2DFD_PROJECT_ROOT", PROJECT_ROOT))
DATASETS_ROOT = Path(os.getenv("X2DFD_DATASETS", DATASETS_ROOT))
WEIGHTS_ROOT = Path(os.getenv("X2DFD_WEIGHTS", WEIGHTS_ROOT))
OUTPUT_ROOT = Path(os.getenv("X2DFD_OUTPUT", OUTPUT_ROOT))
TRAIN_OUTPUT_ROOT = Path(os.getenv("X2DFD_TRAIN_OUTPUT", TRAIN_OUTPUT_ROOT))
CONFIG_ROOT_EVAL = Path(os.getenv("X2DFD_EVAL_CONFIG", CONFIG_ROOT_EVAL))
CONFIG_ROOT_TRAIN = Path(os.getenv("X2DFD_TRAIN_CONFIG", CONFIG_ROOT_TRAIN))
PROMPTS_ROOT = Path(os.getenv("X2DFD_PROMPTS", PROMPTS_ROOT))

def ensure_core_dirs() -> None:
    """Create commonly-used directories if missing at runtime."""
    for path in (DATASETS_ROOT, WEIGHTS_ROOT, OUTPUT_ROOT, TRAIN_OUTPUT_ROOT):
        path.mkdir(parents=True, exist_ok=True)
