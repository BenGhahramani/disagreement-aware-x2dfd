"""Regression tests for expert kwargs built by eval/infer/runner.py.

No model is loaded: ``compute_all_scores`` is monkeypatched so the test only
observes the ``ExpertSpec`` list the runner constructs from a config.

The defect these cover: ``int(e.get("num_workers") or 4)`` silently rewrote a
configured ``num_workers: 0`` to 4. On Windows that made the blending and
diffusion experts unusable, because DataLoader worker processes are spawned and
cannot pickle the detectors' locally-defined Dataset classes
(``AttributeError: Can't pickle local object
'BlendingDetector.infer.<locals>._ImageDataset'``). Zero workers is the only
setting that avoids the spawn entirely.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest

import eval.infer.runner as runner
from utils.model_scoring import ExpertSpec

pytestmark = pytest.mark.unit


def build_specs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, experts_cfg: List[Dict[str, Any]]):
    """Run the conversation builder and return the ExpertSpecs it created."""
    captured: List[ExpertSpec] = []

    def fake_compute_all_scores(experts: List[ExpertSpec], image_paths: List[str]):
        captured.extend(experts)
        return {p: [] for p in image_paths}

    monkeypatch.setattr(runner, "compute_all_scores", fake_compute_all_scores)
    payload = {"Description": str(tmp_path), "images": [{"image_path": "face.jpg"}]}
    runner._build_conversations_multi(payload, experts_cfg=experts_cfg)
    return {spec.provider: spec for spec in captured}


@pytest.mark.parametrize(
    "provider, extra",
    [
        ("blending", {"model_name": "swinv2_base_window16_256", "weights_path": "w.pth"}),
        ("diffusion_detector", {"weights_dir": "weights/", "model": "ours-sync"}),
    ],
)
def test_zero_num_workers_survives(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, provider: str, extra: Dict[str, Any]
) -> None:
    specs = build_specs(
        monkeypatch, tmp_path, [{"provider": provider, "num_workers": 0, **extra}]
    )
    assert specs[provider].kwargs["num_workers"] == 0


@pytest.mark.parametrize("provider", ["blending", "diffusion_detector"])
def test_absent_num_workers_still_defaults_to_four(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, provider: str
) -> None:
    specs = build_specs(monkeypatch, tmp_path, [{"provider": provider}])
    assert specs[provider].kwargs["num_workers"] == 4


@pytest.mark.parametrize("provider", ["blending", "diffusion_detector"])
def test_explicit_num_workers_is_passed_through(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, provider: str
) -> None:
    specs = build_specs(monkeypatch, tmp_path, [{"provider": provider, "num_workers": 3}])
    assert specs[provider].kwargs["num_workers"] == 3


@pytest.mark.parametrize(
    "value, expected", [(0, 0), (None, 7), ("2", 2), ("nonsense", 7), (3.9, 3)]
)
def test_int_or_default(value: Any, expected: int) -> None:
    assert runner._int_or_default(value, 7) == expected
