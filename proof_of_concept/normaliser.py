"""Parse X2DFD runner result files into normalised RunRecords.

The X2DFD evaluation runner (`eval/infer/runner.py`) writes a list of items per
input JSON, each shaped like::

    [{
      "id": "1",
      "image": "<abs path>",
      "conversations": [
        {"from": "human", "value": "<image>\\nIs this image real or fake? And the blending score is 0.812..."},
        {"from": "gpt",   "value": "fake"},
        {"from": "real score", "value": "0.1340"},
        {"from": "fake score", "value": "0.8660"}
      ]
    }]

This module reads such a file and produces one :class:`RunRecord` per file.
The POC handles the single-image case (uses the first item); the same parser
will work on multi-image files in future.

Wall-clock runtime is not emitted by the runner today, so we look up an
optional sidecar ``runtimes.json`` in the same directory, mapping
``run_name -> seconds``. A thin launcher around the real runner can write the
same sidecar for live runs without touching upstream code.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from .schema import RunRecord


_RUN_NAME_TO_EXPERTS: dict[str, list[str]] = {
    "none": [],
    "blending": ["blending"],
    "diffusion": ["diffusion"],
    "blending_diffusion": ["blending", "diffusion"],
}

# e.g. "the blending score is 0.812"  /  "the diffusion score is N/A"
_SCORE_RE = re.compile(
    r"the\s+(?P<alias>[A-Za-z0-9_]+)\s+score\s+is\s+(?P<score>[0-9]*\.?[0-9]+|N/A)",
    re.IGNORECASE,
)
_PRED_RE = re.compile(r"\b(real|fake)\b", re.IGNORECASE)
_INFERENCE_ERROR_PREFIX = "Inference error:"

_DEFAULT_FILES: list[tuple[str, str]] = [
    ("none", "demo_none.json"),
    ("blending", "demo_blending.json"),
    ("diffusion", "demo_diffusion.json"),
    ("blending_diffusion", "demo_blending_diffusion.json"),
]


def _safe_float(value: object) -> Optional[float]:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _extract_prediction(gpt_value: str) -> Optional[str]:
    if not isinstance(gpt_value, str) or not gpt_value:
        return None
    if gpt_value.strip().startswith(_INFERENCE_ERROR_PREFIX):
        return None
    match = _PRED_RE.search(gpt_value)
    if not match:
        return None
    return match.group(1).lower()


def _extract_experts_from_human(human_value: str) -> dict[str, Optional[float]]:
    """Pull e.g. ``{'blending': 0.812, 'diffusion': 0.153}`` from the prompt."""
    out: dict[str, Optional[float]] = {}
    for m in _SCORE_RE.finditer(human_value or ""):
        alias = m.group("alias").lower()
        raw = m.group("score")
        out[alias] = None if raw.upper() == "N/A" else _safe_float(raw)
    return out


def _runtime_for(result_path: Path, run_name: str) -> Optional[float]:
    """Optional sidecar lookup: ``runtimes.json`` next to the result file."""
    sidecar = result_path.parent / "runtimes.json"
    if not sidecar.exists():
        return None
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return _safe_float(data.get(run_name))


def _confidence(real: Optional[float], fake: Optional[float]) -> Optional[float]:
    if real is None or fake is None:
        return None
    if not (0.0 <= real <= 1.0 and 0.0 <= fake <= 1.0):
        return None
    return max(real, fake)


def parse_run_file(path: str | Path, run_name: str) -> RunRecord:
    """Parse one X2DFD-style result JSON into a :class:`RunRecord`.

    The caller supplies ``run_name`` since the X2DFD runner does not record
    which experts produced a given file. The convention assumed by the POC
    launcher is ``demo_<run_name>.json``.
    """
    p = Path(path)
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return RunRecord(
            run_name=run_name,
            experts_used=list(_RUN_NAME_TO_EXPERTS.get(run_name, [])),
            prediction=None,
            real_score=None,
            fake_score=None,
            confidence=None,
            explanation="",
            runtime_seconds=_runtime_for(p, run_name),
            error=f"failed to read {p.name}: {type(exc).__name__}: {exc}",
        )

    if not isinstance(payload, list) or not payload:
        return RunRecord(
            run_name=run_name,
            experts_used=list(_RUN_NAME_TO_EXPERTS.get(run_name, [])),
            prediction=None,
            real_score=None,
            fake_score=None,
            confidence=None,
            explanation="",
            runtime_seconds=_runtime_for(p, run_name),
            error=f"empty or malformed result file: {p.name}",
        )

    item = payload[0] if isinstance(payload[0], dict) else {}
    conv = item.get("conversations") or []
    human_value = ""
    gpt_value = ""
    real_score: Optional[float] = None
    fake_score: Optional[float] = None
    for turn in conv:
        if not isinstance(turn, dict):
            continue
        src = (turn.get("from") or "").strip().lower()
        val = turn.get("value")
        if src == "human" and isinstance(val, str):
            human_value = val
        elif src == "gpt" and isinstance(val, str):
            gpt_value = val
        elif src == "real score":
            real_score = _safe_float(val)
        elif src == "fake score":
            fake_score = _safe_float(val)

    error: Optional[str] = None
    if isinstance(gpt_value, str) and gpt_value.strip().startswith(_INFERENCE_ERROR_PREFIX):
        error = gpt_value.strip()

    if run_name in _RUN_NAME_TO_EXPERTS:
        experts_used = list(_RUN_NAME_TO_EXPERTS[run_name])
    else:
        experts_used = list(_extract_experts_from_human(human_value).keys())

    return RunRecord(
        run_name=run_name,
        experts_used=experts_used,
        prediction=_extract_prediction(gpt_value),
        real_score=real_score,
        fake_score=fake_score,
        confidence=_confidence(real_score, fake_score),
        explanation=gpt_value.strip(),
        runtime_seconds=_runtime_for(p, run_name),
        error=error,
    )


def load_scenario(directory: str | Path) -> dict[str, RunRecord]:
    """Load the four expected result files from a scenario directory.

    Missing files become a :class:`RunRecord` with ``error='file missing: ...'``
    so the evaluator can still classify the comparison as Failed/insufficient.
    """
    base = Path(directory)
    out: dict[str, RunRecord] = {}
    for run_name, fname in _DEFAULT_FILES:
        fpath = base / fname
        if fpath.exists():
            out[run_name] = parse_run_file(fpath, run_name)
        else:
            out[run_name] = RunRecord(
                run_name=run_name,
                experts_used=list(_RUN_NAME_TO_EXPERTS[run_name]),
                prediction=None,
                real_score=None,
                fake_score=None,
                confidence=None,
                explanation="",
                runtime_seconds=_runtime_for(fpath, run_name),
                error=f"file missing: {fpath.name}",
            )
    return out


def primary_image(records: dict[str, RunRecord], directory: str | Path) -> str:
    """Best-effort image path: take the ``image`` field from the first present run file."""
    base = Path(directory)
    for _, fname in _DEFAULT_FILES:
        fpath = base / fname
        if not fpath.exists():
            continue
        try:
            payload = json.loads(fpath.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, list) and payload and isinstance(payload[0], dict):
            img = payload[0].get("image")
            if isinstance(img, str) and img.strip():
                return img
    return "<unknown>"
