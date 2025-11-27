#!/usr/bin/env python3
"""
Shared helpers for the unified pipeline so that steps can be reused standalone.

Exposes:
- load_yaml(path) -> dict
- ensure_dir(path)
- format_score(x) -> str
- build_scored_items(dataset_json, template, provider_name, provider_kwargs) -> items
- merge_annotations(files_with_labels, out_path, use_scores, weak_cfg, question_template) -> Path

Path rules:
- Prefer JSON top-level Description as the only prefix for dataset-style JSONs
- Otherwise require absolute paths (no config-based prefix fallbacks)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple
import os
import json
import random

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

from utils.model_scoring import get_provider, resolve_abs_paths, ExpertSpec, compute_all_scores
from utils.annotation_utils import (
    build_binary_question,
    compose_labeled_response,
    build_blending_prompt,
)


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists() or yaml is None:
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def format_score(sc: Optional[float]) -> str:
    if sc is None:
        return "N/A"
    try:
        return f"{float(sc):.3f}"
    except Exception:
        return "N/A"


def _json_description(payload: Any) -> Optional[str]:
    try:
        if isinstance(payload, dict):
            d = payload.get("Description") or payload.get("description")
            if isinstance(d, str) and d.strip():
                return d.strip()
    except Exception:
        pass
    return None


def build_scored_items(
    dataset_json: Path,
    *,
    template: str,
    provider_name: str,
    provider_kwargs: Dict[str, Any],
) -> List[Dict[str, Any]]:
    with dataset_json.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    local_root = _json_description(payload)

    # Gather image paths from dataset or conversation schema
    rels: List[str] = []
    if isinstance(payload, dict) and isinstance(payload.get("images"), list):
        for it in payload["images"]:
            if isinstance(it, dict):
                p = it.get("image_path") or it.get("path")
                rels.append(p if isinstance(p, str) else "")
    else:
        items = payload if isinstance(payload, list) else []
        rels = [it.get("image", "") for it in items if isinstance(it, dict)]

    if local_root:
        abs_paths = resolve_abs_paths(rels, root_prefix=local_root)
    else:
        # Require absolute when no Description
        for p in rels:
            if p and not os.path.isabs(p):
                raise SystemExit(f"Missing Description while containing relative paths in {dataset_json}: {p}")
        abs_paths = resolve_abs_paths(rels, root_prefix=None)

    provider = get_provider(provider_name, **provider_kwargs)
    score_map = provider.compute_scores(abs_paths)

    items_out: List[Dict[str, Any]] = []
    for idx, abs_p in enumerate(abs_paths, start=1):
        sr = score_map.get(abs_p)
        sc = None if sr is None else sr.score
        alias_val = (provider_kwargs.get("alias", "Blending") or "").strip().lower()
        q = template.format(alias=alias_val, score=format_score(sc))
        items_out.append({
            "id": str(idx),
            "image": abs_p,
            "conversations": [
                {"from": "human", "value": q},
                {"from": "gpt", "value": ""},
            ],
        })
    return items_out


def merge_annotations(
    files_with_labels: List[Tuple[Path, str]],
    out_path: Path,
    *,
    use_scores: bool,
    weak_cfg: Dict[str, Any],
    question_template: str,
) -> Path:
    """Merge per-dataset annotations into a unified training JSON with normalization.

    For each input dataset file (a list of conversation items), normalize to:
      - HUMAN: "<image>\n" + binary question or scored question_template
      - GPT:   "This image is real/fake." + original explanation

    Then shuffle all items and reassign sequential string ids starting from 1.
    """
    normalized: List[Dict[str, Any]] = []
    human_binary = f"<image>\n{build_binary_question()}"

    # Prepare weak provider(s) once if using scores
    provider = None
    # allow custom provider, default to blending for backward-compat
    provider_name = (weak_cfg.get("provider") or "blending").strip().lower()
    # support multi experts when weak_cfg carries 'weak_supplies'
    experts_cfg: List[Dict[str, Any]] = list(weak_cfg.get("weak_supplies") or []) if isinstance(weak_cfg, dict) else []
    multi_mode = use_scores and bool(experts_cfg)
    # default alias per provider; aligner -> Aligner, blending -> Blending
    is_diff = provider_name in ("aligner", "diffusion", "diffusion_detector", "diffdet")
    alias = weak_cfg.get("alias") or ("Diffusion" if is_diff else "Blending")
    metric_word = "diffusion" if is_diff else "blending"
    lo = float(((weak_cfg.get("thresholds") or {}).get("lo")) if isinstance(weak_cfg.get("thresholds"), dict) else weak_cfg.get("lo", 0.3))
    hi = float(((weak_cfg.get("thresholds") or {}).get("hi")) if isinstance(weak_cfg.get("thresholds"), dict) else weak_cfg.get("hi", 0.7))
    if use_scores and (not multi_mode):
        try:
            if is_diff:
                provider = get_provider(
                    provider_name,
                    weights_dir=weak_cfg.get("weights_dir") or "",
                    model=weak_cfg.get("model") or "",
                    device=None,
                    # allow batching knobs for diffusion/aligner single-expert path
                    batch_size=int(weak_cfg.get("batch_size") or 256),
                    num_workers=int(weak_cfg.get("num_workers") or 4),
                    pin_memory=bool(weak_cfg.get("pin_memory") if weak_cfg.get("pin_memory") is not None else True),
                    # optional compile knobs (safe by default)
                    compile_model=bool(weak_cfg.get("compile_model") or False),
                    compile_mode=str(weak_cfg.get("compile_mode") or "safe"),
                )
            else:
                provider = get_provider(
                    "blending",
                    model_name=weak_cfg.get("model_name") or "swinv2_base_window16_256",
                    weights_path=weak_cfg.get("weights_path") or "weights/blending_models/best_gf.pth",
                    img_size=int(weak_cfg.get("img_size") or 256),
                    num_class=int(weak_cfg.get("num_class") or 2),
                    device=None,
                    batch_size=int(weak_cfg.get("batch_size") or 64),
                    num_workers=int(weak_cfg.get("num_workers") or 4),
                    pin_memory=bool(weak_cfg.get("pin_memory") if weak_cfg.get("pin_memory") is not None else True),
                )
        except Exception:
            provider = None
            use_scores = False

    for fp, lbl in files_with_labels:
        try:
            with fp.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        if not isinstance(data, list):
            continue

        # If scoring is enabled, compute scores for this dataset in batch
        dataset_scores: Dict[str, Any] = {}
        abs_paths_for_score: List[str] = []
        if use_scores:
            rel_or_abs = [it.get("image") for it in data if isinstance(it, dict)]
            # Enforce absolute paths; no prefixing here
            abs_paths_for_score = []
            for p in rel_or_abs:
                if isinstance(p, str) and os.path.isabs(p):
                    abs_paths_for_score.append(os.path.normpath(p))
                else:
                    raise ValueError(f"Non-absolute image path encountered in merged annotations {fp}: {p}")
            if multi_mode:
                # Build ExpertSpec list from experts_cfg
                expert_specs: List[ExpertSpec] = []
                # Record per-expert thresholds for later conditional notes
                expert_thresholds: List[Tuple[float, float]] = []  # (lo, hi)
                for e in experts_cfg:
                    prov = (e.get("provider") or "").strip().lower()
                    alias_e = e.get("alias") or ("Diffusion" if prov in ("aligner","diffusion") else ("Blending" if prov == "blending" else prov.title()))
                    kwargs_e: Dict[str, Any] = {}
                    if prov in ("aligner","diffusion","diffusion_detector","diffdet"):
                        # add batching knobs for diffusion/aligner in multi-expert mode
                        kwargs_e = {
                            "weights_dir": e.get("weights_dir") or "",
                            "model": e.get("model") or "",
                            "device": None,
                            "batch_size": int(e.get("batch_size") or 256),
                            "num_workers": int(e.get("num_workers") or 4),
                            "pin_memory": bool(e.get("pin_memory") if e.get("pin_memory") is not None else True),
                            # compile knobs per expert
                            "compile_model": bool(e.get("compile_model") or False),
                            "compile_mode": str(e.get("compile_mode") or "safe"),
                        }
                    elif prov == "blending":
                        kwargs_e = {
                            "model_name": e.get("model_name") or "swinv2_base_window16_256",
                            "weights_path": e.get("weights_path") or "weights/blending_models/best_gf.pth",
                            "img_size": int(e.get("img_size") or 256),
                            "num_class": int(e.get("num_class") or 2),
                            "device": None,
                            "batch_size": int(e.get("batch_size") or 64),
                            "num_workers": int(e.get("num_workers") or 4),
                            "pin_memory": bool(e.get("pin_memory") if e.get("pin_memory") is not None else True),
                        }
                    else:
                        kwargs_e = dict(e)
                        kwargs_e.pop("provider", None)
                        kwargs_e.pop("alias", None)
                    expert_specs.append(ExpertSpec(provider=prov, alias=alias_e, kwargs=kwargs_e))
                    thr = e.get("thresholds") or {}
                    lo_e = float(thr.get("lo", lo)) if isinstance(thr, dict) else lo
                    hi_e = float(thr.get("hi", hi)) if isinstance(thr, dict) else hi
                    expert_thresholds.append((lo_e, hi_e))
                try:
                    dataset_scores = compute_all_scores(expert_specs, abs_paths_for_score)  # {path: [ExpertScore,...]}
                except Exception:
                    dataset_scores = {ap: [] for ap in abs_paths_for_score}
            else:
                if provider is not None:
                    try:
                        score_map = provider.compute_scores(abs_paths_for_score)
                    except Exception:
                        score_map = {}
                else:
                    score_map = {}
                dataset_scores = {ap: (score_map.get(ap).score if score_map.get(ap) is not None else None) for ap in abs_paths_for_score}

        for idx_in_ds, it in enumerate(data):
            if not isinstance(it, dict):
                continue
            image = it.get("image")
            conv = (it.get("conversations") or []) if isinstance(it.get("conversations"), list) else []
            # Find first GPT answer to preserve original explanation
            original_answer = ""
            for turn in conv:
                if isinstance(turn, dict) and turn.get("from") == "gpt":
                    original_answer = turn.get("value") or ""
                    break
            human_text = human_binary
            gpt_value = compose_labeled_response(lbl, original_answer)
            if use_scores and abs_paths_for_score:
                abs_p = abs_paths_for_score[idx_in_ds] if idx_in_ds < len(abs_paths_for_score) else None
                if multi_mode:
                    # Build multi-expert tail: "The blending score is X, and the aligner score is Y."
                    pairs: List[Tuple[str, Optional[float]]] = []
                    if abs_p is not None and isinstance(dataset_scores.get(abs_p), list):
                        for es in dataset_scores.get(abs_p):
                            pairs.append((es.alias, es.score))
                    parts = [f"the {(a or '').strip().lower()} score is {format_score(s)}" for a, s in pairs]
                    if parts:
                        if len(parts) == 1:
                            tail = " And " + parts[0] + "."
                        else:
                            tail = " And " + ", ".join(parts[:-1]) + ", and " + parts[-1] + "."
                        human_text = "<image>\nIs this image real or fake?" + tail
                    # Conditional supporting notes (per expert, concat; group when multiple support)
                    supporters: List[str] = []
                    if abs_p is not None and isinstance(dataset_scores.get(abs_p), list):
                        es_list = dataset_scores.get(abs_p)
                        for idx_e, es in enumerate(es_list):
                            sc = es.score
                            if sc is None:
                                continue
                            lo_e, hi_e = expert_thresholds[idx_e] if idx_e < len(expert_thresholds) else (lo, hi)
                            try:
                                scf = float(sc)
                            except Exception:
                                continue
                            if lbl == "real" and scf <= lo_e:
                                supporters.append(es.alias)
                            elif lbl == "fake" and scf >= hi_e:
                                supporters.append(es.alias)
                    if supporters:
                        # Compose artifact-style sentence per label
                        names = [s.strip().lower() for s in supporters if isinstance(s, str) and s.strip()]
                        if len(names) == 1:
                            conj = names[0]
                        elif len(names) == 2:
                            conj = f"{names[0]} and {names[1]}"
                        else:
                            conj = ", ".join(names[:-1]) + f", and {names[-1]}"
                        if lbl == "fake":
                            gpt_value = gpt_value + f" Additionally, the image shows obvious {conj} artifacts supporting fake."
                        else:  # real
                            gpt_value = gpt_value + f" Additionally, the image shows very few {conj} artifacts supporting real."
                else:
                    sc = None
                    if abs_p is not None:
                        sc = dataset_scores.get(abs_p)
                    try:
                        human_text = question_template.format(alias=(alias or '').strip().lower(), score=format_score(sc))
                    except Exception:
                        alias_l = (alias or '').strip().lower()
                        human_text = (
                            f"<image>\nIs this image real or fake? And the {alias_l} score is {format_score(sc)}."
                        )
                    if sc is not None:
                        try:
                            scf = float(sc)
                            if lbl == "real" and scf <= lo:
                                gpt_value = gpt_value + f" Additionally, the {alias} detector indicates very few artifacts supporting real."
                            elif lbl == "fake" and scf >= hi:
                                gpt_value = gpt_value + f" Additionally, the {alias} detector indicates a lot of artifacts supporting fake."
                        except Exception:
                            pass
            else:
                try:
                    human_text = question_template.format(alias=(alias or '').strip().lower(), score=format_score(None))
                except Exception:
                    # Metric-specific fallback (N/A)
                    alias_l = (alias or '').strip().lower()
                    human_text = (
                        f"<image>\nIs this image real or fake? And the {alias_l} score is N/A."
                    )
            normalized.append({
                "id": "0",
                "image": image,
                "conversations": [
                    {"from": "human", "value": human_text},
                    {"from": "gpt", "value": gpt_value},
                ],
            })

    if normalized:
        random.shuffle(normalized)
    for i, it in enumerate(normalized, start=1):
        it["id"] = str(i)

    ensure_dir(out_path.parent)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(normalized, f, ensure_ascii=False, indent=2)
    return out_path
