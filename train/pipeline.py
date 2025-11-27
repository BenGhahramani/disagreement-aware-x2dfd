#!/usr/bin/env python3
"""
Unified pipeline runner: base-model QA-style annotation -> optional LoRA training -> LoRA testing.

Flow:
  1) Base model generates explanations for training datasets (feature_annotation.process_json_file)
  2) Merge per-dataset annotations into a single training JSON
  3) Optionally launch LoRA training (model_train.py via deepspeed or python)
  4) Test on other datasets using base + LoRA adapter (utils.lora_inference.lora_infer_conversation_items)

Notes:
  - Reuses existing functions without modifying them
  - Writes all artefacts under results/pipeline/runs/pipeline_<timestamp>/
  - Supports Dataset JSON schema: {"Description": "/abs/root", "images": [{"image_path": "rel/or/abs"}, ...]}
    and conversation-list schema (id/image/conversations)
"""

from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

# Reuse existing logic
from eval.tools.feature_annotation import (
    load_prompt_template,
    process_json_file,
)
from utils.model_scoring import (
    get_provider,
    resolve_abs_paths,
    ExpertSpec,
    compute_all_scores,
)
from utils.lora_inference import lora_infer_conversation_items
from utils.annotation_utils import build_binary_question, compose_labeled_response, build_blending_prompt
from utils.paths import (
    CONFIG_ROOT_EVAL,
    CONFIG_ROOT_TRAIN,
    DATASETS_ROOT,
    PROJECT_ROOT,
    PROMPTS_ROOT,
    TRAIN_OUTPUT_ROOT,
    ensure_core_dirs,
)

ensure_core_dirs()

DEFAULT_CONFIG = CONFIG_ROOT_TRAIN / "config.yaml"
# Use the same default as test.sh / eval.infer.runner to avoid config drift
DEFAULT_INFER_CONFIG = CONFIG_ROOT_EVAL / "infer_config.yaml"


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists() or yaml is None:
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}

def _parse_experts(spec: Optional[str]) -> Optional[set[str]]:
    """Parse comma-separated experts to a lowercase set.

    - None -> None (do not filter)
    - "" or "none" -> empty set (disable all)
    - else -> {provider or alias, ...}
    """
    if spec is None:
        return None
    s = spec.strip()
    if not s or s.lower() == "none":
        return set()
    return {t.strip().lower() for t in s.split(',') if t.strip()}

def _filter_experts(cfg_list: List[Dict[str, Any]], names: Optional[set[str]]) -> List[Dict[str, Any]]:
    if names is None:
        return cfg_list
    if len(names) == 0:
        return []
    out: List[Dict[str, Any]] = []
    for e in cfg_list:
        prov = str(e.get('provider') or '').strip().lower()
        alias = str(e.get('alias') or '').strip().lower()
        if prov in names or alias in names:
            out.append(e)
    return out

def _experts_slug(cfg_list: List[Dict[str, Any]]) -> str:
    if not cfg_list:
        return 'base'
    toks: List[str] = []
    for e in cfg_list:
        alias = (e.get('alias') or e.get('provider') or 'expert')
        slug = ''.join(ch if ch.isalnum() else '-' for ch in str(alias).lower()).strip('-')
        if slug:
            toks.append(slug)
    return '+'.join(toks) if toks else 'base'

def _coerce_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _resolve_data_entry(entry: str, data_root: Path) -> Path:
    path = Path(entry).expanduser()
    if path.is_absolute():
        return path
    return (data_root / path).resolve()


def _resolve_cli_path(entry: Optional[str]) -> Optional[Path]:
    if entry is None:
        return None
    path = Path(entry).expanduser()
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _list_annotation_files(directory: Path) -> List[Path]:
    if not directory.exists():
        return []
    files = sorted(p for p in directory.glob("*_annotations.json") if p.is_file())
    if not files:
        files = sorted(p for p in directory.glob("*.json") if p.is_file())
    return files


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _abs_from_desc(rel: str, desc: Optional[str], fallback_root: Optional[str]) -> str:
    if os.path.isabs(rel):
        return os.path.normpath(rel)
    root = desc or fallback_root or ""
    return os.path.normpath(os.path.join(root, rel))


def _format_score(sc: Optional[float]) -> str:
    if sc is None:
        return "N/A"
    try:
        return f"{float(sc):.3f}"
    except Exception:
        return "N/A"


def _format_multi_scores(pairs: List[Tuple[str, Optional[float]]]) -> str:
    """Return a tail string like: " And the blending score is 0.812, and the aligner score is 0.153."

    pairs: list of (alias, score). Alias is rendered in lower-case for style.
    """
    parts: List[str] = []
    for alias, sc in pairs:
        name = (alias or "").strip().lower() or "expert"
        parts.append(f"the {name} score is {_format_score(sc)}")
    if not parts:
        return ""
    if len(parts) == 1:
        return " And " + parts[0] + "."
    return " And " + ", ".join(parts[:-1]) + ", and " + parts[-1] + "."


def _build_scored_items_multi(
    dataset_json: Path,
    *,
    image_root_prefix: Optional[str],
    experts_cfg: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Build conversation items using multiple experts in one prompt.

    The HUMAN becomes:
      "<image>\nIs this image real or fake? The blending score is X, and the aligner score is Y."
    """
    with dataset_json.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    # Root resolution (same rule as single)
    local_root: Optional[str] = None
    try:
        if isinstance(payload, dict):
            desc = payload.get("Description") or payload.get("description")
            if isinstance(desc, str) and desc.strip():
                local_root = desc.strip()
    except Exception:
        pass

    # Collect paths
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
        for p in rels:
            if p and not os.path.isabs(p):
                raise SystemExit(f"Missing Description in {dataset_json} while containing relative paths: {p}")
        abs_paths = resolve_abs_paths(rels, root_prefix=None)

    # Build ExpertSpec list
    experts: List[ExpertSpec] = []
    for e in experts_cfg:
        prov = (e.get("provider") or "").strip().lower()
        alias = e.get("alias") or ("Diffusion" if prov in ("aligner","diffusion") else ("Blending" if prov == "blending" else prov.title()))
        kwargs: Dict[str, Any] = {}
        if prov in ("aligner","diffusion"):
            kwargs = {
                "weights_dir": e.get("weights_dir") or "",
                "model": e.get("model") or "",
                "device": None,
                # Optional batching knobs for diffusion/aligner provider
                "batch_size": int(e.get("batch_size") or 64),
                "num_workers": int(e.get("num_workers") or 4),
                "pin_memory": bool(e.get("pin_memory") if e.get("pin_memory") is not None else True),
                # Optional compile knobs for Aligner
                "compile_model": bool(e.get("compile_model") or False),
                "compile_mode": str(e.get("compile_mode") or "safe"),
            }
        elif prov == "blending":
            kwargs = {
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
            # Fallback: let provider resolve unspecified keys
            kwargs = dict(e)
            kwargs.pop("provider", None)
            kwargs.pop("alias", None)
        experts.append(ExpertSpec(provider=prov, alias=alias, kwargs=kwargs))

    # Compute scores across experts
    score_map = compute_all_scores(experts, abs_paths)

    items_out: List[Dict[str, Any]] = []
    for idx, abs_p in enumerate(abs_paths, start=1):
        pairs = [(experts[i].alias, (score_map.get(abs_p) or [])[i].score if score_map.get(abs_p) and i < len(score_map[abs_p]) else None)
                 for i in range(len(experts))]
        # Build human prompt
        base = "<image>\nIs this image real or fake?"
        tail = _format_multi_scores(pairs)
        q = base + tail
        items_out.append({
            "id": str(idx),
            "image": abs_p,
            "conversations": [
                {"from": "human", "value": q},
                {"from": "gpt", "value": ""},
            ],
        })
    return items_out

def _build_scored_items(
    dataset_json: Path,
    *,
    image_root_prefix: Optional[str],
    template: str,
    provider_name: str,
    provider_kwargs: Dict[str, Any],
) -> List[Dict[str, Any]]:
    with dataset_json.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    # JSON-level Description defines root (no config fallback)
    local_root: Optional[str] = None
    try:
        if isinstance(payload, dict):
            desc = payload.get("Description") or payload.get("description")
            if isinstance(desc, str) and desc.strip():
                local_root = desc.strip()
    except Exception:
        pass

    # Collect rel paths from Dataset schema
    rels: List[str] = []
    if isinstance(payload, dict) and isinstance(payload.get("images"), list):
        for it in payload["images"]:
            if isinstance(it, dict):
                p = it.get("image_path") or it.get("path")
                rels.append(p if isinstance(p, str) else "")
    else:
        # Conversation schema; keep image as-is
        items = payload if isinstance(payload, list) else []
        rels = [it.get("image", "") for it in items if isinstance(it, dict)]

    # Require either Description for relative paths (dataset schema) or absolute paths already (conversation schema)
    if local_root:
        abs_paths = resolve_abs_paths(rels, root_prefix=local_root)
    else:
        # Ensure all non-empty paths are absolute
        for p in rels:
            if p and not os.path.isabs(p):
                raise SystemExit(f"Missing Description in {dataset_json} while containing relative paths: {p}")
        abs_paths = resolve_abs_paths(rels, root_prefix=None)
    provider = get_provider(provider_name, **provider_kwargs)
    score_map = provider.compute_scores(abs_paths)

    items_out: List[Dict[str, Any]] = []
    for idx, (rel, abs_p) in enumerate(zip(rels, abs_paths), start=1):
        sr = score_map.get(abs_p)
        sc = None if sr is None else sr.score
        alias_val = (provider_kwargs.get("alias", "Blending") or "").strip().lower()
        q = template.format(alias=alias_val, score=_format_score(sc))
        items_out.append({
            "id": str(idx),
            "image": abs_p,  # normalize to absolute
            "conversations": [
                {"from": "human", "value": q},
                {"from": "gpt", "value": ""},
            ],
        })
    return items_out


def _merge_annotations(
    files_with_labels: List[Tuple[Path, str]],
    out_path: Path,
    *,
    use_scores: bool,
    weak_cfg: Dict[str, Any],
    image_root_prefix: Optional[str],
    question_template: str,
) -> Path:
    """Merge per-dataset annotations into a unified training JSON with normalization.

    For each input dataset file (a list of conversation items), normalize to:
      - HUMAN: "<image>\n" + binary question (Is this image real or fake?)
      - GPT:   "This image is real/fake." + original explanation

    Then shuffle all items and reassign sequential string ids starting from 1.
    """
    normalized: List[Dict[str, Any]] = []
    human_binary = f"<image>\n{build_binary_question()}"

    # Prepare weak provider once if using scores
    provider = None
    provider_name = (weak_cfg.get("provider") or "blending").strip().lower()
    is_diff = provider_name in ("aligner", "diffusion", "diffusion_detector", "diffdet")
    alias = weak_cfg.get("alias") or ("Diffusion" if is_diff else "Blending")
    metric_word = "diffusion" if is_diff else "blending"
    lo = float(((weak_cfg.get("thresholds") or {}).get("lo")) if isinstance(weak_cfg.get("thresholds"), dict) else weak_cfg.get("lo", 0.3))
    hi = float(((weak_cfg.get("thresholds") or {}).get("hi")) if isinstance(weak_cfg.get("thresholds"), dict) else weak_cfg.get("hi", 0.7))
    # multi-expert when weak_cfg contains 'weak_supplies' list
    experts_cfg: List[Dict[str, Any]] = list(weak_cfg.get("weak_supplies") or []) if isinstance(weak_cfg, dict) else []
    multi_mode = use_scores and bool(experts_cfg)
    if use_scores and (not multi_mode):
        try:
            if is_diff:
                provider = get_provider(
                    provider_name,
                    weights_dir=weak_cfg.get("weights_dir") or "",
                    model=weak_cfg.get("model") or "",
                    device=None,
                    # bump defaults and allow overrides for diffusion/aligner
                    batch_size=int(weak_cfg.get("batch_size") or 256),
                    num_workers=int(weak_cfg.get("num_workers") or 4),
                    pin_memory=bool(weak_cfg.get("pin_memory") if weak_cfg.get("pin_memory") is not None else True),
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
        except Exception as e:
            print(f"[WARN] Failed to initialize weak provider, falling back to no-score merge: {e}")
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
        # Compute scores for this dataset in batch if needed
        dataset_scores: Dict[str, Any] = {}
        abs_paths_for_score: List[str] = []
        expert_thresholds: List[Tuple[float, float]] = []
        if use_scores:
            rel_or_abs = [it.get("image") for it in data if isinstance(it, dict)]
            # Enforce absolute paths in merged annotations; no config-based prefixing
            abs_paths_for_score = []
            for p in rel_or_abs:
                if isinstance(p, str) and os.path.isabs(p):
                    abs_paths_for_score.append(os.path.normpath(p))
                else:
                    raise ValueError(f"Non-absolute image path encountered in merged annotations {fp}: {p}")
            if multi_mode:
                # Build ExpertSpec list and per-expert thresholds
                expert_specs: List[ExpertSpec] = []
                for e in experts_cfg:
                    prov = (e.get("provider") or "").strip().lower()
                    alias_e = e.get("alias") or ("Diffusion" if prov in ("aligner","diffusion") else ("Blending" if prov == "blending" else prov.title()))
                    kwargs_e: Dict[str, Any] = {}
                    if prov in ("aligner","diffusion","diffusion_detector","diffdet"):
                        # add batching knobs for diffusion/aligner experts in multi-expert training
                        kwargs_e = {
                            "weights_dir": e.get("weights_dir") or "",
                            "model": e.get("model") or "",
                            "device": None,
                            "batch_size": int(e.get("batch_size") or 256),
                            "num_workers": int(e.get("num_workers") or 4),
                            "pin_memory": bool(e.get("pin_memory") if e.get("pin_memory") is not None else True),
                            # Optional compile knobs per expert
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
                    dataset_scores = compute_all_scores(expert_specs, abs_paths_for_score)
                except Exception as e:
                    print(f"[WARN] Multi-expert scoring failed for {fp.name}: {e}")
                    dataset_scores = {ap: [] for ap in abs_paths_for_score}
            else:
                try:
                    score_map = provider.compute_scores(abs_paths_for_score) if provider is not None else {}
                except Exception as e:
                    print(f"[WARN] Weak provider scoring failed for {fp.name}: {e}")
                    score_map = {}
                dataset_scores = {ap: (score_map.get(ap).score if score_map.get(ap) is not None else None) for ap in abs_paths_for_score}

        # Iterate and normalize items
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
            # Determine score if enabled
            human_text = human_binary
            gpt_value = compose_labeled_response(lbl, original_answer)
            if use_scores and abs_paths_for_score:
                abs_p = abs_paths_for_score[idx_in_ds] if idx_in_ds < len(abs_paths_for_score) else None
                if multi_mode:
                    # Multi-expert human tail
                    pairs: List[Tuple[str, Optional[float]]] = []
                    if abs_p is not None and isinstance(dataset_scores.get(abs_p), list):
                        for es in dataset_scores.get(abs_p):
                            pairs.append((es.alias, es.score))
                    parts = [f"the {(a or '').strip().lower()} score is {_format_score(s)}" for a, s in pairs]
                    if parts:
                        if len(parts) == 1:
                            tail = " And " + parts[0] + "."
                        else:
                            tail = " And " + ", ".join(parts[:-1]) + ", and " + parts[-1] + "."
                        human_text = "<image>\nIs this image real or fake?" + tail
                    # Grouped supporters
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
                        names = [s.strip().lower() for s in supporters if isinstance(s, str) and s.strip()]
                        if len(names) == 1:
                            conj = names[0]
                        elif len(names) == 2:
                            conj = f"{names[0]} and {names[1]}"
                        else:
                            conj = ", ".join(names[:-1]) + f", and {names[-1]}"
                        if lbl == "fake":
                            gpt_value = gpt_value + f" Additionally, the image shows obvious {conj} artifacts supporting fake."
                        else:
                            gpt_value = gpt_value + f" Additionally, the image shows very few {conj} artifacts supporting real."
                else:
                    sc = None
                    if abs_p is not None:
                        sc = dataset_scores.get(abs_p)
                    try:
                        human_text = question_template.format(alias=alias, score=_format_score(sc))
                    except Exception:
                        human_text = (
                            f"<image>\nIs this image real or fake? And by observation of {alias} expert, "
                            f"the {metric_word} score is {_format_score(sc)}."
                        )
                    supporter = None
                    if sc is not None:
                        try:
                            scf = float(sc)
                            if lbl == "real" and scf <= lo:
                                supporter = alias
                            elif lbl == "fake" and scf >= hi:
                                supporter = alias
                        except Exception:
                            supporter = None
                    if supporter:
                        name = supporter.strip().lower()
                        if lbl == "fake":
                            gpt_value = gpt_value + f" Additionally, the image shows obvious {name} artifacts supporting fake."
                        else:
                            gpt_value = gpt_value + f" Additionally, the image shows very few {name} artifacts supporting real."
            else:
                # No scores in training merge; still reuse template with N/A score if it has placeholders
                try:
                    human_text = question_template.format(alias=alias, score=_format_score(None))
                except Exception:
                    human_text = (
                        f"<image>\nIs this image real or fake? And by observation of {alias} expert, "
                        f"the {metric_word} score is N/A."
                    )
            normalized.append({
                "id": "0",  # temporary; will be reassigned
                "image": image,
                "conversations": [
                    {"from": "human", "value": human_text},
                    {"from": "gpt", "value": gpt_value},
                ],
            })

    # Shuffle to avoid dataset-blocking bias
    if normalized:
        random.shuffle(normalized)

    # Reassign ids sequentially
    for i, it in enumerate(normalized, start=1):
        it["id"] = str(i)

    _ensure_dir(out_path.parent)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(normalized, f, ensure_ascii=False, indent=2)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Unified (staged): annotate -> weak -> train -> test")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Path to uniconfig.yaml (unified)")
    ap.add_argument(
        "--infer-config",
        type=Path,
        default=DEFAULT_INFER_CONFIG,
        help="Optional separate infer config; defaults to eval/configs/infer_config.yaml",
    )
    # Dynamic expert selection
    ap.add_argument("--experts", default=None, help="Comma-separated experts (provider or alias) for training merge; 'none' disables scores")
    ap.add_argument("--test-experts", default=None, help="Comma-separated experts for test stage; default to --experts when omitted")
    # Optional ckpt tag to customize output dir name (appended to slug)
    ap.add_argument("--ckpt-tag", default=None, help="Optional tag appended to ckpt dir name, e.g., v1 or expA")

    # Staging controls
    ap.add_argument("--phase", choices=["annotate", "weak", "train", "test", "full"], default="full",
                    help="Run a specific stage or the full pipeline. 'weak' supplements experts on existing annotations.")
    ap.add_argument("--force-annotate", action="store_true", help="Re-run annotate stage even if .annotations.done exists")
    ap.add_argument("--force-weak", action="store_true", help="Re-run weak stage even if .weak.done exists")
    ap.add_argument("--out-weak-json", type=Path, default=None, help="Output merged JSON for weak stage (default: train/train_weak.json)")

    # Train inputs
    ap.add_argument("--train-jsons", nargs="*", default=None, help="Training dataset JSONs (Dataset or conversation schema)")
    ap.add_argument("--train-labels", nargs="*", default=None, help="Labels for train jsons (real/fake), auto if omitted")
    ap.add_argument("--max-train-items", type=int, default=None, help="Limit items per train dataset")

    # Training control
    ap.add_argument("--run-train", action="store_true", help="Actually launch training (else only prepare data)")
    ap.add_argument("--train-gpus", default=None, help="Comma-separated GPU ids for deepspeed include list")
    ap.add_argument("--deepspeed-config", default=None, help="Path to DeepSpeed config JSON (optional)")
    ap.add_argument("--train-output", type=Path, default=None, help="Output dir for LoRA adapter (training)")

    # Base/LoRA model paths
    ap.add_argument("--base-model", default=None, help="Base model path; default from config.annotations.model_path or infer.model.base")
    ap.add_argument("--adapter-path", default=None, help="LoRA adapter path for test; defaults to --train-output if training")

    # Test inputs
    ap.add_argument("--test-jsons", nargs="*", default=None, help="Testing dataset JSONs (Dataset or conversation schema)")
    ap.add_argument("--question-template", default=None, help="Override test question template; else from infer_config or default")
    ap.add_argument("--max-test-items", type=int, default=None, help="Limit items per test dataset")
    # Training merge scoring options
    ap.add_argument("--train-with-scores", action="store_true", help="Include weak-detector scores in training questions and add supportive GPT notes")
    ap.add_argument("--no-train-scores", action="store_true", help="Disable weak-detector scores in training merge (overrides config)")
    # Reuse existing per-dataset annotations instead of regenerating
    ap.add_argument("--skip-annotate", action="store_true", help="Skip base annotation and reuse existing per-dataset *_annotations.json")
    ap.add_argument("--reuse-datasets-dir", type=Path, default=None, help="Directory containing per-dataset *_annotations.json to merge")
    # Fast path: start directly from LoRA testing (skip annotation + training)
    ap.add_argument("--test-only", action="store_true", help="Skip annotation/merge/training and only run LoRA testing on --test-jsons or infer.inputs")

    args = ap.parse_args()

    cfg = _load_yaml(args.config)
    cfg_infer = _load_yaml(args.infer_config) if args.infer_config else cfg

    paths_cfg = (cfg.get("paths") or {}) if isinstance(cfg, dict) else {}
    data_dir = _coerce_path(paths_cfg.get("data_dir") or (DATASETS_ROOT / "raw" / "data"))
    results_dir_cfg = paths_cfg.get("results_dir") or str(TRAIN_OUTPUT_ROOT)
    results_root = _coerce_path(results_dir_cfg)
    image_root_prefix = paths_cfg.get("image_root_prefix") or None

    # Build a single question template reused by both training merge and testing
    q_template = (
        args.question_template
        or (cfg_infer.get("infer", {}).get("question_template") if isinstance(cfg_infer, dict) else None)
        or "<image>\nIs this image real or fake? And the {alias} score is {score}."
    )

    annotations_cfg = (cfg.get("annotations") or {}) if isinstance(cfg, dict) else {}
    prompt_file = annotations_cfg.get("prompt_file") or str(PROMPTS_ROOT / "Prompt" / "prompt.json")
    prompt_template = load_prompt_template(str(_coerce_path(prompt_file)))

    run_id = datetime.utcnow().strftime("pipeline_%Y%m%dT%H%M%SZ")
    pipeline_root = results_root / "pipeline" / "runs" / run_id
    train_dir = pipeline_root / "train"
    test_dir = pipeline_root / "test"
    train_datasets_dir = train_dir / "datasets"
    test_datasets_dir = test_dir / "datasets"
    for d in (train_datasets_dir, test_datasets_dir):
        _ensure_dir(d)

    annotation_done = train_dir / ".annotations.done"
    merge_done = train_dir / ".merge.done"  # legacy merge marker
    weak_done = train_dir / ".weak.done"
    train_done = train_dir / ".training.done"
    adapter_record_path = train_dir / "adapter_path.txt"
    test_done = test_dir / ".test.done"

    # Resolve base model path
    base_model_entry = (
        args.base_model
        or annotations_cfg.get("model_path")
        or (cfg.get("model", {}).get("base") if isinstance(cfg, dict) else None)
        or (cfg_infer.get("model", {}).get("base") if isinstance(cfg_infer, dict) else None)
        or os.environ.get("X2DFD_BASE_MODEL")
    )
    base_model_path = _resolve_cli_path(base_model_entry)
    if not base_model_path:
        raise SystemExit("Base model path is required (use --base-model or set in config.yaml/infer_config.yaml)")
    base_model = str(base_model_path)

    # Resolve pipeline mode from config (CLI overrides)
    pipeline_cfg = (cfg.get("pipeline") or {}) if isinstance(cfg, dict) else {}
    test_only = bool(args.test_only or pipeline_cfg.get("test_only", False))

    # Prepare variables used in run info regardless of path
    train_jsons: List[str] = []
    labels: List[str] = []
    train_outputs: List[Path] = []
    train_pairs: List[Tuple[Path, str]] = []
    merged_train: Optional[Path] = None
    train_with_scores: bool = False

    # Only run annotate stage when phase is annotate/train/full (weak stage should not touch annotate)
    if not test_only and (args.phase in ("annotate", "train", "full")):
        # Determine training JSONs
        train_json_paths: List[Path] = []
        if args.train_jsons:
            for entry in args.train_jsons:
                resolved = _resolve_cli_path(entry)
                if resolved is None:
                    raise SystemExit(f"Invalid --train-jsons entry: {entry}")
                train_json_paths.append(resolved)
        else:
            real_files = list(annotations_cfg.get("real_files") or [])
            fake_files = list(annotations_cfg.get("fake_files") or [])
            combined = real_files + fake_files
            train_json_paths = [_resolve_data_entry(p, data_dir) for p in combined]

        train_jsons = [str(p) for p in train_json_paths]
        if not train_jsons:
            raise SystemExit("No training dataset JSONs provided (set annotations.real_files/fake_files or pass --train-jsons)")

        # Map labels
        if args.train_labels and len(args.train_labels) == len(train_jsons):
            labels = list(args.train_labels)
        else:
            for p in train_json_paths:
                lower = p.name.lower()
                labels.append("real" if "real" in lower else "fake")

        reuse_dir = _coerce_path(args.reuse_datasets_dir) if args.reuse_datasets_dir else train_datasets_dir

        def _reuse_existing(reason: str) -> None:
            files = _list_annotation_files(reuse_dir)
            if not files:
                raise SystemExit(f"No annotation JSONs found in {reuse_dir} ({reason})")
            for p in files:
                lower = p.name.lower()
                lbl = "real" if "real" in lower else "fake"
                train_outputs.append(p)
                train_pairs.append((p, lbl))

        # 1) Base-model annotations for training datasets (or reuse existing)
        if annotation_done.exists() and not args.skip_annotate and not args.force_annotate:
            print(f"[SKIP] Found existing annotation artefacts in {train_datasets_dir}; reusing per-dataset outputs.")
            _reuse_existing("annotation marker")
        elif args.skip_annotate and not args.force_annotate:
            print("[SKIP] --skip-annotate provided; reusing existing annotation JSONs.")
            _reuse_existing("--skip-annotate")
        else:
            for in_path, lbl in zip(train_jsons, labels):
                name = os.path.splitext(os.path.basename(in_path))[0]
                out_file = train_datasets_dir / f"{name}_annotations.json"
                try:
                    process_json_file(
                        input_file=in_path,
                        output_file=str(out_file),
                        prompt_template=prompt_template,
                        model_path=base_model,
                        max_items=(args.max_train_items if args.max_train_items is not None else (cfg.get("training", {}).get("max_train_items") if isinstance(cfg, dict) else None)),
                        label_override=lbl,
                        image_root_prefix=None,
                        progress_tracker=None,
                        progress_key=None,
                        template_output_file=None,
                    )
                    train_outputs.append(out_file)
                    train_pairs.append((out_file, lbl))
                except Exception as e:
                    print(f"[WARN] Training annotation failed for {in_path}: {e}")
            if train_outputs:
                annotation_done.write_text(datetime.utcnow().isoformat(timespec="seconds"))

        # If only做注释阶段，直接结束
        if args.phase == "annotate":
            print("✅ Annotate stage done.")
            # write minimal run info at end as usual
            pass

        # Merge training datasets
        if not train_pairs:
            raise SystemExit("No annotation JSONs available to merge; ensure annotation stage succeeded.")

        merged_train = train_dir / "train_merged.json"
        out_weak_json = (args.out_weak_json or (train_dir / "train_weak.json"))
        # Determine whether to include scores in training merge
        tr_cfg = (cfg.get("training") or {}) if isinstance(cfg, dict) else {}
        train_with_scores_cfg = bool(tr_cfg.get("train_with_scores", True))
        if args.no_train_scores:
            train_with_scores = False
        elif args.train_with_scores:
            train_with_scores = True
        else:
            train_with_scores = train_with_scores_cfg

        weak_cfg_single = (cfg.get("weak_supply") or {}) if isinstance(cfg, dict) else {}
        weak_cfg_multi_raw = (cfg.get("weak_supplies") or []) if isinstance(cfg, dict) else []
        # Apply expert filter for training merge
        ex_names = _parse_experts(args.experts)
        weak_cfg_multi = _filter_experts(weak_cfg_multi_raw, ex_names)
        # If only single provider configured and it matches filter, include as one-item list
        if not weak_cfg_multi and weak_cfg_single:
            prov = str(weak_cfg_single.get('provider') or '').strip().lower()
            alias = str(weak_cfg_single.get('alias') or '').strip().lower()
            if ex_names is None or (len(ex_names) > 0 and (prov in ex_names or alias in ex_names)):
                weak_cfg_multi = [weak_cfg_single]
        # Blend into one dict so downstream can see 'weak_supplies' when present
        weak_cfg = dict(weak_cfg_single)
        if weak_cfg_multi:
            weak_cfg["weak_supplies"] = weak_cfg_multi
        # Adapt question template wording for provider: aligner/diffusion -> "diffusion score"
        q_template_merge = q_template
        try:
            prov_name_tmp = (weak_cfg.get("provider") or "blending").strip().lower()
            if prov_name_tmp in ("aligner", "diffusion") and isinstance(q_template_merge, str):
                q_template_merge = q_template_merge.replace("blending score", "diffusion score")
        except Exception:
            pass

        # Weak stage: when显式选择 --phase weak，则写出到 train_weak.json 并可单独退出
        if args.phase == "weak":
            if (not args.force_weak) and weak_done.exists() and out_weak_json.exists():
                print(f"[SKIP] Found weak merged dataset at {out_weak_json}; skipping weak stage.")
            else:
                _merge_annotations(
                    train_pairs,
                    out_weak_json,
                    use_scores=train_with_scores,
                    weak_cfg=weak_cfg,
                    image_root_prefix=image_root_prefix,
                    question_template=q_template_merge,
                )
                weak_done.write_text(datetime.utcnow().isoformat(timespec="seconds"))
            print("✅ Weak stage done.")
            # Allow continuing into training when user also passed --run-train; else return early
            if not bool(args.run_train):
                # write run info later
                pass
        else:
            # Legacy merged (kept for backward-compatibility). Training会优先使用 train_weak.json（若存在）。
            if merge_done.exists() and merged_train.exists():
                print(f"[SKIP] Found merged dataset at {merged_train}; skipping merge stage.")
            else:
                _merge_annotations(
                    train_pairs,
                    merged_train,
                    use_scores=train_with_scores,
                    weak_cfg=weak_cfg,
                    image_root_prefix=image_root_prefix,
                    question_template=q_template_merge,
                )
                merge_done.write_text(datetime.utcnow().isoformat(timespec="seconds"))

    # ---- Stage: weak (outside annotate block) ----
    if (not test_only) and (args.phase in ("weak", "train", "full")):
        reuse_dir = _coerce_path(args.reuse_datasets_dir) if args.reuse_datasets_dir else train_datasets_dir
        ann_list = sorted(p for p in reuse_dir.glob("*_annotations.json") if p.is_file())
        if not ann_list:
            raise SystemExit(f"Weak stage requires *_annotations.json under {reuse_dir}. Provide --reuse-datasets-dir or run --phase annotate first.")
        weak_pairs: List[Tuple[Path, str]] = [(p, ("real" if "real" in p.name.lower() else "fake")) for p in ann_list]

        # Determine whether to include scores in weak merge (default True)
        train_with_scores = True if args.train_with_scores else (False if args.no_train_scores else True)

        weak_cfg_single = (cfg.get("weak_supply") or {}) if isinstance(cfg, dict) else {}
        weak_cfg_multi_raw = (cfg.get("weak_supplies") or []) if isinstance(cfg, dict) else []
        ex_names = _parse_experts(args.experts)
        weak_cfg_multi = _filter_experts(weak_cfg_multi_raw, ex_names)
        if not weak_cfg_multi and weak_cfg_single:
            prov = str(weak_cfg_single.get('provider') or '').strip().lower()
            alias = str(weak_cfg_single.get('alias') or '').strip().lower()
            if ex_names is None or (len(ex_names) > 0 and (prov in ex_names or alias in ex_names)):
                weak_cfg_multi = [weak_cfg_single]
        weak_cfg = dict(weak_cfg_single)
        if weak_cfg_multi:
            weak_cfg["weak_supplies"] = weak_cfg_multi
        q_template_merge = q_template
        try:
            prov_name_tmp = (weak_cfg.get("provider") or "blending").strip().lower()
            if prov_name_tmp in ("aligner", "diffusion") and isinstance(q_template_merge, str):
                q_template_merge = q_template_merge.replace("blending score", "diffusion score")
        except Exception:
            pass

        out_weak_json = (args.out_weak_json or (train_dir / "train_weak.json"))
        if (not args.force_weak) and (train_dir / ".weak.done").exists() and out_weak_json.exists():
            print(f"[SKIP] Found weak merged dataset at {out_weak_json}; skipping weak stage.")
        else:
            _merge_annotations(
                weak_pairs,
                out_weak_json,
                use_scores=train_with_scores,
                weak_cfg=weak_cfg,
                image_root_prefix=image_root_prefix,
                question_template=q_template_merge,
            )
            (train_dir / ".weak.done").write_text(datetime.utcnow().isoformat(timespec="seconds"))
        if args.phase == "weak":
            print("✅ Weak stage done.")
            return

    # 2) (Optional) Launch LoRA training
    adapter_path = args.adapter_path
    run_train_cfg = bool(args.run_train or (cfg.get("training", {}).get("run_train") if isinstance(cfg, dict) else False))
    if (not test_only) and run_train_cfg and (args.phase in ("train", "full")):
        # Auto-name ckpt with expert slug when not explicitly provided
        experts_for_slug = weak_cfg.get("weak_supplies") or []
        slug = _experts_slug(experts_for_slug if train_with_scores else [])
        if args.ckpt_tag:
            slug = f"{slug}-{''.join(ch if ch.isalnum() or ch in '-_.' else '-' for ch in args.ckpt_tag)}"
        default_ckpt_dir = pipeline_root / "ckpt" / f"lora-[{slug}]"
        out_dir_value = args.train_output or (cfg.get("training", {}).get("output_dir") if isinstance(cfg, dict) else None) or default_ckpt_dir
        out_dir_path = _coerce_path(out_dir_value)
        _ensure_dir(out_dir_path)
        if train_done.exists() and adapter_record_path.exists():
            stored = adapter_record_path.read_text().strip()
            adapter_path = stored or str(out_dir_path)
            print(f"[SKIP] Training outputs already exist at {adapter_path}; skipping training stage.")
        else:
            adapter_path = str(out_dir_path)
            # Build deepspeed/python command
            include = args.train_gpus or (cfg.get("training", {}).get("gpus") if isinstance(cfg, dict) else None) or "0"
            ds_cfg = args.deepspeed_config or (cfg.get("training", {}).get("deepspeed_config") if isinstance(cfg, dict) else None) or os.environ.get("DEEPSPEED_CONFIG")
            tr = (cfg.get("training") or {}) if isinstance(cfg, dict) else {}
            vision_tower = tr.get("vision_tower") or os.environ.get("VISION_TOWER")
            if not vision_tower:
                print("[WARN] training.vision_tower not set; training may fail. Set in config or VISION_TOWER env.")
            # LoRA + MM defaults (can be overridden via config.training)
            lora_r = str(tr.get("lora_r", 16))
            lora_alpha = str(tr.get("lora_alpha", 32))
            mm_projector_lr = str(tr.get("mm_projector_lr", 2e-5))
            mm_projector_type = str(tr.get("mm_projector_type", "mlp2x_gelu"))
            mm_vision_select_layer = str(tr.get("mm_vision_select_layer", -2))
            mm_use_im_start_end = str(tr.get("mm_use_im_start_end", False))
            mm_use_im_patch_token = str(tr.get("mm_use_im_patch_token", False))
            image_aspect_ratio = str(tr.get("image_aspect_ratio", "pad"))
            group_by_modality_length = str(tr.get("group_by_modality_length", True))
            per_device_train_batch_size = str(tr.get("per_device_train_batch_size", 2))
            per_device_eval_batch_size = str(tr.get("per_device_eval_batch_size", 2))
            gradient_accumulation_steps = str(tr.get("gradient_accumulation_steps", 1))
            num_train_epochs = str(tr.get("num_train_epochs", 1))
            dataloader_num_workers = str(tr.get("dataloader_num_workers", 4))
            model_max_length = str(tr.get("model_max_length", 2048))
            learning_rate = str(tr.get("learning_rate", 2e-4))
            weight_decay = str(tr.get("weight_decay", 0.0))
            warmup_ratio = str(tr.get("warmup_ratio", 0.03))
            lr_scheduler_type = str(tr.get("lr_scheduler_type", "cosine"))
            logging_steps = str(tr.get("logging_steps", 1))

            # Resolve training toggles from config/env
            def _to_bool(v: Any, default: bool = False) -> bool:
                if isinstance(v, bool):
                    return v
                if isinstance(v, (int, float)):
                    return bool(v)
                try:
                    s = str(v).strip().lower()
                except Exception:
                    return default
                if s in ("1", "true", "yes", "y"):  # truthy strings
                    return True
                if s in ("0", "false", "no", "n"):  # falsy strings
                    return False
                return default

            gc_cfg = tr.get("gradient_checkpointing") if isinstance(tr, dict) else None
            if gc_cfg is None:
                gc_cfg = os.environ.get("GRADIENT_CHECKPOINTING", "False")
            gc_flag = _to_bool(gc_cfg, default=False)

            ddp_unused_cfg = tr.get("ddp_find_unused_parameters") if isinstance(tr, dict) else None
            if ddp_unused_cfg is None:
                ddp_unused_cfg = os.environ.get("DDP_FIND_UNUSED_PARAMETERS", "False")
            ddp_unused_flag = _to_bool(ddp_unused_cfg, default=False)

            report_to = str((tr.get("report_to") if isinstance(tr, dict) else None) or os.environ.get("REPORT_TO") or "none")

            # Prefer weak-stage输出作为训练数据集
            dataset_for_train = str((args.out_weak_json or (train_dir / "train_weak.json")))
            if not Path(dataset_for_train).exists():
                dataset_for_train = str(merged_train) if merged_train and merged_train.exists() else None
            if not dataset_for_train:
                raise SystemExit("No training dataset JSON found (expected train_weak.json or train_merged.json)")

            base_cmd = [
                "deepspeed", "--master_port", "29000", "--include", f"localhost:{include}",
                str(Path(__file__).with_name("model_train.py")),
                "--lora_enable", "True",
                "--lora_r", lora_r, "--lora_alpha", lora_alpha, "--mm_projector_lr", mm_projector_lr,
                "--model_name_or_path", base_model,
                "--version", "v1",
                "--data_path", dataset_for_train,
                "--image_folder", "",
                "--vision_tower", vision_tower or "",
                "--mm_projector_type", mm_projector_type,
                "--mm_vision_select_layer", mm_vision_select_layer,
                "--mm_use_im_start_end", mm_use_im_start_end,
                "--mm_use_im_patch_token", mm_use_im_patch_token,
                "--image_aspect_ratio", image_aspect_ratio,
                "--group_by_modality_length", group_by_modality_length,
                "--output_dir", str(out_dir_path),
                "--bf16", "True",
                "--gradient_checkpointing", ("True" if gc_flag else "False"),
                "--ddp_find_unused_parameters", ("True" if ddp_unused_flag else "False"),
                "--num_train_epochs", num_train_epochs,
                "--per_device_train_batch_size", per_device_train_batch_size,
                "--per_device_eval_batch_size", per_device_eval_batch_size,
                "--gradient_accumulation_steps", gradient_accumulation_steps,
                "--evaluation_strategy", "no",
                "--save_strategy", "no",
                "--save_total_limit", "1",
                "--learning_rate", learning_rate,
                "--weight_decay", weight_decay,
                "--warmup_ratio", warmup_ratio,
                "--lr_scheduler_type", lr_scheduler_type,
                "--logging_steps", logging_steps,
                "--tf32", "True",
                "--model_max_length", model_max_length,
                "--dataloader_num_workers", dataloader_num_workers,
                "--lazy_preprocess", "True",
                "--report_to", report_to,
            ]
            if ds_cfg:
                base_cmd += ["--deepspeed", ds_cfg]
            # Launch
            print("[TRAIN] Command:", " ".join(base_cmd))
            try:
                from subprocess import Popen
                proc = Popen(base_cmd)
                rc = proc.wait()
                if rc != 0:
                    print(f"[WARN] Training process exited with code {rc}")
                else:
                    train_done.write_text(datetime.utcnow().isoformat(timespec="seconds"))
                    adapter_record_path.write_text(adapter_path)
            except Exception as e:
                print(f"[WARN] Failed to launch training: {e}")

    # 3) LoRA testing on other datasets
    # Resolve LoRA adapter path
    if not adapter_path:
        adapter_path = (cfg.get("model", {}).get("adapter") if isinstance(cfg, dict) else None) or (cfg_infer.get("model", {}).get("adapter") if isinstance(cfg_infer, dict) else None)
    adapter_path_resolved = _resolve_cli_path(adapter_path) if adapter_path else None
    adapter_path = str(adapter_path_resolved) if adapter_path_resolved else None
    if not adapter_path:
        if test_only:
            raise SystemExit("Adapter path is required for --test-only mode (use --adapter-path or set model.adapter in config)")
        print("[INFO] No adapter path provided; skipping LoRA testing.")
    else:
        # Resolve scoring provider and question template
        weak = (cfg_infer.get("weak_supply") or {}) if isinstance(cfg_infer, dict) else {}
        weak_multi_infer_raw: List[Dict[str, Any]] = (cfg_infer.get("weak_supplies") or []) if isinstance(cfg_infer, dict) else []
        # Apply expert filter for test stage (default to training filter when omitted)
        test_names = _parse_experts(args.test_experts if args.test_experts is not None else args.experts)
        weak_multi_infer: List[Dict[str, Any]] = _filter_experts(weak_multi_infer_raw, test_names)
        if not weak_multi_infer and weak:
            prov = str(weak.get('provider') or '').strip().lower()
            alias = str(weak.get('alias') or '').strip().lower()
            if test_names is None or (len(test_names) > 0 and (prov in test_names or alias in test_names)):
                weak_multi_infer = [weak]
        infer_gen_cfg = (cfg_infer.get("infer") or {}) if isinstance(cfg_infer, dict) else {}
        provider_name = (weak.get("provider") or "blending").strip().lower()
        provider_kwargs: Dict[str, Any] = {}
        if provider_name in ("aligner", "diffusion"):
            provider_kwargs.update({
                "weights_dir": weak.get("weights_dir") or "",
                "model": weak.get("model") or "",
                "device": None,
                # optional batch + compile knobs for test-time scoring
                "batch_size": int(weak.get("batch_size") or 256),
                "num_workers": int(weak.get("num_workers") or 4),
                "pin_memory": bool(weak.get("pin_memory") if weak.get("pin_memory") is not None else True),
                "compile_model": bool(weak.get("compile_model") or False),
                "compile_mode": str(weak.get("compile_mode") or "safe"),
            })
            provider_kwargs["alias"] = weak.get("alias") or "Diffusion"
        else:
            provider_name = "blending"
            provider_kwargs.update({
                "model_name": weak.get("model_name") or "swinv2_base_window16_256",
                "weights_path": weak.get("weights_path") or "weights/blending_models/best_gf.pth",
                "img_size": int(weak.get("img_size") or 256),
                "num_class": int(weak.get("num_class") or 2),
                "device": None,
                "batch_size": int(weak.get("batch_size") or 64),
                "num_workers": int(weak.get("num_workers") or 4),
                "pin_memory": bool(weak.get("pin_memory") if weak.get("pin_memory") is not None else True),
            })
            provider_kwargs["alias"] = weak.get("alias") or "Blending"

        # Adapt test prompt wording: aligner/diffusion -> "diffusion score"
        q_template_test = q_template
        try:
            if provider_name in ("aligner", "diffusion") and isinstance(q_template_test, str):
                q_template_test = q_template_test.replace("blending score", "diffusion score")
        except Exception:
            pass

        test_json_paths: List[Path] = []
        if args.test_jsons:
            for entry in args.test_jsons:
                resolved = _resolve_cli_path(entry)
                if resolved is None:
                    raise SystemExit(f"Invalid --test-jsons entry: {entry}")
                test_json_paths.append(resolved)
        else:
            entries = list((cfg_infer.get("infer", {}).get("inputs") or []) if isinstance(cfg_infer, dict) else [])
            test_json_paths = [_resolve_cli_path(e) or Path(e) for e in entries]

        test_jsons = [str(p) for p in test_json_paths if p is not None]
        if not test_jsons:
            print("[INFO] No test datasets specified; skipping LoRA testing stage.")
        else:
            test_outputs: List[Path] = []
            if test_done.exists() or (args.phase not in ("test", "full")):
                print(f"[SKIP] Existing LoRA test artefacts detected in {test_datasets_dir}; skipping testing stage.")
                test_outputs = sorted(p for p in test_datasets_dir.glob("*_result.json") if p.is_file())
            else:
                for tj in test_jsons:
                    name = os.path.splitext(os.path.basename(tj))[0]
                    out_path = test_datasets_dir / f"{name}_result.json"
                    tj_path = Path(tj)
                    if out_path.exists():
                        print(f"[SKIP] Found existing result for {name}, skipping dataset.")
                        test_outputs.append(out_path)
                        continue

                    # Multi-expert path takes precedence when configured
                    if weak_multi_infer:
                        items = _build_scored_items_multi(
                            tj_path,
                            image_root_prefix=None,
                            experts_cfg=weak_multi_infer,
                        )
                    else:
                        items = _build_scored_items(
                            tj_path,
                            image_root_prefix=None,
                            template=q_template_test,
                            provider_name=provider_name,
                            provider_kwargs=provider_kwargs,
                        )

                    # Limit test items if configured
                    if args.max_test_items is not None:
                        items = items[: args.max_test_items]
                    elif isinstance(cfg, dict) and isinstance(cfg.get("training"), dict) and cfg["training"].get("max_test_items") is not None:
                        try:
                            items = items[: int(cfg["training"]["max_test_items"])]
                        except Exception:
                            pass

                    temperature = infer_gen_cfg.get("temperature")
                    top_p = infer_gen_cfg.get("top_p")
                    num_beams = infer_gen_cfg.get("num_beams")
                    max_new_tokens = infer_gen_cfg.get("max_new_tokens")

                    answered = lora_infer_conversation_items(
                        items,
                        image_root_prefix=None,  # already absolute in items
                        model_path=adapter_path,
                        model_base=base_model,
                        temperature=float(temperature if isinstance(temperature, (int, float)) else 0.0),
                        top_p=float(top_p if isinstance(top_p, (int, float)) else 1.0),
                        num_beams=int(num_beams if isinstance(num_beams, (int, float)) else 1),
                        max_new_tokens=int(max_new_tokens if isinstance(max_new_tokens, (int, float)) else 512),
                        add_scores_turns=True,
                    )
                    with out_path.open("w", encoding="utf-8") as f:
                        json.dump(answered, f, ensure_ascii=False, indent=2)
                    test_outputs.append(out_path)
                if test_outputs:
                    test_done.write_text(datetime.utcnow().isoformat(timespec="seconds"))

    # Write pipeline info + latest pointer
    info = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
        "run_id": run_id,
        "base_model": base_model,
        "adapter_path": adapter_path,
        "image_root_prefix": image_root_prefix,
        "train": {
            "inputs": (train_jsons if not args.skip_annotate else [str(p) for p in train_outputs]),
            "labels": [lbl for (_, lbl) in train_pairs],
            "outputs": [str(p) for p in train_outputs],
            "merged": (str(merged_train) if merged_train else None),
            "train_with_scores": bool(train_with_scores),
            "ran_training": bool(args.run_train and (not test_only)),
            "train_output": str(args.train_output) if args.train_output else None,
        },
        "test": {
            "inputs": (args.test_jsons or ((cfg_infer.get("infer", {}).get("inputs") if isinstance(cfg_infer, dict) else []))),
            "outputs": [str(p) for p in (test_outputs if 'test_outputs' in locals() else [])],
        },
        "artefacts": {
            "pipeline_root": str(pipeline_root),
            "train_dir": str(train_dir),
            "test_dir": str(test_dir),
        },
    }
    with (pipeline_root / "pipeline_info.json").open("w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    latest = results_root / "pipeline" / "latest_run.json"
    _ensure_dir(latest.parent)
    with latest.open("w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print(f"✅ Pipeline done. Root: {pipeline_root}")


if __name__ == "__main__":
    main()
