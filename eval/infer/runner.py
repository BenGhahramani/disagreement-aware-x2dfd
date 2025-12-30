#!/usr/bin/env python3
"""
Quick LoRA inference aligned with pipeline path rules.

Key changes vs older behavior:
- Only use JSON top-level Description as the path prefix for dataset-style JSON.
- No fallback to config.paths.image_root_prefix; if Description is missing and
  relative paths exist, raise an error.
- Items written for inference carry absolute image paths; LoRA inference receives
  no additional image_root_prefix (None).

Input JSON (dataset style):
{
  "Description": "/abs/root",
  "images": [ { "image_path": "rel/or/abs" }, ... ]
}

Output is a conversation-style list with absolute image paths.
"""

from __future__ import annotations

# Reduce noisy deprecation logs from PyTorch storages
import warnings
warnings.filterwarnings(
    "ignore",
    message=r"TypedStorage is deprecated",
    category=UserWarning,
)

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import glob
from datetime import datetime

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore

from utils.lora_inference import single_image_infer_with_scores, lora_infer_conversation_items
from utils.model_scoring import (
    get_provider,
    resolve_abs_paths,
    ExpertSpec,
    compute_all_scores,
)
from utils.paths import OUTPUT_ROOT, PROJECT_ROOT, ensure_core_dirs

ensure_core_dirs()


DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "config.yaml")
DEFAULT_BASE = os.environ.get("X2DFD_BASE_MODEL", "weights/base/llava-v1.5-7b")
DEFAULT_ADAPTER = "weights/checkpoints/ckpt/FR/llava-v1.5-7b-lora-[small]"

DEFAULT_TEMPLATE = (
    "<image>\nIs this image real or fake? And the {alias} score is {score}."
)


def _load_yaml(path: str) -> Dict[str, Any]:
    if not yaml:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _json_description(payload: Any) -> Optional[str]:
    try:
        if isinstance(payload, dict):
            desc = payload.get("Description") or payload.get("description")
            if isinstance(desc, str) and desc.strip():
                return desc.strip()
    except Exception:
        pass
    return None


def _format_score(sc: Optional[float]) -> str:
    if sc is None:
        return "N/A"
    try:
        return f"{float(sc):.3f}"
    except Exception:
        return "N/A"


def _format_multi_scores(pairs: List[tuple[str, Optional[float]]]) -> str:
    """Return tail like: " the blending score is 0.812, and the diffusion score is 0.153.".

    Alias is rendered in lower-case for readability, matching train/pipeline behavior.
    """
    parts: List[str] = []  # type: ignore[assignment]
    for alias, sc in pairs:
        name = (alias or "").strip().lower() or "expert"
        parts.append(f"the {name} score is {_format_score(sc)}")
    if not parts:
        return ""
    if len(parts) == 1:
        return " And " + parts[0] + "."
    return " And " + ", ".join(parts[:-1]) + ", and " + parts[-1] + "."


def _build_conversations_with_scores(
    payload: Any,
    *,
    template: str,
    provider_name: str,
    provider_kwargs: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Build conversation items with absolute image paths using only JSON Description.

    If Description is missing, all image_path entries must already be absolute; otherwise error.
    """
    if not (isinstance(payload, dict) and isinstance(payload.get("images"), list)):
        raise ValueError("Expect dataset JSON with an 'images' list for quick_infer alignment.")

    imgs: List[str] = []
    for it in payload["images"]:
        if isinstance(it, dict):
            p = it.get("image_path") or it.get("path")
            imgs.append(p if isinstance(p, str) else "")
        else:
            imgs.append("")

    local_root = _json_description(payload)
    if local_root:
        abs_paths = resolve_abs_paths(imgs, root_prefix=local_root)
    else:
        # Ensure all are absolute
        for p in imgs:
            if p and not os.path.isabs(p):
                raise SystemExit("Missing JSON Description while containing relative image paths")
        abs_paths = resolve_abs_paths(imgs, root_prefix=None)
    # normalize alias lower-case for template formatting consistency
    if isinstance(provider_kwargs.get("alias"), str):
        provider_kwargs["alias"] = provider_kwargs["alias"].strip().lower()
    provider = get_provider(provider_name, **provider_kwargs)
    score_map = provider.compute_scores(abs_paths)

    items: List[Dict[str, Any]] = []
    for idx, (rel, abs_p) in enumerate(zip(imgs, abs_paths), start=1):
        sc = None
        try:
            sr = score_map.get(abs_p)
            sc = None if sr is None else sr.score
        except Exception:
            sc = None
        question = template.format(alias=provider_kwargs.get("alias", "Blending"), score=_format_score(sc))
        items.append({
            "id": str(idx),
            # Normalize: write absolute image path in outputs
            "image": abs_p,
            "conversations": [
                {"from": "human", "value": question},
                {"from": "gpt", "value": ""},
            ],
        })
    return items


def _build_conversations_multi(
    payload: Any,
    *,
    experts_cfg: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Build conversation items using multiple experts (shared style with training).

    HUMAN: "<image>\nIs this image real or fake?" + tail from multiple experts
    """
    if not (isinstance(payload, dict) and isinstance(payload.get("images"), list)):
        raise ValueError("Expect dataset JSON with an 'images' list for multi-expert inference.")

    # Collect relative/absolute paths from JSON
    imgs: List[str] = []
    for it in payload["images"]:
        if isinstance(it, dict):
            p = it.get("image_path") or it.get("path")
            imgs.append(p if isinstance(p, str) else "")
        else:
            imgs.append("")

    # Resolve to absolute using only JSON Description
    local_root = _json_description(payload)
    if local_root:
        abs_paths = resolve_abs_paths(imgs, root_prefix=local_root)
    else:
        for p in imgs:
            if p and not os.path.isabs(p):
                raise SystemExit("Missing JSON Description while containing relative image paths")
        abs_paths = resolve_abs_paths(imgs, root_prefix=None)

    # Build ExpertSpec list
    experts: List[ExpertSpec] = []
    for e in experts_cfg:
        prov = (e.get("provider") or "").strip().lower()
        alias = e.get("alias") or ("Diffusion" if prov in ("aligner", "diffusion", "diffusion_detector", "diffdet") else ("Blending" if prov == "blending" else prov.title()))
        if prov in ("aligner", "diffusion", "diffusion_detector", "diffdet"):
            # pass batching knobs for diffusion/aligner experts in multi-expert inference
            kwargs = {
                "weights_dir": e.get("weights_dir") or "",
                "model": e.get("model") or "",
                "device": None,
                "batch_size": int(e.get("batch_size") or 256),
                "num_workers": int(e.get("num_workers") or 4),
                "pin_memory": bool(e.get("pin_memory") if e.get("pin_memory") is not None else True),
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
            kwargs = dict(e)
            kwargs.pop("provider", None)
            kwargs.pop("alias", None)
        experts.append(ExpertSpec(provider=prov, alias=alias, kwargs=kwargs))

    # Compute scores across experts
    score_map = compute_all_scores(experts, abs_paths)

    items: List[Dict[str, Any]] = []
    for idx, abs_p in enumerate(abs_paths, start=1):
        pairs: List[tuple[str, Optional[float]]] = []
        es_list = score_map.get(abs_p) or []
        for i, spec in enumerate(experts):
            sc = es_list[i].score if i < len(es_list) else None
            pairs.append((spec.alias, sc))
        base = "<image>\nIs this image real or fake?"
        q = base + _format_multi_scores(pairs)
        items.append({
            "id": str(idx),
            "image": abs_p,
            "conversations": [
                {"from": "human", "value": q},
                {"from": "gpt", "value": ""},
            ],
        })
    return items


def _process_one(
    *,
    json_path: str,
    question_template: str,
    provider_name: str,
    provider_kwargs: Dict[str, Any],
    model_path: str,
    model_base: str,
    temperature: float,
    top_p: float,
    num_beams: int,
    max_new_tokens: int,
    output_path: Optional[str] = None,
) -> str:
    payload = _load_json(json_path)
    items = _build_conversations_with_scores(
        payload,
        template=question_template,
        provider_name=provider_name,
        provider_kwargs=provider_kwargs,
    )

    # Use multi-GPU sharded LoRA inference mirroring utils.inference
    out_items = lora_infer_conversation_items(
        items,
        image_root_prefix=None,  # items already carry absolute image paths
        model_path=model_path,
        model_base=model_base,
        temperature=temperature,
        top_p=top_p,
        num_beams=num_beams,
        max_new_tokens=max(4, max_new_tokens),
        add_scores_turns=True,
    )

    out_path = output_path
    if not out_path:
        base = os.path.splitext(os.path.basename(json_path))[0]
        out_path = os.path.join(os.path.dirname(json_path), f"{base}_result.json")
    _save_json(out_path, out_items)
    return out_path

def main() -> None:
    ap = argparse.ArgumentParser(description="Quick LoRA inference aligned with dp_infer (score->question->answer)")
    ap.add_argument("--config", default=DEFAULT_CONFIG, help="Path to config.yaml")
    ap.add_argument("--json", default=None, help="Optional single input JSON; if omitted, use infer.inputs from config")
    ap.add_argument("--output", default=None, help="Optional single output JSON; else default next to input or infer.output_dir")
    ap.add_argument("--model-path", default=None, help="LoRA adapter or model path (overrides config)")
    ap.add_argument("--model-base", default=None, help="Base model path when using LoRA (overrides config)")
    ap.add_argument("--question-template", default=None, help="Per-item question template with {alias} and {score} (overrides config)")
    ap.add_argument("--experts", default=None, help="Comma-separated experts (provider or alias) to enable; use 'none' to disable scores")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--num-beams", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=None)
    args = ap.parse_args()

    cfg = _load_yaml(args.config)

    # Helper: parse and filter experts
    def _parse_experts(spec: Optional[str]) -> Optional[set[str]]:
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

    # Resolve defaults from config
    weak = (cfg.get("weak_supply") or {}) if isinstance(cfg, dict) else {}
    weak_multi_cfg = list(cfg.get("weak_supplies") or []) if isinstance(cfg, dict) else []
    # Apply CLI expert filter
    ex_names = _parse_experts(args.experts)
    weak_multi = _filter_experts(weak_multi_cfg, ex_names)
    # If no multi config, try to upcast single weak_supply to a list when selected
    if not weak_multi and weak:
        prov0 = str(weak.get('provider') or '').strip().lower()
        alias0 = str(weak.get('alias') or '').strip().lower()
        if ex_names is None or (len(ex_names) > 0 and (prov0 in ex_names or alias0 in ex_names)):
            weak_multi = [weak]
    # Force multi-expert path (even if empty) when user passed --experts
    force_multi = (args.experts is not None)
    # Resolve provider (blending | diffusion/aligner) similar to train/pipeline
    prov = (weak.get("provider") or "blending").strip().lower()
    if prov in ("aligner", "diffusion", "diffusion_detector", "diffdet"):
        provider_name = prov
        alias = weak.get("alias") or "Diffusion"
        provider_kwargs: Dict[str, Any] = {
            "weights_dir": weak.get("weights_dir") or "",
            "model": weak.get("model") or "",
            "device": None,
            # optional batching knobs for diffusion/aligner
            "batch_size": int(weak.get("batch_size") or 64),
            "num_workers": int(weak.get("num_workers") or 4),
            "pin_memory": bool(weak.get("pin_memory") if weak.get("pin_memory") is not None else True),
        }
    else:
        provider_name = "blending"
        alias = weak.get("alias") or "Blending"
        provider_kwargs = {
            "model_name": weak.get("model_name") or "swinv2_base_window16_256",
            "weights_path": weak.get("weights_path") or "weights/blending_models/best_gf.pth",
            "img_size": int(weak.get("img_size") or 256),
            "num_class": int(weak.get("num_class") or 2),
            "device": None,
            # batching controls for detector
            "batch_size": int(weak.get("batch_size") or 64),
            "num_workers": int(weak.get("num_workers") or 4),
            "pin_memory": bool(weak.get("pin_memory") if weak.get("pin_memory") is not None else True),
        }
    provider_kwargs["alias"] = alias

    # Determine settings from config (optional)
    cfg_infer = (cfg.get("infer") or {}) if isinstance(cfg, dict) else {}
    cfg_inputs_raw = list(cfg_infer.get("inputs") or [])
    # Expand directories in inputs to all *.json (recursive), to allow pointing to a test root dir.
    cfg_inputs: List[str | Dict[str, Any]] = []
    for ent in cfg_inputs_raw:
        if isinstance(ent, str):
            # expand environment variables like ${X2DFD_DATASETS}
            ent = os.path.expandvars(ent)
            p_abs = ent if os.path.isabs(ent) else os.path.join(PROJECT_ROOT, ent)
            if os.path.isdir(p_abs):
                for p in sorted(glob.glob(os.path.join(p_abs, "**", "*.json"), recursive=True)):
                    cfg_inputs.append(p)
            else:
                cfg_inputs.append(ent)
        else:
            cfg_inputs.append(ent)
    # Canonical output layout: <results_dir>/infer/runs/<run_id>/datasets
    paths_cfg = (cfg.get("paths") or {}) if isinstance(cfg, dict) else {}
    results_dir_cfg = paths_cfg.get("results_dir") or str(OUTPUT_ROOT)
    results_root = results_dir_cfg if os.path.isabs(results_dir_cfg) else os.path.join(PROJECT_ROOT, results_dir_cfg)
    infer_root = os.path.join(results_root, "infer")
    runs_root = os.path.join(infer_root, "runs")
    os.makedirs(runs_root, exist_ok=True)
    run_id = datetime.utcnow().strftime("infer_%Y%m%dT%H%M%SZ")
    run_dir = os.path.join(runs_root, run_id)
    datasets_dir = os.path.join(run_dir, "datasets")
    os.makedirs(datasets_dir, exist_ok=True)
    question_template = args.question_template or cfg_infer.get("question_template") or DEFAULT_TEMPLATE
    # Align wording when using diffusion/aligner: swap metric term to 'diffusion score'
    try:
        if provider_name in ("aligner", "diffusion") and isinstance(question_template, str):
            question_template = question_template.replace("blending score", "diffusion score")
    except Exception:
        pass
    model_base = args.model_base or (cfg.get("model", {}).get("base") if isinstance(cfg, dict) else None) or DEFAULT_BASE
    model_path = args.model_path or (cfg.get("model", {}).get("adapter") if isinstance(cfg, dict) else None) or DEFAULT_ADAPTER

    def _is_lora_adapter_dir(p: str | None) -> bool:
        if not p:
            return False
        try:
            if not os.path.isdir(p):
                return False
            return os.path.exists(os.path.join(p, "adapter_config.json")) or os.path.exists(os.path.join(p, "adapter_model.safetensors")) or os.path.exists(os.path.join(p, "adapter_model.bin"))
        except Exception:
            return False

    # If user points to a merged checkpoint directory, do NOT require model_base.
    # (LLaVA's builder merges LoRA in load_pretrained_model only when model name contains 'lora'.)
    using_lora = _is_lora_adapter_dir(model_path) or ("lora" in os.path.basename(str(model_path)).lower())
    if not using_lora:
        model_base = None

    # Strict existence checks, but only require base when using LoRA.
    def _looks_remote(p: str | None) -> bool:
        try:
            if not p:
                return False
            s = str(p)
            return ("://" in s) or s.startswith("hf://")
        except Exception:
            return False

    missing_msgs: list[str] = []
    try:
        if isinstance(model_path, str) and not _looks_remote(model_path) and not os.path.exists(model_path):
            missing_msgs.append(f"Model not found: {model_path}")
    except Exception:
        pass
    try:
        if using_lora and isinstance(model_base, str) and not _looks_remote(model_base) and not os.path.exists(model_base):
            missing_msgs.append(f"Base model not found: {model_base}")
    except Exception:
        pass
    if missing_msgs:
        tip_lines = [
            "[X2DFD] Missing required model files:",
            *[f"- {m}" for m in missing_msgs],
            "",
            "How to fix:",
            "- Download the missing files and place them at the paths above, or:",
            "- Edit eval/configs/infer_config.yaml -> model.adapter / model.base, or:",
            "- Override via CLI:",
            "  - LoRA:   --model-path <adapter_dir> --model-base <base_model_dir>",
            "  - Merged: --model-path <merged_model_dir>   (no --model-base needed)",
        ]
        print("\n".join(tip_lines))
        raise SystemExit(2)
    temperature = args.temperature if args.temperature is not None else float(cfg_infer.get("temperature", 0.0))
    top_p = args.top_p if args.top_p is not None else float(cfg_infer.get("top_p", 1.0))
    num_beams = args.num_beams if args.num_beams is not None else int(cfg_infer.get("num_beams", 1))
    max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else int(cfg_infer.get("max_new_tokens", 512))

    def _abs_path(path_str: str) -> str:
        """Return absolute path; expand env vars then resolve relative to project root."""
        path_str = os.path.expandvars(path_str)
        if os.path.isabs(path_str):
            return os.path.normpath(path_str)
        return os.path.normpath(os.path.join(PROJECT_ROOT, path_str))

    def _derive_out(json_path: str) -> str:
        base = os.path.splitext(os.path.basename(json_path))[0]
        if args.output:
            # Explicit output: respect user-provided path
            out_p = args.output
            os.makedirs(os.path.dirname(out_p) or ".", exist_ok=True)
            return out_p
        # Default output: write into this run's datasets directory
        return os.path.join(datasets_dir, f"{base}_result.json")

    # Single input via CLI
    results_printed: List[str] = []
    outputs_collected: List[str] = []
    inputs_used: List[str] = []

    if args.json:
        json_abs = _abs_path(args.json)
        if weak_multi or force_multi:
            # Multi-expert path (preferred when configured)
            payload = _load_json(json_abs)
            items = _build_conversations_multi(payload, experts_cfg=weak_multi)
            out_items = lora_infer_conversation_items(
                items,
                image_root_prefix=None,
                model_path=model_path,
                model_base=model_base,
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                add_scores_turns=True,
            )
            out_path = _derive_out(args.json)
            _save_json(out_path, out_items)
        else:
            out_path = _process_one(
                json_path=json_abs,
                question_template=question_template,
                provider_name=provider_name,
                provider_kwargs=provider_kwargs,
                model_path=model_path,
                model_base=model_base,
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                output_path=_derive_out(args.json),
            )
        outputs_collected.append(out_path)
        inputs_used.append(json_abs)
        # Write run metadata and latest pointer
        info = {
            "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
            "run_id": run_id,
            "model_base": model_base,
            "model_path": model_path,
            "question_template": question_template,
            # Path prefix is defined by JSON Description; not recorded here
            "params": {
                "temperature": temperature,
                "top_p": top_p,
                "num_beams": num_beams,
                "max_new_tokens": max_new_tokens,
            },
            "inputs": inputs_used,
            "outputs": outputs_collected,
            "artefacts": {
                "run_dir": run_dir,
                "datasets_dir": datasets_dir,
            },
        }
        with open(os.path.join(run_dir, "infer_info.json"), "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        # latest pointer
        latest = os.path.join(infer_root, "latest_run.json")
        os.makedirs(os.path.dirname(latest), exist_ok=True)
        with open(latest, "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        print(f"Saved conversation-style results: {out_path}")
        print(f"Run info saved to: {os.path.join(run_dir, 'infer_info.json')}")
        return

    # Multiple inputs via config
    if not cfg_inputs:
        raise SystemExit("No input JSON provided. Use --json or infer.inputs in config.")

    for entry in cfg_inputs:
        if isinstance(entry, str):
            jpath = _abs_path(entry)
            out_path = _derive_out(jpath)
        elif isinstance(entry, dict):
            raw_path = entry.get("json") or entry.get("path")
            jpath = _abs_path(raw_path) if isinstance(raw_path, str) else None
            if not isinstance(jpath, str) or not jpath:
                continue
            out_path = entry.get("output") or _derive_out(jpath)
        else:
            continue
        if weak_multi or force_multi:
            payload = _load_json(jpath)
            items = _build_conversations_multi(payload, experts_cfg=weak_multi)
            out_items = lora_infer_conversation_items(
                items,
                image_root_prefix=None,
                model_path=model_path,
                model_base=model_base,
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                add_scores_turns=True,
            )
            out_done = out_path
            _save_json(out_done, out_items)
        else:
            out_done = _process_one(
                json_path=jpath,
                question_template=question_template,
                provider_name=provider_name,
                provider_kwargs=provider_kwargs,
                model_path=model_path,
                model_base=model_base,
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                output_path=out_path,
            )
        outputs_collected.append(out_done)
        inputs_used.append(jpath)
        print(f"Saved conversation-style results: {out_done}")

    # Write run metadata and latest pointer (multi-input)
    info = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
        "run_id": run_id,
        "model_base": model_base,
        "model_path": model_path,
        "question_template": question_template,
        # Path prefix is defined per-input JSON Description; not recorded here
        "params": {
            "temperature": temperature,
            "top_p": top_p,
            "num_beams": num_beams,
            "max_new_tokens": max_new_tokens,
        },
        "inputs": inputs_used,
        "outputs": outputs_collected,
        "artefacts": {
            "run_dir": run_dir,
            "datasets_dir": datasets_dir,
        },
    }
    with open(os.path.join(run_dir, "infer_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    latest = os.path.join(infer_root, "latest_run.json")
    os.makedirs(os.path.dirname(latest), exist_ok=True)
    with open(latest, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
