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

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
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
from utils.model_scoring import get_provider, resolve_abs_paths
from utils.paths import OUTPUT_ROOT, PROJECT_ROOT, ensure_core_dirs

ensure_core_dirs()


DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "config.yaml")
DEFAULT_BASE = os.environ.get("X2DFD_BASE_MODEL", "weights/base/llava-v1.5-7b")
DEFAULT_ADAPTER = "weights/checkpoints/ckpt/FR/llava-v1.5-7b-lora-[small]"

DEFAULT_TEMPLATE = (
    "<image>\nIs this image real or fake? And by observation of {alias} expert, "
    "the blending score is {score}."
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
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--num-beams", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=None)
    args = ap.parse_args()

    cfg = _load_yaml(args.config)

    # Resolve defaults from config
    weak = (cfg.get("weak_supply") or {}) if isinstance(cfg, dict) else {}
    alias = weak.get("alias") or "Blending"
    provider_name = "blending"
    provider_kwargs: Dict[str, Any] = {
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
    cfg_inputs = list(cfg_infer.get("inputs") or [])
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
    model_base = args.model_base or (cfg.get("model", {}).get("base") if isinstance(cfg, dict) else None) or DEFAULT_BASE
    model_path = args.model_path or (cfg.get("model", {}).get("adapter") if isinstance(cfg, dict) else None) or DEFAULT_ADAPTER
    temperature = args.temperature if args.temperature is not None else float(cfg_infer.get("temperature", 0.0))
    top_p = args.top_p if args.top_p is not None else float(cfg_infer.get("top_p", 1.0))
    num_beams = args.num_beams if args.num_beams is not None else int(cfg_infer.get("num_beams", 1))
    max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else int(cfg_infer.get("max_new_tokens", 512))

    def _abs_path(path_str: str) -> str:
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
