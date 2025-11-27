#!/usr/bin/env python3
"""
Step 5a: Build prompted conversation items for evaluation.

Reads eval/configs/infer_config.yaml (or --config), computes two-expert
scores (blending + diffusion_detector by default), and writes prompted
JSONs next to inputs (with _prompted.json suffix) or into --out-dir.

This script does NOT run LLM inference; it only prepares the items.
"""
from __future__ import annotations

import argparse, json, os, glob
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

from utils.model_scoring import (
    resolve_abs_paths,
    ExpertSpec,
    compute_all_scores,
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
    parts: List[str] = []
    for alias, sc in pairs:
        name = (alias or "").strip().lower() or "expert"
        parts.append(f"the {name} score is {_format_score(sc)}")
    if not parts:
        return ""
    if len(parts) == 1:
        return " And " + parts[0] + "."
    return " And " + ", ".join(parts[:-1]) + ", and " + parts[-1] + "."

def _build_conversations_multi(payload: Any, experts_cfg: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not (isinstance(payload, dict) and isinstance(payload.get("images"), list)):
        raise ValueError("Expect dataset JSON with an 'images' list for multi-expert prompts.")

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
        for p in imgs:
            if p and not os.path.isabs(p):
                raise SystemExit("Missing JSON Description while containing relative image paths")
        abs_paths = resolve_abs_paths(imgs, root_prefix=None)

    experts: List[ExpertSpec] = []
    for e in experts_cfg:
        prov = (e.get("provider") or "").strip().lower()
        alias = e.get("alias") or ("Diffusion" if prov in ("aligner", "diffusion", "diffusion_detector", "diffdet") else ("Blending" if prov == "blending" else prov.title()))
        if prov in ("aligner", "diffusion", "diffusion_detector", "diffdet"):
            kwargs = {
                "weights_dir": e.get("weights_dir") or "",
                "model": e.get("model") or "",
                "device": None,
                "batch_size": int(e.get("batch_size") or 64),
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

def main() -> None:
    ap = argparse.ArgumentParser(description="Build prompts with two weak scores (no inference)")
    ap.add_argument("--config", default="config/test_config.yaml")
    ap.add_argument("--experts", default="blending,diffusion_detector")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    cfg = _load_yaml(args.config)
    weak_multi: List[Dict[str, Any]] = list(cfg.get("weak_supplies") or [])
    if not weak_multi:
        raise SystemExit("Config must provide weak_supplies for multi-expert prompts")
    # Filter by --experts
    names = {t.strip().lower() for t in args.experts.split(',') if t.strip()}
    weak_multi = [e for e in weak_multi if (str(e.get('provider','')).strip().lower() in names) or (str(e.get('alias','')).strip().lower() in names)]
    if not weak_multi:
        raise SystemExit("No experts selected after filtering; check --experts")

    cfg_infer = (cfg.get("infer") or {})
    inputs_raw = list(cfg_infer.get("inputs") or [])
    inputs: List[str] = []
    for ent in inputs_raw:
        if isinstance(ent, str):
            inputs.append(ent)
        else:
            j = ent.get("json") or ent.get("path")
            if isinstance(j, str):
                inputs.append(j)

    if not inputs:
        raise SystemExit("No inputs found in config.infer.inputs")

    out_dir = args.out_dir
    if not out_dir:
        # eval/outputs/prompts/runs/prompt_<ts>/datasets
        results_dir = (cfg.get("paths") or {}).get("results_dir") or "eval/outputs"
        results_root = results_dir if os.path.isabs(results_dir) else os.path.join(os.path.dirname(__file__), "..", results_dir)
        run_id = datetime.utcnow().strftime("prompt_%Y%m%dT%H%M%SZ")
        out_dir = os.path.join(results_root, "prompts", "runs", run_id, "datasets")
    os.makedirs(out_dir, exist_ok=True)

    def _abs(p: str) -> str:
        return p if os.path.isabs(p) else os.path.abspath(p)

    for entry in inputs:
        jpath = _abs(entry)
        payload = _load_json(jpath)
        items = _build_conversations_multi(payload, experts_cfg=weak_multi)
        name = os.path.splitext(os.path.basename(jpath))[0]
        out_path = os.path.join(out_dir, f"{name}_prompted.json")
        _save_json(out_path, items)
        print("[step5a] wrote:", out_path)

if __name__ == "__main__":
    main()
