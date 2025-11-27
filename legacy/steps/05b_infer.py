#!/usr/bin/env python3
"""
Step 5b: Run LoRA inference on pre-built prompted JSONs.

Inputs are *_prompted.json files (conversation-style, with absolute image paths).
Outputs are *_result.json next to inputs (or under --out-dir).
"""
from __future__ import annotations

import argparse, json, os, glob
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List

from utils.lora_inference import lora_infer_conversation_items

def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _save_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def main() -> None:
    ap = argparse.ArgumentParser(description="Run LoRA inference on prompted JSONs")
    ap.add_argument("--inputs", nargs="*", default=None, help="Prompted JSON files or directories to search for *_prompted.json")
    ap.add_argument("--model-path", required=True, help="LoRA adapter or model path")
    ap.add_argument("--model-base", default=None, help="Base model when using LoRA")
    ap.add_argument("--out-dir", default=None, help="Optional output dir; else next to input")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--num-beams", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=4)
    args = ap.parse_args()

    # Collect inputs
    paths: List[str] = []
    for ent in (args.inputs or []):
        p = ent
        if os.path.isdir(p):
            for fp in glob.glob(os.path.join(p, "**", "*_prompted.json"), recursive=True):
                paths.append(fp)
        else:
            paths.append(p)
    # Fallback to eval/outputs/prompts/latest run if not provided
    if not paths:
        base = os.path.join("eval", "outputs", "prompts", "runs")
        try:
            runs = sorted(os.listdir(base), reverse=True)
        except Exception:
            runs = []
        if runs:
            paths = glob.glob(os.path.join(base, runs[0], "datasets", "*_prompted.json"))
    if not paths:
        raise SystemExit("No prompted JSONs provided or found")

    for jpath in paths:
        payload = _load_json(jpath)
        if not isinstance(payload, list):
            raise SystemExit(f"Prompted JSON must be a list: {jpath}")
        out_items = lora_infer_conversation_items(
            payload,
            image_root_prefix=None,
            model_path=args.model_path,
            model_base=args.model_base,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            add_scores_turns=True,
        )
        if args.out_dir:
            os.makedirs(args.out_dir, exist_ok=True)
            name = os.path.splitext(os.path.basename(jpath))[0].replace("_prompted","")
            out_path = os.path.join(args.out_dir, f"{name}_result.json")
        else:
            base, ext = os.path.splitext(jpath)
            out_path = base.replace("_prompted","_result") + (ext or ".json")
        _save_json(out_path, out_items)
        print("[step5b] wrote:", out_path)

if __name__ == "__main__":
    main()
