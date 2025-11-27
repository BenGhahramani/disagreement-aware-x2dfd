#!/usr/bin/env python3
"""
Single-image demo for X2DFD (LoRA required).

Runs LLaVA with a LoRA adapter to answer a binary question for one image and
prints the result as JSON (optionally with legacy real/fake scores).

Usage (LoRA required):
  python demo.py \
    --image /abs/path/to/image.png \
    --model-base weights/base/llava-v1.5-7b \
    --adapter-path weights/checkpoints/ckpt/FR/llava-v1.5-7b-lora-[small] \
    [--question "Is this image real or fake?"]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from utils.lora_inference import single_image_infer_with_scores


def _abs_path(p: str) -> str:
    return os.path.normpath(os.path.abspath(p))


def run_demo(
    image: str,
    question: str,
    *,
    model_base: Optional[str],
    adapter_path: Optional[str],
    temperature: float = 0.0,
    top_p: float = 1.0,
    num_beams: int = 1,
    max_new_tokens: int = 4,
) -> Dict[str, Any]:
    # LoRA-only: require adapter + base
    if not adapter_path:
        raise SystemExit("Adapter is required: provide --adapter-path plus --model-base")
    if not model_base:
        raise SystemExit("Base model is required: provide --model-base together with --adapter-path")
    model_path = adapter_path
    model_base_arg = model_base

    result = single_image_infer_with_scores(
        image_path=image,
        question=question,
        model_path=model_path,
        model_base=model_base_arg,
        temperature=temperature,
        top_p=top_p,
        num_beams=num_beams,
        max_new_tokens=max_new_tokens,
    )
    out: Dict[str, Any] = {
        "image": image,
        "question": question,
        "answer": result.get("answer"),
    }
    if "real_score" in result or "fake_score" in result:
        out["real_score"] = result.get("real_score")
        out["fake_score"] = result.get("fake_score")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Single-image LLaVA demo (LoRA required)")
    ap.add_argument("--image", required=True, help="Absolute path to an image file")
    ap.add_argument("--question", default="Is this image real or fake?", help="Question to ask")
    ap.add_argument("--model-base", default=os.environ.get("X2DFD_BASE_MODEL", "weights/base/llava-v1.5-7b"), help="Base LLaVA model path")
    ap.add_argument("--adapter-path", default=None, help="Optional LoRA adapter path")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--num-beams", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=4)
    args = ap.parse_args()

    img_abs = _abs_path(args.image)
    if not os.path.exists(img_abs):
        raise SystemExit(f"Image does not exist: {img_abs}")

    out = run_demo(
        image=img_abs,
        question=args.question,
        model_base=args.model_base,
        adapter_path=args.adapter_path,
        temperature=args.temperature,
        top_p=args.top_p,
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
