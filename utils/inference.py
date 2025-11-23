"""Utilities for running LLaVA-based inference.

Adds helpers to process a list of conversation-style JSON items:

[
  {
    "id": "1",
    "image": "path/to/img.png",
    "conversations": [
      {"from": "human", "value": "<image>\n..."},
      {"from": "gpt",   "value": ""}
    ]
  },
  ...
]

Functions:
 - infer_conversation_items: fill GPT answers for a list of items (batch)
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, Iterable, List, Tuple, Optional

import torch
from PIL import Image

from llava.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import conv_templates
from llava.mm_utils import process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

try:
    # Optional console progress bar
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore

import argparse
import json
import sys
import tempfile
import time


IMAGE_PLACEHOLDER = "<image>"

DEFAULT_MODEL_PATH = os.environ.get("X2DFD_BASE_MODEL", "weights/base/llava-v1.5-7b")

# Cache keyed by model_path only (no adapter/base handling here)
_MODEL_CACHE: Dict[str, Tuple[Any, ...]] = {}


def get_model_name_from_path(model_path: str) -> str:
    """Return the model name inferred from its path."""

    return os.path.basename(model_path)


def load_images(image_files: Iterable[str]) -> List[Image.Image]:
    """Load and RGB-convert the given image paths."""

    return [Image.open(image_file).convert("RGB") for image_file in image_files]


def _resolve_model_device(model) -> torch.device:
    """Resolve the model device, defaulting to CUDA when available."""

    if hasattr(model, "device"):
        return model.device
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _resolve_model_dtype(model) -> torch.dtype:
    """Resolve a reasonable dtype for inputs that matches the model."""

    try:
        return next(model.parameters()).dtype  # type: ignore[attr-defined]
    except (StopIteration, AttributeError):
        return torch.float16


def _get_or_load_model(model_path: str):
    """Load model components once and reuse across invocations."""
    if model_path not in _MODEL_CACHE:
        disable_torch_init()
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path, None, model_name, False
        )
        model.eval()
        _MODEL_CACHE[model_path] = (tokenizer, model, image_processor, context_len)
    return _MODEL_CACHE[model_path]


def _parse_visible_gpu_ids() -> List[int]:
    """Parse visible GPU ids from CUDA_VISIBLE_DEVICES or by probing torch.

    Mirrors infer_debug/dp_infer.py behavior: prefer explicit list if present,
    otherwise auto-detect via torch.cuda.device_count().
    """
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env:
        ids: List[int] = []
        for x in env.replace(" ", "").split(","):
            if x != "":
                try:
                    ids.append(int(x))
                except Exception:
                    # If mapping is symbolic (e.g., UUID), fall back to index positions
                    pass
        # If parsing failed (UUID style), fall back to index positions length
        if ids:
            return ids
        try:
            cnt = torch.cuda.device_count()
        except Exception:
            cnt = 0
        return list(range(cnt))
    try:
        cnt = torch.cuda.device_count()
    except Exception:
        cnt = 0
    return list(range(cnt))


def _shard_indices(n_items: int, n_shards: int) -> List[range]:
    """Evenly split [0, n_items) into n_shards contiguous ranges.

    Copied from infer_debug/dp_infer.py to ensure identical strategy.
    """
    n_shards = max(int(n_shards), 1)
    n_items = max(int(n_items), 0)
    return [
        range((i * n_items) // n_shards, ((i + 1) * n_items) // n_shards)
        for i in range(n_shards)
    ]


def _generate_answer(
    image_path: str,
    question: str,
    model_path: str = DEFAULT_MODEL_PATH,
    temperature: float = 0,
    top_p: float = 1,
    num_beams: int = 1,
    max_new_tokens: int = 512,
) -> str:
    """Internal: generate answer for one image-question with LLaVA."""
    tokenizer, model, image_processor, _ = _get_or_load_model(model_path)

    model_name = get_model_name_from_path(model_path)
    model_device = _resolve_model_device(model)
    model_dtype = _resolve_model_dtype(model)
    if model_device.type == "cpu" and model_dtype == torch.float16:
        model_dtype = torch.float32

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in question:
        qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, question)
    else:
        qs = image_token_se + question

    images = load_images([image_path])
    image_sizes = [x.size for x in images]
    images_tensor = process_images(images, image_processor, model.config)

    if isinstance(images_tensor, list):
        image_inputs = [
            img.to(device=model_device, dtype=model_dtype)
            for img in images_tensor
        ]
    else:
        image_inputs = images_tensor.to(device=model_device, dtype=model_dtype)

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0)
    input_ids = input_ids.to(model_device)

    with torch.inference_mode():
        generation_output = model.generate(
            input_ids,
            images=image_inputs,
            image_sizes=image_sizes,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )

    answer = tokenizer.batch_decode(
        generation_output, skip_special_tokens=True
    )[0].strip()
    return answer

def _first_human_question(conversations: List[Dict[str, Any]]) -> Optional[str]:
    """Return the first human question value from a conversation list, if any."""
    try:
        for turn in conversations:
            if isinstance(turn, dict) and turn.get("from") == "human":
                val = turn.get("value")
                if isinstance(val, str) and val.strip():
                    return val
    except Exception:
        pass
    return None


def _set_or_append_gpt_answer(
    conversations: List[Dict[str, Any]], answer: str
) -> List[Dict[str, Any]]:
    """Set GPT answer in-place if a GPT turn exists; otherwise append one."""
    try:
        for turn in conversations:
            if isinstance(turn, dict) and turn.get("from") == "gpt":
                turn["value"] = answer
                return conversations
    except Exception:
        pass
    conversations.append({"from": "gpt", "value": answer})
    return conversations


# Public thin wrappers for reuse in experiments
def first_human_question(conversations: List[Dict[str, Any]]) -> Optional[str]:
    """Public helper: return the first human question string if present."""
    return _first_human_question(conversations)


def set_or_append_gpt_answer(
    conversations: List[Dict[str, Any]], answer: str
) -> List[Dict[str, Any]]:
    """Public helper: set or append a GPT turn with the given answer."""
    return _set_or_append_gpt_answer(conversations, answer)


def infer_conversation_items(
    items: List[Dict[str, Any]],
    *,
    image_root_prefix: Optional[str] = None,
    model_path: str = DEFAULT_MODEL_PATH,
    temperature: float = 0,
    top_p: float = 1,
    num_beams: int = 1,
    max_new_tokens: int = 512,
) -> List[Dict[str, Any]]:
    """Fill GPT answers for a list of conversation-style items.

    For each item, uses:
      - image: relative or absolute image path
      - question: first human turn's value (expects optional "<image>" token)

    Returns the same list with GPT answers filled (or appended if missing).
    """
    def _infer_seq(seq: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out_local: List[Dict[str, Any]] = []
        mp = model_path or DEFAULT_MODEL_PATH

        # Progress bar: show per-sequence progress if tqdm available and enabled
        use_bar_env = os.environ.get("USE_PROGRESS_BAR", "1").lower() in {"1", "true", "yes"}
        show_bar = bool(tqdm) and use_bar_env
        iterator = enumerate(seq)
        total = len(seq)
        if show_bar:
            iterator = enumerate(tqdm(seq, desc="Inference", total=total, ncols=80))  # type: ignore

        for idx, it in iterator:  # type: ignore[assignment]
            image_rel = (it or {}).get("image")
            conv = (it or {}).get("conversations") or []
            question = _first_human_question(conv) or ""

            if not isinstance(image_rel, str) or not image_rel:
                out_local.append(it)
                continue

            if image_root_prefix and not os.path.isabs(image_rel):
                image_abs = os.path.join(image_root_prefix, image_rel)
            else:
                image_abs = image_rel

            try:
                answer = _generate_answer(
                    image_path=image_abs,
                    question=question,
                    model_path=mp,
                    temperature=temperature,
                    top_p=top_p,
                    num_beams=num_beams,
                    max_new_tokens=max_new_tokens,
                )
            except Exception as e:
                answer = f"Inference error: {type(e).__name__}: {e}"

            new_conv = _set_or_append_gpt_answer(list(conv), answer)
            new_item = dict(it)
            new_item["conversations"] = new_conv
            out_local.append(new_item)
        return out_local

    # If running as a designated worker, skip data-parallel dispatch to avoid recursion
    if os.environ.get("X2DFD_WORKER", "0") == "1":
        return _infer_seq(items)

    # Data-parallel dispatch using GPU shards if multiple GPUs are available
    gpu_ids = _parse_visible_gpu_ids()
    n = len(items)
    if len(gpu_ids) <= 1 or n <= 1:
        return _infer_seq(items)

    # Create temporary shard inputs and spawn workers using this same module
    shard_ranges = _shard_indices(n, min(len(gpu_ids), n))
    tmp_dir = tempfile.mkdtemp(prefix="infer_shards_")
    worker_specs: List[Tuple[str, str, int]] = []  # (in_json, out_json, gpu_id)
    for i, r in enumerate(shard_ranges):
        shard_items = [items[j] for j in r]
        in_path = os.path.join(tmp_dir, f"shard{i}.json")
        out_path = os.path.join(tmp_dir, f"shard{i}.out.json")
        with open(in_path, "w", encoding="utf-8") as fo:
            json.dump(
                {
                    "items": shard_items,
                    "image_root_prefix": image_root_prefix,
                    "model_path": model_path,
                    "temperature": temperature,
                    "top_p": top_p,
                    "num_beams": num_beams,
                    "max_new_tokens": max_new_tokens,
                },
                fo,
                ensure_ascii=False,
            )
        worker_specs.append((in_path, out_path, gpu_ids[min(i, len(gpu_ids) - 1)]))

    # Launch subprocess per shard
    procs: List[Tuple[Any, str, range]] = []  # (Popen, out_path, range)
    this_py = os.path.abspath(__file__)
    for spec, r in zip(worker_specs, shard_ranges):
        in_json, out_json, gid = spec
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gid)
        env["ACCELERATE_USE_DEEPSPEED"] = "0"
        env["X2DFD_WORKER"] = "1"
        cmd = [sys.executable, this_py, "--worker-input", in_json, "--worker-output", out_json]
        p = os.spawnlp if False else None  # placeholder to help static analysis
        from subprocess import Popen  # local import to keep top clean

        proc = Popen(cmd, env=env)
        procs.append((proc, out_json, r))

    # Aggregate results as shards complete
    merged: List[Optional[Dict[str, Any]]] = [None] * n
    # Optional coarse-grained progress: update on shard completion
    use_bar_env = os.environ.get("USE_PROGRESS_BAR", "1").lower() in {"1", "true", "yes"}
    outer_bar = tqdm(total=n, desc="Shards", ncols=80) if tqdm and use_bar_env else None  # type: ignore
    for (proc, out_json, r) in procs:
        rc = proc.wait()
        # Load shard
        try:
            with open(out_json, "r", encoding="utf-8") as fr:
                jr = json.load(fr)
            shard_out: List[Dict[str, Any]] = jr.get("items", [])
        except Exception as e:  # worker failed or output missing
            shard_out = []
            # Fill with pass-through items but mark error in conversations
            for _idx in r:
                it = items[_idx]
                conv = (it or {}).get("conversations") or []
                new_conv = _set_or_append_gpt_answer(list(conv), f"Shard error: {type(e).__name__}: {e}")
                new_item = dict(it)
                new_item["conversations"] = new_conv
                shard_out.append(new_item)
        # Place back to original positions
        for offset, j in enumerate(range(r.start, r.stop)):
            if offset < len(shard_out):
                merged[j] = shard_out[offset]
        if outer_bar is not None:
            outer_bar.update(r.stop - r.start)  # type: ignore
    if outer_bar is not None:
        outer_bar.close()  # type: ignore

    # Fallback for any None entries
    final_out: List[Dict[str, Any]] = []
    for i in range(n):
        if merged[i] is not None:
            final_out.append(merged[i])  # type: ignore[arg-type]
        else:
            final_out.append(items[i])
    # Best-effort cleanup
    try:
        for in_json, out_json, _ in worker_specs:
            try:
                os.remove(in_json)
            except Exception:
                pass
            try:
                os.remove(out_json)
            except Exception:
                pass
        try:
            os.rmdir(tmp_dir)
        except Exception:
            pass
    except Exception:
        pass

    return final_out


def _worker_cli() -> None:
    ap = argparse.ArgumentParser(description="Worker mode for parallel inference shards")
    ap.add_argument("--worker-input", required=True, help="Input JSON with items and params")
    ap.add_argument("--worker-output", required=True, help="Output JSON path for processed items")
    args = ap.parse_args()

    with open(args.worker_input, "r", encoding="utf-8") as f:
        payload = json.load(f)
    items = payload.get("items", [])
    if not isinstance(items, list):
        raise ValueError("worker-input JSON missing 'items' list")
    params = {
        "image_root_prefix": payload.get("image_root_prefix"),
        "model_path": payload.get("model_path", DEFAULT_MODEL_PATH),
        "temperature": payload.get("temperature", 0.0),
        "top_p": payload.get("top_p", 1.0),
        "num_beams": payload.get("num_beams", 1),
        "max_new_tokens": payload.get("max_new_tokens", 512),
    }
    # Ensure worker flag so we don't recurse into parallel dispatch
    os.environ["X2DFD_WORKER"] = "1"
    started = time.time()
    out_items = infer_conversation_items(items, **params)
    took = time.time() - started
    try:
        with open(args.worker_output, "w", encoding="utf-8") as fo:
            json.dump({"items": out_items, "meta": {"took_sec": took}}, fo, ensure_ascii=False)
    except Exception:
        # Ensure parent can still read partial results
        pass


if __name__ == "__main__":
    # If called as a worker, run worker CLI; else perform a minimal self-test
    if any(x in sys.argv for x in ("--worker-input", "--worker-output")):
        _worker_cli()
    else:
        sample = [{
            "id": "1",
            "image": "datasets/raw/images/example.png",
            "conversations": [
                {"from": "human", "value": "<image>\nPlease describe the content of this image."},
                {"from": "gpt", "value": ""}
            ]
        }]
        out = infer_conversation_items(sample)
        print("Model output:", out[0].get("conversations", [])[-1].get("value", ""))
