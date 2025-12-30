"""LoRA-aware LLaVA inference utilities.

Provides helpers to load a model with optional adapter/base (LoRA) and generate
answers for image-question pairs. Also adds a high-throughput, multi-GPU sharding
path mirroring utils.inference so callers can process a list of conversation
items in parallel via subprocess workers.

Public helpers:
- single_image_infer:                LoRA-aware answer generation for one item
- single_image_infer_with_scores:    LoRA-aware answer + (real/fake) scores
- lora_infer_conversation_items:     Batch items; multi-GPU shard if available

Worker mode is supported via:
  python utils/lora_inference.py --worker-input /tmp/in.json --worker-output /tmp/out.json
which is used internally by lora_infer_conversation_items.
"""

from __future__ import annotations

import os
import re
from typing import Any, Iterable, List, Optional, Tuple, Dict

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
import inspect
import types


IMAGE_PLACEHOLDER = "<image>"

# Cache model by (model_path, model_base)
_MODEL_CACHE: dict[tuple[str, Optional[str]], tuple[Any, ...]] = {}

# Reuse helpers for conversation turn handling
try:
    from .inference import first_human_question, set_or_append_gpt_answer  # type: ignore
except Exception:  # pragma: no cover
    def first_human_question(conversations: List[Dict[str, Any]]) -> Optional[str]:
        try:
            for turn in conversations:
                if isinstance(turn, dict) and turn.get("from") == "human":
                    v = turn.get("value")
                    if isinstance(v, str) and v.strip():
                        return v
        except Exception:
            pass
        return None

    def set_or_append_gpt_answer(conversations: List[Dict[str, Any]], answer: str) -> List[Dict[str, Any]]:
        try:
            for turn in conversations:
                if isinstance(turn, dict) and turn.get("from") == "gpt":
                    turn["value"] = answer
                    return conversations
        except Exception:
            pass
        conversations.append({"from": "gpt", "value": answer})
        return conversations


def _parse_visible_gpu_ids() -> List[int]:
    """Parse visible GPU ids from CUDA_VISIBLE_DEVICES or by probing torch.

    Mirrors utils.inference behavior to ensure identical sharding strategy.
    """
    import torch  # local import
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env:
        ids: List[int] = []
        for x in env.replace(" ", "").split(","):
            if x != "":
                try:
                    ids.append(int(x))
                except Exception:
                    pass
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
    """Evenly split [0, n_items) into n_shards contiguous ranges."""
    n_shards = max(int(n_shards), 1)
    n_items = max(int(n_items), 0)
    return [
        range((i * n_items) // n_shards, ((i + 1) * n_items) // n_shards)
        for i in range(n_shards)
    ]


def get_model_name_from_path(model_path: str) -> str:
    return os.path.basename(model_path)


def load_images(image_files: Iterable[str]) -> List[Image.Image]:
    return [Image.open(image_file).convert("RGB") for image_file in image_files]


def _resolve_model_device(model) -> torch.device:
    if hasattr(model, "device"):
        return model.device
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _resolve_model_dtype(model) -> torch.dtype:
    try:
        return next(model.parameters()).dtype  # type: ignore[attr-defined]
    except (StopIteration, AttributeError):
        return torch.float16


def _patch_forward_cache_position(model: Any) -> None:
    """Make LLaVA models compatible with newer Transformers generation kwargs.

    transformers>=4.43 may pass `cache_position` into forward(); older LLaVA
    model implementations don't accept it. Patch at runtime without modifying
    the upstream LLaVA checkout.
    """
    try:
        sig = inspect.signature(model.forward)
        if "cache_position" in sig.parameters:
            return
    except Exception:
        # If we can't introspect, try patching anyway
        pass

    orig_forward = model.forward

    # Capture original signature so we only forward supported args.
    try:
        sig = inspect.signature(orig_forward)
        accepted = set(sig.parameters.keys())
        has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    except Exception:
        accepted = set()
        has_var_kw = True

    def _forward(  # type: ignore[no-untyped-def]
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        images=None,
        image_sizes=None,
        return_dict=None,
        cache_position=None,
        **kwargs,
    ):
        # Drop unsupported kwarg(s) introduced by newer Transformers.
        kwargs.pop("cache_position", None)

        call_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "inputs_embeds": inputs_embeds,
            "labels": labels,
            "use_cache": use_cache,
            "output_attentions": output_attentions,
            "output_hidden_states": output_hidden_states,
            "images": images,
            "image_sizes": image_sizes,
            "return_dict": return_dict,
        }
        if accepted:
            call_kwargs = {k: v for k, v in call_kwargs.items() if k in accepted}
            if has_var_kw and kwargs:
                call_kwargs.update(kwargs)
        elif kwargs:
            # If we can't introspect, fall back to forwarding kwargs as-is.
            call_kwargs.update(kwargs)

        return orig_forward(**call_kwargs)

    try:
        model.forward = types.MethodType(_forward, model)
    except Exception:
        return


def _get_or_load_model(model_path: str, model_base: Optional[str] = None):
    key = (model_path, model_base)
    if key not in _MODEL_CACHE:
        disable_torch_init()
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path, model_base, model_name, False
        )
        _patch_forward_cache_position(model)
        model.eval()
        _MODEL_CACHE[key] = (tokenizer, model, image_processor, context_len)
    return _MODEL_CACHE[key]


def single_image_infer(
    image_path: str,
    question: str,
    *,
    model_path: str,
    model_base: Optional[str] = None,
    temperature: float = 0.0,
    top_p: float = 1.0,
    num_beams: int = 1,
    max_new_tokens: int = 512,
) -> str:
    """Run LoRA-aware inference on a single image-question pair using LLaVA."""

    tokenizer, model, image_processor, _ = _get_or_load_model(model_path, model_base)

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
        qs = image_token_se + "\n" + question

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
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=model_device)

    with torch.inference_mode():
        generation_output = model.generate(
            input_ids,
            images=image_inputs,
            image_sizes=image_sizes,
            attention_mask=attention_mask,
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


def _token_id_for_word(tokenizer, word: str) -> Optional[int]:
    """Best-effort resolve a single token id for a literal word.

    Tries variants with/without leading space and capitalization; if the
    tokenizer() API returns a sequence with BOS, skips it.
    """
    variants = [word, " " + word, word.capitalize(), " " + word.capitalize()]
    for v in variants:
        try:
            ids = tokenizer.encode(v, add_special_tokens=False)
            if isinstance(ids, list) and len(ids) == 1:
                return ids[0]
        except Exception:
            pass
    try:
        ids_full = tokenizer(word).input_ids
        if len(ids_full) >= 2:
            return ids_full[1]
        if len(ids_full) == 1:
            return ids_full[0]
    except Exception:
        pass
    return None


def single_image_infer_with_scores(
    image_path: str,
    question: str,
    *,
    model_path: str,
    model_base: Optional[str] = None,
    temperature: float = 0.0,
    top_p: float = 1.0,
    num_beams: int = 1,
    max_new_tokens: int = 4,
) -> Dict[str, Any]:
    """Run inference and compute real/fake token probabilities (legacy style).

    Uses fixed step index and tokenizer indices similar to the user's snippet.
    Returns a dict with keys: answer, real_score, fake_score.
    """
    tokenizer, model, image_processor, _ = _get_or_load_model(model_path, model_base)

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
        qs = image_token_se + "\n" + question

    images = load_images([image_path])
    image_sizes = [x.size for x in images]
    images_tensor = process_images(images, image_processor, model.config)
    if isinstance(images_tensor, list):
        image_inputs = [img.to(device=model_device, dtype=model_dtype) for img in images_tensor]
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
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=model_device)

    with torch.inference_mode():
        generation_output = model.generate(
            input_ids,
            images=image_inputs,
            image_sizes=image_sizes,
            attention_mask=attention_mask,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            max_new_tokens=max(max_new_tokens, 1),
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=True,
        )

    # Decode answer text
    answer = tokenizer.batch_decode(
        generation_output.sequences, skip_special_tokens=True
    )[0].strip()

    # Optional perplexity, legacy style
    def compute_perplexity(sentence: str) -> Optional[float]:
        try:
            ids = tokenizer.encode(sentence, return_tensors="pt").to(model_device)
            with torch.no_grad():
                out = model(ids, labels=ids)
            loss = out.loss
            return float(torch.exp(loss).item())
        except Exception:
            return None

    _ = compute_perplexity(answer)  # computed but not returned (matches snippet intent)

    # Legacy probability extraction: fixed step [3], direct tokenizer indices.
    # Make it robust to shorter generations by falling back to the last available step.
    real_prob: Optional[float] = None
    fake_prob: Optional[float] = None
    try:
        scores = getattr(generation_output, "scores", None)
        if not isinstance(scores, (list, tuple)) or len(scores) == 0:
            raise ValueError("No generation scores available")
        step_idx = 3 if len(scores) > 3 else (len(scores) - 1)
        logits = scores[step_idx][0]
        probs = torch.nn.functional.softmax(
            torch.tensor([
                logits[tokenizer("real").input_ids[1]],
                logits[tokenizer("fake").input_ids[1]],
            ]),
            dim=0,
        ).detach().cpu().numpy()
        real_prob = float(probs[0])
        fake_prob = float(probs[1])
    except Exception:
        real_prob, fake_prob = None, None

    out: Dict[str, Any] = {"answer": answer}
    if real_prob is not None and fake_prob is not None:
        out["real_score"] = round(real_prob, 8)
        out["fake_score"] = round(fake_prob, 8)
    return out


def lora_infer_conversation_items(
    items: List[Dict[str, Any]],
    *,
    image_root_prefix: Optional[str] = None,
    model_path: str = "",
    model_base: Optional[str] = None,
    temperature: float = 0.0,
    top_p: float = 1.0,
    num_beams: int = 1,
    max_new_tokens: int = 512,
    add_scores_turns: bool = True,
) -> List[Dict[str, Any]]:
    """Fill GPT answers (and optionally score turns) for conversation items.

    Strategy mirrors utils.inference.infer_conversation_items:
      - If multiple GPUs are visible, shard items across GPUs via subprocesses
      - Else run sequentially within this process

    When add_scores_turns is True, append two extra turns per item when
    available:
      {"from": "real score", "value": "<prob>"}
      {"from": "fake score", "value": "<prob>"}
    """

    def _infer_seq(seq: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out_local: List[Dict[str, Any]] = []
        # Progress bar
        use_bar_env = os.environ.get("USE_PROGRESS_BAR", "1").lower() in {"1", "true", "yes"}
        show_bar = bool(tqdm) and use_bar_env
        iterator = enumerate(seq)
        total = len(seq)
        if show_bar:
            iterator = enumerate(tqdm(seq, desc="LoRA Inference", total=total, ncols=80))  # type: ignore

        for idx, it in iterator:  # type: ignore[assignment]
            image_rel = (it or {}).get("image")
            conv = (it or {}).get("conversations") or []
            question = first_human_question(conv) or ""

            if not isinstance(image_rel, str) or not image_rel:
                out_local.append(it)
                continue

            if image_root_prefix and not os.path.isabs(image_rel):
                image_abs = os.path.join(image_root_prefix, image_rel)
            else:
                image_abs = image_rel

            try:
                meta = single_image_infer_with_scores(
                    image_path=image_abs,
                    question=question,
                    model_path=model_path,
                    model_base=model_base,
                    temperature=temperature,
                    top_p=top_p,
                    num_beams=num_beams,
                    max_new_tokens=max_new_tokens,
                )
                answer = meta.get("answer", "")
                real_score = meta.get("real_score")
                fake_score = meta.get("fake_score")
            except Exception as e:
                answer = f"Inference error: {type(e).__name__}: {e}"
                real_score = None
                fake_score = None

            new_conv = set_or_append_gpt_answer(list(conv), answer)
            if add_scores_turns:
                if real_score is not None:
                    try:
                        new_conv.append({"from": "real score", "value": f"{float(real_score):.4f}"})
                    except Exception:
                        new_conv.append({"from": "real score", "value": str(real_score)})
                if fake_score is not None:
                    try:
                        new_conv.append({"from": "fake score", "value": f"{float(fake_score):.4f}"})
                    except Exception:
                        new_conv.append({"from": "fake score", "value": str(fake_score)})

            new_item = dict(it)
            new_item["conversations"] = new_conv
            out_local.append(new_item)
        return out_local

    # Worker short-circuit to avoid recursion
    if os.environ.get("X2DFD_WORKER", "0") == "1":
        return _infer_seq(items)

    # Data-parallel dispatch
    gpu_ids = _parse_visible_gpu_ids()
    n = len(items)
    if len(gpu_ids) <= 1 or n <= 1:
        return _infer_seq(items)

    shard_ranges = _shard_indices(n, min(len(gpu_ids), n))
    tmp_dir = tempfile.mkdtemp(prefix="lora_infer_shards_")
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
                    "model_base": model_base,
                    "temperature": temperature,
                    "top_p": top_p,
                    "num_beams": num_beams,
                    "max_new_tokens": max_new_tokens,
                    "add_scores_turns": add_scores_turns,
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
        from subprocess import Popen  # local import
        proc = Popen(cmd, env=env)
        procs.append((proc, out_json, r))

    # Aggregate results
    merged: List[Optional[Dict[str, Any]]] = [None] * n
    use_bar_env = os.environ.get("USE_PROGRESS_BAR", "1").lower() in {"1", "true", "yes"}
    outer_bar = tqdm(total=n, desc="LoRA Shards", ncols=80) if tqdm and use_bar_env else None  # type: ignore
    for (proc, out_json, r) in procs:
        rc = proc.wait()
        try:
            with open(out_json, "r", encoding="utf-8") as fr:
                jr = json.load(fr)
            shard_out: List[Dict[str, Any]] = jr.get("items", [])
        except Exception as e:
            shard_out = []
            # Fill pass-through with error marks
            for _idx in r:
                it = items[_idx]
                conv = (it or {}).get("conversations") or []
                new_conv = set_or_append_gpt_answer(list(conv), f"Shard error: {type(e).__name__}: {e}")
                new_item = dict(it)
                new_item["conversations"] = new_conv
                shard_out.append(new_item)
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
    ap = argparse.ArgumentParser(description="LoRA worker mode for parallel inference shards")
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
        "model_path": payload.get("model_path", ""),
        "model_base": payload.get("model_base"),
        "temperature": payload.get("temperature", 0.0),
        "top_p": payload.get("top_p", 1.0),
        "num_beams": payload.get("num_beams", 1),
        "max_new_tokens": payload.get("max_new_tokens", 512),
        "add_scores_turns": bool(payload.get("add_scores_turns", True)),
    }
    os.environ["X2DFD_WORKER"] = "1"
    started = time.time()
    out_items = lora_infer_conversation_items(items, **params)
    took = time.time() - started
    try:
        with open(args.worker_output, "w", encoding="utf-8") as fo:
            json.dump({"items": out_items, "meta": {"took_sec": took}}, fo, ensure_ascii=False)
    except Exception:
        pass


if __name__ == "__main__":
    # Only worker mode is exposed in this module; batch API is imported directly
    if any(x in sys.argv for x in ("--worker-input", "--worker-output")):
        _worker_cli()
