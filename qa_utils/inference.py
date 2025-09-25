"""Utilities for running LLaVA-based inference on single images."""

from __future__ import annotations

import os
import re
from typing import Any, Dict, Iterable, List, Tuple

import random

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


IMAGE_PLACEHOLDER = "<image>"

DEFAULT_MODEL_PATH = "/data/250010183/hetao/Weight/llava-v1.5-7b"

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


def single_image_infer(
    image_path: str,
    question: str,
    model_path: str = DEFAULT_MODEL_PATH,
    temperature: float = 0,
    top_p: float = 1,
    num_beams: int = 1,
    max_new_tokens: int = 512,
) -> str:
    """Run inference on a single image-question pair using a LLaVA model."""

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

def describe_image_answer(
    rng: random.Random,
    dataset_label: str,
    question: str,
    answer: str,
) -> str:
    """Compose a narrative response combining dataset label, question, and answer."""

    prefix_map = {
        "real": "This image is real.",
        "fake": "This image is fake.",
        "unlabeled": "This image is unlabeled.",
    }
    prefix = prefix_map.get(dataset_label, prefix_map["unlabeled"])

    filler = rng.choice(
        [
            "Answer generated using the vision-language model.",
            "Model response provided for analysis purposes.",
            "Narrative generated alongside the model prediction.",
        ]
    )

    return f"{prefix} {filler} Question: {question} Answer: {answer}."


if __name__ == "__main__":
    image_path = (
        "/data/250010183/Datasets/Celeb-DF-v3-process/Celeb-synthesis/"
        "FaceSwap/Celeb-DF-v2/id0_id1_0002/011.png"
    )
    question = "请描述这张图像中的内容"
    result = single_image_infer(image_path, question)
    print("模型输出：", result)
