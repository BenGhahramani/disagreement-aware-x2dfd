# X2DFD - Face Forgery Detection QA System

## Overview
X2DFD is a LLaVA-based system for deepfake detection with two main capabilities:
- QA accuracy evaluation across real/fake datasets.
- Image annotation generation with structured explanations, optionally augmented with a traditional detector score.

## Quick Start

1) Environment Installation
- Run: `bash install.sh`
- Activate: `conda activate X2DFD`

2) Weight Download
- LLaVA-1.5-7B base model (Hugging Face):
  https://huggingface.co/liuhaotian/llava-v1.5-7b
  - Place contents at: `weights/base/llava-v1.5-7b`
  - Or set: `X2DFD_BASE_MODEL=/path/to/llava-v1.5-7b`
- CLIP ViT-L/14-336 vision tower (Hugging Face):
  https://huggingface.co/openai/clip-vit-large-patch14-336
  - Place contents at: `weights/base/clip-vit-large-patch14-336`
  - Example absolute path: `/data/250010183/workspace/X2DFD/weights/base/clip-vit-large-patch14-336`

3) Data Preparation
- Images: `datasets/raw/images` (replicate dataset structures, e.g., FaceForensics++)
- Metadata JSONs: `datasets/raw/data/train/*.json` for training and `datasets/raw/data/test/*.json` for testing
- See “Dataset & Path Rules” for JSON schema.

4) Train & Test
- Train pipeline (annotation -> merge -> LoRA train -> LoRA test):
  `./train.sh --run-train`
  - Test only: `./train.sh --test-only`
  - Reuse existing annotations: `./train.sh --skip-annotate`
- Quick test (LoRA inference + AUC):
  `./test.sh`

5) Single-Image Demo
- Run with base only:
  `python demo.py --image /abs/path/to/image.png --model-base weights/base/llava-v1.5-7b`
- Run with LoRA adapter:
  `python demo.py --image /abs/path/to/image.png --model-base weights/base/llava-v1.5-7b --adapter-path weights/checkpoints/ckpt/FR/llava-v1.5-7b-lora-[small]`

## Dataset & Path Rules
- Dataset-style JSON: top-level `Description` (absolute image root), images under `images[].image_path`.
- Conversation-style JSON: each entry contains absolute `image` and `conversations` turns.
- Write-out behavior: every produced result (evaluation/inference/annotation) records absolute image paths; no global prefixing, `Description` is not kept in outputs.

Minimal dataset JSON example:
```
{
  "Description": "/abs/path/to/datasets/raw/images/FaceForensics++",
  "images": [
    { "image_path": "manipulated_sequences/Deepfakes/c23/00001.png" },
    { "image_path": "/abs/path/to/another/image.png" }
  ]
}
```
- If `Description` is missing, all `images[].image_path` must be absolute.
- Conversation-style example: `[{"id":"1","image":"/abs/img.png","conversations":[{"from":"human","value":"<image>\n..."},{"from":"gpt","value":""}]}]` (image must be absolute).
