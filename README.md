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

## Output Locations (Unified)
- QA evaluation: `eval/outputs/qa_runs/` (+ `latest_run.json`)
- Inference: `eval/outputs/infer/runs/infer_*/datasets/` (+ `latest_run.json`)
- Annotation: `eval/outputs/annotations/runs/ann_*/`
- Pipeline (end-to-end): `train/outputs/pipeline/runs/pipeline_*/`

## Common Environment Variables
- `X2DFD_BASE_MODEL`: override base model path.
- `X2DFD_DATASETS`: custom datasets root.
- `X2DFD_OUTPUT`: evaluation/inference output root (default `eval/outputs`).
- `X2DFD_TRAIN_OUTPUT`: pipeline output root (default `train/outputs`).

## 训练/测试 JSON 规范
- 放置位置
  - 训练集 JSON：`/data/250010183/workspace/X2DFD/datasets/raw/data/train/`（由 `train/configs/config.yaml: paths.data_dir` 指定）
  - 测试/推理 JSON：`/data/250010183/workspace/X2DFD/datasets/raw/data/test/`（由 `eval/configs/infer_config.yaml` 或 `train/configs/config.yaml: infer.inputs` 指定）

- 文件命名与分组
  - 训练集分组通过配置列表划分标签：
    - `train/configs/config.yaml: annotations.real_files` 列表内文件视为 real
    - `train/configs/config.yaml: annotations.fake_files` 列表内文件视为 fake
  - 测试集用于 AUC 统计时，建议测试 JSON 基名包含 `real` 或 `fake` 以便分组：
    - 例如：`Real.json` → 结果 `Real_result.json` 归入 real；`Deepfakes_tiny.json` 含 `fake` 子串 → 归入 fake
    - 若基名不含 `fake`（如 `Faceswap_tiny.json`），其结果会被 AUC 分组忽略（不影响其它数据集参与计算）

- 字段要求（Dataset 风格，推荐）
  - 顶层 `Description`：必须是图片根目录的绝对路径，例如：`/data/250010183/Datasets/FaceForensics++` 或 `/data/250010183/Datasets`
  - `images[].image_path`：可以是相对 `Description` 的相对路径，或直接为绝对路径；必须指向存在的文件
  - 列表非空；建议每个 JSON 表示单一标签的数据集（real 或 fake）

- 写出行为（适用于训练注释、测试推理与 QA 评估）
  - 结果 JSON 中的 `image` 字段写为绝对路径，不再保留 `Description`
  - 单次运行的所有产物集中写入统一的 run 目录，并提供 `latest_run.json` 指针
