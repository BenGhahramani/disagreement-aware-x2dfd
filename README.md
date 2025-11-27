<div align="center">
  <h2>
    <img src="figs/fig_teaser.png" alt="Image Alt Text" width="50" height="50" align="absmiddle">
    X2-DFD: A framework for eXplainable and eXtendable Deepfake Detection
  </h2>
</div>

<div align="center">

Yize Chen<sup>1*</sup>, Zhiyuan Yan<sup>3*</sup>, Guangliang Cheng<sup>4</sup>, Kangran Zhao<sup>1</sup>,<br>
Siwei Lyu<sup>5</sup>, Baoyuan Wu<sup>1†</sup>

<br>
<sup>1</sup> The Chinese University of Hong Kong, Shenzhen, Guangdong, 518172, P.R. China <br>
<sup>3</sup> School of Electronic and Computer Engineering, Peking University, P.R. China<br>
<sup>4</sup> Department of Computer Science, University of Liverpool, Liverpool, L69 7ZX, UK <br>
<sup>5</sup> Department of Computer Science and Engineering, University at Buffalo, State University of New York, Buffalo, NY, USA

<br>
<sup>*</sup> Equal contribution &nbsp;&nbsp;&nbsp; <sup>†</sup> Corresponding author

</div>

<div align="center">

[![NeurIPS 2025](https://img.shields.io/badge/NeurIPS%202025-Poster-34d058)](https://neurips.cc/)
[![arXiv](https://img.shields.io/badge/Arxiv-2410.06126-AD1C18.svg?logo=arXiv)](https://arxiv.org/abs/2410.06126)
[![GitHub issues](https://img.shields.io/github/issues/chenyize111/X2DFD?color=critical&label=Issues)](https://github.com/chenyize111/X2DFD/issues)
[![GitHub Stars](https://img.shields.io/github/stars/chenyize111/X2DFD?style=social)](https://github.com/chenyize111/X2DFD/stargazers)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-info-lightgrey)](#dataset)
[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-checkpoints%20soon-9cf)](#model)
[![License](https://img.shields.io/badge/License-MIT-yellow)](#license)

</div>

## Contents
- [Install](#install)
- [Weights](#llava-weights)
- [Demo](#demo)
- [Quick Inference](#quick-infer)
- [Model Zoo](#model-zoo)
- [Dataset](#dataset)
- [Train](#train)
- [Evaluation](#evaluation)

## 📰 News
- [2025.11.27]: 🎉 X2-DFD was accepted to NeurIPS 2025 (Poster)!
- [2025.11.27]: 🚀 Public release of the X2-DFD codebase and docs.
- [2024.10.XX]: 📝 Preprint available on arXiv: 2410.06126.

## <img id="overview_icon" width="3%" src="https://cdn-icons-png.flaticon.com/256/599/599205.png"> X2-DFD Overview

X2‑DFD is an eXplainable and eXtendable deepfake detection framework built on MLLMs. It outputs both a binary verdict (real/fake) and concise, human‑readable explanations. At its core, X2‑DFD focuses on two principles:

- **Prompt‑guided explainability.** We design prompts that steer the model to attend to the features it is naturally good at **(strong feature)**, and use the generated rationales to construct an explainable dataset.
- **Extensibility via Specific Feature Detectors (SFDs).** We keep the system plug‑and‑play by supplementing low‑level artifact cues (e.g., blending/diffusion trace) with dedicated detectors **(weak feature)**, further enhancing the model’s capability.


<div align="center">
<img src="figs/fig_framework_overview.png" alt="X2-DFD framework: experts + LLaVA reasoning pipeline" width="90%"/>
</div>

## <img id="contrib_icon" width="3%" src="https://cdn-icons-png.flaticon.com/256/2435/2435606.png"> Contributions

- **Systematic assessment of MLLMs’ intrinsic capabilities for deepfake detection.** We provide an in‑depth feature‑wise analysis revealing that pretrained MLLMs tend to discriminate **semantic** cues well (e.g., skin/contours) but are weaker on **signal‑level** artifacts (e.g., blending/lighting), motivating targeted strengthening.
- **Enhancing explainability by reinforcing strong features.** We design **SFS** to strengthen the MLLM’s well‑learned features and fine‑tune it to generate clear, artifact‑aware rationales, improving both detectability and explainability.
- **Supplementing weaknesses with Specific Feature Detectors.** Through **WFS**, we incorporate **SFDs** to complement the model’s limitations, yielding a robust and **plug‑and‑play** system that supports future MLLMs and detectors.

<a id="install"></a>

## 🛠️ Installation

1) Environment (Conda, Python 3.10 recommended)
```bash
bash install.sh
conda activate X2DFD
```

Troubleshooting / 安装问题排查
- 遇到环境安装或依赖冲突（如 CUDA、PyTorch、Deepspeed、bitsandbytes 等）？LLaVA 社区有大量成熟讨论与解答，click here: https://github.com/haotian-liu/LLaVA/issues


<a id="llava-weights"></a>

2) **Weights**
- **Base model:** LLaVA-1.5-7B — download from [here](https://huggingface.co/liuhaotian/llava-v1.5-7b), then place under `weights/base/llava-v1.5-7b`.
  - Or set env var: `X2DFD_BASE_MODEL=/abs/path/to/llava-v1.5-7b`
- **Vision tower:** CLIP ViT-L/14-336 — download from [here](https://huggingface.co/openai/clip-vit-large-patch14-336) and place under `weights/base/clip-vit-large-patch14-336`.
- Feature detectors: blending feature detector; diffusion feature detector.

<a id="demo"></a>

3) Single-Image Demo (LoRA required)
```bash
python demo.py --image /abs/img.png \
  --model-base weights/base/llava-v1.5-7b \
  --adapter-path weights/checkpoints/ckpt/FR/llava-v1.5-7b-lora-[small]
```

<a id="quick-infer"></a>

## ⚡ Quick Inference

Two quick runs with different adapters and Specific Feature Detectors (SFDs):

```bash
# (A) Blending SFD + blending-tuned adapter
python -m eval.infer.runner \
  --config eval/configs/infer_config.yaml \
  --model-path weights/checkpoints/ckpt/FR/llava-v1.5-7b-lora-blending \
  --experts blending

# (B) Diffusion SFD + diffusion-tuned adapter
python -m eval.infer.runner \
  --config eval/configs/infer_config.yaml \
  --model-path weights/checkpoints/ckpt/FR/llava-v1.5-7b-lora-diffusion \
  --experts diffusion_detector
```

Tip: you can also run both SFDs simultaneously via `--experts blending,diffusion_detector`.

## 📥 Required Weights

| Component | Where to get | Put under (default) | Env var override | Used in |
| --- | --- | --- | --- | --- |
| **LLaVA-1.5-7B (base)** | Hugging Face: liuhaotian/llava-v1.5-7b | `weights/base/llava-v1.5-7b` | `X2DFD_BASE_MODEL` | annotation, training, evaluation |
| **CLIP ViT-L/14-336 (vision tower)** | Hugging Face: openai/clip-vit-large-patch14-336 | `weights/base/clip-vit-large-patch14-336` | `VISION_TOWER` (training), or via config | training |
| **Blending detector (SwinV2-B, 256)** | [Coming soon](#coming-soon) (checkpoint file, e.g., `best_gf.pth`) | `weights/blending_models/best_gf.pth` | set in config `weak_supplies[].weights_path` | weak-signal scores (optional) |
| **Diffusion/aligner detector (ours-sync)** | [Coming soon](#coming-soon) (folder with config/ckpt) | `weights/ours-sync/` | set via `weak_supplies[].weights_dir` + `model: ours-sync` | weak-signal scores (optional) |

Notes
- If某个专家（SFD）权重缺失，可在 config 的 `weak_supplies` 中移除该专家，仅用剩余专家或不带专家的提示运行（仍需 LoRA）。
- Paths can be absolute; environment variables in configs are expanded at runtime.

### 🔀 Optional Expert Variants

We provide two ready-to-use expert settings:

- **Blending-only variant** (lighter): requires only the blending detector checkpoint.
  - Download blending weights: [Coming soon](#coming-soon)
  - Place at: `weights/blending_models/best_gf.pth`
  - Run with blending only (examples):
    - Eval: `python -m eval.infer.runner --config eval/configs/infer_config.yaml --experts blending`
    - Train: `python -m train.pipeline --config train/configs/config.yaml --experts blending --run-train`

- **Blending + Diffusion variant** (stronger): uses both experts.
  - Download blending weights: [Coming soon](#coming-soon) → `weights/blending_models/best_gf.pth`
  - Download diffusion/aligner package: [Coming soon](#coming-soon) → `weights/ours-sync/`
  - Eval/Train (default configs already include both experts). You can also pass `--experts blending,diffusion_detector` explicitly.

## 🚀 Usage

<a id="evaluation"></a>

### 1) **Evaluation (one-liner)**
- One-liner (LoRA inference):
```bash
./test.sh
```
  - Requirement: a valid LoRA adapter must be available at the path configured in `eval/configs/infer_config.yaml -> model.adapter` (default: `weights/checkpoints/ckpt/FR/llava-v1.5-7b-lora-[small]`).
    - If the adapter (or base model) is missing, the runner will stop with a clear message and instructions to download/configure the paths.
  - Inference: `eval/outputs/infer/latest_run.json`

### 2) **Evaluation (manual)**
```bash
python -m eval.infer.runner --config eval/configs/infer_config.yaml
```

Question template (example, includes expert score):
```
<image>
Is this image real or fake? And the {alias} score is {score}.
```

<a id="train"></a>

### 3) **Training (staged)**
End-to-end (annotation → weak merge → LoRA train → LoRA test):
```bash
./train.sh --run-train [--train-gpus 0,1]
```
- More scripts: `train/` and `train/scripts/`; legacy scripts are archived in `legacy/` (not recommended).

Stages mirror our methodology:
- **Stage 2 — Explainable annotation** (base LLaVA generates rationales)
- **Stage 3 — Weak feature supply** (multi-expert artifact scores merged into prompts)
- **Stage 4 — LoRA training** (fine-tune lightweight adapter)

---

## 🧭 Project Structure
- `src/` core code
  - `diffusion/`: networks, preprocessing, detectors
  - `blending/`: detectors and utils
  - `EXPERTS_GUIDE.md`: registry pattern for experts/providers
- `utils/`: shared helpers (`model_scoring.py`, `lora_inference.py`, `paths.py`, `pipeline_utils.py`)
- `train/`: pipeline + training (`pipeline.py`, `model_train.py`, `configs/`, `outputs/`)
- `eval/`: inference (`infer/runner.py`, `configs/`, `outputs/`)
- `datasets/`: raw images, metadata JSONs, prompts; `weights/`: model files (not tracked)

Coding style: Python 3.10, PEP 8, 4-space indent; prefer type hints and f-strings; keep functions small with minimal side effects.

---

## ⚙️ Config & Environment Variables
- Common env vars (see `utils/paths.py`):
  - `X2DFD_BASE_MODEL`, `X2DFD_WEIGHTS`, `X2DFD_OUTPUT`, `X2DFD_DATASETS`
- GPU auto-detection; override with `GPUS` or `--train-gpus`.
- All outputs record absolute image paths. Dataset JSON may use a top-level `Description` as image root.

<a id="dataset"></a>

## 📦 Dataset
Main datasets used in our experiments:

| Dataset | Link | Notes |
| --- | --- | --- |
| **FF++ (FaceForensics++)** | find [here](https://github.com/ondyari/FaceForensics) | Widely used benchmark for face manipulation detection (train/eval). |
| DiFF (diffusion images) | [here](https://github.com/xaCheng1996/DiFF) | A small subset of diffusion data was used; inference supports both SFDs. |

  Minimal dataset JSON example:
```json
{
  "Description": "/abs/path/to/datasets/raw/images/FaceForensics++",
  "images": [
    { "image_path": "manipulated_sequences/Deepfakes/c23/00001.png" },
    { "image_path": "/abs/path/to/another/image.png" }
  ]
}
```
- If `Description` is missing, every `images[].image_path` must be absolute.
- Outputs (evaluation/inference/annotation) always contain absolute image paths.

Tiny sanity JSONs are provided under `datasets/raw/data/test/Tiny_Test/`.
For large-scale evaluation, set `X2DFD_DATASETS` and use `eval/configs/infer_config.example.yaml`.

---

<a id="model-zoo"></a>

## 🧩 Model Zoo
- X2‑DFD LoRA (Blending‑only): [Coming soon](#coming-soon). See “Optional Expert Variants” for how to run with `--experts blending`.
- X2‑DFD LoRA (Blending + Diffusion): [Coming soon](#coming-soon). Default configs use both experts; or pass `--experts blending,diffusion_detector`.

<a id="model"></a>

## 🧠 Model
- **Base:** LLaVA-1.5-7B (`weights/base/llava-v1.5-7b`).
- **Checkpoints:** **LoRA** adapters will be released soon. Paths default to `weights/checkpoints/ckpt/...`.
- You can point to your own adapters via CLI override or config:
  - demo: `--adapter-path`
  - runner: `--model-path` or config `model.adapter`.

<a id="citation"></a>

## Citation
If you find this repository useful, please cite:

```bibtex
@article{chen2024x2,
  title={X2-dfd: A framework for explainable and extendable deepfake detection},
  author={Chen, Yize and Yan, Zhiyuan and Cheng, Guangliang and Zhao, Kangran and Lyu, Siwei and Wu, Baoyuan},
  journal={arXiv preprint arXiv:2410.06126},
  year={2024}
}
```

A `CITATION.cff` file is also provided for GitHub's "Cite this repository" widget.

---

## 😄 Acknowledgements
- We thank the **LLaVA** team for their open-source project and training/inference pipeline:
  https://github.com/haotian-liu/LLaVA
- Parts of this repo build on community practices (fusion, detectors, training tools). If we missed an attribution, please open an issue.

---


<a id="license"></a>

## 📝 License
MIT License. See `LICENSE`.

---

<a id="coming-soon"></a>

## ⏳ Coming Soon
- We will release the expert checkpoints (blending and diffusion/aligner) and provide verified download links here.
- Once published, the links in the Required Weights table and Optional Expert Variants will point to the release assets.
