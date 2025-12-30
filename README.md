<div align="center">
  <h2>
    <img src="figs/fig_teaser.png" alt="Image Alt Text" width="50" height="50" align="absmiddle">
    X2-DFD: A framework for eXplainable and eXtendable Deepfake Detection
  </h2>
</div>

<div align="center">

Yize Chen<sup>1*</sup>, Zhiyuan Yan<sup>2*</sup>, Guangliang Cheng<sup>3</sup>, Kangran Zhao<sup>1</sup>,<br>
Siwei Lyu<sup>4</sup>, Baoyuan Wu<sup>1†</sup>

<br>
<sup>1</sup> The Chinese University of Hong Kong, Shenzhen, Guangdong, 518172, P.R. China <br>
<sup>2</sup> School of Electronic and Computer Engineering, Peking University, P.R. China<br>
<sup>3</sup> Department of Computer Science, University of Liverpool, Liverpool, L69 7ZX, UK <br>
<sup>4</sup> Department of Computer Science and Engineering, University at Buffalo, State University of New York, Buffalo, NY, USA

<br>
<sup>*</sup> Equal contribution &nbsp;&nbsp;&nbsp; <sup>†</sup> Corresponding author

</div>

<div align="center">

[![NeurIPS 2025](https://img.shields.io/badge/NeurIPS%202025-Poster-34d058)](https://neurips.cc/)
[![arXiv](https://img.shields.io/badge/Arxiv-2410.06126-AD1C18.svg?logo=arXiv)](https://arxiv.org/abs/2410.06126)
[![GitHub issues](https://img.shields.io/github/issues/chenyize111/X2DFD?color=critical&label=Issues)](https://github.com/chenyize111/X2DFD/issues)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-info-lightgrey)](#dataset)
[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-checkpoints%20soon-9cf)](#model)
[![License](https://img.shields.io/badge/License-MIT-yellow)](#license)

</div>
## 📰 News

- [2025.11.27]: 🚀 Public release of the X2-DFD codebase and docs.
- [2025.9.24]: 🎉 X2-DFD was accepted to NeurIPS 2025!

## TODO

- [x] Public code release
- [ ] Upload checkpoints (Ready to Release)

- ## Contents
- [Install](#install)
- [Quick Inference](#quick-infer)
- [Usage](#🚀-usage)
  - [Training](#1-training-staged)
  - [Evaluation](#2-evaluation)
- [Dataset](#dataset)
- [Model](#model)



## <img id="overview_icon" width="3%" src="https://cdn-icons-png.flaticon.com/256/599/599205.png"> X2‑DFD Overview

X2‑DFD is an eXplainable and eXtendable deepfake detection framework built on MLLMs. It outputs both a binary verdict (real/fake) and concise, human‑readable explanations. At its core, X2‑DFD focuses on two principles:

- **Prompt‑guided explainability.** We design prompts that steer the model to attend to the features it is naturally good at **(strong feature)**, and use the generated rationales to construct an explainable dataset.
- **Extensibility via Specific Feature Detectors (SFDs).** We keep the system plug‑and‑play by supplementing low‑level artifact cues (e.g., blending/diffusion trace) with dedicated detectors **(weak feature)**, further enhancing the model’s capability.


<div align="center">
<img src="figs/fig_framework_overview.png" alt="X2-DFD framework: experts + LLaVA reasoning pipeline" width="90%"/>
</div>

## <img id="contrib_icon" width="3%" src="https://cdn-icons-png.flaticon.com/256/2435/2435606.png"> Contributions

- **Systematic assessment of MLLMs’ intrinsic capabilities for deepfake detection.** We provide an in‑depth, feature‑wise analysis revealing that pretrained MLLMs tend to discriminate **semantic** cues well (e.g., skin/contours) but are weaker on **signal‑level** artifacts (e.g., blending/lighting), motivating targeted strengthening.
- **Enhancing explainability by reinforcing strong features.** We design **SFS** to strengthen the MLLM’s well‑learned features and fine‑tune it to generate clear, artifact‑aware rationales, improving both detectability and explainability.
- **Supplementing weaknesses with Specific Feature Detectors.** Through **WFS**, we incorporate **SFDs** to complement the model’s limitations, yielding a robust and **plug‑and‑play** system that supports future MLLMs and detectors.

<a id="install"></a>

## 🛠️ Installation

1) Environment (Conda, Python 3.10 recommended)
```bash
git clone https://github.com/chenyize111/X2DFD.git
cd X2DFD
bash install.sh
conda activate X2DFD
```

Troubleshooting
- Having issues with environment setup or dependency conflicts (e.g., CUDA, PyTorch, DeepSpeed, bitsandbytes)? The LLaVA community has many resolved discussions and helpful tips — click here: https://github.com/haotian-liu/LLaVA/issues


Before You Run: Prepare Weights
- Download and place the following weights before running (paths are defaults; you can override via env vars or config):

| Component | Where to get | Put under (default) | Override (env/CLI) | Used in |
| --- | --- | --- | --- | --- |
| LLaVA-1.5-7B (base) | Hugging Face: liuhaotian/llava-v1.5-7b | `weights/base/llava-v1.5-7b` | `X2DFD_BASE_MODEL` | annotation, training, evaluation |
| CLIP ViT-L/14-336 (vision tower) | Hugging Face: openai/clip-vit-large-patch14-336 | `weights/base/clip-vit-large-patch14-336` | `VISION_TOWER` (training) or config | training |
| X2‑DFD LoRA (Blending+Diffusion, recommended) | Baidu Netdisk (see below) | `weights/checkpoints/ckpt/llava-v1.5-7b-lora-[ble-diff]` | runner: `--model-path` / config `model.adapter` | inference, training |
| Blending detector (SwinV2-B, 256) | Baidu Netdisk (see below) | `weights/blending_models/best_gf.pth` | config `weak_supplies[].weights_path` | weak-signal scores |
| Diffusion detector (ours-sync) | Baidu Netdisk (see below) | `weights/ours-sync/` | config `weak_supplies[].weights_dir` + `model` | weak-signal scores |

### 📥 Baidu Netdisk Links
Download and place files to the paths above.
- X2‑DFD LoRA (ble+diff): 链接：`https://pan.baidu.com/s/1jeBFWS_pAK2vfoI2Z69CTg?pwd=x39g`  提取码：`x39g`
- Blending detector weights (`best_gf.pth`): 链接：`https://pan.baidu.com/s/1jeBFWS_pAK2vfoI2Z69CTg?pwd=x39g`  提取码：`x39g`
- Diffusion detector weights (`weights/ours-sync/`): 链接：`https://pan.baidu.com/s/1jeBFWS_pAK2vfoI2Z69CTg?pwd=x39g`  提取码：`x39g`

- Preprocessing: follow [DeepfakeBench](https://github.com/SCLBD/DeepfakeBench).

## ⚡ Quick Inference

Recommended quick run (two experts + ble+diff LoRA):

```bash
python -m eval.infer.runner \
  --config eval/configs/infer_config.yaml \
  --model-path weights/checkpoints/ckpt/llava-v1.5-7b-lora-[ble-diff] \
  --model-base weights/base/llava-v1.5-7b \
  --experts blending,diffusion_detector
```

Tip:
- Use `--experts blending` for blending only; `--experts diffusion_detector` for diffusion only.
- Use `--experts none` to disable expert scores in the prompt.

<a id="demo"></a>

3) Single-Image Demo (LoRA required)
```bash
python demo.py --image /abs/img.png \
  --model-base weights/base/llava-v1.5-7b \
  --adapter-path weights/checkpoints/ckpt/llava-v1.5-7b-lora-[ble-diff]
```

<a id="quick-infer"></a>

<!-- weights table moved up to Installation -> Before You Run: Prepare Weights -->

### 🔀 Optional Expert Variants

- Blending only (light)
  - Weights: `weights/blending_models/best_gf.pth`
  - Eval: `python -m eval.infer.runner --config eval/configs/infer_config.yaml --experts blending`
  - Train: `python -m train.pipeline --config train/configs/config.yaml --experts blending --run-train`

- Blending + Diffusion (strong)
  - Weights: blending → `weights/blending_models/best_gf.pth`; diffusion → `weights/ours-sync/`
  - Use default configs (both experts enabled), or pass `--experts blending,diffusion_detector`

## 🚀 Usage

<a id="train"></a>

### 1) **Training (staged)**
End-to-end (annotation → weak merge → LoRA train → LoRA test):
```bash
./train.sh --run-train 
```
<!-- - More scripts: `train/` and `train/scripts/`; legacy scripts are archived in `legacy/` (not recommended).

Stages mirror our methodology:
- **Stage 2 — Explainable annotation** (base LLaVA generates rationales)
- **Stage 3 — Weak feature supply** (multi-expert artifact scores merged into prompts)
- **Stage 4 — LoRA training** (fine-tune lightweight adapter) -->

---

<a id="evaluation"></a>

### 2) **Evaluation**

#### Script to Start
```bash
./test.sh
```
- Output: `eval/outputs/infer/latest_run.json`. Ensure `eval/configs/infer_config.yaml -> model.adapter` points to your LoRA.

#### How Others Can Test (minimal reproduction)
1) Install: `bash install.sh && conda activate X2DFD`
2) Download weights (Baidu links above) and place them to:
   - LoRA: `weights/checkpoints/ckpt/llava-v1.5-7b-lora-[ble-diff]`
   - Blending: `weights/blending_models/best_gf.pth`
   - Diffusion: `weights/ours-sync/`
   - Base: `weights/base/llava-v1.5-7b`
3) Run the tiny eval set: `./test.sh`
4) Check:
   - Run metadata: `eval/outputs/infer/latest_run.json`
   - Results JSON(s): listed in `latest_run.json -> outputs` (each item includes the full answer plus `real score`/`fake score` turns)

#### Config to Start
```bash
python -m eval.infer.runner --config eval/configs/infer_config.yaml
```
- Edit `model.adapter` (LoRA path) and `infer.inputs` in the YAML. Optional: `--experts blending` or `--experts blending,diffusion_detector`.

#### Batch Inference (config-driven)
- Set multiple JSONs or a directory under `infer.inputs` in `eval/configs/infer_config.yaml` (directories expand to all `*.json` recursively), then run the command above. Ensure `model.adapter` points to your LoRA, and `weak_supplies` lists the experts you want (e.g., `blending`, `diffusion_detector`).
- Weak-feature supplementation: we include a blending detector as a weak-feature supplement; we also experimented with a diffusion-trace feature detector. Enable them via `--experts` or `weak_supplies`.

#### Question template (example, includes expert score)
```
<image>
Is this image real or fake? And the {alias} score is {score}.
```

<!-- Project Structure & Config sections removed per simplification request -->

<a id="dataset"></a>

## 📦 Dataset
Main datasets used in our experiments:

| Dataset | Link | Notes |
| --- | --- | --- |
| **FF++ (FaceForensics++)** | find [here](https://github.com/ondyari/FaceForensics) | The primary training data in our main table comes from this dataset. |
| DiFF (diffusion images) | [here](https://github.com/xaCheng1996/DiFF) | To tackle diffusion traces, we also developed and integrated both blending and diffusion experts. |

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


---

<a id="model-zoo"></a>

## 🧩 Model Zoo
- Entries moved to the weights table in Installation → "Before You Run: Prepare Weights".

<a id="model"></a>

## 🧠 Model
- **Base:** LLaVA-1.5-7B (`weights/base/llava-v1.5-7b`).
- **Checkpoints:** **LoRA** adapters live under `weights/checkpoints/ckpt/...` (see Baidu Netdisk links above).
- You can point to your own adapters via CLI override or config:
  - demo: `--adapter-path`
  - runner: `--model-path` or config `model.adapter`.

## 😄 Acknowledgements
- We thank the **LLaVA** team for their open-source project and training/inference pipeline:
  https://github.com/haotian-liu/LLaVA
 - Parts of this repo build on community practices (fusion, detectors, training tools). If we missed an attribution, please contact 223040004@link,cuhk.edu.cn.

---


<a id="license"></a>

## 📝 License
MIT License. See `LICENSE`.

---

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
