# Weights setup

Everything the inherited X2DFD pipeline needs to load before a single image can
be scored. Nothing here has been downloaded yet — this document records the
exact sources, destinations and verification steps so the download can be done
deliberately.

`weights/` is gitignored (`.gitignore`), so these files never enter version
control.

**Size column caveat:** no component has been downloaded, so sizes are
*estimates* derived from the architecture (parameter count × dtype) or from the
published Hugging Face repository, not measured values. Treat them as planning
figures only.

---

## Download sources

| Source | URL | Covers |
| --- | --- | --- |
| Hugging Face | `liuhaotian/llava-v1.5-7b` | base model |
| Hugging Face | `openai/clip-vit-large-patch14-336` | vision tower (training only) |
| Baidu Netdisk | https://pan.baidu.com/s/1V5u2xDULlFBOOB_PJxWBfA?pwd=9pbq | LoRA + blending + diffusion |
| Google Drive | https://drive.google.com/file/d/1RkzA3ZIxxrO2YmUtW5mvVoQNPI47y-O1/view?usp=sharing | "All weights package" |

Both the Baidu and Google Drive links come from `README.md` (added in commits
`7c8cc8b` and `0b4eb31`, April 2026). The README does not state what the Drive
archive contains beyond "All weights package", so whether it includes the LLaVA
base model is **unverified**.

---

## The four required components

### 1. LLaVA-1.5-7B base model

| Field | Value |
| --- | --- |
| Source | Hugging Face `liuhaotian/llava-v1.5-7b` |
| Destination | `weights/base/llava-v1.5-7b/` (directory) |
| Approx. size | ~13–14 GB (7B parameters, fp16) — estimated |
| Referenced by | `eval/configs/infer_config.yaml → model.base`; also `config/test_config.yaml`, `config/train_config.yaml`, `train/configs/config.yaml`, `eval/configs/config.yaml` |
| Overrides | env `X2DFD_BASE_MODEL`, CLI `--model-base` (runner and `demo.py`) |

Suggested download:

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download liuhaotian/llava-v1.5-7b --local-dir weights/base/llava-v1.5-7b
```

Verify: the directory should contain `config.json`, a tokenizer
(`tokenizer.model` / `tokenizer_config.json`) and the weight shards
(`*.safetensors` or `pytorch_model-*.bin`) plus their index file.

```bash
.venv\Scripts\python.exe -c "import json,pathlib; p=pathlib.Path('weights/base/llava-v1.5-7b'); print(sorted(x.name for x in p.iterdir())[:12]); print(json.load(open(p/'config.json'))['model_type'])"
```

### 2. X2DFD LoRA adapter (blending + diffusion)

| Field | Value |
| --- | --- |
| Source | Baidu Netdisk / Google Drive package |
| Destination | `weights/checkpoints/ckpt/llava-v1.5-7b-lora-[ble-diff]/` (directory, brackets are part of the name) |
| Approx. size | ~100–700 MB depending on LoRA rank — **unverified estimate** |
| Referenced by | `eval/configs/infer_config.yaml → model.adapter` |
| Overrides | CLI `--model-path` (runner), `--adapter-path` (`demo.py`) |

Other configs name different adapters that are **not** required for evaluation:
`eval/configs/config.yaml` points at `.../ckpt/FR/llava-v1.5-7b-lora-[small]`,
and `train/configs/config.yaml` uses
`.../ckpt/llava-v1.5-7b-lora-[x2dfdfinal]` as both adapter and training output.

Verify: `eval/infer/runner.py::_is_lora_adapter_dir` treats the path as a LoRA
adapter only when the directory contains `adapter_config.json`,
`adapter_model.safetensors` or `adapter_model.bin`. If none are present the
runner will try to load it as a merged checkpoint and drop `--model-base`.

```bash
.venv\Scripts\python.exe -c "import pathlib; p=pathlib.Path('weights/checkpoints/ckpt/llava-v1.5-7b-lora-[ble-diff]'); print([f.name for f in p.iterdir()])"
```

### 3. Blending detector (SwinV2-B, 256px)

| Field | Value |
| --- | --- |
| Source | Baidu Netdisk / Google Drive package |
| Destination | `weights/blending_models/best_gf.pth` (single file) |
| Approx. size | ~350 MB (SwinV2-B ≈ 88M params, fp32 state dict) — estimated |
| Referenced by | `eval/configs/infer_config.yaml → weak_supplies[0].weights_path`; same key in `config/test_config.yaml`, `config/train_config.yaml`, `train/configs/config.yaml`, `eval/configs/config.yaml` |
| Loaded by | `src/blending/detector.py::BlendingDetector._load_network` via `timm.create_model("swinv2_base_window16_256", num_classes=2)` |

Verify (loads the state dict only, no GPU needed):

```bash
.venv\Scripts\python.exe -c "import torch; sd=torch.load('weights/blending_models/best_gf.pth', map_location='cpu'); print(type(sd), len(sd)); print(list(sd)[:3])"
```

The loader strips a `module.` prefix if the checkpoint was saved under
DataParallel, then calls `load_state_dict` strictly, so key names must match the
timm SwinV2-B graph.

### 4. Diffusion detector (`ours-sync`)

| Field | Value |
| --- | --- |
| Source | Baidu Netdisk / Google Drive package |
| Destination | `weights/ours-sync/` containing `config.yaml` **and** the checkpoint file it names |
| Approx. size | ~100–200 MB (ResNet-50-class backbone) — estimated |
| Referenced by | `eval/configs/infer_config.yaml → weak_supplies[1].weights_dir: weights/` + `model: ours-sync` |

Note the indirection: the config gives `weights_dir: weights/` and
`model: ours-sync`, and `src/diffusion/core.py::Aligner` then opens
`<weights_dir>/<model>/config.yaml`. So the effective directory is
`weights/ours-sync/`.

`config.yaml` must define these keys (read directly in `core.py`, and documented
in `src/EXPERTS_GUIDE.md`):

```yaml
arch: res50            # or res50nodown / opencliplinear_*
norm_type: resnet      # resnet | clip | xception | spec | fft2 | residue3 | npr | cooc
patch_size: 224        # int | [H,W] | Clip224 | null
weights_file: best.pth # file inside weights/ours-sync/
```

`Aligner.__init__` raises `FileNotFoundError` for a missing `config.yaml` and
again for a missing `weights_file`, so both must be present.

Verify:

```bash
.venv\Scripts\python.exe -c "import yaml,pathlib; d=pathlib.Path('weights/ours-sync'); c=yaml.safe_load((d/'config.yaml').read_text()); print(c); print('weights present:', (d/c['weights_file']).exists())"
```

---

## Not required for Stage 2 evaluation

**CLIP ViT-L/14-336** — `weights/base/clip-vit-large-patch14-336`, ~1.7 GB,
from `openai/clip-vit-large-patch14-336`. Referenced by
`config/train_config.yaml → vision_tower` and `train/configs/config.yaml`, i.e.
training only. The base LLaVA checkpoint already bundles its vision tower for
inference.

---

## Verifying the whole set

The environment checker validates every path above in one pass:

```bash
.venv\Scripts\python.exe -m tools.check_environment
```

Success looks like these five lines flipping to `PASS`:

```text
[PASS] weights.model.adapter
[PASS] weights.model.base
[PASS] weights.expert.Blending
[PASS] weights.expert.Diffusion.dir
[PASS] weights.expert.Diffusion.config
```

To check only the weights while dataset images are still unavailable:

```bash
.venv\Scripts\python.exe -m tools.check_environment --skip-datasets
```

The checker only tests existence and file-vs-directory type. It does **not**
load the checkpoints, so a corrupt or truncated download will still report
`PASS`; the per-component verification commands above are what catch that.

---

## Known risks before downloading

1. **Total footprint is roughly 14–15 GB**, dominated by the base model. There
   is ~276 GB free on `C:`, so disk is not a constraint.
2. **10 GiB VRAM.** fp16 LLaVA-1.5-7B needs ~14 GiB before an expert detector is
   loaded alongside it. 4-bit loading via `bitsandbytes` (installed, 0.45.5) or
   CPU offload will likely be required, and that is a deviation from the
   published setup worth recording in the thesis.
3. **Baidu Netdisk** typically requires an account and its own client for large
   files; the Google Drive package may be the more practical route, but its
   contents are unverified.
4. **Adapter layout is unverified.** If the archive ships a merged checkpoint
   rather than a LoRA adapter directory, the runner takes a different code path
   (`_is_lora_adapter_dir` returns `False` and `--model-base` is discarded).
   Check the extracted contents before running.
