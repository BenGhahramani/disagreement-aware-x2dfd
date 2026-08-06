# Weights setup

Everything the inherited X2DFD pipeline needs to load before a single image can
be scored, with the exact sources, destinations and verification steps.

`weights/` is gitignored (`.gitignore`), so these files never enter version
control.

**Status:** the Google Drive archive has been downloaded and unpacked (see
[What the Google Drive archive contains](#what-the-google-drive-archive-contains)),
which supplied the LoRA adapter, the blending detector and the diffusion
checkpoint. Sizes for those three are measured. The base model is not in the
archive and comes from Hugging Face separately.

---

## Download sources

| Source | URL | Covers |
| --- | --- | --- |
| Hugging Face | `liuhaotian/llava-v1.5-7b` | base model |
| Hugging Face | `openai/clip-vit-large-patch14-336` | vision tower (training only) |
| Baidu Netdisk | https://pan.baidu.com/s/1V5u2xDULlFBOOB_PJxWBfA?pwd=9pbq | LoRA + blending + diffusion |
| Google Drive | https://drive.google.com/file/d/1RkzA3ZIxxrO2YmUtW5mvVoQNPI47y-O1/view?usp=sharing | "All weights package" |

Both the Baidu and Google Drive links come from `README.md` (added in commits
`7c8cc8b` and `0b4eb31`, April 2026).

---

## What the Google Drive archive contains

Downloaded non-interactively with `gdown` (no browser step was needed):

```bash
.venv\Scripts\python.exe -m pip install gdown
.venv\Scripts\python.exe -m gdown 1RkzA3ZIxxrO2YmUtW5mvVoQNPI47y-O1
```

| Property | Value |
| --- | --- |
| Filename | `weight.zip` |
| Size | 1,054,614,325 bytes (1.05 GB) |
| Integrity | `zipfile.testzip()` → all CRCs OK |
| Entries | 36 (18 real + 18 `__MACOSX/` resource forks, discarded on extract) |
| Uncompressed | 1.05 GiB |

Contents under a single `weight/` root:

| Archive path | Size | Maps to |
| --- | --- | --- |
| `checkpoints/ckpt/llava-v1.5-7b-lora-[ble-diff]/adapter_model.safetensors` | 152.56 MiB | LoRA adapter |
| `checkpoints/ckpt/llava-v1.5-7b-lora-[ble-diff]/non_lora_trainables.bin` | 40.02 MiB | LoRA adapter (mm projector) |
| `checkpoints/ckpt/llava-v1.5-7b-lora-[ble-diff]/{adapter_config,config,trainer_state}.json`, `README.md` | < 3 MiB | LoRA adapter metadata |
| `blending_models/best_gf.pth` | 336.63 MiB | `weights/blending_models/best_gf.pth` |
| `ours-sync/ours-sync.pth` | 269.49 MiB | `weights/ours-sync/ours-sync.pth` |
| `weights2/buffalo_l.zip` | 275.25 MiB | InsightFace face-analysis pack, **not referenced anywhere in this repo** |

The archive also ships `adapter_config(1).json` and `config(1).json`, which are
byte-identical duplicates of their unsuffixed counterparts (verified by SHA-256)
and were not copied into `weights/`.

Two things the archive does **not** contain:

1. **The LLaVA-1.5-7B base model.** At 1.05 GB the archive is far too small to
   hold a 7B checkpoint. The base model must be fetched from Hugging Face.
2. **`weights/ours-sync/config.yaml`.** Only the `.pth` is shipped, but
   `src/diffusion/core.py::Aligner` raises `FileNotFoundError` without the YAML,
   so it had to be authored locally (see component 4).

`buffalo_l.zip` was copied to `weights/weights2/buffalo_l.zip` to avoid a second
download if upstream face-cropping preprocessing is ever needed; nothing in this
repository reads it.

---

## The four required components

### 1. LLaVA-1.5-7B base model

| Field | Value |
| --- | --- |
| Source | Hugging Face `liuhaotian/llava-v1.5-7b` |
| Destination | `weights/base/llava-v1.5-7b/` (directory) |
| Size | 12.63 GiB across 11 files (from the HF file listing) |
| Referenced by | `eval/configs/infer_config.yaml → model.base`; also `config/test_config.yaml`, `config/train_config.yaml`, `train/configs/config.yaml`, `eval/configs/config.yaml` |
| Overrides | env `X2DFD_BASE_MODEL`, CLI `--model-base` (runner and `demo.py`) |

Not present in the Drive archive, so it was downloaded separately with the
`huggingface_hub` already installed in the venv:

```bash
.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('liuhaotian/llava-v1.5-7b', local_dir='weights/base/llava-v1.5-7b')"
```

The bulk is two `pytorch_model-0000n-of-00002.bin` shards (9.29 GiB + 3.30 GiB);
there are no `.safetensors` in this repository, plus `mm_projector.bin`,
`tokenizer.model` and the small JSON configs.

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
| Size | 195.1 MiB across 6 files (measured) |
| Referenced by | `eval/configs/infer_config.yaml → model.adapter` |
| Overrides | CLI `--model-path` (runner), `--adapter-path` (`demo.py`) |

Other configs name different adapters that are **not** required for evaluation:
`eval/configs/config.yaml` points at `.../ckpt/FR/llava-v1.5-7b-lora-[small]`,
and `train/configs/config.yaml` uses
`.../ckpt/llava-v1.5-7b-lora-[x2dfdfinal]` as both adapter and training output.

Verify: `_is_lora_adapter_dir`, a helper nested inside
`eval/infer/runner.py::main` (not importable), treats the path as a LoRA adapter
only when the directory contains `adapter_config.json`,
`adapter_model.safetensors` or `adapter_model.bin`. If none are present the
runner will try to load it as a merged checkpoint and drop `--model-base`. The
adjacent `"lora" in basename` test also matches this directory name, so the LoRA
path is taken either way.

```bash
.venv\Scripts\python.exe -c "import pathlib; p=pathlib.Path('weights/checkpoints/ckpt/llava-v1.5-7b-lora-[ble-diff]'); print([f.name for f in p.iterdir()])"
```

Observed: `README.md`, `adapter_config.json`, `adapter_model.safetensors`
(448 tensors, first key `base_model.model.model.layers.0.mlp.down_proj.lora_A.weight`),
`config.json`, `non_lora_trainables.bin`, `trainer_state.json`. The adapter is
rank 16, alpha 32, targeting the seven attention/MLP projections, over base
`weights/base/llava-v1.5-7b`.

#### PEFT version mismatch (resolved)

As shipped, `adapter_config.json` was written by PEFT 0.17.1 and carried eight
fields the pinned PEFT 0.10.0 does not know, so the config could not be parsed
at all:

```text
PeftConfig.from_pretrained(...) -> TypeError: LoraConfig.__init__() got an
unexpected keyword argument 'corda_config'
```

The tensors were never the problem — only the config parse. Rather than
upgrading PEFT (which would put the pinned `llava` / `transformers 4.37.2`
combination at risk), the eight fields were removed with
`tools/make_peft_compatible_config.py`, which drops an unsupported field only
when it is provably inert and refuses to write anything otherwise:

| Removed key | Value | Why it is inert |
| --- | --- | --- |
| `corda_config` | `null` | unset |
| `eva_config` | `null` | unset |
| `exclude_modules` | `null` | unset |
| `target_parameters` | `null` | unset |
| `trainable_token_indices` | `null` | unset |
| `lora_bias` | `false` | feature disabled |
| `use_qalora` | `false` | feature disabled |
| `qalora_group_size` | `16` | PEFT's own default, and its help text says "Only used when `use_qalora=True`", which is false here |

Corroborating evidence from `adapter_model.safetensors`: all 448 tensors are
`lora_A.weight` / `lora_B.weight` pairs with no bias tensors (consistent with
`lora_bias: false`), no embedding or token tensors (consistent with
`trainable_token_indices: null`) and no parameter-level targets (consistent with
`target_parameters: null`). None of the removed fields left a trace in the
adapter's architecture.

The pristine file is kept alongside as `adapter_config.original.json`. All 24
supported fields were copied through with no value drift, and the config now
parses:

```bash
.venv\Scripts\python.exe -c "from peft import PeftConfig; c=PeftConfig.from_pretrained('weights/checkpoints/ckpt/llava-v1.5-7b-lora-[ble-diff]'); print(c.peft_type, c.task_type, c.r, c.lora_alpha, sorted(c.target_modules))"
```

To reproduce the conversion from the preserved original:

```bash
.venv\Scripts\python.exe -m tools.make_peft_compatible_config "weights/checkpoints/ckpt/llava-v1.5-7b-lora-[ble-diff]" --in-place
```

### 3. Blending detector (SwinV2-B, 256px)

| Field | Value |
| --- | --- |
| Source | Baidu Netdisk / Google Drive package |
| Destination | `weights/blending_models/best_gf.pth` (single file) |
| Size | 336.63 MiB (measured) |
| Referenced by | `eval/configs/infer_config.yaml → weak_supplies[0].weights_path`; same key in `config/test_config.yaml`, `config/train_config.yaml`, `train/configs/config.yaml`, `eval/configs/config.yaml` |
| Loaded by | `src/blending/detector.py::BlendingDetector._load_network` via `timm.create_model("swinv2_base_window16_256", num_classes=2)` |

Verify (loads the state dict only, no GPU needed):

```bash
.venv\Scripts\python.exe -c "import torch; sd=torch.load('weights/blending_models/best_gf.pth', map_location='cpu'); print(type(sd), len(sd)); print(list(sd)[:3])"
```

The loader strips a `module.` prefix if the checkpoint was saved under
DataParallel, then calls `load_state_dict` strictly, so key names must match the
timm SwinV2-B graph.

Observed: an `OrderedDict` of 427 entries beginning `patch_embed.proj.weight`,
with no `module.` prefix. It strict-loads into
`timm.create_model("swinv2_base_window16_256", num_classes=2)` and a CPU forward
pass on a `1×3×256×256` tensor returns logits of shape `(1, 2)`.

### 4. Diffusion detector (`ours-sync`)

| Field | Value |
| --- | --- |
| Source | Baidu Netdisk / Google Drive package |
| Destination | `weights/ours-sync/` containing `config.yaml` **and** the checkpoint file it names |
| Size | 269.49 MiB for `ours-sync.pth` (measured) |
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

#### `config.yaml` was authored locally, not downloaded

The Drive archive ships only `ours-sync.pth`, so `weights/ours-sync/config.yaml`
was written by hand from the `ours-sync` example in `src/EXPERTS_GUIDE.md`, with
`weights_file` changed from the guide's `best.pth` to the name the archive
actually uses:

```yaml
arch: res50
norm_type: resnet
patch_size: 224
weights_file: ours-sync.pth
```

Evidence for `arch`: `ours-sync.pth` is a dict of
`{'model', 'optimizer', 'total_steps'}`; its 320-entry `model` state dict starts
at `conv1.weight` `(64, 3, 7, 7)` and ends at `fc.weight` `(1, 2048)`, i.e. a
ResNet-50 with a single-logit head, which `_logits_to_prob` maps through a plain
sigmoid. `load_weights` handles the `'model'` wrapper and the keys carry no
`module.` prefix.

**Caveat:** the state dict strict-loads into *both* `res50` (conv1 stride 2) and
`res50nodown` (conv1 stride 1), because they differ only in stride and dropout,
neither of which changes tensor shapes. `res50` was chosen because it is what
the guide documents for this model name. `norm_type` and `patch_size` are
likewise taken from the guide and are **unverified** against published scores —
a wrong choice here would silently degrade the diffusion expert rather than
raise an error.

Verify:

```bash
.venv\Scripts\python.exe -c "import yaml,pathlib; d=pathlib.Path('weights/ours-sync'); c=yaml.safe_load((d/'config.yaml').read_text()); print(c); print('weights present:', (d/c['weights_file']).exists())"
```

Stronger check — actually build the expert (CPU, no inference):

```bash
.venv\Scripts\python.exe -c "import sys; sys.path.insert(0,'.'); from src.diffusion.core import Aligner; a=Aligner(['ours-sync'],'weights/',device='cpu'); print('models =', list(a.models))"
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

## Known risks

Resolved:

- **Drive archive contents.** No longer unverified — see the contents table
  above. Baidu Netdisk was never needed; `gdown` fetched the Drive file without
  any browser step.
- **Adapter layout.** The archive ships a genuine LoRA adapter directory, not a
  merged checkpoint, so the runner keeps `--model-base`.
- **PEFT version mismatch.** Eight inert PEFT 0.17 fields were stripped from
  `adapter_config.json`; `PeftConfig.from_pretrained` now returns a `LoraConfig`
  (details in component 2). Only the config parse was verified — the adapter has
  not yet been attached to the base model.

Outstanding:

1. **10 GiB VRAM.** fp16 LLaVA-1.5-7B needs ~14 GiB before an expert detector is
   loaded alongside it. 4-bit loading via `bitsandbytes` (installed, 0.45.5) or
   CPU offload will likely be required, and that is a deviation from the
   published setup worth recording in the thesis.
2. **Diffusion expert hyperparameters are a documented guess.** `arch`,
   `norm_type` and `patch_size` come from `src/EXPERTS_GUIDE.md`, not from the
   archive; a wrong value degrades scores silently (component 4).
3. **Vision tower is fetched at load time.** The base model's `config.json`
   names `openai/clip-vit-large-patch14-336` as `mm_vision_tower`, so the first
   inference run will pull ~1.7 GB from Hugging Face unless it is cached or
   staged locally first.
