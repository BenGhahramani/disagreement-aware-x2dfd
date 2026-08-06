# Runtime setup for X2DFD one-image inference

This document records what the inherited X2DFD pipeline requires to run a
single image, and what the current development machine actually provides. It is
written from repository inspection plus the automated checker in
`tools/check_environment.py`.

Nothing here is aspirational: every "verified" line was produced by running the
command shown.

---

## 1. Automated check

Run it with the project virtual environment (see section 4):

```bash
.venv\Scripts\python.exe -m tools.check_environment                   # text report, exit 0/1
.venv\Scripts\python.exe -m tools.check_environment --strict          # warnings also fail
.venv\Scripts\python.exe -m tools.check_environment --json            # machine-readable
.venv\Scripts\python.exe -m tools.check_environment --json --output eval/outputs/env_check.json
.venv\Scripts\python.exe -m tools.check_environment --skip-weights --skip-datasets
.venv\Scripts\python.exe -m tools.check_environment --help
```

Exit codes: `0` = all required checks passed, `1` = at least one required check
failed, `2` = bad arguments or unreadable config.

The checker validates: Python version, imports of every package the inference
path touches, PyTorch version, CUDA availability, GPU name and VRAM, the weight
paths named by the eval config, config file existence, writability of the
results directory, and whether the dataset JSONs listed in `infer.inputs`
resolve to image files that exist on this machine.

---

## 2. What the pipeline requires

### Interpreter and environment

| Requirement | Source |
| --- | --- |
| Python 3.10 | `install.sh` creates conda env `X2DFD` with `python=3.10`; `AGENTS.md` states Python 3.10 |
| Conda env named `X2DFD` | `install.sh`, `AGENTS.md` |

Conda is not installed on this machine, so an equivalent Python 3.10 venv is
used instead (section 4). No upstream file was modified to achieve this.

### Packages (pinned in `install.sh`)

Required for the inference path: `torch`, `torchvision`,
`transformers==4.37.2`, `accelerate==0.28.0`, `peft==0.10.0`,
`llava==1.2.2.post1`, `sentencepiece==0.1.99`, `tokenizers==0.15.1`,
`safetensors==0.4.5`, `timm==0.6.13`, `opencv-python==4.9.0.80`,
`numpy==1.26.4`, `Pillow==10.4.0`, `PyYAML==6.0.2`, `tqdm==4.67.1`,
`einops==0.7.0`.

Optional: `bitsandbytes` (4/8-bit loading), `deepspeed` (training only).

`utils/lora_inference.py` imports `llava.constants`, `llava.conversation`,
`llava.mm_utils`, `llava.model.builder` and `llava.utils` at module import
time, so the `llava` package is a hard requirement — the runner cannot start
without it.

### Weights

Not tracked in git. See `docs/WEIGHTS_SETUP.md` for sources, destination paths,
sizes and per-component verification.

### Environment variables (optional overrides, see `utils/paths.py`)

`X2DFD_PROJECT_ROOT`, `X2DFD_DATASETS`, `X2DFD_WEIGHTS`, `X2DFD_OUTPUT`,
`X2DFD_TRAIN_OUTPUT`, `X2DFD_EVAL_CONFIG`, `X2DFD_TRAIN_CONFIG`,
`X2DFD_PROMPTS`, `X2DFD_BASE_MODEL` (default `--model-base` for `demo.py` and
the runner), `VISION_TOWER` (training), `GPUS`.

None are required if the weights sit at the default paths.

### Data format

```json
{
  "Description": "/absolute/root/directory",
  "images": [{ "image_path": "relative/or/absolute/image.png" }]
}
```

`eval/infer/runner.py` resolves each `image_path` against the top-level
`Description`; there is no fallback prefix.

---

## 3. Smallest one-image commands

Both entry points are inherited upstream code; neither has been executed
successfully yet, because weights and images are still missing.

**Option A — the runner used by the proof-of-concept launcher** (preferred: it
writes the conversation-style JSON the POC already parses):

```bash
.venv\Scripts\python.exe -m eval.infer.runner ^
  --config eval/configs/infer_config.yaml ^
  --json datasets/raw/data/poc/demo_one.json ^
  --experts none ^
  --output eval/outputs/poc/demo_none.json
```

Runner behaviour (read from `eval/infer/runner.py`):

- `--experts` accepts provider or alias names, comma separated; `none` disables
  expert scores. Config aliases are `Blending` and `Diffusion`, so
  `--experts blending`, `--experts diffusion` and `--experts blending,diffusion`
  all match the shipped config.
- Without `--output`, results go to
  `eval/outputs/infer/runs/infer_<UTC timestamp>/datasets/<input>_result.json`,
  with run metadata in `eval/outputs/infer/latest_run.json`.
- Exit code `2` plus a "Missing required model files" message when
  `model.adapter` / `model.base` do not exist.
- Output is a list of items with `id`, `image` and a `conversations` list
  holding `human`, `gpt`, `real score`, `fake score` turns.

**Option B — single-image demo:**

```bash
.venv\Scripts\python.exe demo.py --image /abs/path/to/image.png ^
  --model-base weights/base/llava-v1.5-7b ^
  --adapter-path "weights/checkpoints/ckpt/llava-v1.5-7b-lora-[ble-diff]"
```

`demo.py` prints JSON to stdout and writes no result file, so it fits the
disagreement POC less well than Option A.

---

## 4. Project environment (created 2026-08-06)

Conda is unavailable, so an isolated **Python 3.10.4 venv** was created at
`.venv/` (already covered by `.gitignore`). Nothing was installed globally and
no upstream source file was changed.

```bash
py -3.10 -m venv .venv
.venv\Scripts\python.exe -m pip install -U pip setuptools wheel
.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2
.venv\Scripts\python.exe -m pip install transformers==4.37.2 accelerate==0.28.0 peft==0.10.0 sentencepiece==0.1.99 tokenizers==0.15.1 safetensors==0.4.5
.venv\Scripts\python.exe -m pip install bitsandbytes==0.45.5
.venv\Scripts\python.exe -m pip install timm==0.6.13 opencv-python==4.9.0.80 numpy==1.26.4 Pillow==10.4.0 PyYAML==6.0.2 tqdm==4.67.1 einops==0.7.0
.venv\Scripts\python.exe -m pip install "git+https://github.com/haotian-liu/LLaVA.git#egg=llava"
.venv\Scripts\python.exe -m pip install pytest==8.3.5
```

Two deviations from `install.sh`, both deliberate:

1. **Torch is pinned to `2.1.2+cu121` and installed first.** `install.sh`
   installs torch unpinned and `llava` last. LLaVA requires `torch==2.1.2`, and
   on Windows PyPI serves CPU-only torch wheels — installing `llava` against an
   unpinned torch can silently replace the CUDA build. Pre-pinning `2.1.2+cu121`
   satisfies LLaVA's specifier (a local version tag still matches `==2.1.2`) and
   torch was confirmed untouched by the `llava` install.
2. **`llava` came from GitHub, not PyPI.** `pip install llava==1.2.2.post1`
   fails: PyPI only publishes `llava` `0.0.1.dev0`. The GitHub fallback is the
   one `install.sh` itself documents. Installed commit: `c121f043`.

Resulting versions (`pip freeze`, abridged):

```text
torch==2.1.2+cu121        torchvision==0.16.2+cu121   transformers==4.37.2
llava==1.2.2.post1        peft==0.10.0                accelerate==0.21.0
timm==0.6.13              opencv-python==4.9.0.80     numpy==1.26.4
Pillow==10.4.0            PyYAML==6.0.2               tqdm==4.67.1
einops==0.6.1             sentencepiece==0.1.99       tokenizers==0.15.1
safetensors==0.4.5        bitsandbytes==0.45.5        pytest==8.3.5
```

`accelerate` (0.28.0 → 0.21.0) and `einops` (0.7.0 → 0.6.1) were downgraded by
the `llava` install. This also happens with `install.sh` as written, since it
installs `llava` after its own pins, so this venv matches the upstream end
state rather than diverging from it. `pip check` reports no broken
requirements.

Verified after installation:

```text
$ .venv\Scripts\python.exe -c "import utils.lora_inference"
utils.lora_inference OK

$ .venv\Scripts\python.exe -c "import src.blending.detector, src.diffusion.detector, utils.model_scoring"
blending OK, diffusion OK,
providers: ['aligner','blending','diffdet','diffusion','diffusion_detector','forensics']

$ .venv\Scripts\python.exe -m pytest -m unit
34 passed in 0.21s
```

---

## 5. Verified machine state (2026-08-06)

`.venv\Scripts\python.exe -m tools.check_environment` → **exit 1 (FAIL)**,
`PASS=22 WARN=3 FAIL=9`.

With `--skip-weights --skip-datasets` → **exit 0 (PASS)**, `PASS=23 WARN=2
FAIL=0`. The software layer is complete; only data is missing.

| Area | Observed | Verdict |
| --- | --- | --- |
| Python | 3.10.4 in `.venv` | PASS |
| PyTorch | 2.1.2+cu121 | PASS |
| CUDA | available (driver CUDA 13.2) | PASS |
| GPU | NVIDIA GeForce RTX 3080, 10.0 GiB VRAM | WARN — fp16 LLaVA-1.5-7B needs ~14 GiB |
| Required packages | all 15 importable, including `llava` | PASS |
| `deepspeed` | not installed | WARN — training only, not needed for Stages 2–5 |
| Weights | `weights/` does not exist (all four components) | FAIL |
| Dataset images | JSONs exist, but `Description` root is `/data/250010183/Datasets` (upstream cluster path) | FAIL |
| Output dir | `eval/outputs` writable | PASS |

---

## 6. What must change before Stage 2 can run

1. **Download the weights** — four components, ~14–15 GB. See
   `docs/WEIGHTS_SETUP.md`. Not started; awaiting a go-ahead.
2. **Provide at least one real image.** The bundled dataset JSONs point at a
   cluster path. Either place images locally and write a one-image JSON with a
   local `Description`, or set `X2DFD_DATASETS` and rewrite `infer.inputs`.
3. **Plan for 10 GiB VRAM.** Expect to need 4-bit loading (`bitsandbytes` is
   installed) or CPU offload, and record that as a deviation from the published
   configuration.

Re-run the checker after each step. Stage 2 starts only when it exits `0`.
