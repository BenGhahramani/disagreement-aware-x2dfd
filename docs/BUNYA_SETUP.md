# UQ Bunya setup guide

This document prepares the repository for a **minimal one-image GPU smoke test** on
[UQ Bunya](https://github.com/UQ-RCC/hpc-docs/blob/main/guides/Bunya-User-Guide.md).
It does not change model behaviour, thresholds, or evaluation logic.

## Bunya rules that affect this project

| Rule | Implication for X2DFD |
| --- | --- |
| **No compute on login nodes** | Do not run inference, `pip install`, `conda create`, or environment checks on login nodes. Use `sbatch` or `salloc` on compute nodes only. |
| **Active work on `/scratch`** | Clone the repo, create `.venv`, run jobs, and write logs/outputs under `/scratch/$USER/…`. |
| **Do not run jobs from `/QRISdata`** | Copy or rsync weights and durable results to/from QRIS; never set `SLURM_SUBMIT_DIR` under `/QRISdata`. |
| **Accounting required** | Every job needs `--account=a_your_group` (from the `groups` command). |

## 1. Clone the repository onto scratch

On a **login node** (git operations only):

```bash
cd /scratch/$USER
git clone <your-remote-url> disagreement-aware-x2dfd
cd disagreement-aware-x2dfd
```

Keep `$HOME` for ssh keys and small dotfiles; put the working copy on scratch.

## 2. Stage weights and the smoke-test image

Weights are **not** tracked in git. Copy them from your local machine or QRIS storage
into scratch, for example:

```text
/scratch/$USER/x2dfd/weights/
  base/llava-v1.5-7b/
  checkpoints/ckpt/llava-v1.5-7b-lora-[ble-diff]/
  blending_models/best_gf.pth
  ours-sync/
```

Also copy the known smoke-test image (not tracked):

```text
datasets/raw/images/poc/real_face_01.jpg
```

See [docs/WEIGHTS_SETUP.md](WEIGHTS_SETUP.md) for download instructions used on the
verified local Windows setup.

The smoke-test Slurm script can symlink `/scratch/$USER/x2dfd/weights` into
`<repo>/weights` automatically when `weights/` is absent.

## 3. Create the Python environment (compute node only)

**Do not** create the venv on a login node. Request an interactive GPU session.
Environment setup (pip/conda) usually needs longer than the 30-minute debug
window, so use `qos=gpu` for install sessions:

```bash
salloc --account=AccountString \
  --partition=gpu_cuda --qos=gpu --gres=gpu:l40s:1 \
  --cpus-per-task=4 --mem=32G --time=02:00:00
```

Replace `AccountString` with your `a_*` group from `groups`. For install-only
work you may request a different GPU type (`l40s`, `a100`, `h100`) if your
account has access and you need more VRAM.

On the **compute node** allocated by `salloc`:

```bash
cd /scratch/$USER/disagreement-aware-x2dfd

# Option A — venv (matches local Windows workflow)
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2
python -m pip install transformers==4.37.2 accelerate==0.28.0 peft==0.10.0 \
  sentencepiece==0.1.99 tokenizers==0.15.1 safetensors==0.4.5
python -m pip install bitsandbytes==0.45.5
python -m pip install timm==0.6.13 opencv-python==4.9.0.80 numpy==1.26.4 \
  Pillow==10.4.0 PyYAML==6.0.2 tqdm==4.67.1 einops==0.7.0
python -m pip install "git+https://github.com/haotian-liu/LLaVA.git#egg=llava"
python -m pip install pytest==8.3.5

# PEFT adapter compatibility (same as local)
python -m tools.make_peft_compatible_config \
  weights/checkpoints/ckpt/llava-v1.5-7b-lora-[ble-diff] --in-place
```

Option B — use `bash install.sh` inside the same interactive GPU job if you prefer
Conda (`X2DFD` env). The project’s verified local path used a `.venv`; either works
as long as packages match [docs/RUNTIME_SETUP.md](RUNTIME_SETUP.md).

Verify on the compute node:

```bash
source .venv/bin/activate
python -m tools.check_bunya_environment --require-bitsandbytes --skip-datasets
```

## 4. Configure the smoke-test job

Edit [bunya/run_smoke_test.slurm](../bunya/run_smoke_test.slurm):

1. Replace `#SBATCH --account=AccountString` with your `a_*` accounting group.
2. Optionally adjust path variables at the top of the script body:
   - `PROJECT_DIR` — repo on scratch (default `/scratch/$USER/disagreement-aware-x2dfd`)
   - `WEIGHTS_DIR` — staged weights (default `/scratch/$USER/x2dfd/weights`)
   - `OUTPUT_DIR` — scratch outputs (default `/scratch/$USER/x2dfd/runs/smoke/$SLURM_JOB_ID`)

### Resource request (initial smoke test)

The batch script [bunya/run_smoke_test.slurm](../bunya/run_smoke_test.slurm) uses
UQ Bunya’s **debug** QoS for a short first validation run:

| Slurm directive | Value | Rationale |
| --- | --- | --- |
| `--partition` | `gpu_cuda` | NVIDIA GPUs for PyTorch/CUDA stack |
| `--qos` | `debug` | Short initial smoke test only (30 min limit) |
| `--gres` | `gpu:l40s:1` | One L40S GPU for single-image LLaVA-7B + experts |
| `--cpus-per-task` | `4` | DataLoader + preprocessing headroom |
| `--mem` | `32G` | Host RAM for model load and experts |
| `--time` | `00:30:00` | Sufficient for one-image smoke test on debug QoS |

**QoS and GPU choices after the smoke test**

- **`debug`** is for this short initial smoke test only. It is not appropriate for
  production evaluation batches.
- **Production evaluation jobs** should use `--qos=gpu` with a longer `--time`
  (for example `02:00:00` or more for multi-image labelled runs).
- **GPU type** may later be changed to `a100` or `h100` (for example
  `--gres=gpu:a100:1`, `--gres=gpu:h100:1`) depending on memory requirements
  and experiment needs. The smoke script pins `l40s` as the default for the
  first CUDA validation on current Bunya capacity.

The local pilot used **4-bit loading** on a 10 GiB RTX 3080. Bunya nodes may have
more VRAM; the first smoke test still uses `--load-4bit` to match the verified local
setup. You can retry with fp16 later on larger GPUs without changing this guide.

### Infer config

Use [eval/configs/infer_config.bunya.yaml](../eval/configs/infer_config.bunya.yaml).
It mirrors the verified [infer_config.windows.yaml](../eval/configs/infer_config.windows.yaml)
settings with Linux-appropriate `num_workers`. Override paths via env vars documented
in `utils/paths.py` (`X2DFD_WEIGHTS`, `X2DFD_OUTPUT`, etc.) if weights are not
symlinked into the clone.

## 5. Submit the smoke test

From the repo on scratch (login node — **scheduling only**):

```bash
cd /scratch/$USER/disagreement-aware-x2dfd
sbatch bunya/run_smoke_test.slurm
```

Monitor:

```bash
squeue --me
```

Logs default to `x2dfd-smoke-<jobid>.out` / `.err` in the submit directory.
Structured outputs land under `OUTPUT_DIR` (see script): `env_check.json`,
`smoke_test_summary.json`, `demo_one_result.json`.

## 6. Copy results to durable storage

After the job succeeds, copy summaries off scratch before purge policies apply:

```bash
mkdir -p /QRISdata/QXXXX/$USER/x2dfd/smoke/
rsync -av /scratch/$USER/x2dfd/runs/smoke/ /QRISdata/QXXXX/$USER/x2dfd/smoke/
```

Replace `QXXXX` with your RDM id. Do not run `sbatch` from `/QRISdata`.

## 7. Manual one-image command (interactive compute node)

Equivalent to the Slurm script body, for debugging on an `salloc` session:

```bash
cd /scratch/$USER/disagreement-aware-x2dfd
source .venv/bin/activate
export X2DFD_OUTPUT=/scratch/$USER/x2dfd/runs/smoke/manual

python -m tools.check_bunya_environment --require-bitsandbytes --skip-datasets

python -m tools.run_smoke_test \
  --manifest /scratch/$USER/x2dfd/runs/smoke/manual/demo_one_manifest.json \
  --config eval/configs/infer_config.bunya.yaml \
  --output "$X2DFD_OUTPUT/demo_one_result.json" \
  --summary "$X2DFD_OUTPUT/smoke_test_summary.json" \
  --experts none \
  --load-4bit \
  --project-root "$PWD"
```

Create the manifest first (Linux path — do **not** use `datasets/raw/data/poc/demo_one.json`,
which embeds a Windows `Description` path):

```bash
mkdir -p "$X2DFD_OUTPUT"
cat > "$X2DFD_OUTPUT/demo_one_manifest.json" <<EOF
{
  "Description": "$(pwd)/datasets/raw/images/poc",
  "images": [{"image_path": "real_face_01.jpg"}]
}
EOF
```

## Windows-specific assumptions in this repository

These are fine locally but need the Bunya path above:

| Location | Windows assumption |
| --- | --- |
| `eval/configs/infer_config.windows.yaml` | `num_workers: 0` for DataLoader spawn/pickle |
| `dashboard/live_analysis.py` | Default config is `infer_config.windows.yaml` |
| `run_demo.ps1`, several docs | PowerShell / `.venv\Scripts\python.exe` |
| `datasets/raw/data/poc/demo_one.json` | `Description` uses `C:/Users/...` |
| Local pilot docs | 4-bit via `bitsandbytes` on 10 GiB RTX 3080 |
| `eval/infer/runner.py` | Documents Windows `num_workers: 0`; Linux fork path unchanged |

Cross-platform pieces already in place: `utils/paths.py` env overrides,
`tools/run_smoke_test.py` (`sys.executable`, `subprocess.run`),
`tools/check_environment.py`, and `utils/runtime_env.py`.

## Apptainer / Singularity (future option)

Bunya provides Apptainer on compute nodes only ([RCC docs](https://github.com/UQ-RCC/hpc-docs/blob/main/guides/Bunya-User-Guide.md)).
Containerisation is **not required** for the first smoke test. If dependency drift
becomes painful, a future Apptainer definition could pin the verified PyTorch +
LLaVA stack; that is out of scope until the bare-metal venv smoke test passes.

## Related files

| File | Purpose |
| --- | --- |
| [bunya/run_smoke_test.slurm](../bunya/run_smoke_test.slurm) | Batch smoke test |
| [bunya/submit_labelled_evaluation.sh](../bunya/submit_labelled_evaluation.sh) | One-command labelled FP16 submit + analysis dependency |
| [bunya/run_labelled_evaluation.slurm](../bunya/run_labelled_evaluation.slurm) | Production labelled evaluation (FP16) |
| [bunya/run_labelled_analysis.slurm](../bunya/run_labelled_analysis.slurm) | Post-inference analysis (`afterok`) |
| [docs/BUNYA_LABELLED_EVALUATION.md](BUNYA_LABELLED_EVALUATION.md) | SCP + submit workflow for thesis datasets |
| [eval/configs/infer_config.bunya.yaml](../eval/configs/infer_config.bunya.yaml) | Linux/Bunya infer config |
| [tools/check_bunya_environment.py](../tools/check_bunya_environment.py) | Bunya-oriented env check |
| [docs/WEIGHTS_SETUP.md](WEIGHTS_SETUP.md) | Weight download and layout |
| [docs/RUNTIME_SETUP.md](RUNTIME_SETUP.md) | Verified local package versions |
