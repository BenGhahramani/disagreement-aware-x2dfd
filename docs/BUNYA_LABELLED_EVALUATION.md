# Bunya labelled evaluation workflow

Production path for thesis labelled evaluation on UQ Bunya: **git pull → SCP
prepared dataset → one submit command**. Slurm queues inference (FP16) and, on
success, analysis. Do **not** use interactive `salloc` for evaluation runs.

Related: [BUNYA_SETUP.md](BUNYA_SETUP.md) (venv / weights / smoke test).

## Prerequisites on Bunya scratch

1. Repo clone at `/scratch/user/s4749229/disagreement-aware-x2dfd` (or set
   `X2DFD_PROJECT_DIR`).
2. Working `.venv` created on a **compute** node (see BUNYA_SETUP).
3. Weights under `$PROJECT/weights` including:
   - `base/llava-v1.5-7b`
   - `checkpoints/ckpt/llava-v1.5-7b-lora-[ble-diff]`
   - `blending_models/best_gf.pth`
   - `ours-sync/`
   - `base/clip-vit-large-patch14-336` (vision tower; required by the production
     job so inference does not hit Hugging Face at runtime)
4. Optional: `python -m pip install matplotlib` in the venv (figures).

## Wall time

The validated smoke script uses **30 minutes** (single-image). That is **not**
appropriate for full labelled runs.

Measured local **4-bit** 4-image pilot: ~92 s/image for all four configs
(~6.2 min total). Extrapolating to 240 images ≈ **6.1 h** of GPU time. There is
**no measured FP16 A100 pilot** in-repo, so the production default wall time is
**12:00:00**, overridable at submit:

```bash
TIME_LIMIT=16:00:00 bash bunya/submit_labelled_evaluation.sh <manifest> <output_dir>
```

Resume is on by default; a wall-time kill can be continued by re-submitting the
same output directory.

## Windows → Bunya: SCP prepared evaluation data only

Copy **prepared** subsets only. Do **not** SCP DeepFakeFace ZIPs or raw Celeb-DF
videos.

From **Windows PowerShell** (OpenSSH `scp`):

```powershell
# DeepFakeFace prepared subset (images + manifest + provenance)
scp -r `
  "C:\Users\Ben\Desktop\UNI\REIT\root\disagreement-aware-x2dfd\datasets\evaluation\deepfakeface_final" `
  s4749229@bunya.rcc.uq.edu.au:/scratch/user/s4749229/datasets/

# Celeb-DF-v2 prepared subset (frames + manifest + provenance)
scp -r `
  "C:\Users\Ben\Desktop\UNI\REIT\root\disagreement-aware-x2dfd\datasets\evaluation\celebdf_v2_final" `
  s4749229@bunya.rcc.uq.edu.au:/scratch/user/s4749229/datasets/
```

Expected remote layout:

```text
/scratch/user/s4749229/datasets/deepfakeface_final/
  deepfakeface_manifest.json
  sampling_provenance.json
  images/...

/scratch/user/s4749229/datasets/celebdf_v2_final/
  celebdf_v2_manifest.json
  sampling_provenance.json
  frames/...
```

After SCP, manifests use relative image paths, so they resolve against the
manifest directory on Bunya without editing.

## Bunya: pull code and submit

```bash
cd /scratch/user/s4749229/disagreement-aware-x2dfd
git pull origin newLayer
```

### DeepFakeFace (240 images × 4 configs, FP16)

```bash
bash bunya/submit_labelled_evaluation.sh \
  /scratch/user/s4749229/datasets/deepfakeface_final/deepfakeface_manifest.json \
  /scratch/user/s4749229/eval_outputs/deepfakeface_fp16
```

### Celeb-DF-v2 (120 images × 4 configs, FP16)

```bash
bash bunya/submit_labelled_evaluation.sh \
  /scratch/user/s4749229/datasets/celebdf_v2_final/celebdf_v2_manifest.json \
  /scratch/user/s4749229/eval_outputs/celebdf_v2_fp16
```

The wrapper:

1. Validates arguments and creates the output / `slurm_logs/` directories.
2. Submits `bunya/run_labelled_evaluation.slurm` via `sbatch` (returns immediately).
3. Queues `bunya/run_labelled_analysis.slurm` with `--dependency=afterok:<jobid>`.

Skip analysis: `SKIP_ANALYSIS=1 bash bunya/submit_labelled_evaluation.sh …`

If the analysis CPU partition differs on your account:

```bash
ANALYSIS_PARTITION=general ANALYSIS_QOS=normal \
  bash bunya/submit_labelled_evaluation.sh <manifest> <output_dir>
```

## What the inference job runs

- Same account / partition / QoS / A100 MIG request as the validated smoke profile
  (`a_css`, `gpu_cuda`, `qos=gpu`, A100 40 GB MIG).
- Modules: `python/3.10.4-…`, `cuda/12.1.1`
- `tools.check_bunya_environment` (weights; datasets skipped — labelled images
  come from the prepared manifest)
- Manifest dry-run into `$OUTPUT/_preflight_dry_run`
- `python -m tools.run_labelled_evaluation … --no-4bit --resume` (canonical four
  configs; no new CLI flags)

## Outputs and logs

| Path | Contents |
| --- | --- |
| `$OUTPUT/aggregate.json` | Labelled batch aggregate |
| `$OUTPUT/batch_meta.json` | Provenance / quantisation / git |
| `$OUTPUT/<image_id>/` | Per-image crops, matrix JSONs, provenance |
| `$OUTPUT/analysis/` | `analysis_summary.json`, CSVs, `figures/` (afterok job) |
| `$OUTPUT/slurm_logs/labelled-eval-<jobid>.out` | Inference stdout |
| `$OUTPUT/slurm_logs/labelled-analysis-<jobid>.out` | Analysis stdout |

Monitor:

```bash
squeue --me
squeue -j <jobid>
sacct -j <jobid> --format=JobID,State,Elapsed,ExitCode,MaxRSS -P
tail -f /scratch/user/s4749229/eval_outputs/deepfakeface_fp16/slurm_logs/labelled-eval-<jobid>.out
```

## Dataset independence

The same Slurm scripts work for any labelled manifest (DeepFakeFace, Celeb-DF-v2,
future sets). Change only the manifest path and output directory.
