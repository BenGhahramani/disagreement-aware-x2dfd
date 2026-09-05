# Celeb-DF-v2 evaluation subset (image-level)

Celeb-DF-v2 is a **video** deepfake corpus. X2DFD inference and the labelled
evaluation runner operate at the **image** level. This repository therefore
builds a fixed image subset by extracting **one deterministic full frame per
selected video** from the official Celeb-DF-v2 test split.

## Why one frame per video?

- Matches X2DFD’s image-level input (not a video detector API).
- One sample per video reduces within-video dependence (correlated frames from
  the same clip are not treated as independent evaluation units).
- This is an **image-level evaluation**, not a video-level detector benchmark.

## What is extracted?

- **Full frames** (PNG), with `already_cropped: false`.
- Face cropping is **not** applied here; it is performed later by
  `tools.run_labelled_evaluation` using the same Haar crop path as the rest of
  the project.

### Label-blind usable-frame policy

For each selected video, candidate timestamps are evaluated in this **exact**
order (fractions of duration):

`50% → 40% → 60% → 30% → 70%`

The first candidate that passes all of the following is kept:

1. Frame decodes successfully.
2. At least one face is found via `tools.make_face_crop.plan_crop` (same Haar
   defaults as evaluation cropping).
3. The largest face is sufficiently inside the image bounds (documented edge
   margin).
4. The frame is not severely blurred: full-frame variance of Laplacian
   ≥ `25.0` (documented constant in `eval/celebdf_v2_prepare.py`; calibrated for
compressed Celeb-DF-v2 video frames — a still-photo heuristic of 100 rejected
too many synthesis clips to reach 60 fake samples).

Ground-truth labels are used only for class quotas (60 real / 60 fake). Frame
choice never uses X2DFD predictions or blending/diffusion detector scores.

If no candidate passes, the video is marked unsuitable and **deterministically
replaced** by the next unused official-test video from the same class (sorted
by path). The run never silently emits fewer than the requested counts.

## Official split only

Videos are taken only from `List_of_testing_videos.txt` (official test list).
The preparation script does **not** invent train/val splits.

Official layout expected under `--dataset-root`:

```text
<dataset-root>/
  Celeb-real/
  YouTube-real/
  Celeb-synthesis/
  List_of_testing_videos.txt
```

Test-list line format: `<label> <relative_path>` where `1` = real and `0` = fake.

## Prepare the subset

From the repository root (after extracting Celeb-DF-v2):

```powershell
# Dry-run: selection only (no frames)
.venv\Scripts\python.exe -m tools.prepare_celebdf_evaluation `
  --dataset-root "C:\Users\Ben\Desktop\UNI\REIT" `
  --dry-run

# Write frames + manifest (default: 60 real + 60 fake, seed 20260305)
.venv\Scripts\python.exe -m tools.prepare_celebdf_evaluation `
  --dataset-root "C:\Users\Ben\Desktop\UNI\REIT" `
  --output-dir datasets/evaluation/celebdf_v2_final
```

Optional: `--skip-video-hash` speeds up preparation (frame hashes are still written).

## Outputs

```text
datasets/evaluation/celebdf_v2_final/
  frames/real/*.png
  frames/fake/*.png
  celebdf_v2_manifest.json      # labelled-evaluation manifest
  sampling_provenance.json      # seed, algorithm, hashes, OpenCV version
  celebdf_v2_summary.csv
```

Run labelled evaluation later (FP16 on Bunya example):

```bash
python -m tools.run_labelled_evaluation \
  --manifest datasets/evaluation/celebdf_v2_final/celebdf_v2_manifest.json \
  --output-dir eval/outputs/labelled_evaluation/celebdf_v2_final \
  --config eval/configs/infer_config.bunya.yaml \
  --no-4bit
```

Do **not** commit Celeb-DF videos or extracted PNG frames to Git. Manifest and
provenance JSON are suitable for version control when licensing permits.
