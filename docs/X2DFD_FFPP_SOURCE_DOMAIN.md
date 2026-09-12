# FF++ c23 source-domain sanity check

Separate from the thesis **labelled evaluation** (DeepFakeFace 240 + Celeb-DF-v2
120). After weak cross-dataset results, this workflow asks whether the
**inherited public X2-DFD 7B setup** behaves broadly in line with the paper’s
strong **FF++ in-domain** result — using a lightweight subsample of the official
**held-out test** split (not training videos).

## Provenance

| Item | Inherited setup |
| --- | --- |
| Base | `weights/base/llava-v1.5-7b` |
| Adapter | `weights/checkpoints/ckpt/llava-v1.5-7b-lora-[ble-diff]` |
| Specialists | **Both** `blending` + `diffusion_detector` |
| Compression | **c23** |
| Split | Official FaceForensics `test.json` (vendored under `datasets/evaluation/ffpp_c23_source/official_splits/`) |

**Comparison mode:** `source_domain_sanity_check` — **not** an exact reproduction.

Why not exact:

1. Default run **subsamples** the held-out test set (100 / 700 videos).
2. Default uses **8** frames/video (DeepfakeBench / paper testing uses **32**).
3. Haar crop (`tools.make_face_crop`) ≠ DeepfakeBench dlib/RetinaFace aligner.
4. Public `[ble-diff]` weights, not an independent retrain.

### Paper metadata only

Appendix C.2 Table 9 (“Ours”) FF++c23 AUC **0.966** (plus per-method columns).
Never a pass/fail gate.

## Default protocol (lightweight)

| Setting | Default |
| --- | --- |
| Videos | **100** (50 genuine + 50 manipulated) |
| Fake stratification | 13 / 13 / 12 / 12 across DF / F2F / FS / NT |
| Seed | **4842** (recorded in manifest `subset_selection`) |
| Frames / video | **8** evenly spaced |
| Planned frames | **≈ 800** |

CLI to expand later:

```powershell
# Full held-out test catalogue, 32 frames/video
.venv\Scripts\python.exe -m tools.prepare_ffpp_source_domain `
  --dataset-root "PATH\TO\FaceForensics++" `
  --output-dir datasets/evaluation/ffpp_c23_source `
  --full-test --frames-per-video 32
```

## Data to download

From [FaceForensics++](https://github.com/ondyari/FaceForensics) (**c23**):

- `original_sequences/youtube/c23/`
- `manipulated_sequences/Deepfakes/c23/`
- `manipulated_sequences/Face2Face/c23/`
- `manipulated_sequences/FaceSwap/c23/`
- `manipulated_sequences/NeuralTextures/c23/`

Prep resolves only IDs selected from the official test split (sanity subset or
`--full-test`). Train/val are never scored.

## Local CPU validation

```powershell
.venv\Scripts\python.exe -m pytest tests/test_x2dfd_ffpp_source_domain.py -q

.venv\Scripts\python.exe -m tools.prepare_ffpp_source_domain `
  --dataset-root "PATH\TO\FaceForensics++" `
  --output-dir datasets/evaluation/ffpp_c23_source `
  --dry-run
```

Do **not** run GPU inference locally for this experiment.

## Prepare + transfer

```powershell
.venv\Scripts\python.exe -m tools.prepare_ffpp_source_domain `
  --dataset-root "PATH\TO\FaceForensics++" `
  --output-dir datasets/evaluation/ffpp_c23_source

scp -r `
  "...\datasets\evaluation\ffpp_c23_source" `
  s4749229@bunya.rcc.uq.edu.au:/scratch/user/s4749229/datasets/
```

## Bunya submit

```bash
cd /scratch/user/s4749229/disagreement-aware-x2dfd
bash bunya/submit_ffpp_source_domain.sh \
  /scratch/user/s4749229/datasets/ffpp_c23_source \
  /scratch/user/s4749229/eval_outputs/x2dfd_ffpp_source_domain
```

Default wall time is **12h** (enough for ~800 frames). Override if needed:

```bash
TIME_LIMIT=16:00:00 bash bunya/submit_ffpp_source_domain.sh ...
```

## Runtime estimate (~800 frames)

~800 × 15–25 s/frame ≈ **3–6 GPU-hours**. Plan **8–12 h** wall with resume.

## Outputs

| Path | Role |
| --- | --- |
| `datasets/evaluation/ffpp_c23_source/` | Prep + `subset_selection` in manifest |
| `eval/outputs/x2dfd_ffpp_source_domain/` | Scores + analysis (separate from labelled eval) |
