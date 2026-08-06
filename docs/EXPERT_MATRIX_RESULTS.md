# Stage 3 — expert configuration matrix

Four sequential inference runs on one face crop, using the same verified
4-bit loading path as Stage 2. Each configuration ran in its own subprocess so
GPU memory was released between runs.

**Result: PASS** (4/4 configurations validated). Disagreement-aware status from
the existing proof-of-concept evaluator: **Uncertain**.

---

## Input

| Field | Value |
| --- | --- |
| Image | `datasets/raw/images/poc/real_face_01_crop.jpg` (256×256 Haar crop) |
| Manifest | `datasets/raw/data/poc/demo_one_crop.json` |
| Ground truth | real (public-domain NASA portrait) |
| Loading | 4-bit NF4 via `X2DFD_LOAD_4BIT=1` |
| Config | `eval/configs/infer_config.windows.yaml` (`num_workers: 0` for Windows spawn/pickle) |
| Started | 2026-08-06T11:57:01Z |
| Finished | 2026-08-06T11:58:27Z |

Face crop provenance and recipe: `docs/TEST_IMAGE_PROVENANCE.md`. Re-running
`tools.make_face_crop` reproduces the crop byte-for-byte
(SHA-256 `e8bd4ea46bd95e8cf7ac82eaabf0a8ab4adb55506d7ea8b7ca9420cc75142fd8`).

---

## Command

```bash
.venv\Scripts\python.exe -m tools.run_expert_matrix \
  --manifest datasets/raw/data/poc/demo_one_crop.json \
  --config   eval/configs/infer_config.windows.yaml \
  --load-4bit
```

Default configurations (in order): `none`, `blending`, `diffusion`,
`blending,diffusion`. Each launches:

```bash
.venv\Scripts\python.exe -m eval.infer.runner \
  --config <windows yaml> \
  --json   datasets/raw/data/poc/demo_one_crop.json \
  --output eval/outputs/expert_matrix/demo_one_crop/demo_<run_name>.json \
  --experts <none|blending|diffusion|blending,diffusion>
```

---

## Results

| Config | Label | Real | Fake | Expert scores in prompt | Runtime (s) | Peak VRAM (MiB) | Exit | Validation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `none` | fake | 0.2393 | 0.7607 | — | 19.881 | 8387 | 0 | valid |
| `blending` | fake | 0.3594 | 0.6406 | blending=0.009 | 21.666 | 8915 | 0 | valid |
| `diffusion` | fake | 0.2878 | 0.7122 | diffusion=0.048 | 21.616 | 8840 | 0 | valid |
| `blending,diffusion` | fake | 0.3141 | 0.6859 | blending=0.009, diffusion=0.048 | 22.896 | 9125 | 0 | valid |

Totals from `eval/outputs/expert_matrix_summary.json`:

| Field | Value |
| --- | --- |
| Passed / failed | 4 / 0 |
| Wall-clock sum of runtimes | 86.059 s |
| Max peak device VRAM | 9125 MiB of 10240 |
| Distinct LM labels | `{fake}` (agree) |

Outputs live under `eval/outputs/expert_matrix/demo_one_crop/` as
`demo_none.json`, `demo_blending.json`, `demo_diffusion.json`,
`demo_blending_diffusion.json`, plus `runtimes.json` and per-run logs.

Prediction texts were all `"This image is fake"`.

---

## Proof-of-concept comparison

```bash
.venv\Scripts\python.exe -m proof_of_concept.run_demo \
  --scenario-dir eval/outputs/expert_matrix/demo_one_crop \
  --output proof_of_concept/outputs/real_example_report.md
```

| Field | Value |
| --- | --- |
| Evidence status | **Uncertain** |
| Rationale | All four runs agree on `fake`, but combined-experts confidence 0.686 is below the 0.70 Stable threshold |
| Report | `proof_of_concept/outputs/real_example_report.md` |

Generated from the real matrix directory, not from mock fixtures.

---

## Do the experts add useful information?

On the LM label axis alone: **no disagreement**. Every configuration produced
`fake`, so the existing PoC rules classify the case as Uncertain rather than
Contested.

On the detector-score axis: **yes, and the signal is strong**. Both expert
detectors report very low fake-likelihood scores (blending 0.009, diffusion
0.048) for an image that is genuinely real, while the language model still
emits `fake` with moderate confidence. Injecting those scores into the prompt
did move the LM's real/fake token probabilities (real rose from 0.24 with no
expert to 0.36 with blending alone) but never flipped the label.

That is useful for the thesis in two ways:

1. It shows the disagreement-aware layer can already surface a borderline case
   (Uncertain) rather than rubber-stamping a single wrong verdict as Stable.
2. It also shows a gap: expert-vs-LM conflict currently lives only in the
   prompt tail and the matrix summary's `expert_scores` field. The PoC
   evaluator does not yet treat low expert scores contradicting the LM label as
   Contested. That is a natural Stage 4 refinement, not a Stage 3 defect.

Caveats that limit how far one image can be pushed: Haar crop ≠ DeepfakeBench
alignment; 4-bit LoRA merge is lossy; authored `ours-sync/config.yaml`
hyperparameters remain a documented guess; VRAM peaks include other desktop
processes on the same GPU.

---

## Tooling and tests

| Artefact | Role |
| --- | --- |
| `tools/make_face_crop.py` | Deterministic Haar face crop |
| `tools/run_expert_matrix.py` | Sequential multi-config runner; reuses smoke-test helpers |
| `tests/test_expert_matrix.py` | 44 mocked unit tests |
| `eval/configs/infer_config.windows.yaml` | Windows-safe `num_workers: 0` (spawn cannot pickle nested Dataset classes) |

```bash
.venv\Scripts\python.exe -m pytest tests/test_expert_matrix.py -q
# 44 passed
.venv\Scripts\python.exe -m pytest -m unit -q
# 164 passed
```

Unit coverage includes command construction, configuration normalisation,
distinct POC-shaped output paths, valid outputs, malformed JSON, hidden
inference errors, partial failure, CUDA OOM, timeout, and summary / runtimes
sidecar generation. No GPU required for those tests.

---

## Remaining blockers for a larger batch (Stage 5)

1. Only one labelled image is available; no batch dataset has been assembled.
2. Expert-vs-LM score conflict is recorded but not yet a first-class PoC status
   rule.
3. Peak VRAM at 9125 MiB leaves little headroom; keep one-config-per-process.
4. Do not begin a larger evaluation batch until more images and labels are
   deliberately chosen.
