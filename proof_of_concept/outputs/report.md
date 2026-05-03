# Disagreement-aware X2DFD comparison — POC report

> **Proof-of-concept notice.** This report is produced by a *disagreement-aware comparison layer* built on top of the existing X2DFD pipeline. It is **not a new deepfake detector** and does not modify any X2DFD model code. It re-uses the same image, same model, same experts — the only thing it adds is running each image under several expert settings and applying a small rule-based evaluator over the resulting JSONs to surface agreement, disagreement, and low-confidence cases.

## Scenario: `stable`

**Image:** `/data/poc/demo_stable.png`

| run | experts | prediction | real | fake | confidence | runtime (s) | error |
|-----|---------|------------|------|------|------------|-------------|-------|
| `none` | — | fake | 0.110 | 0.890 | 0.890 | 7.40 | — |
| `blending` | blending | fake | 0.090 | 0.910 | 0.910 | 11.20 | — |
| `diffusion` | diffusion | fake | 0.130 | 0.870 | 0.870 | 10.80 | — |
| `blending_diffusion` | blending,diffusion | fake | 0.080 | 0.920 | 0.920 | 14.60 | — |


**Evidence status:** `Stable`

**Rationale:** All four runs agree on 'fake'; combined-experts confidence 0.92 >= 0.70.


**Per-run answer excerpts:**

- `none`: fake. The face shows visible blending artefacts along the jawline and inconsistent lighting across the forehead.
- `blending`: fake. The blending detector reports a high score, consistent with seam artefacts around the cheek-jaw boundary.
- `diffusion`: fake. The diffusion-trace expert reports strong synthetic signal in mid-frequency residuals.
- `blending_diffusion`: fake. Both experts agree: blending seam plus residual diffusion trace point to manipulation.

## Scenario: `contested`

**Image:** `/data/poc/demo_contested.png`

| run | experts | prediction | real | fake | confidence | runtime (s) | error |
|-----|---------|------------|------|------|------------|-------------|-------|
| `none` | — | real | 0.780 | 0.220 | 0.780 | 7.60 | — |
| `blending` | blending | real | 0.620 | 0.380 | 0.620 | 11.50 | — |
| `diffusion` | diffusion | fake | 0.320 | 0.680 | 0.680 | 11.10 | — |
| `blending_diffusion` | blending,diffusion | fake | 0.340 | 0.660 | 0.660 | 14.90 | — |


**Evidence status:** `Contested`

**Rationale:** Disagreement across runs (none=real, blending=real, diffusion=fake, blending_diffusion=fake). Sources of disagreement: adding experts flips bare-LLaVA verdict; blending and diffusion experts disagree.


**Per-run answer excerpts:**

- `none`: real. The face appears natural with consistent skin texture and no obvious seams.
- `blending`: real. Low blending score and natural skin tone support a real verdict.
- `diffusion`: fake. High diffusion score indicates synthetic generation despite the face looking visually plausible.
- `blending_diffusion`: fake. The diffusion expert overrides blending: residual signal indicates synthetic generation.

## Scenario: `uncertain`

**Image:** `/data/poc/demo_uncertain.png`

| run | experts | prediction | real | fake | confidence | runtime (s) | error |
|-----|---------|------------|------|------|------------|-------------|-------|
| `none` | — | fake | 0.470 | 0.530 | 0.530 | 7.30 | — |
| `blending` | blending | fake | 0.460 | 0.540 | 0.540 | 11.00 | — |
| `diffusion` | diffusion | fake | 0.480 | 0.520 | 0.520 | 10.90 | — |
| `blending_diffusion` | blending,diffusion | fake | 0.460 | 0.540 | 0.540 | 14.50 | — |


**Evidence status:** `Uncertain`

**Rationale:** All four runs agree on 'fake' but combined-experts confidence 0.54 is below 0.70; treat as borderline.


**Per-run answer excerpts:**

- `none`: fake. There may be subtle manipulation but the cues are weak.
- `blending`: fake. Blending score is mid-range; verdict is borderline.
- `diffusion`: fake. Diffusion expert is near its decision threshold.
- `blending_diffusion`: fake. Both experts hover near their thresholds; treat the verdict as low confidence.

## Scenario: `failed`

**Image:** `/data/poc/demo_failed.png`

| run | experts | prediction | real | fake | confidence | runtime (s) | error |
|-----|---------|------------|------|------|------------|-------------|-------|
| `none` | — | fake | 0.200 | 0.800 | 0.800 | 7.50 | — |
| `blending` | blending | fake | 0.180 | 0.820 | 0.820 | 11.30 | — |
| `diffusion` | diffusion | — | — | — | — | 3.10 | Inference error: RuntimeError: CUDA out… |
| `blending_diffusion` | blending,diffusion | — | — | — | — | 3.40 | Inference error: RuntimeError: CUDA out… |


**Evidence status:** `Failed/insufficient`

**Rationale:** Insufficient signal: 2/4 runs failed and/or required scores are missing for ['diffusion', 'blending_diffusion'].


**Per-run answer excerpts:**

- `none`: fake. Asymmetric facial features suggest manipulation.
- `blending`: fake. Blending detector flags high seam likelihood.
- `diffusion`: Inference error: RuntimeError: CUDA out of memory while loading diffusion expert
- `blending_diffusion`: Inference error: RuntimeError: CUDA out of memory while loading diffusion expert

## How this supports the thesis

The thesis investigates **explainable** deepfake detection. A single verdict + rationale from one MLLM run is hard to trust on its own. By running the same image under multiple expert configurations and comparing the results, this layer gives a supervisor / end user three extra pieces of information that an isolated X2DFD call cannot:

1. **Robustness** — does the verdict survive removing or changing experts?
2. **Source attribution** — when verdicts differ, *which* expert flipped it?
3. **Calibration cue** — is the confidence high enough to act on, or should the case be flagged for human review?

The Markdown report above is intentionally simple so each rule can be audited; a richer dashboard view is a follow-up, not a POC requirement.
