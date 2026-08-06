# Disagreement-aware X2DFD comparison — POC report

> **Proof-of-concept notice.** This report is produced by a *disagreement-aware comparison layer* built on top of the existing X2DFD pipeline. It is **not a new deepfake detector** and does not modify any X2DFD model code. It re-uses the same image, same model, same experts — the only thing it adds is running each image under several expert settings and applying a small rule-based evaluator over the resulting JSONs to surface agreement, disagreement, and low-confidence cases.

## Scenario: `demo_one_crop`

**Image:** `C:\Users\Ben\Desktop\UNI\REIT\root\disagreement-aware-x2dfd\datasets\raw\images\poc\real_face_01_crop.jpg`

| run | experts | prediction | real | fake | confidence | runtime (s) | error |
|-----|---------|------------|------|------|------------|-------------|-------|
| `none` | — | fake | 0.239 | 0.761 | 0.761 | 19.88 | — |
| `blending` | blending | fake | 0.359 | 0.641 | 0.641 | 21.67 | — |
| `diffusion` | diffusion | fake | 0.288 | 0.712 | 0.712 | 21.62 | — |
| `blending_diffusion` | blending,diffusion | fake | 0.314 | 0.686 | 0.686 | 22.90 | — |


**Evidence status:** `Uncertain`

**Rationale:** All four runs agree on 'fake' but combined-experts confidence 0.69 is below 0.70; treat as borderline.


**Per-run answer excerpts:**

- `none`: This image is fake
- `blending`: This image is fake
- `diffusion`: This image is fake
- `blending_diffusion`: This image is fake

## How this supports the thesis

The thesis investigates **explainable** deepfake detection. A single verdict + rationale from one MLLM run is hard to trust on its own. By running the same image under multiple expert configurations and comparing the results, this layer gives a supervisor / end user three extra pieces of information that an isolated X2DFD call cannot:

1. **Robustness** — does the verdict survive removing or changing experts?
2. **Source attribution** — when verdicts differ, *which* expert flipped it?
3. **Calibration cue** — is the confidence high enough to act on, or should the case be flagged for human review?

The Markdown report above is intentionally simple so each rule can be audited; a richer dashboard view is a follow-up, not a POC requirement.
