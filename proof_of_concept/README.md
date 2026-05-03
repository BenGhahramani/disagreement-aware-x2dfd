# Disagreement-aware X2DFD comparison — Proof of Concept

A small, **self-contained** layer on top of X2DFD that runs the **same image
under multiple expert settings** and reports an **evidence status** plus a
short rationale based on agreement/disagreement across runs.

> This POC does **not** train or modify any X2DFD code. It only consumes the
> conversation-style JSONs that `eval/infer/runner.py` already writes.

---

## How this supports the thesis

The thesis investigates **explainable** deepfake detection. A single verdict
plus rationale from one MLLM run is hard to trust on its own — it is one
sample of one configuration of one model. This POC layer adds three signals
that an isolated X2DFD call cannot provide:

1. **Robustness across expert settings.** Running the same image with
   `none` / `blending` / `diffusion` / `blending+diffusion` shows whether
   the verdict survives the removal or addition of each expert.
2. **Source attribution for disagreement.** When the four runs do not all
   agree, the rule layer flags *which* comparison flipped the verdict
   (bare LLaVA vs combined experts, or blending vs diffusion).
3. **Calibration cue.** When all four runs agree but the combined-experts
   confidence is below a threshold, the case is surfaced as `Uncertain` —
   exactly the kind of borderline image that benefits from human review or
   from being highlighted in an explainability dashboard.

The whole layer is intentionally **rule-based and auditable** so the
disagreement logic can be defended in a thesis discussion, not treated as a
black box. Replacing it with a learned aggregator is a follow-up question,
not a POC-week question.

---

## What this demo does

1. Reads four X2DFD-style result JSONs for the same image, one per setting:

   | run name             | X2DFD CLI flag                   |
   |----------------------|----------------------------------|
   | `none`               | `--experts none`                 |
   | `blending`           | `--experts blending`             |
   | `diffusion`          | `--experts diffusion`            |
   | `blending_diffusion` | `--experts blending,diffusion`   |

2. Normalises each into a `RunRecord` with a fixed schema:
   `run_name, experts_used, prediction, real_score, fake_score, confidence,
   explanation, runtime_seconds, error`.

3. Applies a tiny rule-based evaluator that classifies the comparison as one
   of: **`Stable`**, **`Contested`**, **`Uncertain`**, **`Failed/insufficient`**.

4. Renders a Markdown report with a per-run table, the evidence status, the
   generated rationale, and per-run answer excerpts.

The demo ships with **mock fixtures** for all four statuses so the
disagreement layer can be shown to a supervisor even before model weights or
GPU access are ready.

---

## How to run

From the repo root, with **Python 3.10+** (no third-party packages required —
standard library only):

```bash
# Render all four built-in scenarios to stdout
python -m proof_of_concept.run_demo --all

# Render a single scenario to stdout
python -m proof_of_concept.run_demo --scenario contested

# Write the full report to a file
python -m proof_of_concept.run_demo --all \
  --output proof_of_concept/outputs/report.md
```

Open `proof_of_concept/outputs/report.md` in any Markdown viewer (the Cursor
preview pane works well for the supervisor demo).

---

## Folder layout

```
proof_of_concept/
├── README.md                 # this file
├── SUPERVISOR_DEMO.md        # short script for the meeting
├── __init__.py               # package marker
├── schema.py                 # RunRecord, ImageComparison, Status
├── normaliser.py             # parse X2DFD result JSONs -> RunRecord
├── evaluator.py              # rule-based status + rationale
├── report.py                 # Markdown rendering
├── run_demo.py               # CLI: render a report from existing result JSONs
├── launcher.py               # CLI: run the four X2DFD invocations + write runtimes.json
└── fixtures/
    ├── stable/
    │   ├── demo_none.json
    │   ├── demo_blending.json
    │   ├── demo_diffusion.json
    │   ├── demo_blending_diffusion.json
    │   └── runtimes.json     # optional sidecar: {run_name: seconds}
    ├── contested/  ...
    ├── uncertain/  ...
    └── failed/     ...
```

---

## How mock outputs map to real X2DFD outputs

Each `demo_<run>.json` is **byte-for-byte the same shape** as a real
`*_result.json` produced by `eval/infer/runner.py`:

```jsonc
[
  {
    "id": "1",
    "image": "/abs/path/to/image.png",
    "conversations": [
      {"from": "human", "value": "<image>\nIs this image real or fake? And the blending score is 0.812."},
      {"from": "gpt",   "value": "fake"},
      {"from": "real score", "value": "0.1340"},
      {"from": "fake score", "value": "0.8660"}
    ]
  }
]
```

Mock-only deviations (none affect the parser):

- `image` is a placeholder path like `/data/poc/demo_stable.png`.
- `gpt` answers are slightly longer than a real `max_new_tokens=4` run; this
  matches what you would get if you bumped `max_new_tokens` to ~64 in the
  POC config (recommended for a richer demo).
- The error-case `gpt` value uses the exact `Inference error: ...` string
  that `utils/lora_inference.py` emits on failure, so the parser's error
  detection is exercised.

The optional `runtimes.json` sidecar (`{run_name: seconds}`) is a POC
convention. The X2DFD runner does not currently emit per-image runtimes; the
sidecar lets the report show wall-clock numbers without modifying upstream
code.

---

## How to switch to real outputs

Once the X2DFD weights are in place and a one-image run works, there are two
paths: a one-shot launcher (recommended) and a manual four-call sequence.

### Recommended: use the launcher

`proof_of_concept/launcher.py` runs all four `--experts` settings, names each
output to match the POC convention, captures per-run wall-clock to
`runtimes.json`, and (with `--render`) renders the Markdown report in the
same directory.

```bash
# Single image, defaults to a timestamped output dir, then render the report
python -m proof_of_concept.launcher --image /abs/path/to/image.png --render

# Existing dataset JSON, custom output dir
python -m proof_of_concept.launcher \
  --json datasets/raw/data/poc/demo_one.json \
  --output-dir eval/outputs/poc/real_run_001 \
  --render

# Show the four commands without invoking X2DFD (good for verifying weights/config)
python -m proof_of_concept.launcher --image /abs/path/to/image.png --dry-run
```

Failed runner calls do not abort the launcher: it records the wall-clock for
each attempt and the POC parser will mark the missing/errored runs as
`Failed/insufficient` automatically (exactly like the bundled `failed/`
fixture).

### Manual: four runner calls + render

If you'd rather see every step explicitly:

1. Create a one-image dataset JSON, e.g. `datasets/raw/data/poc/demo_one.json`:

   ```json
   {
     "Description": "/abs/path/to/datasets/raw/data/poc",
     "images": [{"image_path": "demo.png"}]
   }
   ```

2. Run the X2DFD runner four times, naming the outputs to match the POC
   convention (`demo_<run>.json`):

   ```bash
   mkdir -p eval/outputs/poc/real_run

   python -m eval.infer.runner --config eval/configs/infer_config.yaml \
     --json datasets/raw/data/poc/demo_one.json --experts none \
     --output eval/outputs/poc/real_run/demo_none.json

   python -m eval.infer.runner --config eval/configs/infer_config.yaml \
     --json datasets/raw/data/poc/demo_one.json --experts blending \
     --output eval/outputs/poc/real_run/demo_blending.json

   python -m eval.infer.runner --config eval/configs/infer_config.yaml \
     --json datasets/raw/data/poc/demo_one.json --experts diffusion \
     --output eval/outputs/poc/real_run/demo_diffusion.json

   python -m eval.infer.runner --config eval/configs/infer_config.yaml \
     --json datasets/raw/data/poc/demo_one.json --experts blending,diffusion \
     --output eval/outputs/poc/real_run/demo_blending_diffusion.json
   ```

3. (Optional) Write `eval/outputs/poc/real_run/runtimes.json`:

   ```json
   {"none": 8.1, "blending": 12.4, "diffusion": 11.9, "blending_diffusion": 15.7}
   ```

4. Render the report against the real directory — same CLI, no code change:

   ```bash
   python -m proof_of_concept.run_demo \
     --scenario-dir eval/outputs/poc/real_run \
     --output proof_of_concept/outputs/real_report.md
   ```

Either path produces the same `demo_<run>.json` + `runtimes.json` layout that
the POC parser already understands.

---

## Evidence-status rules (short)

Defined in `evaluator.py`, with thresholds at the top of the file
(`CONF_HIGH=0.70`, `CONF_LOW=0.55`). Decision order, first match wins:

1. **Failed/insufficient** — combined-experts run errored, has no scores, or
   ≥ 2 of the 4 runs failed.
2. **Contested** — at least two valid runs disagree on real/fake.
3. **Stable** — all runs agree and combined-experts confidence ≥ `CONF_HIGH`.
4. **Uncertain** — all runs agree but confidence is below `CONF_HIGH`.

The rationale string also flags **which** comparison flipped the verdict
(e.g. "adding experts flips bare-LLaVA verdict" or "blending and diffusion
experts disagree").

---

## What this is not

- Not a new deepfake detector.
- Not a replacement for the X2DFD pipeline.
- Not a calibrated probabilistic ensemble — the evaluator is intentionally a
  small, auditable rule-based layer suitable for a thesis proof of concept.

---

## Next step: connect to real X2DFD outputs

The full mock → real handoff is documented above in
[*How to switch to real outputs*](#how-to-switch-to-real-outputs). The short
version is:

1. Place the X2DFD weights at the paths in `eval/configs/infer_config.yaml`
   (LLaVA base, LoRA adapter, blending detector, diffusion detector).
2. Create a one-image dataset JSON and run `eval/infer/runner.py` four times
   with `--experts none`, `blending`, `diffusion`, `blending,diffusion`,
   writing each output as `demo_<run>.json` into a single directory.
3. (Optional) Write a sibling `runtimes.json` mapping `run_name -> seconds`
   so the report shows wall-clock numbers; without it that column shows `—`.
4. Render the report against the real directory:

   ```bash
   python -m proof_of_concept.run_demo \
     --scenario-dir eval/outputs/poc/real_run \
     --output proof_of_concept/outputs/real_report.md
   ```

No code in this folder needs to change for the swap. If a real run errors
(e.g. CUDA OOM), the same parser will pick up the runner's `Inference error:`
string and the comparison will be classified as `Failed/insufficient`, just
like the bundled `failed/` fixture demonstrates.
