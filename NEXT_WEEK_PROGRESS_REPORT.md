# Next-week progress report

**Date:** 2026-08-06
**Repository:** `disagreement-aware-x2dfd` (branch `main`, HEAD `0a031bf`)
**Scope of this session:** Stage 1 — environment and dependency verification,
plus creation of the isolated project environment.

---

## 1. Stages completed

| Stage | Outcome |
| --- | --- |
| Stage 1 — Environment and dependency verification | **Tooling complete and unit-tested. Software layer now PASSES.** The full check still fails on missing weights and missing dataset images. |
| Stage 2 — Single-image smoke test | **Not started.** Blocked: no weights, no locally resolvable image. |
| Stage 3 — Expert configuration matrix | **Not started.** Depends on Stage 2. |
| Stage 4 — Connect real outputs to the proof of concept | **Not started.** Depends on Stage 3. |
| Stage 5 — Small evaluation batch | **Not started.** Depends on Stage 4. |

Nothing was committed. No upstream source file was modified. Nothing was
deleted, reset or rewritten. No packages were installed globally.

### Files added

| Path | Purpose |
| --- | --- |
| `tools/check_environment.py` | Environment/dependency/weight/dataset checker with `--help`, JSON output, non-zero exit on missing requirements |
| `tools/__init__.py` | Makes `tools/` importable from tests |
| `tests/conftest.py` | Puts the repository root on `sys.path` |
| `tests/test_check_environment.py` | 34 unit tests; no weights, no GPU |
| `pytest.ini` | Markers `unit` / `integration` / `gpu` / `slow` |
| `docs/RUNTIME_SETUP.md` | Requirements, one-image commands, venv build, verified machine state |
| `docs/WEIGHTS_SETUP.md` | The four weight components: source, path, size, config reference, verification |
| `TESTING.md` | Commands per test category and per planned stage |
| `NEXT_WEEK_PROGRESS_REPORT.md` | This report |
| `.venv/` (untracked, gitignored) | Isolated Python 3.10.4 environment |

---

## 2. Environment built this session

Conda is not installed, so `install.sh` could not run as written. An equivalent
isolated **Python 3.10.4 venv** was created at `.venv/` and the pinned
dependency set installed into it. Full command list and resulting versions are
in `docs/RUNTIME_SETUP.md` section 4.

Two deviations from `install.sh`, both deliberate and documented:

1. **Torch pinned to `2.1.2+cu121` and installed first.** LLaVA requires
   `torch==2.1.2`, and on Windows PyPI serves CPU-only torch wheels, so
   installing `llava` against an unpinned torch risks silently replacing the
   CUDA build. Pre-pinning satisfies LLaVA's specifier and torch was confirmed
   untouched afterwards (`torch 2.1.2+cu121 cuda True`).
2. **`llava` installed from GitHub, not PyPI.** `pip install llava==1.2.2.post1`
   fails — PyPI publishes only `llava` `0.0.1.dev0`. The GitHub source is the
   fallback `install.sh` itself documents. Installed commit `c121f043`.

`accelerate` (0.28.0 → 0.21.0) and `einops` (0.7.0 → 0.6.1) were downgraded by
the `llava` install. `install.sh` installs `llava` after its own pins too, so
this venv matches the upstream end state rather than diverging from it.
`pip check` reports no broken requirements.

---

## 3. Tests run and results

### Unit tests (in the venv)

```text
$ .venv\Scripts\python.exe -m pytest -m unit
platform win32 -- Python 3.10.4, pytest-8.3.5, pluggy-1.6.0
collected 34 items
tests\test_check_environment.py ..................................  [100%]
============================= 34 passed in 0.21s =============================
```

Coverage: Python-version pass/warn/fail bands; required vs optional imports;
torch import failure; CUDA unavailable; VRAM above and below the fp16
threshold; a CUDA probe raising `RuntimeError`; YAML errors (missing file,
non-mapping); weight-requirement derivation for model, blending and diffusion
experts; env-var expansion; file-vs-directory checks; writable-directory
success and failure; dataset JSON checks (image present, image missing, missing
JSON, malformed JSON, no inputs); summary logic including `--strict`; text
rendering; and `main()` exit codes and JSON report. Dependency state is mocked
throughout, so these never need weights, GPU or network.

### Import smoke checks (real upstream modules, no weights)

```text
$ .venv\Scripts\python.exe -c "import utils.lora_inference"
utils.lora_inference OK

$ .venv\Scripts\python.exe -c "import src.blending.detector, src.diffusion.detector, utils.model_scoring"
blending OK, diffusion OK,
providers: ['aligner','blending','diffdet','diffusion','diffusion_detector','forensics']
```

This matters because `utils/lora_inference.py` imports `llava.*` at module
level: the runner's import chain is now satisfied. It does **not** mean the
model loads or runs.

### Real environment check

```text
$ .venv\Scripts\python.exe -m tools.check_environment
PASS=23 WARN=2 FAIL=9   ->  exit 1  (FAIL)
blocking: weights.model.adapter, weights.model.base, weights.expert.Blending,
          weights.expert.Diffusion.dir, weights.expert.Diffusion.config,
          datasets.DFDCP_real.json, datasets.DFDCP_fake.json,
          datasets.DFDC_real.json, datasets.DFDC_fake.json

$ .venv\Scripts\python.exe -m tools.check_environment --skip-weights --skip-datasets
PASS=23 WARN=2 FAIL=0   ->  exit 0  (PASS)
```

Remaining warnings: `deepspeed` absent (training only, optional) and the GPU's
10.0 GiB VRAM. Machine-readable evidence: `eval/outputs/env_check.json`.

**No package or version failures remain.**

---

## 4. Real inference evidence

**None.** No model has been loaded and no image has been scored. The
proof-of-concept demo still runs on mock fixtures only. No claim of working
inference is made in this report.

---

## 5. Measured runtimes

Tooling only, since no inference ran:

| Command | Wall clock |
| --- | --- |
| `pytest -m unit` (34 tests) | ~0.2 s |
| `tools.check_environment` | ~5 s (dominated by importing torch for the CUDA probe) |
| venv build: CUDA torch install | ~3.5 min |
| venv build: pinned stack + `llava` from source | ~3.5 min |

No model load time, per-image inference time, or expert-scoring time has been
measured.

---

## 6. Observed expert agreement or disagreement

**Not observable yet.** The four configurations have not run against a real
image. The only agreement/disagreement behaviour shown so far comes from the
hand-written fixtures in `proof_of_concept/fixtures/`, which demonstrate that
the rule layer classifies inputs as designed — not that the models behave that
way.

---

## 7. Current blockers

### ~~Blocker A — missing Python packages~~ RESOLVED

All 15 required packages import in `.venv`, including `llava`. The software
layer of the checker exits `0`.

### Blocker B — no model weights present (open)

`weights/` does not exist. Four components required by
`eval/configs/infer_config.yaml`: LLaVA-1.5-7B base, LoRA adapter
`llava-v1.5-7b-lora-[ble-diff]`, blending detector `best_gf.pth`, and
`ours-sync/` with its `config.yaml`. Sources, destinations, approximate sizes
(~14–15 GB total) and verification commands are in `docs/WEIGHTS_SETUP.md`.
**Not downloaded — awaiting explicit go-ahead.**

### Blocker C — no images resolve locally (open)

The `infer.inputs` JSONs are valid but their `Description` root is
`/data/250010183/Datasets`, an upstream cluster path. Every sampled image is
absent here, so even with weights there is nothing to run on. Stage 2 needs
either a local image folder plus a one-image JSON, or cluster access.

### Blocker D — VRAM headroom (open, mitigable)

RTX 3080 with 10.0 GiB. fp16 LLaVA-1.5-7B needs ~14 GiB before a detector loads
alongside it. `bitsandbytes` 0.45.5 is installed, so 4-bit loading is available,
but that is a deviation from the published configuration and should be recorded
as such in the thesis.

---

## 8. Does the framework currently appear suitable?

**Still undetermined, though one risk has been retired.** The dependency stack
— historically the hardest part of LLaVA-based repositories, especially on
Windows — now installs cleanly and every upstream inference module imports.
That is real evidence the codebase is workable on this machine.

The structural fit also remains good: the runner exposes an `--experts` flag
accepting `none`, `blending`, `diffusion` or both, a `--json` single-input
path, and an `--output` override, and it writes the conversation-style JSON the
existing POC parser understands.

What still cannot be judged is whether the model runs within 10 GiB, how long
an image takes, and whether the four configurations disagree in informative
ways. Those need one real inference.

---

## 9. Exact next technical step

Two inputs are needed from the supervisor/author before Stage 2:

1. **Approval to download ~14–15 GB of weights** into `weights/` per
   `docs/WEIGHTS_SETUP.md`. The base model can be pulled non-interactively from
   Hugging Face; the LoRA and both detectors come from Baidu Netdisk or Google
   Drive and will likely need a manual browser download.
2. **A local path to at least one real face image** (or confirmation that work
   should move to the cluster where the dataset already lives).

Once both exist: write a one-image dataset JSON with a local `Description`, run
`.venv\Scripts\python.exe -m tools.check_environment` until it exits `0`, then
begin Stage 2 by building `tools/run_smoke_test.py` and its unit tests, and
attempt the first real single-image run with `--experts none` to isolate the
LLaVA path before adding detectors.
