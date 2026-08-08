# Testing guide

Test framework: **pytest** (configured in `pytest.ini`). The repository had no
prior automated test suite, so pytest is the established choice going forward.

Run everything with the project virtual environment created in
`docs/RUNTIME_SETUP.md`:

```bash
.venv\Scripts\python.exe -m pytest -m unit      # Windows
.venv/bin/python -m pytest -m unit              # Linux / WSL
```

The bare `python -m pytest` form below assumes that venv is active.

Markers:

| Marker | Meaning |
| --- | --- |
| `unit` | No model weights, no GPU, no subprocess inference. Must always be runnable. |
| `integration` | Exercises real repository entry points end to end. |
| `gpu` | Requires a working CUDA device. |
| `slow` | Takes more than a few seconds, typically real model inference. |

---

## Unit tests only (no weights, no GPU)

```bash
python -m pytest -m unit
```

Current unit coverage (164 tests):

- `tests/test_check_environment.py` — environment checker (mocked imports, fake
  torch, tmp-path weights/datasets, exit-code mapping, protobuf regression).
- `tests/test_make_peft_compatible_config.py` — adapter-config down-converter.
- `tests/test_smoke_test_helpers.py` — Stage 2 smoke-test wrapper (mocked
  subprocess) plus the quantisation switch.
- `tests/test_expert_matrix.py` — 44 tests for Stage 3 (config normalisation,
  distinct POC-shaped outputs, valid/malformed/hidden-error cases, partial
  failure, CUDA OOM, summary generation).

## Integration tests

```bash
python -m pytest -m integration
```

None exist yet. Integration tests will be added alongside the Stage 2 smoke
test, and will be skipped automatically when weights are absent.

## GPU / slow tests

```bash
python -m pytest -m "gpu or slow"
```

None exist yet.

## Everything except model-dependent tests

```bash
python -m pytest -m "not gpu and not slow"
```

---

## Real environment check (Stage 1)

```bash
python -m tools.check_environment
python -m tools.check_environment --json --output eval/outputs/env_check.json
python -m tools.check_environment --skip-weights --skip-datasets   # software layer only
```

Exit `0` means the machine can attempt real inference. Exit `1` lists the
blocking items. See `docs/RUNTIME_SETUP.md` for the current machine status.

As of 2026-08-06 all weights are installed and
`python -m tools.check_environment --skip-datasets` exits `0`; the full check
fails only on the dataset images named in `eval/configs/infer_config.yaml`.

## Adapter-config compatibility

```bash
python -m tools.make_peft_compatible_config <adapter_dir>              # writes adapter_config.compatible.json
python -m tools.make_peft_compatible_config <adapter_dir> --in-place   # rewrites after backing up
python -m tools.make_peft_compatible_config <adapter_dir> --json
```

Exit `0` converted or already compatible, `1` blocked because an unsupported
field held a meaningful value, `2` usage error. Background in
`docs/WEIGHTS_SETUP.md`.

## Real smoke test (Stage 2)

```bash
python -m tools.run_smoke_test --manifest datasets/raw/data/poc/demo_one.json --experts none --load-4bit
```

Requires the weights and a GPU. `--load-4bit` is effectively mandatory on a
10 GiB card: the fp16 path needs CPU offload, which the pinned PEFT/accelerate
pair cannot combine with a LoRA adapter. Exit `0` means the run produced a
prediction with both scores; a passing run is recorded in
`docs/SMOKE_TEST_RESULTS.md` and `eval/outputs/smoke_test_summary.json`.

## Expert matrix (Stage 3)

```bash
python -m tools.run_expert_matrix --manifest datasets/raw/data/poc/demo_one_crop.json --config eval/configs/infer_config.windows.yaml --load-4bit
```

Requires weights and a GPU. Runs `none`, `blending`, `diffusion`, and
`blending,diffusion` in separate subprocesses. Results:
`eval/outputs/expert_matrix_summary.json` and `docs/EXPERT_MATRIX_RESULTS.md`.

Feed the matrix directory into the existing proof of concept:

```bash
python -m proof_of_concept.run_demo --scenario-dir eval/outputs/expert_matrix/demo_one_crop --output proof_of_concept/outputs/real_example_report.md
```

## Supervisor dashboard (saved outputs only)

Preferred Thursday launch (checks venv + saved Stage 3 files, then opens the browser):

```powershell
.\run_demo.ps1
```

Equivalent manual command:

```powershell
.venv\Scripts\python.exe -m streamlit run dashboard/app.py
```

Loads `eval/outputs/expert_matrix/demo_one_crop/` and
`eval/outputs/expert_matrix_summary.json`. No upload or live inference.
View-model unit tests: `tests/test_dashboard_view_model.py`.

## Small evaluation batch (Stage 5)

Not implemented yet. Planned entry point: `tools/run_small_evaluation.py`.

---

## Existing proof-of-concept demo (no GPU required)

The inherited-plus-thesis proof of concept renders Markdown from bundled mock
fixtures using only the standard library:

```bash
python -m proof_of_concept.run_demo --all
python -m proof_of_concept.run_demo --scenario contested
```

This is a demo, not a test; it has no automated assertions yet. Test coverage
for the POC is planned for Stage 4.

---

## Conventions for new test files

- Put unit tests in `tests/`, name them `test_<subject>.py`, and mark the
  module with `pytestmark = pytest.mark.unit` when every test in the file is a
  unit test.
- Mock subprocesses and model loading. Never require `weights/` in a unit test.
- Use `tmp_path` for any filesystem interaction.
