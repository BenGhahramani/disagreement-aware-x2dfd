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

Current unit coverage:

- `tests/test_check_environment.py` — 34 tests for the Stage 1 environment
  checker (mocked imports, fake torch module, tmp-path weights/datasets,
  exit-code mapping).

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

As of 2026-08-06 the software layer passes (exit `0` with the skip flags) and
the full check fails only on missing weights and missing dataset images.

## Real smoke test (Stage 2)

Not implemented yet — blocked because no weights are installed
(`docs/WEIGHTS_SETUP.md`) and no dataset image resolves locally. The planned
entry point is `tools/run_smoke_test.py` with unit tests in
`tests/test_smoke_test_helpers.py`.

## Expert matrix (Stage 3)

Not implemented yet. Planned entry point: `tools/run_expert_matrix.py`.

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
