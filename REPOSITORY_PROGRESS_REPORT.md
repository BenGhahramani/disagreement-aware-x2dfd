# Repository Summary

This repository is a local copy of the upstream **X2-DFD** (eXplainable and eXtendable Deepfake Detection) research codebase, hosted under the GitHub remote `BenGhahramani/disagreement-aware-x2dfd`. The inherited system is a Python research pipeline for image deepfake detection: Specific Feature Detectors (blending and diffusion) supply scores that are injected into LLaVA LoRA prompts; the model returns a real/fake verdict plus optional short text and real/fake token scores. There is no web frontend, no REST API, and no media-upload service in the tree.

On top of that inherited codebase, a single post-import commit by **BenGhahramani** (2026-05-03, `0a031bf`, message `proof of concept`) adds a self-contained `proof_of_concept/` layer. That layer compares four expert configurations for the same image and assigns an evidence status (`Stable` / `Contested` / `Uncertain` / `Failed/insufficient`). The POC is runnable end-to-end on bundled mock fixtures using the Python standard library only. Live connection through the real X2DFD runner is implemented as a launcher CLI, but this workspace has no `weights/` directory, so live GPU inference was not verified here.

Overall progress: upstream X2DFD code is present and documented; thesis-specific disagreement-aware comparison is implemented as a mock-demo POC with a real-runner launcher scaffold; dashboard/UI, video, heatmaps, automated unit tests for the POC, and verified live multi-expert runs are not present in repository evidence.

# Repository Provenance

| Item | Evidence |
| --- | --- |
| Remotes | `origin` → `https://github.com/BenGhahramani/disagreement-aware-x2dfd.git` (fetch/push). No separate `upstream` remote is configured locally. |
| Likely upstream project | **X2-DFD / X2DFD** by Chen Yize et al. (NeurIPS 2025 / arXiv:2410.06126). README clone URL is `https://github.com/chenyize111/X2DFD.git`; issues badge and citation point to that project. Scripts under `tools/` reference push/mirror targets such as `chenyize111/X2DFD2.git`. |
| Likely fork or import point | Not a GitHub “fork parent” metadata check (GitHub CLI unavailable here). History is a linear import of the X2DFD commit chain into this remote name. **Practical import/baseline for thesis work:** tip of inherited history on `main` before Ben’s commit = `0b4eb31` (`docs(readme): add google drive weights link`, Chen Yize, 2026-04-09). |
| Baseline commit used for comparison | `0b4eb31f79744d5357db354066e8948b823a9501` (last commit on `main` before `proof of concept`). Cumulative upstream body also descends from initial commit `b9acb03` (2025-09-25). |
| Current branch and HEAD | Branch `main`, tracking `origin/main`. HEAD = `0a031bfc63de68d3dec5a06d0177b0fb8c48bb25` (`proof of concept`, BenGhahramani, 2026-05-03). |
| Working tree | Clean (`git status`: nothing to commit). |
| Other remote branches | `origin/master` at `215247c` (2025-12-30); `origin/refactor/train-test-paths` at `718c62c` (one-line `max_new_tokens` fix, not merged into `main`). |
| Tags | None. |
| Merges | No merge commits in history. |
| Authors (all commits) | X2DFD Maintainer 33; Chen Yize 6; BenGhahramani 1. |
| Uncommitted / staged | None. |

**Uncertainty:** Whether Ben’s GitHub repository was created via GitHub’s Fork button, `git clone` + remote rename, or mirror push cannot be confirmed without GitHub API access. Content and authorship of history before `0a031bf` are clearly the X2DFD project, not Ben’s.

# Commit History Summary

Focus after the thesis-relevant baseline. Full history has **40 commits** across all refs (no merges). Early history is condensed; post-public-release and post-import commits are listed in more detail.

| Date | Hash | Author | Message | Files (summary) | Actual change |
| --- | --- | --- | --- | --- | --- |
| 2025-09-25 | `b9acb03` | Chen Yize | Initial commit | `.gitattributes`, `rank_questions.py` | Seed repo. |
| 2025-09-25 | `cf81461` | Chen Yize | 更新MFA | Removes `rank_questions.py` | Early MFA cleanup. |
| 2025-09-25 | `9b8b647` | Chen Yize | MFA finish | MFA scripts, `qa_utils/`, sample results | Model feature assessment tooling. |
| 2025-09-25 | `e809965` | Chen Yize | inference added | `qa_utils/inference.py` expanded | Inference path for QA evaluation. |
| 2025-11-23 | `6d9ff3d` | X2DFD Maintainer | Refactor: unify result outputs… | Large restructure → `eval/`, `train/`, `datasets/`, `src/`, `utils/` | Public pipeline layout established. |
| 2025-11-23 | `b9e86f4`–`aef74d9` | X2DFD Maintainer | Env: install.sh… | `install.sh`; remove `requirements.txt` | Conda env `X2DFD`, pinned deps, LLaVA install. |
| 2025-11-23 | `7d8b7af` | X2DFD Maintainer | Docs + `demo.py` | README, `demo.py` | Single-image demo entry point. |
| 2025-11-23 | `d9c7023`–`cee5c28` | X2DFD Maintainer | I18N / train JSON paths | Configs, comments, train annotations | English i18n; train data wiring. |
| 2025-11-23 | `02c37ba` / `fbe4002` | X2DFD Maintainer | Add/remove `run_envs_and_train_test.sh` | Script add then delete | Transient helper. |
| 2025-11-27 | `a5085b8` | X2DFD Maintainer | chore(release): public cleanup… | MIT, `CITATION.cff`, `AGENTS.md`, dataset JSONs, legacy archive | Public release packaging (very large JSON metadata add). |
| 2025-11-27 | `2611746`–`b590cd8` | X2DFD Maintainer | docs(readme): … | README, `tools/*.sh` | Docs polish; mirror-push helper scripts. |
| 2025-11-27 | `2f28d5a` | X2DFD Maintainer | refactor(eval): LoRA-only; remove AUC | Deletes `eval/tools/compute_auc.py`; updates runner/demo/test | Evaluation simplified to LoRA inference. |
| 2025-11-28 | `6fe9bf0`–`385a751` | X2DFD Maintainer | docs(readme): … | README | Docs; TODO notes checkpoints pending. |
| 2025-12-30 | `83f7ec9` | X2DFD Maintainer | fix(infer): improve inference robustness | `demo.py`, `runner.py`, `lora_inference.py`, configs | Error handling / robustness for inference. |
| 2025-12-30 | `4d25afb`–`215247c` | X2DFD Maintainer | docs(readme): Baidu link notes | README | Weight download link edits (`origin/master` tip). |
| 2025-12-30 | `718c62c` | X2DFD Maintainer | fix(eval): default max_new_tokens=4 | `eval/configs/config.yaml` | Only on `origin/refactor/train-test-paths`, not on `main`. |
| 2026-04-09 | `7c8cc8b` | Chen Yize | docs(readme): update baidu netdisk links | README | Link update on `main`. |
| 2026-04-09 | `0b4eb31` | Chen Yize | docs(readme): add google drive weights link | README (+4 lines) | **Baseline immediately before thesis POC.** |
| 2026-05-03 | `0a031bf` | BenGhahramani | proof of concept | **+30 files / +1627 lines under `proof_of_concept/` only** | **Only commit attributable to Ben in this clone.** Adds disagreement-aware POC (schema, normaliser, evaluator, report, fixtures, launcher, docs). Does not modify upstream X2DFD source. |

# Files Added, Modified, Renamed and Deleted

## Relative to baseline `0b4eb31` → HEAD `0a031bf` (committed thesis delta)

### Added (all under `proof_of_concept/`)

| Path | Role |
| --- | --- |
| `proof_of_concept/schema.py` | `Status`, `RunRecord`, `ImageComparison` dataclasses. |
| `proof_of_concept/normaliser.py` | Parse X2DFD conversation-style result JSON → `RunRecord`; load four-run scenarios. |
| `proof_of_concept/evaluator.py` | Rule-based status + rationale (`CONF_HIGH=0.70`, `CONF_LOW=0.55` reserved). |
| `proof_of_concept/report.py` | Markdown report renderer. |
| `proof_of_concept/run_demo.py` | CLI for fixtures / custom scenario dirs. |
| `proof_of_concept/launcher.py` | Subprocess wrapper: four `eval.infer.runner` calls + `runtimes.json` + optional render. |
| `proof_of_concept/__init__.py` | Package marker. |
| `proof_of_concept/README.md`, `SUPERVISOR_DEMO.md` | Design notes and supervisor demo script. |
| `proof_of_concept/fixtures/{stable,contested,uncertain,failed}/` | Mock `demo_*.json` + `runtimes.json` for four statuses. |
| `proof_of_concept/outputs/report.md` | Pre-rendered sample Markdown report (generated artifact checked in). |

### Modified / renamed / deleted (post-baseline)

None in `0b4eb31..0a031bf`. Ben’s commit is additive only.

### Uncommitted

None (working tree clean). Note: writing this report file after analysis will create a new untracked file; it is not part of prior history.

## Inherited upstream (present at baseline; not Ben’s work)

Representative areas (established mainly by `6d9ff3d`, `a5085b8`, and follow-ups):

- `src/blending/`, `src/diffusion/` — SFD detectors and networks.
- `utils/` — LoRA inference, scoring registry, paths, evaluation helpers.
- `train/` — staged annotation → weak supply → LoRA training.
- `eval/infer/runner.py`, `eval/configs/`, `demo.py`, `install.sh`, `train.sh`, `test.sh`.
- `datasets/` — prompt JSONs and large annotation/metadata JSON trees (images themselves gitignored under `datasets/raw/images/`).
- `legacy/` — archived older scripts.
- `README.md`, `LICENSE` (MIT), `CITATION.cff`, `AGENTS.md`, `figs/`.

**Generated vs source:** Large JSON under `datasets/` and `eval/debug/` are data/fixtures from upstream release, not model code. `proof_of_concept/outputs/report.md` is a committed generated demo report. `weights/` is gitignored and absent locally. `eval/outputs/` is gitignored and absent.

# Features Added

## 1. Disagreement-aware comparison POC (Ben — complete for mock path; partial for live path)

Runs four conceptual expert settings (`none`, `blending`, `diffusion`, `blending_diffusion`), normalises runner-shaped JSON, applies auditable rules, and emits Markdown.

- **Files:** `proof_of_concept/{schema,normaliser,evaluator,report,run_demo}.py`, fixtures, docs.
- **Symbols:** `Status`, `RunRecord`, `ImageComparison`; `parse_run_file`, `load_scenario`; `evaluate`; `render`; `main` in `run_demo`.
- **Commit:** `0a031bf`.
- **Status:** **Complete** for offline fixture demo (verified in this analysis: `python -m proof_of_concept.run_demo --all` produced Stable/Contested/Uncertain/Failed sections). **Partial** for production use: depends on external weights and successful runner invocations.

## 2. Multi-run launcher wrapping upstream inference (Ben — scaffolded / unverified live)

`launcher.py` builds one-image dataset JSON, invokes `python -m eval.infer.runner` four times with `--experts none|blending|diffusion|blending,diffusion`, writes `runtimes.json`, optionally renders.

- **Files:** `proof_of_concept/launcher.py`.
- **Symbols:** `_RUN_PLAN`, `_make_one_image_json`, `_run_one`, `main`.
- **Commit:** `0a031bf`.
- **Status:** **Scaffolded and code-complete as a CLI**, but **not verified live** here (`weights/` missing). Filtering uses provider/alias names; config alias `Diffusion` matches CLI `diffusion` (evidence in `eval/infer/runner.py` `_filter_experts`).

## 3. Inherited X2DFD features (upstream — present; live success not verified in this workspace)

| Feature | Location | Status in this clone |
| --- | --- | --- |
| Blending SFD (timm SwinV2) | `src/blending/detector.py`, provider `blending` | Code present; needs `weights/blending_models/best_gf.pth` |
| Diffusion SFD | `src/diffusion/{detector,core,processing,networks}` | Code present; needs `weights/ours-sync/` |
| Expert registry | `utils/model_scoring.py` | Present; docs also mention `src/forensics` which is **absent** |
| LoRA / LLaVA inference | `utils/lora_inference.py`, `demo.py`, `eval/infer/runner.py` | Present; needs base + LoRA weights |
| Staged training | `train/pipeline.py`, `train.sh` | Present as upstream pipeline |
| Prompt-guided explainability (text rationales) | Training/annotation stages; inference `max_new_tokens` often 4 | Inherited design; short answers by default |
| Heatmaps / Grad-CAM | — | **Not found** |
| Web dashboard / upload API | — | **Not found** |
| Video pipeline | — | **Not found** (image paths only) |

# Current Architecture and Technology Stack

**Languages:** Python 3.10 (documented); Bash scripts for install/train/test.

**Frameworks / libraries (from `install.sh` and imports):** Conda env `X2DFD`; PyTorch (+ CUDA 12.1 when NVIDIA present); `transformers`, `accelerate`, `peft`, `sentencepiece`, `tokenizers`, `safetensors`; optional `bitsandbytes`, `deepspeed`; LLaVA (`llava==1.2.2.post1` or GitHub); `timm`, OpenCV, NumPy, Pillow, PyYAML, tqdm, einops.

**Model repositories / assets (configured, not in git):** Hugging Face LLaVA-1.5-7B and CLIP ViT-L/14-336; X2DFD LoRA and detector weights via Baidu / Google Drive links in README; local layout under `weights/`.

**Frontend:** None (CLI + Markdown reports only).

**Backend / API:** None (no FastAPI/Flask/Django/Streamlit/Gradio app found).

**Build / package managers:** Conda + pip via `install.sh` (no `requirements.txt` after upstream env commit `b9e86f4`).

**Storage:** Local filesystem JSON datasets and outputs; no database.

**Runtime / hardware assumptions:** GPU preferred for LLaVA + detectors; CPU possible for POC fixtures and (slower) detector paths; env vars `X2DFD_BASE_MODEL`, `X2DFD_WEIGHTS`, `X2DFD_OUTPUT`, `X2DFD_DATASETS`, `GPUS`.

**Thesis POC stack:** Standard library only for `run_demo` / evaluator path.

# Current Execution Flow

## A. Inherited single-path inference (upstream)

1. Prepare weights and dataset JSON (`Description` + `images[].image_path`).
2. Entry: `python -m eval.infer.runner --config eval/configs/infer_config.yaml` or `./test.sh` or `demo.py --image …`.
3. Load `weak_supplies` (blending / diffusion_detector); optional `--experts` filter.
4. Resolve absolute image paths; run expert `infer` batches → P(fake) scores.
5. Build conversation prompts embedding scores; LoRA LLaVA generation (`max_new_tokens` typically 4).
6. Write conversation-style result JSON (human / gpt / real score / fake score turns) under `eval/outputs/` (gitignored).

**Breakpoint:** Without weights and images, this path cannot complete in the current workspace.

## B. Thesis disagreement POC — mock (implemented and runnable)

1. `python -m proof_of_concept.run_demo --all [--output …]`.
2. Load four fixture JSONs per scenario → `RunRecord`s.
3. `evaluate(runs)` → status + rationale.
4. `render` → Markdown to stdout or file.

**Status:** End-to-end verified offline during this analysis.

## C. Thesis disagreement POC — live (code present; not verified)

1. `python -m proof_of_concept.launcher --image /abs/img.png --render` (or `--json` / `--dry-run`).
2. Four subprocess calls to `eval.infer.runner` with different `--experts`.
3. Write `demo_*.json` + `runtimes.json`; optionally call `run_demo` for `report.md`.

**Breakpoints:** Missing weights; GPU/OOM risk; no committed evidence of a successful live `eval/outputs/poc/` run; dataset path `datasets/raw/data/poc/` referenced in docs but **not present** in the tree.

# Work In Progress and Missing Functionality

Evidence-backed only.

| Item | Classification |
| --- | --- |
| Mock disagreement statuses + Markdown report | **Implemented** (fixtures + `run_demo`; verified) |
| JSON normaliser compatible with runner output shape | **Implemented** (shape match documented; live parse not exercised here) |
| Launcher for four real runner calls | **Partially implemented** (CLI complete; live success unproven; no weights) |
| Rule thresholds / `CONF_LOW` “Stable (weak)” bucket | **Partially implemented** (`CONF_LOW` reserved; collapsed into Uncertain) |
| Dashboard / UI for disagreement statuses | **Planned in comments/docs** only (`report.py`, README mention “follow-up”) |
| Learned aggregator replacing rules | **Planned in docs** as non-POC follow-up |
| Heatmap / spatial explainability | **Referenced but absent** |
| Media upload validation service | **Absent** |
| Video preprocessing / inference | **Absent** |
| Automated pytest/unittest suite for POC | **Absent** |
| Upstream checkpoint upload | **Planned in upstream README TODO** (“Upload checkpoints”) |
| `src/forensics/` legacy package | **Referenced** in `EXPERTS_GUIDE.md` / providers; **directory absent** (diffusion path used instead) |
| `origin/refactor/train-test-paths` | **Unmerged** remote branch |
| Integration of POC into top-level README / `AGENTS.md` | **Absent** (POC not mentioned outside its folder) |
| Modification of core X2DFD detectors/train for thesis | **Not done** (POC explicitly does not modify upstream code) |

# Tests and Validation

| Evidence | Coverage | Runnable? |
| --- | --- | --- |
| `src/diffusion/test_compare_json.py` | Compares old vs new forensics aligners; imports `forensics.aligner` | Likely **broken as-is** (`src/forensics` missing; script paths assume old layout) |
| `train/tools/sanity_train_setup.py` | Train setup sanity helper | Upstream utility; not a unit test suite |
| `eval/qa/run.py` | QA evaluation runner | Upstream eval path; needs models/data |
| POC fixtures + `run_demo` | Manual/demo validation of four statuses | **Runnable** without GPU (executed successfully in this analysis) |
| Pre-rendered `proof_of_concept/outputs/report.md` | Sample supervisor demo output | Present |
| pytest / CI configs | — | **Not found** |

**Missing coverage:** No automated tests for `evaluate`, normaliser edge cases, or launcher dry-run. No evidence in-repo of a successful live multi-expert POC run.

# Thesis-Ready Progress Summary

The current repository combines an inherited open-source deepfake detection framework with a thin, thesis-specific proof-of-concept layer. The inherited system is X2-DFD, a research codebase for explainable and extendable image deepfake detection built around a LLaVA LoRA model and plug-in Specific Feature Detectors for blending and diffusion artefacts. That codebase supplies training and evaluation pipelines, expert score injection into prompts, and conversation-style JSON outputs. It does not provide a product-style interface, upload API, video pipeline, or visual heatmap explainability, and the local checkout does not contain model weights, so full upstream inference has not been demonstrated in this environment.

Original thesis work in this repository is concentrated in a single commit that adds the `proof_of_concept` package without altering upstream detector or training code. That package implements a disagreement-aware comparison over four expert settings for one image, normalises runner-shaped results, applies an auditable rule set that yields Stable, Contested, Uncertain, or Failed/insufficient status with a short rationale, and renders a Markdown report. Bundled fixtures allow the comparison logic to be demonstrated offline using only the Python standard library, and that fixture path has been executed successfully. A launcher that can invoke the real evaluation runner four times and then render the same report is present as working scaffolding, but connecting it to live models remains unverified because required checkpoints and a one-image POC dataset directory are not in the working tree.

The system is therefore still a work in progress relative to a complete disagreement-aware explainability toolchain. Concrete next implementation steps supported by the repository’s own documentation and gaps are: obtain and place X2DFD weights; run the launcher dry-run then live on a small set of images; confirm that expert CLI filters and output filenames align with production configs; optionally raise `max_new_tokens` when longer rationales are needed for demos; and only then extend beyond the rule layer toward richer presentation or evaluation—without treating the current Markdown POC as a finished dashboard.

# Evidence Appendix

## Exact paths (thesis addition)

- `proof_of_concept/schema.py` — `Status`, `RunRecord`, `ImageComparison`
- `proof_of_concept/normaliser.py` — `parse_run_file`, `load_scenario`, `primary_image`
- `proof_of_concept/evaluator.py` — `evaluate`, `CONF_HIGH`, `CONF_LOW`
- `proof_of_concept/report.py` — `render`
- `proof_of_concept/run_demo.py` — CLI entry
- `proof_of_concept/launcher.py` — `_RUN_PLAN`, `main`
- `proof_of_concept/fixtures/*/demo_*.json`, `runtimes.json`
- `proof_of_concept/README.md`, `SUPERVISOR_DEMO.md`, `outputs/report.md`

## Exact paths (inherited core)

- `demo.py`, `eval/infer/runner.py`, `utils/lora_inference.py`, `utils/model_scoring.py`
- `src/blending/detector.py`, `src/diffusion/detector.py`, `src/diffusion/core.py`
- `train/pipeline.py`, `install.sh`, `test.sh`, `train.sh`
- `eval/configs/infer_config.yaml`, `README.md`, `CITATION.cff`, `LICENSE`

## Important commits

- Baseline before thesis work: `0b4eb31f79744d5357db354066e8948b823a9501`
- Thesis POC: `0a031bfc63de68d3dec5a06d0177b0fb8c48bb25`
- Public upstream release packaging: `a5085b8301517286d309bbe215ddb3d727a39b09`
- Major pipeline refactor: `6d9ff3d…`
- Initial upstream seed: `b9acb034aed970c87dbcf69a70fce1cb75595766`

## Commands used (read-only analysis)

```text
git status
git remote -v
git branch -a
git tag -l
git rev-parse HEAD
git log -1 --format=...
git shortlog -sn --all
git reflog -20
git log --all --decorate --oneline / --pretty=format:...
git log --all --graph --decorate --oneline
git show <commit> --stat / --name-status
git diff 0b4eb31..0a031bf --stat
git log --all --merges
git merge-base main origin/master
git log origin/master..main / main..origin/master / …refactor…
git ls-tree -r --name-only HEAD
git log --all --pretty=format:... --numstat
git blame -L 1,20 --line-porcelain proof_of_concept/evaluator.py
git log --diff-filter=A --summary -- proof_of_concept/**
python -m proof_of_concept.run_demo --all
```

(No fetch, checkout, reset, clean, commit, or push was performed. `gh` was unavailable for fork-parent metadata.)

## Attribution summary

| Category | Attribution |
| --- | --- |
| X2DFD detectors, train/eval, demo, install, datasets metadata, docs/license | Upstream (Chen Yize / X2DFD Maintainer) |
| Entire `proof_of_concept/` tree | BenGhahramani (`0a031bf`), high confidence |
| Repo rename / remote under `disagreement-aware-x2dfd` | Likely Ben’s hosting choice; **cannot confirm** fork mechanism |
| Whether live weights were ever run successfully by Ben | **Cannot determine** from this clone (no `weights/`, no `eval/outputs/poc/` artefacts) |

## Uncertainties and limitations

1. No GitHub API/`gh` access: fork-parent and import method unconfirmed.
2. Upstream remote not configured; comparison is against in-repo history tip `0b4eb31`, not a freshly fetched `chenyize111/X2DFD`.
3. Live inference and launcher success were not executed (no weights).
4. Code existence ≠ runtime correctness for GPU paths.
5. Author identity for “X2DFD Maintainer `<devnull@example.com>`” may be a release identity rather than a personal mailbox; treated as upstream project commits.
6. Analysis date context: HEAD commit dated 2026-05-03; report based on that history and a clean working tree at inspection time.
