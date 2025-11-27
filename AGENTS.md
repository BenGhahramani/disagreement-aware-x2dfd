# Repository Guidelines

## Project Structure & Module Organization
- `src/` core code: `diffusion/` (networks, processing, detector), `blending/` (detector, utils), `__init__.py`.
- `utils/` shared helpers: `model_scoring.py`, `lora_inference.py`, `paths.py`, `pipeline_utils.py`.
- `train/` training pipeline (`pipeline.py`, `model_train.py`, `configs/`, `outputs/`).
- `eval/` inference + metrics (`infer/runner.py`, `tools/compute_auc.py`, `configs/`, `outputs/`).
- `datasets/` raw images, metadata JSONs, prompts; `weights/` model files (not tracked).

## Build, Test, and Development Commands
- Environment: `bash install.sh && conda activate X2DFD`.
- Quick eval (LoRA inference + AUC): `./test.sh`.
- Manual eval:
  ```
  python -m eval.infer.runner --config eval/configs/infer_config.yaml
  python -m eval.tools.compute_auc
  ```
- Demo (single image):
  ```
  python demo.py --image /abs/img.png --model-base weights/base/llava-v1.5-7b \
                 --adapter-path weights/checkpoints/ckpt/FR/llava-v1.5-7b-lora-[small]
  ```
- Train pipeline: `./train.sh --run-train [--train-gpus 0,1]`.

## Coding Style & Naming Conventions
- Python 3.10; 4‑space indent; follow PEP 8; prefer type hints and f‑strings.
- Names: `snake_case` for functions/vars, `PascalCase` for classes; modules live under `src/<domain>/`.
- Outputs should record absolute image paths; dataset JSON uses top‑level `Description` as root.
- Keep functions small and side‑effect–light; use dataclasses for simple configs.

## Testing Guidelines
- Primary check: compute ROC AUC from “fake score” turns.
  - Results land in `eval/outputs/…`; latest run: `eval/outputs/infer/latest_run.json`; metrics: `eval/outputs/metrics/auc.json`.
- Add ad‑hoc validator scripts near modules (e.g., `src/diffusion/test_compare_json.py`).
- Expected score range is `[0,1]`. Sanity‑test on 2–3 images before large runs.

## Commit & Pull Request Guidelines
- Use Conventional Commits (`feat|fix|docs|refactor|test|chore`); scope by module, e.g. `feat(diffusion): batched detector infer`.
- PRs must include: clear description, reproduction commands, AUC before/after (paths to `eval/outputs`), linked issues, and focused diffs.

## Security & Configuration Tips
- Never commit datasets or weights; place under `datasets/` and `weights/` locally.
- Useful env vars: `X2DFD_BASE_MODEL`, `X2DFD_WEIGHTS`, `X2DFD_OUTPUT`, `X2DFD_DATASETS` (see `utils/paths.py`).
- GPU is auto‑detected; override with `GPUS` or `--train-gpus`.
- Adding experts/providers? See `src/EXPERTS_GUIDE.md` for the registry pattern.
