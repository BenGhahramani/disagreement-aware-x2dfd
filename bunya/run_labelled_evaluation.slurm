#!/bin/bash
#SBATCH --job-name=x2dfd-labelled
#SBATCH --account=a_css
#SBATCH --partition=gpu_cuda
#SBATCH --qos=gpu
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=labelled-eval-%j.out
#SBATCH --error=labelled-eval-%j.err
#
# Production labelled evaluation on UQ Bunya (FP16 / --no-4bit).
#
# Prefer the submit wrapper (sets paths, log dir, wall time, analysis dependency):
#   bash bunya/submit_labelled_evaluation.sh <manifest.json> <output_dir>
#
# Required environment (exported by the submit wrapper):
#   X2DFD_MANIFEST      Absolute path to a labelled evaluation manifest
#   X2DFD_OUTPUT_DIR    Absolute scratch output directory for this run
#
# Optional environment:
#   X2DFD_PROJECT_DIR   Repo on scratch (default: $SCRATCH/disagreement-aware-x2dfd)
#   X2DFD_WEIGHTS       Weights root (default: $PROJECT_DIR/weights)
#   X2DFD_VENV          Python venv (default: $PROJECT_DIR/.venv)
#   X2DFD_INFER_CONFIG  Infer YAML (default: eval/configs/infer_config.bunya.yaml)
#   X2DFD_VISION_TOWER  Optional explicit CLIP vision-tower directory
#
# Wall time: default 12:00:00 above; override at submit with
#   TIME_LIMIT=HH:MM:SS bash bunya/submit_labelled_evaluation.sh …
# Extrapolation from the measured 4-bit 4-image pilot (~92 s/image for four
# configs) suggests ~6 h for 240 images; FP16 on A100 is unmeasured, so 12 h
# is the conservative default. The runner resume flag remains enabled.

set -euo pipefail

module use /sw/auto/rocky9a/epyc3_a100/modules/all
module load python/3.10.4-gcccore-11.3.0-bare
module load cuda/12.1.1

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ -n "${X2DFD_MANIFEST:-}" ]] || fail "X2DFD_MANIFEST is not set"
[[ -n "${X2DFD_OUTPUT_DIR:-}" ]] || fail "X2DFD_OUTPUT_DIR is not set"

SCRATCH_BASE="${SCRATCH:-/scratch/user/${USER}}"
PROJECT_DIR="${X2DFD_PROJECT_DIR:-${SCRATCH_BASE}/disagreement-aware-x2dfd}"
WEIGHTS_DIR="${X2DFD_WEIGHTS:-${PROJECT_DIR}/weights}"
VENV_DIR="${X2DFD_VENV:-${PROJECT_DIR}/.venv}"
PYTHON="${VENV_DIR}/bin/python"
INFER_CONFIG="${X2DFD_INFER_CONFIG:-${PROJECT_DIR}/eval/configs/infer_config.bunya.yaml}"
MANIFEST="$(readlink -f "${X2DFD_MANIFEST}")"
OUTPUT_DIR="$(mkdir -p "${X2DFD_OUTPUT_DIR}" && readlink -f "${X2DFD_OUTPUT_DIR}")"
VISION_TOWER_DEFAULT="${PROJECT_DIR}/weights/base/clip-vit-large-patch14-336"
VISION_TOWER="${X2DFD_VISION_TOWER:-${VISION_TOWER_DEFAULT}}"

export X2DFD_PROJECT_ROOT="${PROJECT_DIR}"
export X2DFD_WEIGHTS="${WEIGHTS_DIR}"
export X2DFD_OUTPUT="${OUTPUT_DIR}"
export X2DFD_DATASETS="${PROJECT_DIR}/datasets"

echo "=== X2DFD labelled evaluation (FP16 / --no-4bit) ==="
echo "host: $(hostname)"
echo "date: $(date -Is)"
echo "slurm_job_id: ${SLURM_JOB_ID:-unset}"
echo "slurm_job_name: ${SLURM_JOB_NAME:-unset}"
echo "cuda_visible_devices: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "project: ${PROJECT_DIR}"
echo "weights: ${WEIGHTS_DIR}"
echo "venv: ${VENV_DIR}"
echo "manifest: ${MANIFEST}"
echo "output: ${OUTPUT_DIR}"
echo "infer_config: ${INFER_CONFIG}"
echo "vision_tower: ${VISION_TOWER}"

[[ -d "${PROJECT_DIR}" ]] || fail "project directory not found: ${PROJECT_DIR}"
[[ -x "${PYTHON}" ]] || fail \
  "venv python not found at ${PYTHON}. Create the environment on a compute node; see docs/BUNYA_SETUP.md"
[[ -f "${MANIFEST}" ]] || fail "manifest not found: ${MANIFEST}"
[[ -f "${INFER_CONFIG}" ]] || fail "infer config not found: ${INFER_CONFIG}"

source "${VENV_DIR}/bin/activate"
cd "${PROJECT_DIR}"

if [[ -d "${WEIGHTS_DIR}" && ! -e "${PROJECT_DIR}/weights" ]]; then
  echo "linking ${WEIGHTS_DIR} -> ${PROJECT_DIR}/weights"
  ln -sfn "${WEIGHTS_DIR}" "${PROJECT_DIR}/weights"
fi
[[ -d "${PROJECT_DIR}/weights" ]] || fail \
  "weights not found. Set X2DFD_WEIGHTS or symlink/copy weights to ${PROJECT_DIR}/weights"

# Vision tower: LLaVA may fetch openai/clip-vit-large-patch14-336 at load time.
# Prefer a staged local copy so jobs do not depend on Hugging Face at runtime.
if [[ ! -d "${VISION_TOWER}" ]]; then
  fail "vision tower directory not found: ${VISION_TOWER} (set X2DFD_VISION_TOWER or stage weights/base/clip-vit-large-patch14-336)"
fi

GIT_COMMIT="$(git -C "${PROJECT_DIR}" rev-parse HEAD 2>/dev/null || echo unknown)"
GIT_DIRTY="$(git -C "${PROJECT_DIR}" status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
echo "git_commit: ${GIT_COMMIT}"
echo "git_dirty_paths: ${GIT_DIRTY}"

echo "=== GPU allocation ==="
nvidia-smi || fail "nvidia-smi failed"

echo "=== Python / torch / CUDA ==="
"${PYTHON}" - <<'PY'
import sys
print("python:", sys.version.replace("\n", " "))
import torch
print("torch:", torch.__version__)
print("torch.cuda.is_available:", torch.cuda.is_available())
print("torch.version.cuda:", torch.version.cuda)
if torch.cuda.is_available():
    print("gpu0:", torch.cuda.get_device_name(0))
PY

echo "=== Environment check (no inference) ==="
"${PYTHON}" -m tools.check_bunya_environment \
  --project-root "${PROJECT_DIR}" \
  --config "${INFER_CONFIG}" \
  --skip-datasets \
  --json \
  --output "${OUTPUT_DIR}/env_check.json" \
  || fail "environment check failed — see ${OUTPUT_DIR}/env_check.json"

echo "=== Manifest preflight (dry-run into sidecar dir) ==="
PREFLIGHT_DIR="${OUTPUT_DIR}/_preflight_dry_run"
"${PYTHON}" -m tools.run_labelled_evaluation \
  --manifest "${MANIFEST}" \
  --output-dir "${PREFLIGHT_DIR}" \
  --config "${INFER_CONFIG}" \
  --project-root "${PROJECT_DIR}" \
  --no-4bit \
  --dry-run \
  || fail "manifest dry-run failed"

echo "=== Labelled evaluation (FP16, resume on) ==="
"${PYTHON}" -m tools.run_labelled_evaluation \
  --manifest "${MANIFEST}" \
  --output-dir "${OUTPUT_DIR}" \
  --config "${INFER_CONFIG}" \
  --project-root "${PROJECT_DIR}" \
  --no-4bit \
  --resume

echo "=== labelled evaluation complete ==="
echo "aggregate: ${OUTPUT_DIR}/aggregate.json"
echo "batch_meta: ${OUTPUT_DIR}/batch_meta.json"
