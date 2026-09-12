#!/bin/bash
#SBATCH --job-name=x2dfd-ffpp-src
#SBATCH --account=a_css
#SBATCH --partition=gpu_cuda
#SBATCH --qos=gpu
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=ffpp-src-%j.out
#SBATCH --error=ffpp-src-%j.err
#
# FF++ c23 source-domain sanity check (FP16, blending+diffusion).
# Default prep is ~100 videos × 8 frames (~800 frames). Separate from the
# thesis labelled evaluation (360-image) outputs.
#
# Prefer:
#   bash bunya/submit_ffpp_source_domain.sh
#
# Required env:
#   X2DFD_PREP_DIR      Prepared frames/crops + preparation_manifest.json
#   X2DFD_FFPP_OUTPUT   Scratch output directory for scores/analysis

set -euo pipefail

module use /sw/auto/rocky9a/epyc3_a100/modules/all
module load python/3.10.4-gcccore-11.3.0-bare
module load cuda/12.1.1

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ -n "${X2DFD_PREP_DIR:-}" ]] || fail "X2DFD_PREP_DIR is not set"
[[ -n "${X2DFD_FFPP_OUTPUT:-}" ]] || fail "X2DFD_FFPP_OUTPUT is not set"

SCRATCH_BASE="${SCRATCH:-/scratch/user/${USER}}"
PROJECT_DIR="${X2DFD_PROJECT_DIR:-${SCRATCH_BASE}/disagreement-aware-x2dfd}"
WEIGHTS_DIR="${X2DFD_WEIGHTS:-${PROJECT_DIR}/weights}"
VENV_DIR="${X2DFD_VENV:-${PROJECT_DIR}/.venv}"
PYTHON="${VENV_DIR}/bin/python"
PREP_DIR="$(readlink -f "${X2DFD_PREP_DIR}")"
OUTPUT_DIR="$(mkdir -p "${X2DFD_FFPP_OUTPUT}" && readlink -f "${X2DFD_FFPP_OUTPUT}")"
CONFIG="${PROJECT_DIR}/eval/configs/infer_config.bunya.yaml"
VISION_TOWER="${X2DFD_VISION_TOWER:-${PROJECT_DIR}/weights/base/clip-vit-large-patch14-336}"

export X2DFD_PROJECT_ROOT="${PROJECT_DIR}"
export X2DFD_WEIGHTS="${WEIGHTS_DIR}"
export X2DFD_OUTPUT="${OUTPUT_DIR}"
unset X2DFD_LOAD_4BIT || true

echo "=== X2-DFD FF++ c23 source-domain (FP16, blending+diffusion) ==="
echo "host: $(hostname)"
echo "date: $(date -Is)"
echo "slurm_job_id: ${SLURM_JOB_ID:-unset}"
echo "project: ${PROJECT_DIR}"
echo "prep: ${PREP_DIR}"
echo "output: ${OUTPUT_DIR}"
echo "git: $(git -C "${PROJECT_DIR}" rev-parse HEAD 2>/dev/null || echo unknown)"

[[ -d "${PROJECT_DIR}" ]] || fail "project missing: ${PROJECT_DIR}"
[[ -x "${PYTHON}" ]] || fail "venv python missing: ${PYTHON}"
[[ -f "${PREP_DIR}/preparation_manifest.json" ]] || fail "prep manifest missing"
[[ -f "${PREP_DIR}/infer_input.json" ]] || fail "infer_input.json missing"
[[ -d "${VISION_TOWER}" ]] || fail "vision tower missing: ${VISION_TOWER}"

source "${VENV_DIR}/bin/activate"
cd "${PROJECT_DIR}"

if [[ -d "${WEIGHTS_DIR}" && ! -e "${PROJECT_DIR}/weights" ]]; then
  ln -sfn "${WEIGHTS_DIR}" "${PROJECT_DIR}/weights"
fi
[[ -d "${PROJECT_DIR}/weights" ]] || fail "weights missing"

nvidia-smi || fail "nvidia-smi failed"

"${PYTHON}" -m tools.check_bunya_environment \
  --project-root "${PROJECT_DIR}" \
  --config "${CONFIG}" \
  --skip-datasets \
  --json \
  --output "${OUTPUT_DIR}/env_check.json" \
  || fail "environment check failed"

"${PYTHON}" -m tools.run_ffpp_source_domain \
  --prep-dir "${PREP_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --project-root "${PROJECT_DIR}" \
  --config "${CONFIG}" \
  --experts blending,diffusion_detector \
  --chunk-size 64 \
  --resume

echo "=== FF++ source-domain inference complete ==="
echo "scores: ${OUTPUT_DIR}/frame_scores.jsonl"
