#!/bin/bash
#SBATCH --job-name=x2dfd-ffpp-ana
#SBATCH --account=a_css
#SBATCH --partition=general
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=ffpp-src-analysis-%j.out
#SBATCH --error=ffpp-src-analysis-%j.err
#
# CPU analysis for FF++ c23 source-domain scores.
# Submitted with --dependency=afterok:<inference_job_id>.

set -euo pipefail

module use /sw/auto/rocky9a/epyc3_a100/modules/all
module load python/3.10.4-gcccore-11.3.0-bare

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ -n "${X2DFD_PREP_DIR:-}" ]] || fail "X2DFD_PREP_DIR is not set"
[[ -n "${X2DFD_FFPP_OUTPUT:-}" ]] || fail "X2DFD_FFPP_OUTPUT is not set"

SCRATCH_BASE="${SCRATCH:-/scratch/user/${USER}}"
PROJECT_DIR="${X2DFD_PROJECT_DIR:-${SCRATCH_BASE}/disagreement-aware-x2dfd}"
VENV_DIR="${X2DFD_VENV:-${PROJECT_DIR}/.venv}"
PYTHON="${VENV_DIR}/bin/python"
PREP_DIR="$(readlink -f "${X2DFD_PREP_DIR}")"
OUTPUT_DIR="$(readlink -f "${X2DFD_FFPP_OUTPUT}")"

echo "=== FF++ c23 source-domain analysis ==="
echo "host: $(hostname)"
echo "slurm_job_id: ${SLURM_JOB_ID:-unset}"
echo "prep: ${PREP_DIR}"
echo "output: ${OUTPUT_DIR}"

[[ -x "${PYTHON}" ]] || fail "venv python missing"
[[ -f "${OUTPUT_DIR}/frame_scores.jsonl" ]] || fail "frame_scores.jsonl missing"

source "${VENV_DIR}/bin/activate"
cd "${PROJECT_DIR}"

"${PYTHON}" -m tools.analyse_ffpp_source_domain \
  --prep-dir "${PREP_DIR}" \
  --output-dir "${OUTPUT_DIR}"

echo "=== analysis complete ==="
echo "summary: ${OUTPUT_DIR}/analysis/summary.json"
