#!/bin/bash
#SBATCH --job-name=x2dfd-analyse
#SBATCH --account=a_css
#SBATCH --partition=general
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=labelled-analysis-%j.out
#SBATCH --error=labelled-analysis-%j.err
#
# Post-inference analysis for a labelled evaluation output directory.
# No GPU and no model loading — reads aggregate.json and writes metrics/figures.
#
# Required environment:
#   X2DFD_OUTPUT_DIR   Absolute path to the labelled-evaluation output directory
#                      (must contain aggregate.json after a successful inference job)
#
# Optional:
#   X2DFD_PROJECT_DIR  Repo on scratch
#   X2DFD_VENV         Python venv
#   X2DFD_ANALYSIS_DIR Analysis output (default: $X2DFD_OUTPUT_DIR/analysis)
#
# Normally submitted automatically by bunya/submit_labelled_evaluation.sh with
#   --dependency=afterok:<inference_job_id>
#
# If partition "general" is unavailable for your account, override at submit:
#   ANALYSIS_PARTITION=... ANALYSIS_QOS=... bash bunya/submit_labelled_evaluation.sh …

set -euo pipefail

module use /sw/auto/rocky9a/epyc3_a100/modules/all
module load python/3.10.4-gcccore-11.3.0-bare

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ -n "${X2DFD_OUTPUT_DIR:-}" ]] || fail "X2DFD_OUTPUT_DIR is not set"

SCRATCH_BASE="${SCRATCH:-/scratch/user/${USER}}"
PROJECT_DIR="${X2DFD_PROJECT_DIR:-${SCRATCH_BASE}/disagreement-aware-x2dfd}"
VENV_DIR="${X2DFD_VENV:-${PROJECT_DIR}/.venv}"
PYTHON="${VENV_DIR}/bin/python"
OUTPUT_DIR="$(readlink -f "${X2DFD_OUTPUT_DIR}")"
ANALYSIS_DIR="${X2DFD_ANALYSIS_DIR:-${OUTPUT_DIR}/analysis}"
AGGREGATE="${OUTPUT_DIR}/aggregate.json"

echo "=== X2DFD labelled analysis ==="
echo "host: $(hostname)"
echo "date: $(date -Is)"
echo "slurm_job_id: ${SLURM_JOB_ID:-unset}"
echo "project: ${PROJECT_DIR}"
echo "output: ${OUTPUT_DIR}"
echo "aggregate: ${AGGREGATE}"
echo "analysis_dir: ${ANALYSIS_DIR}"

[[ -d "${PROJECT_DIR}" ]] || fail "project directory not found: ${PROJECT_DIR}"
[[ -x "${PYTHON}" ]] || fail "venv python not found at ${PYTHON}"
[[ -f "${AGGREGATE}" ]] || fail "aggregate.json missing (inference may have failed): ${AGGREGATE}"

source "${VENV_DIR}/bin/activate"
cd "${PROJECT_DIR}"

GIT_COMMIT="$(git -C "${PROJECT_DIR}" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "git_commit: ${GIT_COMMIT}"

"${PYTHON}" - <<'PY'
import importlib.util
import sys
missing = [m for m in ("matplotlib",) if importlib.util.find_spec(m) is None]
if missing:
    print(
        "ERROR: missing packages required for analysis figures: "
        + ", ".join(missing)
        + ". Install on a compute node into the project venv, e.g. "
        "python -m pip install matplotlib",
        file=sys.stderr,
    )
    raise SystemExit(1)
print("matplotlib: ok")
PY

mkdir -p "${ANALYSIS_DIR}"
"${PYTHON}" -m tools.analyse_labelled_evaluation \
  --aggregate "${AGGREGATE}" \
  --output-dir "${ANALYSIS_DIR}"

echo "=== analysis complete ==="
echo "summary: ${ANALYSIS_DIR}/analysis_summary.json"
