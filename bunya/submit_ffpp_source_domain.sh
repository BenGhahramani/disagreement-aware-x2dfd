#!/bin/bash
# Submit FF++ c23 source-domain consistency inference (+ optional analysis).
#
# Usage (login node):
#   bash bunya/submit_ffpp_source_domain.sh \
#     /scratch/user/s4749229/datasets/ffpp_c23_source \
#     /scratch/user/s4749229/eval_outputs/x2dfd_ffpp_source_domain
#
# Optional:
#   TIME_LIMIT=16:00:00
#   SKIP_ANALYSIS=1
#   ANALYSIS_PARTITION=general

set -euo pipefail

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage:
  bash bunya/submit_ffpp_source_domain.sh <prep_dir> <output_dir>
EOF
}

[[ $# -eq 2 ]] || { usage >&2; exit 2; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
EVAL_SLURM="${SCRIPT_DIR}/run_ffpp_source_domain.slurm"
ANALYSIS_SLURM="${SCRIPT_DIR}/run_ffpp_source_domain_analysis.slurm"

PREP_IN="$1"
OUT_IN="$2"
[[ -d "${PREP_IN}" ]] || fail "prep dir not found: ${PREP_IN}"
[[ -f "${PREP_IN}/preparation_manifest.json" ]] || fail "missing preparation_manifest.json"

PREP_DIR="$(readlink -f "${PREP_IN}")"
mkdir -p "${OUT_IN}"
OUTPUT_DIR="$(readlink -f "${OUT_IN}")"
LOG_DIR="${OUTPUT_DIR}/slurm_logs"
mkdir -p "${LOG_DIR}"

TIME_LIMIT="${TIME_LIMIT:-12:00:00}"
SCRATCH_BASE="${SCRATCH:-/scratch/user/${USER}}"
export X2DFD_PROJECT_DIR="${X2DFD_PROJECT_DIR:-${SCRATCH_BASE}/disagreement-aware-x2dfd}"
export X2DFD_WEIGHTS="${X2DFD_WEIGHTS:-${X2DFD_PROJECT_DIR}/weights}"
export X2DFD_VENV="${X2DFD_VENV:-${X2DFD_PROJECT_DIR}/.venv}"
export X2DFD_PREP_DIR="${PREP_DIR}"
export X2DFD_FFPP_OUTPUT="${OUTPUT_DIR}"
export X2DFD_VISION_TOWER="${X2DFD_VISION_TOWER:-${X2DFD_PROJECT_DIR}/weights/base/clip-vit-large-patch14-336}"

echo "=== submit FF++ c23 source-domain sanity check ==="
echo "prep:    ${PREP_DIR}"
echo "output:  ${OUTPUT_DIR}"
echo "logs:    ${LOG_DIR}"
echo "time:    ${TIME_LIMIT}"

cd "${REPO_DIR}"

INF_EXPORT="ALL,X2DFD_PREP_DIR,X2DFD_FFPP_OUTPUT,X2DFD_PROJECT_DIR,X2DFD_WEIGHTS,X2DFD_VENV,X2DFD_VISION_TOWER"
INF_JOB="$(
  sbatch --parsable \
    --time="${TIME_LIMIT}" \
    --output="${LOG_DIR}/ffpp-src-%j.out" \
    --error="${LOG_DIR}/ffpp-src-%j.err" \
    --export="${INF_EXPORT}" \
    "${EVAL_SLURM}"
)"
[[ -n "${INF_JOB}" ]] || fail "sbatch did not return a job id"
echo "inference_job_id: ${INF_JOB}"
echo "tail: tail -f ${LOG_DIR}/ffpp-src-${INF_JOB}.out"

if [[ "${SKIP_ANALYSIS:-0}" != "1" ]]; then
  ANALYSIS_PARTITION="${ANALYSIS_PARTITION:-general}"
  ANA_EXPORT="ALL,X2DFD_PREP_DIR,X2DFD_FFPP_OUTPUT,X2DFD_PROJECT_DIR,X2DFD_VENV"
  ANA_JOB="$(
    sbatch --parsable \
      --dependency="afterok:${INF_JOB}" \
      --account=a_css \
      --partition="${ANALYSIS_PARTITION}" \
      --output="${LOG_DIR}/ffpp-src-analysis-%j.out" \
      --error="${LOG_DIR}/ffpp-src-analysis-%j.err" \
      --export="${ANA_EXPORT}" \
      "${ANALYSIS_SLURM}"
  )"
  echo "analysis_job_id: ${ANA_JOB} (afterok:${INF_JOB})"
fi

echo "submitted — returning immediately"
