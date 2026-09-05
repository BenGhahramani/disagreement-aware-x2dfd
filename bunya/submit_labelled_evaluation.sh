#!/bin/bash
# Submit a labelled FP16 evaluation job (and dependent analysis) on UQ Bunya.
#
# Usage (from the repo on scratch, login node — scheduling only):
#   bash bunya/submit_labelled_evaluation.sh \
#     /scratch/user/s4749229/datasets/deepfakeface_final/deepfakeface_manifest.json \
#     /scratch/user/s4749229/eval_outputs/deepfakeface_fp16
#
# Optional environment overrides:
#   TIME_LIMIT=16:00:00          Slurm wall time for the GPU inference job
#   X2DFD_PROJECT_DIR=...        Repo path on scratch
#   X2DFD_WEIGHTS=...            Weights root
#   X2DFD_VENV=...               Python venv
#   X2DFD_VISION_TOWER=...       CLIP vision-tower directory
#   SKIP_ANALYSIS=1              Do not queue the afterok analysis job
#   ANALYSIS_PARTITION=general   CPU partition for analysis
#   ANALYSIS_QOS=                Optional QoS for analysis (omit if unused)
#   ANALYSIS_ACCOUNT=a_css       Account for analysis job
#
# Returns immediately after Slurm accepts the job(s). Does not wait for GPUs.

set -euo pipefail

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage:
  bash bunya/submit_labelled_evaluation.sh <manifest.json> <output_dir>

Example:
  bash bunya/submit_labelled_evaluation.sh \
    /scratch/user/s4749229/datasets/deepfakeface_final/deepfakeface_manifest.json \
    /scratch/user/s4749229/eval_outputs/deepfakeface_fp16
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

[[ $# -eq 2 ]] || { usage >&2; exit 2; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
EVAL_SLURM="${SCRIPT_DIR}/run_labelled_evaluation.slurm"
ANALYSIS_SLURM="${SCRIPT_DIR}/run_labelled_analysis.slurm"

[[ -f "${EVAL_SLURM}" ]] || fail "missing ${EVAL_SLURM}"
[[ -f "${ANALYSIS_SLURM}" ]] || fail "missing ${ANALYSIS_SLURM}"

MANIFEST_IN="$1"
OUTPUT_IN="$2"

[[ -e "${MANIFEST_IN}" ]] || fail "manifest not found: ${MANIFEST_IN}"
MANIFEST="$(readlink -f "${MANIFEST_IN}")"
[[ -f "${MANIFEST}" ]] || fail "manifest is not a file: ${MANIFEST}"

mkdir -p "${OUTPUT_IN}"
OUTPUT_DIR="$(readlink -f "${OUTPUT_IN}")"
LOG_DIR="${OUTPUT_DIR}/slurm_logs"
mkdir -p "${LOG_DIR}"

TIME_LIMIT="${TIME_LIMIT:-12:00:00}"
SCRATCH_BASE="${SCRATCH:-/scratch/user/${USER}}"
export X2DFD_PROJECT_DIR="${X2DFD_PROJECT_DIR:-${SCRATCH_BASE}/disagreement-aware-x2dfd}"
export X2DFD_WEIGHTS="${X2DFD_WEIGHTS:-${X2DFD_PROJECT_DIR}/weights}"
export X2DFD_VENV="${X2DFD_VENV:-${X2DFD_PROJECT_DIR}/.venv}"
export X2DFD_MANIFEST="${MANIFEST}"
export X2DFD_OUTPUT_DIR="${OUTPUT_DIR}"
export X2DFD_VISION_TOWER="${X2DFD_VISION_TOWER:-${X2DFD_PROJECT_DIR}/weights/base/clip-vit-large-patch14-336}"

echo "=== submit labelled evaluation ==="
echo "repo (cwd for relative scripts): ${REPO_DIR}"
echo "manifest: ${MANIFEST}"
echo "output:   ${OUTPUT_DIR}"
echo "logs:     ${LOG_DIR}"
echo "time:     ${TIME_LIMIT}"
echo "project:  ${X2DFD_PROJECT_DIR}"
echo "weights:  ${X2DFD_WEIGHTS}"
echo "venv:     ${X2DFD_VENV}"

cd "${REPO_DIR}"

INF_EXPORT="ALL,X2DFD_MANIFEST,X2DFD_OUTPUT_DIR,X2DFD_PROJECT_DIR,X2DFD_WEIGHTS,X2DFD_VENV,X2DFD_VISION_TOWER"
if [[ -n "${X2DFD_INFER_CONFIG:-}" ]]; then
  INF_EXPORT="${INF_EXPORT},X2DFD_INFER_CONFIG"
fi

INF_JOB="$(
  sbatch --parsable \
    --time="${TIME_LIMIT}" \
    --output="${LOG_DIR}/labelled-eval-%j.out" \
    --error="${LOG_DIR}/labelled-eval-%j.err" \
    --export="${INF_EXPORT}" \
    "${EVAL_SLURM}"
)"
[[ -n "${INF_JOB}" ]] || fail "sbatch did not return an inference job id"

echo "inference_job_id: ${INF_JOB}"
echo "squeue:  squeue -j ${INF_JOB}"
echo "sacct:   sacct -j ${INF_JOB} --format=JobID,State,Elapsed,ExitCode,MaxRSS -P"
echo "tail:    tail -f ${LOG_DIR}/labelled-eval-${INF_JOB}.out"

ANALYSIS_JOB=""
if [[ "${SKIP_ANALYSIS:-0}" != "1" ]]; then
  ANALYSIS_PARTITION="${ANALYSIS_PARTITION:-general}"
  ANALYSIS_ACCOUNT="${ANALYSIS_ACCOUNT:-a_css}"
  ANA_EXPORT="ALL,X2DFD_OUTPUT_DIR,X2DFD_PROJECT_DIR,X2DFD_VENV"
  if [[ -n "${X2DFD_ANALYSIS_DIR:-}" ]]; then
    ANA_EXPORT="${ANA_EXPORT},X2DFD_ANALYSIS_DIR"
  fi
  ANALYSIS_ARGS=(
    --parsable
    --dependency="afterok:${INF_JOB}"
    --account="${ANALYSIS_ACCOUNT}"
    --partition="${ANALYSIS_PARTITION}"
    --output="${LOG_DIR}/labelled-analysis-%j.out"
    --error="${LOG_DIR}/labelled-analysis-%j.err"
    --export="${ANA_EXPORT}"
  )
  if [[ -n "${ANALYSIS_QOS:-}" ]]; then
    ANALYSIS_ARGS+=(--qos="${ANALYSIS_QOS}")
  fi
  ANALYSIS_JOB="$(sbatch "${ANALYSIS_ARGS[@]}" "${ANALYSIS_SLURM}")" \
    || fail "failed to queue analysis job (inference ${INF_JOB} was still accepted)"
  echo "analysis_job_id: ${ANALYSIS_JOB} (afterok:${INF_JOB})"
  echo "analysis_tail: tail -f ${LOG_DIR}/labelled-analysis-${ANALYSIS_JOB}.out"
else
  echo "analysis: skipped (SKIP_ANALYSIS=1)"
fi

echo "submitted — returning immediately (Slurm will run when resources are free)"
