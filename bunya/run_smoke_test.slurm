#!/bin/bash
#SBATCH --job-name=x2dfd-smoke
#SBATCH --account=AccountString
#SBATCH --partition=gpu_cuda
#SBATCH --qos=debug
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# Minimal one-image X2DFD smoke test for UQ Bunya.
#
# Resource profile: short initial smoke test only (--qos=debug, 30 min, L40S).
# Production evaluation jobs should use --qos=gpu and a longer --time; GPU type
# may be changed to a100 or h100 depending on memory and experiment needs.
# See docs/BUNYA_SETUP.md.
#
# BEFORE SUBMITTING:
#   1. Replace AccountString with your a_* accounting group from `groups`.
#   2. Stage the repo and weights under /scratch (see docs/BUNYA_SETUP.md).
#   3. Create the Python venv inside an interactive GPU compute job.
#   4. Copy datasets/raw/images/poc/real_face_01.jpg into the clone (not tracked).
#
# Submit from a login node (scheduling only — compute runs on a GPU node):
#   sbatch bunya/run_smoke_test.slurm
#
# Do not submit from /QRISdata and do not write job outputs there.

set -euo pipefail

echo "=== X2DFD Bunya smoke test ==="
echo "host: $(hostname)"
echo "date: $(date -Is)"
echo "slurm_job_id: ${SLURM_JOB_ID:-unset}"

# ---------------------------------------------------------------------------
# User-editable paths (keep active work on scratch)
# ---------------------------------------------------------------------------
SCRATCH_BASE="${SCRATCH:-/scratch/${USER}}"
PROJECT_DIR="${X2DFD_PROJECT_DIR:-${SCRATCH_BASE}/disagreement-aware-x2dfd}"
WEIGHTS_DIR="${X2DFD_WEIGHTS:-${SCRATCH_BASE}/x2dfd/weights}"
OUTPUT_DIR="${X2DFD_OUTPUT:-${SCRATCH_BASE}/x2dfd/runs/smoke/${SLURM_JOB_ID:-manual}}"
VENV_DIR="${X2DFD_VENV:-${PROJECT_DIR}/.venv}"
PYTHON="${VENV_DIR}/bin/python"

export X2DFD_PROJECT_ROOT="${PROJECT_DIR}"
export X2DFD_WEIGHTS="${WEIGHTS_DIR}"
export X2DFD_OUTPUT="${OUTPUT_DIR}"
export X2DFD_DATASETS="${PROJECT_DIR}/datasets"

echo "project: ${PROJECT_DIR}"
echo "weights: ${WEIGHTS_DIR}"
echo "output:  ${OUTPUT_DIR}"
echo "venv:    ${VENV_DIR}"

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ -d "${PROJECT_DIR}" ]] || fail "project directory not found: ${PROJECT_DIR}"
[[ -x "${PYTHON}" ]] || fail \
  "venv python not found at ${PYTHON}. Create the environment on a compute node; see docs/BUNYA_SETUP.md"

cd "${PROJECT_DIR}"
mkdir -p "${OUTPUT_DIR}"

# Optional: weights live outside the clone (recommended on scratch).
if [[ -d "${WEIGHTS_DIR}" && ! -e "${PROJECT_DIR}/weights" ]]; then
  echo "linking ${WEIGHTS_DIR} -> ${PROJECT_DIR}/weights"
  ln -sfn "${WEIGHTS_DIR}" "${PROJECT_DIR}/weights"
fi
[[ -d "${PROJECT_DIR}/weights" ]] || fail \
  "weights not found. Set WEIGHTS_DIR or symlink/copy weights to ${PROJECT_DIR}/weights"

# Preflight: CUDA, packages, and weight paths (no inference yet).
"${PYTHON}" -m tools.check_bunya_environment \
  --project-root "${PROJECT_DIR}" \
  --require-bitsandbytes \
  --skip-datasets \
  --json \
  --output "${OUTPUT_DIR}/env_check.json" \
  || fail "environment check failed — see ${OUTPUT_DIR}/env_check.json"

# Linux-safe manifest for the known POC image (avoid Windows paths in demo_one.json).
IMAGE_ROOT="${PROJECT_DIR}/datasets/raw/images/poc"
SMOKE_IMAGE="${IMAGE_ROOT}/real_face_01.jpg"
MANIFEST="${OUTPUT_DIR}/demo_one_manifest.json"

[[ -f "${SMOKE_IMAGE}" ]] || fail \
  "smoke-test image missing: ${SMOKE_IMAGE}. Copy it from local datasets/raw/images/poc/"

cat > "${MANIFEST}" <<EOF
{
  "Description": "${IMAGE_ROOT}",
  "images": [
    {"image_path": "real_face_01.jpg"}
  ]
}
EOF

echo "manifest: ${MANIFEST}"
echo "running single-image smoke test (experts=none, 4-bit)..."

"${PYTHON}" -m tools.run_smoke_test \
  --manifest "${MANIFEST}" \
  --config eval/configs/infer_config.bunya.yaml \
  --output "${OUTPUT_DIR}/demo_one_result.json" \
  --summary "${OUTPUT_DIR}/smoke_test_summary.json" \
  --experts none \
  --load-4bit \
  --project-root "${PROJECT_DIR}" \
  --timeout 3600

echo "=== smoke test complete ==="
echo "summary: ${OUTPUT_DIR}/smoke_test_summary.json"
echo "result:  ${OUTPUT_DIR}/demo_one_result.json"
