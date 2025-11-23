#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./quick_infer.sh                    # use config.infer.inputs
#   ./quick_infer.sh input.json         # use a dataset JSON; absolutize image_path using Description
#   ./quick_infer.sh input.json out.json

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PY="${REPO_ROOT}/eval/infer/runner.py"
CFG="${REPO_ROOT}/eval/configs/infer_config.yaml"

INPUT_JSON="${1:-}"
OUTPUT_JSON="${2:-}"

if [[ -z "${INPUT_JSON}" ]];
then
  # No explicit input JSON: fall back to config-driven multi-input run
  python "${PY}" --config "${CFG}"
  exit 0
fi

# Normalize to absolute paths inside the dataset JSON
TMP_DIR="$(mktemp -d -t qi_abs_XXXXXX)"
ABS_JSON="${TMP_DIR}/abs_input.json"

python - "$INPUT_JSON" "$ABS_JSON" <<'PYCODE'
import json, os, sys
src = sys.argv[1]
dst = sys.argv[2]
with open(src, 'r', encoding='utf-8') as f:
    payload = json.load(f)
root = ''
if isinstance(payload, dict):
    root = payload.get('Description') or payload.get('description') or ''
    imgs = payload.get('images') or []
    out_imgs = []
    for it in imgs:
        if not isinstance(it, dict):
            continue
        p = it.get('image_path') or it.get('path')
        if not isinstance(p, str) or not p:
            continue
        abs_p = os.path.normpath(p if os.path.isabs(p) else os.path.join(root, p))
        out = dict(it)
        out['image_path'] = abs_p
        out_imgs.append(out)
    payload['images'] = out_imgs
with open(dst, 'w', encoding='utf-8') as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)
print(dst)
PYCODE

ARGS=( --config "${CFG}" --json "${ABS_JSON}" )

if [[ -n "${OUTPUT_JSON}" ]]; then
  ARGS+=( --output "${OUTPUT_JSON}" )
fi

python "${PY}" "${ARGS[@]}"

# Cleanup temp dir
rm -rf "${TMP_DIR}"
