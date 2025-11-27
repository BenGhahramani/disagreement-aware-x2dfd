#!/usr/bin/env bash
# Step 1: Model Feature Assessment (placeholder)
# Records basic environment/model metadata for future feature assessment.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}" )/.." && pwd)"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/results/steps/01_feature_assess}"
BASE_MODEL="${BASE_MODEL:-weights/base/llava-v1.5-7b}"

mkdir -p "$OUT_ROOT"
INFO_JSON="${OUT_ROOT}/info.json"

python - "$OUT_ROOT" "$BASE_MODEL" << 'PY'
import json, os, sys, time, platform, subprocess
out_dir, base_model = sys.argv[1], sys.argv[2]

def _git(cmd):
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None

meta = {
    "timestamp": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    "base_model": base_model,
    "hostname": platform.node(),
    "python": platform.python_version(),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "git": {
        "commit": _git(["git","rev-parse","--short","HEAD"]) ,
        "branch": _git(["git","rev-parse","--abbrev-ref","HEAD"]) ,
    },
}
with open(os.path.join(out_dir, 'info.json'), 'w', encoding='utf-8') as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
print("[step1] wrote:", os.path.join(out_dir, 'info.json'))
PY

echo "[step1] done."

