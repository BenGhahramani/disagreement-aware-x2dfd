#!/usr/bin/env bash
# 小批量 图像注释生成 测试脚本

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$SCRIPT_DIR"

echo "=== 小批量 图像注释生成（测试） ==="
echo "工作目录: $SCRIPT_DIR"

if command -v conda >/dev/null 2>&1; then
  source ~/.bashrc || true
  conda init || true
  echo "激活conda环境: llava"
  eval "$(conda shell.bash hook)"
  source activate llava || conda activate llava || true
  echo "✅ 环境: ${CONDA_DEFAULT_ENV:-system}"
else
  echo "⚠️ 未检测到conda，使用系统Python"
fi

# 使用方式：
#  1) 不带参数：走 feature_annotation + config.yaml 的流程
#  2) 带 dataset.json：读取 {Description, images[].image_path}，拼出绝对路径并进行 quick_infer（输出中 image 为绝对路径）
#  3) dataset.json + out.json：指定 quick_infer 输出文件

INPUT_JSON="${1:-}"
OUTPUT_JSON="${2:-}"

if [[ -z "${INPUT_JSON}" ]]; then
  cd "$REPO_ROOT"
  python eval/tools/feature_annotation.py --config eval/configs/config.yaml
  exit 0
fi

cd "$REPO_ROOT"

# 将 dataset.json 内的相对路径转为绝对路径（Description 作为根）
TMP_DIR="$(mktemp -d -t run1_abs_XXXXXX)"
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

# 走 quick_infer（多卡会自动分片），输出中的 image 字段即为绝对路径
ARGS=( --config "eval/configs/infer_config.yaml" --json "${ABS_JSON}" )
if [[ -n "${OUTPUT_JSON}" ]]; then
  ARGS+=( --output "${OUTPUT_JSON}" )
fi
python eval/infer/runner.py "${ARGS[@]}"

rm -rf "${TMP_DIR}"
