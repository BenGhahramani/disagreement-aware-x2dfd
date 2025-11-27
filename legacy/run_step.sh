#!/usr/bin/env bash
# Quick runner for stepwise pipeline
# Usage:
#   bash run_step.sh <step> [args...]
# Steps:
#   1 | feature | assess          -> steps/01_feature_assess.sh
#   2 | annotate | annotation     -> steps/02_annotation.sh
#   3 | weak | merge              -> steps/03_weak.sh
#   4 | train | deepspeed         -> steps/04_train.sh
#   5a | prompts | eval-prompts   -> steps/05a_build_prompts.py
#   5b | infer | eval-infer       -> steps/05b_infer.py
#   5 | eval                      -> run 5a then 5b
#   all                           -> run 1 -> 2 -> 3 -> 4 -> 5a -> 5b

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"
cd "$REPO_ROOT"

STEP="${1:-}"
if [[ -z "$STEP" ]] || [[ "$STEP" == "-h" ]] || [[ "$STEP" == "--help" ]]; then
  sed -n '2,999p' "$0" | grep -E '^#' | sed 's/^# \{0,1\}//'
  exit 0
fi
shift || true

case "$STEP" in
  1|feature|assess)
    exec bash steps/01_feature_assess.sh "$@" ;;
  2|annotate|annotation)
    exec bash steps/02_annotation.sh "$@" ;;
  3|weak|merge)
    exec bash steps/03_weak.sh "$@" ;;
  4|train|deepspeed)
    exec bash steps/04_train.sh "$@" ;;
  5a|prompts|eval-prompts)
    exec python steps/05a_build_prompts.py "$@" ;;
  5b|infer|eval-infer)
    exec python steps/05b_infer.py "$@" ;;
  5|eval)
    python steps/05a_build_prompts.py "$@" && python steps/05b_infer.py "$@" ;;
  all)
    bash steps/01_feature_assess.sh "$@"
    bash steps/02_annotation.sh "$@"
    bash steps/03_weak.sh "$@"
    bash steps/04_train.sh "$@"
    python steps/05a_build_prompts.py "$@"
    python steps/05b_infer.py "$@"
    ;;
  *)
    echo "Unknown step: $STEP" >&2
    exit 2 ;;
esac

