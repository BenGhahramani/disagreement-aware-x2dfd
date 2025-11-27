#!/usr/bin/env bash
# Configure a new remote and push code (mirror or normal)
# Usage:
#   bash tools/push_to_new_repo.sh <remote-url> [--mirror]
# Example:
#   bash tools/push_to_new_repo.sh https://github.com/chenyize111/X2DFD2.git --mirror

set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <remote-url> [--mirror]" >&2; exit 2
fi
REMOTE_URL="$1"; shift || true
MODE="normal"
if [[ "${1:-}" == "--mirror" ]]; then MODE="mirror"; fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

REMOTE_NAME="x2dfd2"
if git remote get-url "$REMOTE_NAME" >/dev/null 2>&1; then
  git remote set-url "$REMOTE_NAME" "$REMOTE_URL"
else
  git remote add "$REMOTE_NAME" "$REMOTE_URL"
fi

echo "[info] Remote $REMOTE_NAME => $(git remote get-url "$REMOTE_NAME")"

# Ensure main/master exist and point to HEAD (helps new repos)
for BR in main master; do
  if git show-ref --verify --quiet "refs/heads/$BR"; then
    git branch -f "$BR" >/dev/null
  else
    git branch "$BR" >/dev/null || true
    git branch -f "$BR" >/dev/null
  fi
done

if [[ "$MODE" == "mirror" ]]; then
  echo "[plan] git push --force --mirror $REMOTE_NAME"
  read -r -p "Type 'MIRROR' to proceed: " ACK
  [[ "$ACK" == "MIRROR" ]] || { echo "Aborted."; exit 3; }
  git push --force --mirror "$REMOTE_NAME"
else
  echo "[plan] push current branch and main/master"
  CUR=$(git rev-parse --abbrev-ref HEAD)
  git push -u "$REMOTE_NAME" "$CUR":main
  git push -u "$REMOTE_NAME" "$CUR":master || true
  echo "[done] Pushed $CUR to main/master"
fi
