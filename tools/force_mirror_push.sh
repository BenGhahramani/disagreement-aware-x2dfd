#!/usr/bin/env bash
# Force mirror-push local repository to the remote (DESTROYS remote-only refs).
# Usage:
#   bash tools/force_mirror_push.sh [remote]
# Notes:
#   - This will overwrite the entire remote refs (branches & tags) to match local.
#   - Remote-only branches/tags will be deleted. Issues/PRs remain but may desync.
#   - Disable branch protection (e.g., main/master) temporarily on GitHub before running.

set -euo pipefail
REMOTE="${1:-origin}"

# Make repo safe in case of containerized ownership
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}" )/.." && pwd)"
cd "$REPO_ROOT"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "[ERR] Not a git repo: $REPO_ROOT" >&2
  exit 2
fi

# Optional identity
if ! git config --get user.name >/dev/null; then
  git config user.name "X2DFD Dev"
fi
if ! git config --get user.email >/dev/null; then
  git config user.email "x2dfd@users.noreply.github.com"
fi

# Ensure default branch name exists locally to avoid GitHub default-branch issues
# We create/refresh both main and master to current HEAD.
if git show-ref --verify --quiet refs/heads/main; then
  git branch -f main >/dev/null
else
  git branch main >/dev/null || true
  git branch -f main >/dev/null
fi
if git show-ref --verify --quiet refs/heads/master; then
  git branch -f master >/dev/null
else
  git branch master >/dev/null || true
  git branch -f master >/dev/null
fi

# Show summary and ask for confirmation
set +e
CURRENT="$(git rev-parse --short HEAD 2>/dev/null)"
set -e
cat <<MSG
[PLAN]
- Repo:     $REPO_ROOT
- Remote:   $REMOTE ($(git remote get-url "$REMOTE" 2>/dev/null || echo UNKNOWN))
- Commit:   $CURRENT
- Action:   git push --force --mirror $REMOTE
- WARNING:  Remote-only branches/tags WILL be deleted.
MSG

read -r -p "Type 'OVERRIDE' to proceed: " ACK
if [[ "$ACK" != "OVERRIDE" ]]; then
  echo "Aborted."; exit 3
fi

echo "[PUSH] Executing mirror push..."
git push --force --mirror "$REMOTE"
echo "[DONE] Mirror push completed. Re-enable branch protection on GitHub."
