#!/usr/bin/env bash
# Session resume context — printed by the SessionStart hook (.claude/settings.json) so a
# fresh or post-compaction session auto-loads the current state. Read-only, no secrets.
set -euo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"

echo "==================== QuantEdge — session resume ===================="
echo "branch: $(git branch --show-current 2>/dev/null || echo '?')"
echo
echo "----- CONTINUITY.md -----"
[ -f CONTINUITY.md ] && cat CONTINUITY.md || echo "(no CONTINUITY.md)"
echo
echo "----- Open improvements (unchecked) -----"
if [ -f IMPROVEMENTS.md ]; then
  grep -nE '^\s*-\s*\[ \]' IMPROVEMENTS.md || echo "(none unchecked)"
else
  echo "(no IMPROVEMENTS.md)"
fi
echo
echo "----- Recent commits -----"
git log --oneline -10 2>/dev/null || true
echo "===================================================================="
