#!/usr/bin/env python3
"""Improvements Worker — the queue works ITSELF, on a schedule.

Closes the loop the user asked for ("complete all work in loop"): the scouts
and sessions keep APPENDING to IMPROVEMENTS.md; this worker keeps DRAINING it
with no human/session involved:

  IMPROVEMENTS.md `- [ ]` item
    → GitHub issue labelled `agent-fix-needed`  (deduped by title)
      → Free-Agent Engineer picks it up (label trigger + 4h sweep)
        → PR via the reward gate → merged when CI is green

Blocked items (user unlocks, missing data feeds) are recognized and skipped —
filing unactionable issues would just poison the engineer's queue. Top
MAX_ISSUES_PER_RUN actionable items are filed per run; a digest goes to
Discord #squad-backend.
"""
from __future__ import annotations

import json
import os
import re
import sys
import urllib.request
from pathlib import Path

SCRIPTS = Path(__file__).parent
REPO_ROOT = SCRIPTS.parent.parent
sys.path.insert(0, str(SCRIPTS))

IMPROVEMENTS = REPO_ROOT / "IMPROVEMENTS.md"
GH_TOKEN = os.environ.get("GH_TOKEN", "")
GH_REPO = os.environ.get("GH_REPO", "")
AGENT_FIX_LABEL = "agent-fix-needed"
MAX_ISSUES_PER_RUN = 2

# An item containing any of these is waiting on a human/key/feed — not
# actionable by the engineer. Keep in sync with reality, not hope.
_BLOCKED_MARKERS = (
    "blocked", "oa_session_cookie", "polymarket_private_key", "workflow_pat",
    "needs a feed", "needs a data", "needs fundamentals", "geo-block",
    "needs intraday", "needs l2", "universe mismatch", "share inventory",
    "user unlock", "pending a data source",
)


def parse_open_items(md_text: str) -> list[str]:
    """All unchecked `- [ ]` items, multi-line bodies folded to one line."""
    items: list[str] = []
    current: list[str] | None = None
    for line in md_text.splitlines():
        if line.startswith("- [ ]"):
            if current:
                items.append(" ".join(current))
            current = [line[5:].strip()]
        elif current is not None and line.startswith("  ") and line.strip():
            current.append(line.strip())
        else:
            if current:
                items.append(" ".join(current))
            current = None
    if current:
        items.append(" ".join(current))
    return items


def is_blocked(item: str) -> bool:
    low = item.lower()
    return any(m in low for m in _BLOCKED_MARKERS)


def issue_title(item: str) -> str:
    """Stable, deduplicable title from an item's bold heading or first words."""
    m = re.search(r"\*\*(.+?)\*\*", item)
    core = m.group(1) if m else item
    core = re.sub(r"\s+", " ", core).strip().rstrip(":—- ")
    return f"improvement: {core}"[:120]


def _gh(method: str, path: str, body: dict | None = None) -> dict | list:
    req = urllib.request.Request(
        f"https://api.github.com{path}",
        data=json.dumps(body).encode() if body is not None else None,
        method=method,
        headers={
            "Authorization": f"Bearer {GH_TOKEN}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "QuantEdge-ImprovementsWorker/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read())


def existing_issue_titles() -> set[str]:
    """Open + recently-closed issue titles with our label (dedup set)."""
    titles: set[str] = set()
    for state in ("open", "closed"):
        try:
            issues = _gh("GET", f"/repos/{GH_REPO}/issues?labels={AGENT_FIX_LABEL}"
                                f"&state={state}&per_page=100")
            titles |= {str(i.get("title", "")) for i in issues if isinstance(i, dict)}
        except Exception as exc:  # noqa: BLE001 — dedup degrades, worker continues
            print(f"  ⚠ issue listing ({state}) failed: {str(exc)[:80]}", flush=True)
    return titles


def main() -> int:
    if not GH_TOKEN or not GH_REPO:
        print("Improvements Worker: GH_TOKEN/GH_REPO absent — skipping honestly.")
        return 0
    if not IMPROVEMENTS.exists():
        print("Improvements Worker: no IMPROVEMENTS.md found.")
        return 0

    items = parse_open_items(IMPROVEMENTS.read_text())
    actionable = [i for i in items if not is_blocked(i)]
    blocked_n = len(items) - len(actionable)
    print(f"queue: {len(items)} open, {len(actionable)} actionable, {blocked_n} blocked")

    known = existing_issue_titles()
    filed: list[str] = []
    for item in actionable:
        if len(filed) >= MAX_ISSUES_PER_RUN:
            break
        title = issue_title(item)
        if title in known:
            continue
        body = (
            f"From the standing improvement queue (IMPROVEMENTS.md):\n\n> {item}\n\n"
            f"Definition of done: implement with tests, tick the item in "
            f"IMPROVEMENTS.md in the same PR, keep TRADING_MODE=paper, never touch "
            f"base.py / risk/manager.py / main.py.\n\n"
            f"_Filed by the Improvements Worker; label `{AGENT_FIX_LABEL}` triggers "
            f"the Free-Agent Engineer._"
        )
        try:
            out = _gh("POST", f"/repos/{GH_REPO}/issues",
                      {"title": title, "body": body, "labels": [AGENT_FIX_LABEL]})
            num = out.get("number") if isinstance(out, dict) else "?"
            filed.append(f"#{num} {title}")
            print(f"  ✓ filed #{num}: {title}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  ✗ filing failed for {title!r}: {str(exc)[:100]}", flush=True)

    if not filed:
        print("nothing new to file (all actionable items already have issues)")

    try:
        import notify
        notify.post("#squad-backend",
                    f"🛠️ Improvements Worker: {len(items)} queue items "
                    f"({len(actionable)} actionable, {blocked_n} blocked). "
                    + (f"Filed: {', '.join(filed)}" if filed else "No new issues — engineer queue is current."),
                    username="QuantEdge Improvements Worker")
    except Exception as exc:  # noqa: BLE001
        print(f"notify skipped: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
