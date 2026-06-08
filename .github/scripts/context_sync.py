"""Distill agent_memory.json into a compact shared_context.md for all agents."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT   = Path(__file__).resolve().parents[2]
STATE_DIR   = REPO_ROOT / ".github" / "state"
MEMORY_FILE = STATE_DIR / "agent_memory.json"
OUTPUT_FILE = STATE_DIR / "shared_context.md"


def _load_memory() -> dict:
    if MEMORY_FILE.exists():
        try:
            return json.loads(MEMORY_FILE.read_text())
        except Exception:
            pass
    return {}


def main() -> None:
    mem = _load_memory()
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        f"# QuantEdge Shared Context — {date_str}",
        "",
        "## Platform Status",
    ]

    metrics = mem.get("platform_metrics", {})
    if metrics:
        lines.append(
            f"Strategies live: {metrics.get('strategies_live', '?')} | "
            f"Sharpe: {metrics.get('last_sharpe', '?')} | "
            f"Drawdown: {metrics.get('last_drawdown_pct', '?')}% | "
            f"Commits today: {metrics.get('commits_today', '?')}"
        )
    else:
        lines.append("No metrics available.")

    lines += ["", "## Recent Learnings (last 5)"]
    learnings = mem.get("peer_learnings", [])[-50:]
    for entry in learnings[-5:]:
        if isinstance(entry, dict):
            text = entry.get("learning") or entry.get("text") or str(entry)
        else:
            text = str(entry)
        lines.append(f"- {text[:120]}")
    if not learnings:
        lines.append("- No learnings recorded yet.")

    lines += ["", "## Conversation Highlights (last 10)"]
    convs = mem.get("conversations", {})
    recent = sorted(convs.items())[-20:]
    shown = 0
    for _ts, entry in recent:
        if shown >= 10:
            break
        if not isinstance(entry, dict):
            continue
        speaker = entry.get("speaker", "?")
        msg = (entry.get("message") or "")[:100]
        lines.append(f"**{speaker}**: {msg}")
        shown += 1
    if shown == 0:
        lines.append("No conversations recorded yet.")

    OUTPUT_FILE.write_text("\n".join(lines) + "\n")
    print(f"[context_sync] wrote {OUTPUT_FILE}", flush=True)


if __name__ == "__main__":
    main()
