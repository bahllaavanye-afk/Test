"""Team Lead — reviews, assigns, reports. Deterministic, evidence-only.

The management layer the org was missing (improver run #298 merged an edit to
app/risk/manager.py — a protected file — and broke main; nothing reviewed it).

Every run:
  1. REVIEW  — open `automerge` PRs: any that touch PROTECTED paths lose the
               automerge label + get a comment explaining why (a human or a
               dedicated fix must bless them). This is a hard gate, not advice.
  2. ASSIGN  — open `agent-fix-needed` issues get an area label mapped to the
               employee workflow that owns that path, so pickers stop guessing.
  3. REPORT  — a facts-only digest to Discord #leadership-summary: merged in
               the last window, open PRs + CI state, blocked items, assignments.
               No fabricated metrics — only what the GitHub API returned.

Runs on GITHUB_TOKEN (issues: write, pull-requests: write). Realistic by
construction: every line in the report links to a real PR/issue/run.
"""
from __future__ import annotations

import json
import os
import urllib.request
from datetime import datetime, timedelta, timezone

REPO = os.environ.get("GITHUB_REPOSITORY", "bahllaavanye-afk/Test")
TOKEN = os.environ.get("GITHUB_TOKEN", "")
API = f"https://api.github.com/repos/{REPO}"

# Files no autonomous PR may touch (module CLAUDE.md "Do NOT Modify" lists).
PROTECTED = (
    "backend/app/risk/manager.py",
    "backend/app/brokers/base.py",
    "backend/app/main.py",
    "backend/app/models/",
    "backend/app/strategies/base.py",
)

# path prefix → owning employee (area label applied to issues)
AREA_OWNERS = {
    "backend/app/ml": "area:ml-engineer",
    "backend/app/strategies": "area:quant-researcher",
    "backend/app/brokers": "area:broker-engineer",
    "backend/app/api": "area:backend-engineer",
    "frontend/": "area:frontend-engineer",
    ".github/": "area:infra-engineer",
    "backend/": "area:backend-engineer",
}

BOT_AUTHORS = {"github-actions[bot]"}


def _req(method: str, url: str, body: dict | None = None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={
        "Authorization": f"Bearer {TOKEN}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "QuantEdge-TeamLead/1.0",
    })
    with urllib.request.urlopen(req, timeout=20) as r:
        raw = r.read()
        return json.loads(raw) if raw else {}


def review_prs() -> tuple[list[str], list[str]]:
    """Demote autonomous PRs touching protected files. Returns (ok, blocked) lines."""
    ok, blocked = [], []
    prs = _req("GET", f"{API}/pulls?state=open&per_page=30")
    for pr in prs:
        n = pr["number"]
        labels = {l["name"] for l in pr.get("labels", [])}
        author = pr.get("user", {}).get("login", "")
        if "automerge" not in labels:
            continue
        files = _req("GET", f"{API}/pulls/{n}/files?per_page=100")
        touched = [f["filename"] for f in files
                   if any(f["filename"].startswith(p) or f["filename"] == p for p in PROTECTED)]
        if touched and author in BOT_AUTHORS:
            _req("DELETE", f"{API}/issues/{n}/labels/automerge")
            _req("POST", f"{API}/issues/{n}/comments", {
                "body": "🛑 **Team Lead review — automerge revoked.**\n\n"
                        "This autonomous PR modifies protected file(s):\n"
                        + "\n".join(f"- `{f}`" for f in touched)
                        + "\n\nProtected paths (risk manager, broker/strategy bases, models, "
                          "app factory) require explicit human review — an equivalent "
                          "autonomous edit (#298) broke main. Re-add the label only after review."
            })
            blocked.append(f"PR #{n} — blocked (touches {', '.join(touched[:3])})")
        else:
            state = "draft" if pr.get("draft") else "ready"
            ok.append(f"PR #{n} ({state}) — {pr['title'][:60]}")
    return ok, blocked


def assign_issues() -> list[str]:
    """Label open agent-fix-needed issues with their owning area."""
    out = []
    issues = _req("GET", f"{API}/issues?state=open&labels=agent-fix-needed&per_page=30")
    for it in issues:
        if "pull_request" in it:
            continue
        labels = {l["name"] for l in it.get("labels", [])}
        if any(l.startswith("area:") for l in labels):
            continue  # already assigned
        text = (it.get("title", "") + " " + (it.get("body") or ""))[:4000]
        owner = next((lab for prefix, lab in AREA_OWNERS.items() if prefix in text),
                     "area:backend-engineer")
        _req("POST", f"{API}/issues/{it['number']}/labels", {"labels": [owner]})
        out.append(f"#{it['number']} → {owner} — {it['title'][:60]}")
    return out


def sync_improvements_to_queue(max_new: int = 3) -> list[str]:
    """Feed the worker queue from IMPROVEMENTS.md — the always-improving loop.

    The LLM workers (free-agent-engineer, continuous-improvement) consume
    `agent-fix-needed` issues; IMPROVEMENTS.md is where humans/research park
    intent. This bridges them: top unchecked [P0]/[P1] items become issues
    (deduped by title), so the backlog is ALWAYS being worked, not just read.
    """
    import re
    from pathlib import Path

    md = Path("IMPROVEMENTS.md")
    if not md.exists():
        return []
    items = re.findall(r"^- \[ \] \*\*(\[P[01]\][^*]+)\*\*", md.read_text(), re.MULTILINE)
    if not items:
        return []

    existing = _req("GET", f"{API}/issues?state=all&labels=agent-fix-needed&per_page=100")
    have = {i.get("title", "") for i in existing}
    created = []
    for item in items:
        title = f"[backlog] {item.strip()[:120]}"
        if title in have or len(created) >= max_new:
            continue
        _req("POST", f"{API}/issues", {
            "title": title,
            "body": f"Auto-filed by the Team Lead from IMPROVEMENTS.md:\n\n> {item.strip()}\n\n"
                    "Definition of done: the IMPROVEMENTS.md checkbox is ticked in the same PR "
                    "that implements it, with tests. Reward-gated via the automerge label.",
            "labels": ["agent-fix-needed"],
        })
        created.append(title)
    return created


def merged_recently(hours: int = 6) -> list[str]:
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    prs = _req("GET", f"{API}/pulls?state=closed&sort=updated&direction=desc&per_page=20")
    return [f"PR #{p['number']} — {p['title'][:60]}" for p in prs
            if p.get("merged_at") and datetime.fromisoformat(
                p["merged_at"].replace("Z", "+00:00")) >= since]


def main() -> int:
    shipped = merged_recently()
    ok, blocked = review_prs()
    assigned = assign_issues()
    queued = sync_improvements_to_queue()

    lines = ["**📋 Team Lead report** (facts from the GitHub API — nothing synthesized)"]
    lines.append(f"\n**Shipped (last 6h):** {len(shipped)}")
    lines += [f"• {s}" for s in shipped[:6]]
    lines.append(f"\n**In review:** {len(ok)} open automerge PRs")
    lines += [f"• {s}" for s in ok[:6]]
    if blocked:
        lines.append(f"\n**🛑 Blocked by review:** {len(blocked)}")
        lines += [f"• {s}" for s in blocked]
    if assigned:
        lines.append(f"\n**Assigned:** {len(assigned)}")
        lines += [f"• {s}" for s in assigned[:6]]
    if queued:
        lines.append(f"\n**Queued from IMPROVEMENTS.md:** {len(queued)}")
        lines += [f"• {s}" for s in queued]

    report = "\n".join(lines)
    print(report)
    try:
        import sys
        sys.path.insert(0, os.path.dirname(__file__))
        from notify import discord_post
        discord_post("leadership-summary", report, username="QuantEdge Team Lead")
    except Exception as e:  # noqa: BLE001
        print(f"(Discord notify skipped: {e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
