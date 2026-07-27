"""Discord error triage — read what is actually failing, and drive it to a fix.

The fleet posts errors into Discord all day. Nothing read them. `channel_monitor`
watches for channel SILENCE, `p0_watchdog` watches issue age, `render_failure_issue`
watches deploys — but the error TEXT itself, the richest signal there is, was
write-only. Every failure was announced to a room with nobody in it.

This closes that loop:

  1. read the error channels via the bot API
  2. reduce each message to a stable SIGNATURE (ids, timestamps, prices,
     quantities stripped) so 200 copies of one bug collapse into one row
  3. rank by occurrence count
  4. file/update ONE GitHub issue per recurring signature, labelled so the
     existing reward-gated improver fleet can pick it up
  5. post a ranked digest back to Discord

Dedupe is by signature hash in the issue title, so re-running never spams: an
existing open issue is updated with a new count instead of duplicated.

Run:
    python .github/scripts/discord_error_triage.py            # report + file issues
    python .github/scripts/discord_error_triage.py --dry-run  # report only
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Channels that carry failure signal. Ordered by how actionable they are.
ERROR_CHANNELS = [
    "ci-failures",
    "incidents",
    "infra-alerts",
    "risk-alerts",
    "bot-fleet",
]

# A message is an error if it carries one of these markers.
ERROR_MARKERS = (
    "🚨", "❌", "✗", "⚠", "Traceback", "Error", "ERROR", "error:",
    "failed", "FAILED", "Failure", "exception", "Exception",
)

# Substrings that mean "this is noise, not a defect".
IGNORE_MARKERS = (
    "delivery self-test",
    "if you can read this",
)

MIN_OCCURRENCES = 2          # file an issue only for something that recurs
MAX_ISSUES_PER_RUN = 5       # never flood the tracker
ISSUE_LABEL = "auto-triage"

GH_TOKEN = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN", "")
GH_REPO = os.environ.get("GH_REPO", "bahllaavanye-afk/test")


# ── signature ────────────────────────────────────────────────────────────────

_SUBS = [
    (re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I), "<uuid>"),
    (re.compile(r"\b[0-9a-f]{7,40}\b", re.I), "<sha>"),
    (re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2})?(\.\d+)?Z?"), "<ts>"),
    (re.compile(r"\d{2}:\d{2}(:\d{2})?"), "<time>"),
    (re.compile(r"https?://\S+"), "<url>"),
    (re.compile(r"-?\$?\d[\d,]*\.\d+"), "<num>"),
    (re.compile(r"\b\d+\b"), "<n>"),
]


def signature(text: str) -> str:
    """Collapse a message to what makes it *this kind* of failure.

    Strips ids, timestamps, urls and numbers so that
    "requested: 134.58, available: 6.71" and
    "requested: 92.10, available: 4.02" become the same row.
    """
    s = " ".join(text.split())
    for pattern, repl in _SUBS:
        s = pattern.sub(repl, s)
    return s[:300].strip()


def is_error(text: str) -> bool:
    if not text or not text.strip():
        return False
    if any(m in text for m in IGNORE_MARKERS):
        return False
    return any(m in text for m in ERROR_MARKERS)


def sig_hash(sig: str) -> str:
    return hashlib.sha1(sig.encode("utf-8")).hexdigest()[:10]


# ── collection ───────────────────────────────────────────────────────────────

def collect(limit_per_channel: int = 100) -> list[dict]:
    """Read the error channels. Returns [{channel, content, author, ts}]."""
    try:
        import notify
    except Exception as exc:  # pragma: no cover - import guard
        print(f"notify unavailable: {exc}", flush=True)
        return []

    found: list[dict] = []
    for channel in ERROR_CHANNELS:
        try:
            msgs = notify.read_channel_recent(channel, limit=limit_per_channel)
        except Exception as exc:  # noqa: BLE001 — one bad channel must not stop triage
            print(f"  ⚠ {channel}: read failed ({exc})", flush=True)
            continue
        hits = [m for m in msgs if is_error(m.get("content", ""))]
        print(f"  #{channel}: {len(msgs)} messages, {len(hits)} errors", flush=True)
        for m in hits:
            found.append({"channel": channel, **m})
    return found


def rank(messages: list[dict]) -> list[tuple[str, int, dict]]:
    """[(signature, count, example)] most frequent first."""
    counts: Counter = Counter()
    example: dict[str, dict] = {}
    for m in messages:
        sig = signature(m.get("content", ""))
        if not sig:
            continue
        counts[sig] += 1
        example.setdefault(sig, m)
    return [(sig, n, example[sig]) for sig, n in counts.most_common()]


# ── GitHub issues ────────────────────────────────────────────────────────────

def _gh(method: str, path: str, payload: dict | None = None):
    req = urllib.request.Request(
        f"https://api.github.com{path}",
        data=json.dumps(payload).encode() if payload else None,
        method=method,
        headers={
            "Authorization": f"Bearer {GH_TOKEN}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=45) as resp:
        return json.load(resp)


def existing_issues() -> dict[str, dict]:
    """{sig_hash: issue} for open auto-triage issues."""
    if not GH_TOKEN:
        return {}
    try:
        items = _gh("GET", f"/repos/{GH_REPO}/issues?state=open&labels={ISSUE_LABEL}&per_page=100")
    except Exception as exc:  # noqa: BLE001
        print(f"  ⚠ could not list existing issues: {exc}", flush=True)
        return {}
    out = {}
    for it in items:
        m = re.search(r"\[([0-9a-f]{10})\]", it.get("title", ""))
        if m:
            out[m.group(1)] = it
    return out


def file_issue(sig: str, count: int, example: dict, existing: dict) -> str:
    """Create or update one issue for this signature. Returns an action word."""
    h = sig_hash(sig)
    title = f"[auto-triage] [{h}] {sig[:90]}"
    body = (
        f"Seen **{count}×** in the current Discord window.\n\n"
        f"**Channel:** `#{example.get('channel')}`\n"
        f"**Most recent:** {example.get('ts') or 'unknown'}\n\n"
        f"**Signature** (ids/timestamps/numbers normalised):\n```\n{sig}\n```\n\n"
        f"**Example message:**\n```\n{(example.get('content') or '')[:1200]}\n```\n\n"
        f"---\nFiled automatically by `discord_error_triage.py`. "
        f"The signature hash `{h}` is the dedupe key — this issue is updated, "
        f"never duplicated."
    )

    if h in existing:
        num = existing[h]["number"]
        try:
            _gh("POST", f"/repos/{GH_REPO}/issues/{num}/comments",
                {"body": f"Still occurring — **{count}×** as of "
                         f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}."})
            return f"updated #{num}"
        except Exception as exc:  # noqa: BLE001
            return f"update failed ({exc})"

    try:
        issue = _gh("POST", f"/repos/{GH_REPO}/issues",
                    {"title": title, "body": body, "labels": [ISSUE_LABEL, "bug"]})
        return f"opened #{issue['number']}"
    except Exception as exc:  # noqa: BLE001
        return f"create failed ({exc})"


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    dry = "--dry-run" in sys.argv
    print(f"QuantEdge Discord error triage — "
          f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}", flush=True)

    messages = collect()
    if not messages:
        print("No error messages found (or Discord unreadable) — nothing to triage.",
              flush=True)
        return 0

    ranked = rank(messages)
    print(f"\n{len(messages)} error messages → {len(ranked)} distinct signatures\n",
          flush=True)

    lines = []
    for i, (sig, n, ex) in enumerate(ranked[:15], 1):
        line = f"{i:2d}. {n:3d}×  #{ex['channel']:<14} {sig[:110]}"
        print(line, flush=True)
        lines.append(f"`{n}×` **#{ex['channel']}** — {sig[:150]}")

    recurring = [r for r in ranked if r[1] >= MIN_OCCURRENCES][:MAX_ISSUES_PER_RUN]
    if dry:
        print(f"\n[dry-run] would file/update {len(recurring)} issue(s)", flush=True)
    elif not GH_TOKEN:
        print("\nNo GH_TOKEN — skipping issue filing", flush=True)
    else:
        existing = existing_issues()
        print(f"\nFiling issues for {len(recurring)} recurring signature(s):", flush=True)
        for sig, n, ex in recurring:
            print(f"   {file_issue(sig, n, ex, existing)}  ({n}× {sig[:60]})", flush=True)

    # Digest back to Discord so the ranking is visible where the errors landed.
    if not dry:
        try:
            import notify
            digest = (
                f"🔎 **Error triage** — {len(messages)} error messages, "
                f"{len(ranked)} distinct causes\n\n" + "\n".join(lines[:10])
            )
            notify.post("incidents", digest, username="QuantEdge Triage")
        except Exception as exc:  # noqa: BLE001
            print(f"digest post failed: {exc}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
