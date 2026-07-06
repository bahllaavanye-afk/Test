"""OA Scout — watch Options Alpha for new bots/templates and queue them for cloning.

Verified 2026-07-06: optionalpha.com's template library, community bots, and
leaderboards ALL redirect to app.optionalpha.com/login — there is no public
surface left to scrape. So the scout runs in two honest modes:

  AUTH MODE (needs the OA_SESSION_COOKIE secret — the operator's own account):
    fetches the community/template/leaderboard pages, extracts bot/template
    names, diffs against .github/state/oa_library.json, and for every NEW find:
      • posts it to Discord #alpha-research (QuantEdge OA Scout persona)
      • files an `agent-fix-needed` issue with the source link so the worker
        fleet implements it as a BOT_TEMPLATES clone (reward-gated PR)
    State updates are committed by the workflow, so every find is remembered.

  PUBLIC MODE (no cookie): probes the public URLs anyway and reports the
    auth-wall status — so if OA ever re-opens the library, the scout starts
    harvesting the same day without a code change.

Cadence: daily cron + every CI completion (event-driven — cron is starved in
this repo), which satisfies "employees do this every day, all day". Fetches
are gentle (one pass, ~6 URLs, browser UA) — this reads the operator's own
paid account; keep it that way.
"""
from __future__ import annotations

import json
import os
import re
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

STATE_FILE = Path(__file__).resolve().parent.parent / "state" / "oa_library.json"
COOKIE = os.environ.get("OA_SESSION_COOKIE", "").strip()

WATCH_URLS = [
    "https://app.optionalpha.com/community/templates",
    "https://app.optionalpha.com/community/bots",
    "https://app.optionalpha.com/community/leaderboards",
    "https://optionalpha.com/templates",
    "https://optionalpha.com/templates/sample-templates",
    "https://optionalpha.com/templates/options-strategies",
]

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"


def _fetch(url: str) -> tuple[int, str, str]:
    """(status, final_url, body) — never raises."""
    headers = {"User-Agent": UA, "Accept": "text/html,application/json"}
    if COOKIE:
        headers["Cookie"] = COOKIE
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=25) as r:
            return r.status, r.geturl(), r.read().decode("utf-8", "replace")
    except Exception as exc:  # noqa: BLE001
        return 0, url, f"__error__ {exc}"


def extract_candidates(body: str) -> set[str]:
    """Bot/template names from HTML or JSON — tolerant of markup drift."""
    found: set[str] = set()
    # JSON payloads (the app is SPA-ish; API responses may embed name fields)
    for m in re.finditer(r'"name"\s*:\s*"([^"]{4,80})"', body):
        found.add(m.group(1).strip())
    # HTML link slugs to /bots/<slug> or /templates/<slug> with link text
    for m in re.finditer(r'href="/(?:bots|templates)/([a-z0-9-]{4,80})"[^>]*>([^<]{0,80})', body):
        text = m.group(2).strip()
        found.add(text if len(text) > 3 else m.group(1).replace("-", " "))
    # Filter obvious non-bot noise
    noise = re.compile(r"^(login|sign|home|about|pricing|help|terms|privacy|blog)\b", re.I)
    return {f for f in found if not noise.match(f)}


def _load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except json.JSONDecodeError:
            pass
    return {"known": [], "last_run": None, "auth_wall": None}


def _file_issue(name: str, source: str) -> None:
    token, repo = os.environ.get("GITHUB_TOKEN", ""), os.environ.get("GITHUB_REPOSITORY", "")
    if not (token and repo):
        return
    body = {
        "title": f"[oa-scout] Clone new Options Alpha bot: {name[:80]}",
        "body": f"OA Scout found a new bot/template on Options Alpha:\n\n"
                f"**{name}**\nSource: {source}\n\n"
                "Task: implement it as a BOT_TEMPLATES entry (1m trigger, 2.5% size, "
                "no_position guard, TP + stop on premium sellers — follow the oa_* pattern "
                "in app/bots/templates.py), with a test in test_oa_clone_templates.py. "
                "If parameters aren't visible from the page, use OA community defaults and "
                "flag assumptions in the description.",
        "labels": ["agent-fix-needed", "area:quant-researcher"],
    }
    try:
        req = urllib.request.Request(
            f"https://api.github.com/repos/{repo}/issues",
            data=json.dumps(body).encode(), method="POST",
            headers={"Authorization": f"Bearer {token}",
                     "Accept": "application/vnd.github+json", "User-Agent": UA},
        )
        with urllib.request.urlopen(req, timeout=15) as r:
            print(f"  filed issue #{json.loads(r.read()).get('number')} for {name!r}")
    except Exception as exc:  # noqa: BLE001
        print(f"  (issue filing failed: {exc})")


def main() -> int:
    state = _load_state()
    known = set(state.get("known", []))
    new_finds: list[tuple[str, str]] = []
    auth_walled = 0

    print(f"OA Scout — {'AUTH' if COOKIE else 'PUBLIC'} mode, {len(WATCH_URLS)} URLs")
    for url in WATCH_URLS:
        status, final, body = _fetch(url)
        if "login" in final:
            auth_walled += 1
            print(f"  🔒 {url} → auth wall")
            continue
        if status != 200:
            print(f"  ⚠ {url} → HTTP {status}")
            continue
        cands = extract_candidates(body)
        fresh = sorted(c for c in cands if c not in known)
        print(f"  ✓ {url} → {len(cands)} candidates, {len(fresh)} new")
        for name in fresh:
            new_finds.append((name, url))
            known.add(name)

    for name, source in new_finds[:10]:  # bounded per run
        _file_issue(name, source)

    state.update({
        "known": sorted(known),
        "last_run": datetime.now(timezone.utc).isoformat(),
        "auth_wall": auth_walled == len(WATCH_URLS),
    })
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=1))

    summary = (f"🔭 **OA Scout**: {len(new_finds)} new bot(s) found"
               + (f" — {', '.join(n for n, _ in new_finds[:5])}" if new_finds else "")
               ) if new_finds else None
    if auth_walled == len(WATCH_URLS) and not COOKIE:
        summary = ("🔭 **OA Scout**: every Options Alpha page is behind login. "
                   "Add the `OA_SESSION_COOKIE` repo secret (Cookie header from a logged-in "
                   "browser session) to enable daily library/leaderboard harvesting.")
    if summary:
        try:
            from notify import discord_post

            discord_post("alpha-research", summary, username="QuantEdge OA Scout")
        except Exception as exc:  # noqa: BLE001
            print(f"(Discord notify skipped: {exc})")
    print(f"\nDone: {len(new_finds)} new, {len(known)} known, auth_wall={auth_walled}/{len(WATCH_URLS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
