"""Three Vercel projects in this account answer HTTP 200 and one is the platform.

    quantedge-eight.vercel.app  "QuantEdge — Institutional Trading Platform"  ✅
    quantedge.vercel.app        "Create Next App"          abandoned stub
    quant-edge-nine.vercel.app  "My Google AI Studio App"  unrelated

Every consumer of a frontend URL had been pointed at the middle one: the Discord
dashboard screenshots, the page-reporter workflow's fallback, the third-party
uptime monitor, and `verify_live.py` — the script whose entire job is to confirm
the live system. All four succeeded, because a stub returns 200 exactly like the
real app. The uptime monitor also accepted 404 as healthy, so it could not have
failed under any condition.

A session was once spent reporting "frontend 200 ✓" against the wrong app
(CONTINUITY 2026-07-xx). The rule that came out of it — **verify a frontend by
its `<title>`, not its status code** — was written down and then not applied to
the code that does the verifying. This file applies it.

No network: these are static assertions about what the repo points at.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

REAL = "quantedge-eight.vercel.app"
STUBS = ("quantedge.vercel.app", "quant-edge-nine.vercel.app")

# Every place that resolves a frontend URL for a machine to visit.
CONSUMERS = (
    "backend/app/notifications/screenshot.py",
    "scripts/verify_live.py",
    ".github/scripts/third_party_monitor.py",
    ".github/workflows/page-reporter.yml",
)


def _text(rel: str) -> str:
    return (REPO / rel).read_text()


@pytest.mark.parametrize("rel", CONSUMERS)
def test_no_consumer_points_at_a_stub(rel: str):
    """`quantedge.vercel.app` is a substring of `quantedge-eight...`? No — the
    stub host ends at `.app`, so a word-boundary match is exact."""
    src = _text(rel)
    for stub in STUBS:
        hits = [
            line.strip()
            for line in src.splitlines()
            # A comment explaining the stub is the point of this fix, not a use.
            if re.search(rf"(?<!-){re.escape(stub)}", line)
            and not line.lstrip().startswith(("#", "//"))
        ]
        assert not hits, f"{rel} still resolves {stub}: {hits}"


@pytest.mark.parametrize("rel", CONSUMERS)
def test_every_consumer_names_the_real_app(rel: str):
    assert REAL in _text(rel), f"{rel} does not reference {REAL}"


def test_the_uptime_monitor_can_actually_fail():
    """It accepted `[200, 301, 302, 404]`. With 404 healthy there is no response
    a dead deployment could give that this monitor would reject — a check that
    cannot fail is not a check."""
    src = _text(".github/scripts/third_party_monitor.py")
    block = src.split("Vercel (Frontend)", 1)[1][:600]
    expected = re.search(r"expected_status\":\s*\[([^\]]*)\]", block)
    assert expected, "the Vercel entry lost its expected_status"
    codes = {c.strip() for c in expected.group(1).split(",") if c.strip()}
    assert "404" not in codes, "404 is still accepted as a healthy frontend"


def test_the_screenshot_default_is_overridable_not_hardcoded():
    """Render sets FRONTEND_URL; a hardcoded default is how this drifted in the
    first place and how it would drift again after the next domain change."""
    src = _text("backend/app/notifications/screenshot.py")
    assert "FRONTEND_URL" in src
    assert 'os.environ.get("FRONTEND_URL"' in src
    # Both entry points must take the shared default, not their own literal.
    for fn in ("def capture_dashboard(", "async def capture_all_pages("):
        sig = src.split(fn, 1)[1].split(")", 1)[0]
        assert "vercel.app" not in sig, f"{fn} still hardcodes a URL: {sig}"


def test_the_dead_render_host_is_not_a_documented_url():
    """`quantedge-api.onrender.com` (bare) 404s on /health — live-checked
    2026-08-05. It is fine as a *fallback* in verify_live's host list, which
    exists precisely to prove which host answers; it is not fine as the URL a
    human is told to use."""
    guide = _text("scripts/CLAUDE.md")
    for line in guide.splitlines():
        if "URL:" in line:
            assert "quantedge-api.onrender.com" not in line, f"stale host in: {line.strip()}"
