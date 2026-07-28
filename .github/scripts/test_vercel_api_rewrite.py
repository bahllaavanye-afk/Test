"""Every dashboard list view 404'd at the CDN and never reached the backend.

Measured 2026-07-28 against the live deployment:

    direct to backend   /api/v1/positions/           -> 200
    through Vercel      /api/v1/positions/           -> 404 NOT_FOUND (iad1::…)
    through Vercel      /api/v1/scanners/polymarket  -> 200

The rewrite source was `/api/:path*`. Vercel's named-segment matcher splits on
`/`, so a TRAILING SLASH leaves an empty final segment and the rule does not
match — the request falls through to the SPA fallback and 404s instead of
proxying to Render.

That matters because FastAPI declares every collection endpoint with a
trailing slash (`@router.get("/")`), and the frontend's axios client is built
with `baseURL: "/api/v1"` then called as `.get("/positions/")`. So positions,
trades, bots, strategies, orders and analytics ALL 404'd in the browser while
the identical paths returned data when called directly against Render. Only
`scanners/{desk}` worked, because it has no trailing slash.

This is why the dashboard read as empty regardless of the backend fixes
shipped the same day — those requests were never arriving. It was reported
three times ("still no trades", "still empty trades") and each time I looked
at the backend, which was answering correctly the whole time.

`(.*)` captures the remainder verbatim, trailing slash included.

Lives here rather than in `frontend/` because the frontend has no test runner
wired — no `vitest` dependency and no `test` script — and adding a .test.ts
file breaks `tsc && vite build`, which is what CI actually runs. This suite
runs in CI today.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

_VERCEL = Path(__file__).resolve().parents[2] / "frontend" / "vercel.json"


def _config() -> dict:
    return json.loads(_VERCEL.read_text())


def _api_rule() -> dict:
    for r in _config().get("rewrites", []):
        if r.get("source", "").startswith("/api"):
            return r
    raise AssertionError("no /api rewrite rule found in frontend/vercel.json")


def _matches(source: str, path: str) -> bool:
    """Mirror Vercel's behaviour for the two source syntaxes in play."""
    if ":path*" in source:
        prefix = source.replace("/:path*", "")
        if not path.startswith(prefix + "/"):
            return False
        rest = path[len(prefix) + 1:]
        # An empty final segment (trailing slash) fails to match.
        return bool(rest) and not rest.endswith("/")
    return re.fullmatch(source, path) is not None


def test_the_config_exists_and_targets_the_live_backend():
    rule = _api_rule()
    assert "quantedge-api-9jz0.onrender.com" in rule["destination"], rule


@pytest.mark.parametrize("path", [
    "/api/v1/positions/",
    "/api/v1/trades/",
    "/api/v1/bots/",
    "/api/v1/strategies/",
    "/api/v1/orders/",
    "/api/v1/analytics/",
])
def test_the_trailing_slash_collection_routes_are_proxied(path):
    """THE BUG: these are every list view in the dashboard."""
    assert _matches(_api_rule()["source"], path), path


@pytest.mark.parametrize("path", [
    "/api/v1/scanners/polymarket",
    "/api/v1/auth/demo",
    "/api/v1/market-data/quote/SPY",
    "/api/v1/positions",
])
def test_slashless_routes_still_proxied(path):
    assert _matches(_api_rule()["source"], path), path


def test_the_captured_path_is_forwarded_verbatim():
    """The trailing slash must survive into the destination.

    FastAPI 307-redirects the slashless form, so dropping the slash would turn
    one request into two and break CORS-sensitive clients.
    """
    rule = _api_rule()
    m = re.fullmatch(rule["source"], "/api/v1/positions/")
    assert m, "source did not match"
    assert rule["destination"].replace("$1", m.group(1)) == (
        "https://quantedge-api-9jz0.onrender.com/api/v1/positions/"
    )


def test_the_old_named_segment_form_would_have_dropped_them():
    """Pins the regression: `:path*` fails exactly on the trailing slash."""
    assert not _matches("/api/:path*", "/api/v1/positions/")
    assert _matches("/api/:path*", "/api/v1/scanners/polymarket")


@pytest.mark.parametrize("path", ["/dashboard", "/equity", "/", "/login"])
def test_non_api_routes_are_not_swallowed(path):
    """The SPA still has to render its own routes."""
    assert not _matches(_api_rule()["source"], path), path


def test_the_spa_fallback_still_excludes_api():
    spa = [r for r in _config()["rewrites"] if r["destination"] == "/index.html"]
    assert spa, "SPA fallback rewrite missing"
    assert "?!api" in spa[0]["source"]


def test_security_headers_are_not_lost():
    """The same file carries them; a careless rewrite edit could drop them."""
    keys = {
        h["key"]
        for entry in _config().get("headers", [])
        for h in entry.get("headers", [])
    }
    assert {"X-Content-Type-Options", "X-Frame-Options"} <= keys


# ── build-rate-limit guard ───────────────────────────────────────────────────
# Vercel rebuilt on EVERY commit to main. The bot fleet writes state files
# constantly — over the last 40 commits only ONE touched frontend/ — so the
# free-tier Hobby quota was exhausted and Vercel returned:
#
#     Deployment rate limited — retry in 24 hours.
#
# which blocked the CDN rewrite fix above from reaching production at all. A
# correct fix that cannot deploy is not a fix.
#
# `ignoreCommand` exits 0 to SKIP the build. `git diff --quiet` exits 0 when
# there is NO change in the project directory, so unrelated commits skip and
# frontend commits build. It also fails SAFE: any other error (shallow clone,
# missing HEAD^) is non-zero, which builds.


def test_an_ignore_command_is_configured():
    """Without it the deploy quota is spent on commits that change no UI."""
    assert _config().get("ignoreCommand"), (
        "frontend/vercel.json needs an ignoreCommand or every bot state commit "
        "burns a Vercel deployment (measured: 39 of the last 40 commits)"
    )


def test_the_ignore_command_skips_only_on_an_unchanged_project_dir():
    cmd = _config()["ignoreCommand"]
    assert "git diff" in cmd and "--quiet" in cmd, cmd
    assert "HEAD^" in cmd and "HEAD" in cmd, cmd
