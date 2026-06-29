"""Endpoint smoke guard — critical GET routes must RESOLVE (never 404/500).

Catches the "router mounted but has no handlers / handler explodes" class of bug
— e.g. the leaderboard `/entries` 404 that was red for ages. Auth-protected routes
returning 401/403 are fine; the point is they must not 404 (no handler) or 500
(handler crashes on a basic GET).

Route enumeration via `app.routes` is unreliable here (the API is under a mount,
so it reports 0 routes), so this hits a curated list of must-exist endpoints the
same way the rest of the suite does — through a real TestClient request.
"""
from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from app.main import app

_client = TestClient(app, raise_server_exceptions=False)

# Must-exist GET endpoints. Unauthenticated → 401/403 is acceptable; 404/500 is a bug.
CRITICAL_GETS = [
    "/health",
    "/api/v1/leaderboard/entries",   # regressed to 404 once — guard it
    "/api/v1/leaderboard/summary",
    "/api/v1/strategies/",
    "/api/v1/strategies/available",
    "/api/v1/strategies/active",
    "/api/v1/strategies/desks",
    "/api/v1/risk/",
    "/api/v1/analytics/",
    "/api/v1/backtests/scenarios",
    "/api/v1/bots/",
]


@pytest.mark.parametrize("path", CRITICAL_GETS)
def test_critical_get_resolves(path):
    resp = _client.get(path)
    assert resp.status_code != 404, f"GET {path} → 404 (router mounted but no handler?)"
    assert resp.status_code != 500, f"GET {path} → 500 (handler errored on a basic GET)"
    assert resp.status_code in (200, 401, 403, 422), f"GET {path} → unexpected {resp.status_code}"
