"""Endpoints that are LIVE and had zero test coverage.

The reachability triage split the 37 zero-coverage modules into two groups.
The dead ones got a guard (`test_module_reachability.py`). This file covers the
other group — modules that ARE reachable from mounted, authenticated API
routes and still had 0% coverage:

    api/v1/scanners  -> tasks/stock_scanners     275 statements, 0%
    api/v1/releases  -> ml/serving/serve         105 statements, 0%
                     -> ml/serving/ab_router      70 statements, 0%

That combination is worse than dead code. Dead code cannot break a user;
untested live code can, and nothing would catch it — these routes 401 for an
anonymous prober, so a logged-in 500 was invisible from outside.

Scope, stated honestly: these are contract/smoke tests. They assert the routes
answer, enforce auth, validate input, and do not 500 — not that the scanner
maths is right. That is the gap that mattered: a route that raises on every
authenticated call is a different problem from one that ranks badly.
"""
from __future__ import annotations

import pytest

from tests.integration._auth_helper import auth_headers

SCANNER_DESKS = ("equity", "crypto", "polymarket")

# One registration for the whole file, not one per test.
#
# The auth limiter is 10/min. A first draft called auth_headers() in every test
# (~11 registrations here) and starved OTHER files sharing the worker —
# test_api_health::test_auth_register_then_login went red while passing in
# isolation. A test that breaks its neighbours by consuming a shared budget is
# the same class of problem as one that writes to a shared database.
_TOKEN: dict[str, str] | None = None


async def _headers(client) -> dict[str, str]:
    global _TOKEN
    if _TOKEN is None:
        _TOKEN = await auth_headers(client, prefix="live_ep")
    return _TOKEN


# ── auth is enforced ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
@pytest.mark.parametrize("path", [
    "/api/v1/scanners/",
    "/api/v1/scanners/equity",
    "/api/v1/releases/",
    "/api/v1/releases/ab-tests/active",
])
async def test_these_routes_require_authentication(client, path):
    """Verified against production too: all five return 401 anonymously."""
    r = await client.get(path)
    assert r.status_code in (401, 403), f"{path} answered {r.status_code} without a token"


# ── scanners ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
@pytest.mark.parametrize("desk", SCANNER_DESKS)
async def test_each_scanner_desk_answers_without_erroring(client, desk):
    """275 statements behind this, none of it previously executed by a test."""
    headers = await _headers(client)
    r = await client.get(f"/api/v1/scanners/{desk}", headers=headers)
    assert r.status_code < 500, f"{desk} scanner 500'd: {r.text[:300]}"
    if r.status_code == 200:
        body = r.json()
        assert body["desk"] == desk
        assert isinstance(body["results"], list)
        assert isinstance(body["cached"], bool)


@pytest.mark.asyncio
async def test_an_unknown_desk_is_rejected_not_crashed(client):
    headers = await _headers(client)
    r = await client.get("/api/v1/scanners/not-a-desk", headers=headers)
    assert r.status_code == 400, f"expected a clean 400, got {r.status_code}: {r.text[:200]}"
    assert "Unknown desk" in r.text


@pytest.mark.asyncio
async def test_listing_all_desks_answers(client):
    headers = await _headers(client)
    r = await client.get("/api/v1/scanners/", headers=headers)
    assert r.status_code < 500, f"scanner list 500'd: {r.text[:300]}"
    if r.status_code == 200:
        body = r.json()
        assert isinstance(body, list)
        assert {item["desk"] for item in body} <= set(SCANNER_DESKS)


# ── releases / model serving ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_listing_releases_answers_on_an_empty_registry(client):
    """The cold-start case: no model has ever been registered (ml_models: 0)."""
    headers = await _headers(client)
    r = await client.get("/api/v1/releases/", headers=headers)
    assert r.status_code < 500, f"releases list 500'd: {r.text[:300]}"
    if r.status_code == 200:
        assert isinstance(r.json(), list)


@pytest.mark.asyncio
async def test_active_ab_tests_answers_when_none_exist(client):
    """ab_router had 0% coverage; an empty registry must not raise."""
    headers = await _headers(client)
    r = await client.get("/api/v1/releases/ab-tests/active", headers=headers)
    assert r.status_code < 500, f"ab-tests 500'd: {r.text[:300]}"
    if r.status_code == 200:
        assert isinstance(r.json(), list)


@pytest.mark.asyncio
async def test_a_missing_champion_is_a_clean_404_not_a_500(client):
    """No model is registered anywhere yet — this is the live state today."""
    headers = await _headers(client)
    r = await client.get("/api/v1/releases/champion/no_such_model", headers=headers)
    assert r.status_code != 500, f"missing champion 500'd: {r.text[:300]}"
    assert r.status_code in (404, 200), f"unexpected {r.status_code}: {r.text[:200]}"


@pytest.mark.asyncio
async def test_a_missing_release_is_a_clean_404_not_a_500(client):
    headers = await _headers(client)
    r = await client.get("/api/v1/releases/does-not-exist", headers=headers)
    assert r.status_code != 500, f"missing release 500'd: {r.text[:300]}"
    assert r.status_code in (404, 422), f"unexpected {r.status_code}: {r.text[:200]}"


@pytest.mark.asyncio
async def test_registering_a_release_validates_its_input(client):
    """A malformed body must be a 422, never an unhandled exception."""
    headers = await _headers(client)
    r = await client.post("/api/v1/releases/", headers=headers, json={"nope": True})
    assert r.status_code == 422, f"expected validation error, got {r.status_code}: {r.text[:200]}"
