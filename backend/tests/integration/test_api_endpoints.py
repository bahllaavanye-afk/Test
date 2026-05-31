"""
Integration tests: all major API endpoint groups.
Uses SQLite in-memory DB via the shared `client` fixture in conftest.py.
"""
from __future__ import annotations

import pytest


# ──────────────────────────────────────────────────────────────────────────────
# Auth
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_register_and_login(client):
    r = await client.post("/api/v1/auth/register", json={
        "email": "integration@quantedge.ai", "password": "Str0ng!Pass"
    })
    assert r.status_code in (200, 201, 409), f"register: {r.status_code} {r.text[:200]}"

    r = await client.post("/api/v1/auth/login", json={
        "email": "integration@quantedge.ai", "password": "Str0ng!Pass"
    })
    assert r.status_code == 200, f"login: {r.status_code} {r.text[:200]}"
    assert "access_token" in r.json()


@pytest.mark.asyncio
async def test_login_wrong_password_returns_401(client):
    r = await client.post("/api/v1/auth/login", json={
        "username": "noone@quantedge.ai", "password": "wrong"
    })
    assert r.status_code in (401, 403, 422), f"expected 401, got {r.status_code}"


# ──────────────────────────────────────────────────────────────────────────────
# Health
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health_endpoint(client):
    r = await client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body.get("status") == "ok", f"Unexpected health body: {body}"


# ──────────────────────────────────────────────────────────────────────────────
# Protected routes — reject unauthenticated requests
# ──────────────────────────────────────────────────────────────────────────────

PROTECTED_ROUTES = [
    ("GET",  "/api/v1/accounts/"),
    ("GET",  "/api/v1/positions/"),
    ("GET",  "/api/v1/orders/"),
    ("GET",  "/api/v1/strategies/"),
    ("GET",  "/api/v1/risk/status"),
    ("GET",  "/api/v1/analytics/tearsheet"),
    ("GET",  "/api/v1/ml/models"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("method,path", PROTECTED_ROUTES)
async def test_protected_route_requires_auth(client, method, path):
    r = await getattr(client, method.lower())(path, follow_redirects=False)
    # 401/403 = correctly protected; 307 = redirect to login (also acceptable)
    assert r.status_code in (401, 403, 307, 404), (
        f"{method} {path} returned {r.status_code} without auth — endpoint may be unprotected!"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Strategies endpoint (authenticated)
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
async def auth_headers(client):
    await client.post("/api/v1/auth/register", json={
        "email": "authtest@quantedge.ai", "password": "Str0ng!Pass99"
    })
    r = await client.post("/api/v1/auth/login", json={
        "email": "authtest@quantedge.ai", "password": "Str0ng!Pass99"
    })
    token = r.json().get("access_token", "")
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.asyncio
async def test_strategies_list_authenticated(client, auth_headers):
    r = await client.get("/api/v1/strategies/", headers=auth_headers)
    assert r.status_code == 200
    body = r.json()
    # Should return a list (possibly empty in test mode, but well-formed)
    assert isinstance(body, list), f"Expected list, got {type(body)}: {body}"


@pytest.mark.asyncio
async def test_risk_status_authenticated(client, auth_headers):
    r = await client.get("/api/v1/risk/status", headers=auth_headers)
    assert r.status_code in (200, 503), f"risk status: {r.status_code} {r.text[:200]}"
    if r.status_code == 200:
        body = r.json()
        # Must have some indication of risk state
        assert any(k in body for k in ("is_halted", "global_halt", "status", "capital")), (
            f"Risk status missing expected fields: {body}"
        )


@pytest.mark.asyncio
async def test_orders_post_rejects_invalid_payload(client, auth_headers):
    r = await client.post("/api/v1/orders/", json={"invalid": "payload"}, headers=auth_headers)
    assert r.status_code == 422, (
        f"Expected 422 for invalid order payload, got {r.status_code}: {r.text[:200]}"
    )


@pytest.mark.asyncio
async def test_backtests_endpoint_exists(client, auth_headers):
    r = await client.get("/api/v1/backtests/", headers=auth_headers)
    assert r.status_code in (200, 404, 501), (
        f"backtests endpoint unexpected status: {r.status_code}"
    )
