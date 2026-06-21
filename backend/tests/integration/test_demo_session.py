"""Demo session makes the login-free app functional: /auth/demo issues a token that
unblocks the JWT-gated endpoints (the fix for 'all pages blank / no buttons')."""
import pytest


@pytest.mark.asyncio
async def test_demo_login_issues_token(client):
    r = await client.post("/api/v1/auth/demo")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body.get("access_token") and body.get("refresh_token")


@pytest.mark.asyncio
async def test_demo_token_unblocks_protected_endpoints(client):
    tok = (await client.post("/api/v1/auth/demo")).json()["access_token"]
    headers = {"Authorization": f"Bearer {tok}"}
    # Previously these returned 401 (blank data); with a demo token they must succeed.
    for path in ("/api/v1/positions/", "/api/v1/strategies/"):
        r = await client.get(path, headers=headers)
        assert r.status_code == 200, f"{path} -> {r.status_code}: {r.text}"


@pytest.mark.asyncio
async def test_demo_login_is_idempotent(client):
    a = (await client.post("/api/v1/auth/demo")).json()["access_token"]
    b = (await client.post("/api/v1/auth/demo")).json()["access_token"]
    assert a and b  # second call reuses the same demo user, no unique-email crash
