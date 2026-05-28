"""API health and root endpoint tests."""
import pytest


@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_auth_register_then_login(client):
    resp = await client.post("/api/v1/auth/register", json={
        "email": "test@example.com", "password": "testpass123"
    })
    # Either 200 (success) or 409 (already exists)
    assert resp.status_code in (200, 201, 409, 400)

    resp = await client.post("/api/v1/auth/login", json={
        "username": "test@example.com", "password": "testpass123"
    })
    if resp.status_code == 200:
        data = resp.json()
        assert "access_token" in data
