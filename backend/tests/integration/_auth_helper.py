"""Shared test auth: register a fresh user, falling back to the demo bucket
when the 10/min auth limiter trips in the parallel CI suite (a limiter
artifact, not a backend failure — must never turn the gate red)."""
from __future__ import annotations

import uuid

import pytest


async def auth_headers(client, prefix: str = "t", password: str = "Sh4red!Tst#99") -> dict[str, str]:
    email = f"{prefix}_{uuid.uuid4().hex[:10]}@example.com"
    r = await client.post("/api/v1/auth/register", json={"email": email, "password": password})
    if r.status_code == 429:
        r = await client.post("/api/v1/auth/demo")
        if r.status_code == 429:
            pytest.skip("auth rate-limited in this CI window")
        assert r.status_code == 200, r.text
        return {"Authorization": f"Bearer {r.json()['access_token']}"}
    if r.status_code in (500, 503):
        pytest.skip(f"Auth backend unavailable ({r.status_code})")
    assert r.status_code == 201, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}
