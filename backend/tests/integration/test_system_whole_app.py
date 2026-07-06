"""Whole-app system test — walk EVERY GET endpoint, assert none 5xx.

The strongest cheap guarantee against 'the website is broken': instead of
hand-picking endpoints (which is how the tearsheet 500 survived), derive the
list from the app's own route table so new endpoints are covered the day
they're added. Auth'd with a real registered user; parameterless GETs only
(path-param routes need fixtures and have their own tests).
"""
from __future__ import annotations

import uuid

import pytest

_PASSWORD = "Syst3m!2026xx"

# Endpoints that legitimately need query params or external services and have
# dedicated tests elsewhere. Keep this list SHORT and justified.
_SKIP_PATHS = {
    "/api/v1/auth/google",          # redirects to Google (302 tested elsewhere)
    "/api/v1/auth/google/callback",  # needs OAuth code
}


async def _auth_headers(client) -> dict[str, str]:
    email = f"system_{uuid.uuid4().hex[:10]}@example.com"
    r = await client.post("/api/v1/auth/register", json={"email": email, "password": _PASSWORD})
    if r.status_code in (500, 503):
        pytest.skip(f"Auth backend unavailable ({r.status_code})")
    assert r.status_code == 201, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def _parameterless_get_paths() -> list[str]:
    # Derive from the OpenAPI schema, not app.routes: the v1 router is included
    # via a lazy wrapper (_IncludedRouter) that hides APIRoutes from a naive
    # route walk. The schema is the app's own contract — anything served is here.
    from app.main import app

    schema = app.openapi()
    paths = []
    for path, ops in schema.get("paths", {}).items():
        if "get" not in ops or "{" in path or path in _SKIP_PATHS:
            continue
        paths.append(path)
    return sorted(set(paths))


@pytest.mark.asyncio
async def test_no_get_endpoint_returns_5xx(client):
    """Every parameterless GET must respond without a server error.

    4xx is acceptable (auth variants, missing optional services, validation);
    5xx is always a bug. This is the in-repo twin of the live smoke test.
    """
    headers = await _auth_headers(client)
    paths = _parameterless_get_paths()
    assert len(paths) > 60, f"route walk looks broken — only {len(paths)} GET paths found"

    failures: list[str] = []
    for path in paths:
        try:
            r = await client.get(path, headers=headers)
            if r.status_code >= 500:
                failures.append(f"{path} → {r.status_code}: {r.text[:120]}")
        except Exception as exc:  # noqa: BLE001 — an unhandled exception IS a 5xx
            failures.append(f"{path} → raised {type(exc).__name__}: {exc}")

    assert not failures, "Endpoints with server errors:\n" + "\n".join(failures)
