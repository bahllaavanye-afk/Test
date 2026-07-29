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
from pydantic import BaseModel, Field, validator
from starlette.testclient import TestClient

from app.main import app

_client = TestClient(app, raise_server_exceptions=False)


class EndpointInfo(BaseModel):
    """Schema representing a critical GET endpoint that must be reachable."""

    path: str = Field(
        ...,
        description="Relative URL path for the critical GET endpoint.",
        example="/health",
    )

    @validator("path")
    def must_start_with_slash(cls, v: str) -> str:
        """Ensure the endpoint path is an absolute path starting with '/'."""
        if not v.startswith("/"):
            raise ValueError("endpoint path must start with '/'")
        return v


# Must-exist GET endpoints. Unauthenticated → 401/403 is acceptable; 404/500 is a bug.
CRITICAL_GETS = [
    EndpointInfo(path="/health"),
    EndpointInfo(path="/api/v1/leaderboard/entries"),
    EndpointInfo(path="/api/v1/leaderboard/summary"),
    EndpointInfo(path="/api/v1/strategies/"),
    EndpointInfo(path="/api/v1/strategies/available"),
    EndpointInfo(path="/api/v1/strategies/active"),
    EndpointInfo(path="/api/v1/strategies/desks"),
    EndpointInfo(path="/api/v1/risk/"),
    EndpointInfo(path="/api/v1/analytics/"),
    EndpointInfo(path="/api/v1/backtests/scenarios"),
    EndpointInfo(path="/api/v1/bots/"),
]


@pytest.mark.parametrize("endpoint", CRITICAL_GETS)
def test_critical_get_resolves(endpoint: EndpointInfo):
    resp = _client.get(endpoint.path)
    assert resp.status_code != 404, f"GET {endpoint.path} → 404 (router mounted but no handler?)"
    assert resp.status_code != 500, f"GET {endpoint.path} → 500 (handler errored on a basic GET)"
    assert resp.status_code in (200, 401, 403, 422), f"GET {endpoint.path} → unexpected {resp.status_code}"