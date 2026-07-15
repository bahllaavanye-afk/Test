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

import logging
from typing import List

import pytest
from starlette.testclient import TestClient

from app.main import app

# Configure a module‑level logger for structured error output.
_logger = logging.getLogger(__name__)
_handler = logging.StreamHandler()
_formatter = logging.Formatter(
    fmt='%(asctime)s %(levelname)s %(name)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
_handler.setFormatter(_formatter)
_logger.addHandler(_handler)
_logger.setLevel(logging.INFO)

_client = TestClient(app, raise_server_exceptions=False)

# Must-exist GET endpoints. Unauthenticated → 401/403 is acceptable; 404/500 is a bug.
CRITICAL_GETS: List[str] = [
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


def _perform_get(path: str):
    """Execute a GET request against the test client with robust error handling.

    Args:
        path: The endpoint path to request.

    Returns:
        The Starlette response object.

    Raises:
        AssertionError: If the request raises an unexpected exception.
    """
    try:
        response = _client.get(path)
        return response
    except Exception as exc:  # pragma: no cover – defensive programming
        # Log the exception with a structured payload for easier debugging.
        _logger.error(
            "Error during GET request",
            extra={
                "path": path,
                "exception_type": exc.__class__.__name__,
                "exception_message": str(exc),
            },
        )
        raise AssertionError(f"GET {path} raised an unexpected exception: {exc}") from exc


@pytest.mark.parametrize("path", CRITICAL_GETS)
def test_critical_get_resolves(path: str) -> None:
    resp = _perform_get(path)

    # Explicit checks with clear failure messages.
    assert resp.status_code != 404, f"GET {path} → 404 (router mounted but no handler?)"
    assert resp.status_code != 500, f"GET {path} → 500 (handler errored on a basic GET)"
    assert resp.status_code in (200, 401, 403, 422), (
        f"GET {path} → unexpected status code {resp.status_code}"
    )