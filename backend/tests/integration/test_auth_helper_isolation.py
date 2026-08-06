"""The demo fallback must never silently substitute a shared identity.

`auth_headers` falls back to `/api/v1/auth/demo` when the 10/min auth limiter
trips, so a limiter artifact does not turn CI red. But that endpoint is a
get-or-create on ONE user (`demo@quantedge.app`), so two tests that both fall
back become the same user and their accounts merge.

Measured 2026-08-06 (CI run 31073616242): `test_tearsheet_clean_404_when_no_trades`
asked for a 404 on a user with no trades and got 200 with `n_trades: 5`. The
equity curve was verbatim the seed from the test two functions above it:

    +120, -40, +80, -20, +200   →   100120, 100080, 100160, 100140, 100340

A guard written so a flake could not fail the build had turned a flake into a
failure by another route.
"""
from __future__ import annotations

import asyncio

import pytest

from tests.integration._auth_helper import auth_headers


class _Resp:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = str(payload)

    def json(self):
        return self._payload


class _Client:
    """Registers always rate-limit; the demo bucket always succeeds."""

    def __init__(self):
        self.paths = []

    async def post(self, path, json=None):
        self.paths.append(path)
        if path.endswith("/auth/demo"):
            return _Resp(200, {"access_token": "demo-token"})
        return _Resp(429)


def test_isolated_skips_rather_than_sharing_the_demo_user():
    # `Skipped` is what pytest.skip() raises. Caught explicitly, or it would
    # skip THIS test — which would look like a pass and assert nothing.
    from _pytest.outcomes import Skipped

    client = _Client()
    with pytest.raises(Skipped) as exc:
        asyncio.run(auth_headers(client, prefix="x", isolated=True))
    assert "own user" in str(exc.value)
    assert not any(p.endswith("/auth/demo") for p in client.paths), (
        "an isolated test must not reach the shared demo user at all")


def test_non_isolated_still_falls_back_so_a_limiter_trip_stays_green():
    """The fallback's original purpose is intact for tests that only need SOME
    authenticated identity — removing it would trade one flake class for another.

    The signature default is asserted separately: flipping it to True makes this
    test SKIP rather than fail, and a skip proves nothing."""
    import inspect

    assert inspect.signature(auth_headers).parameters["isolated"].default is False, (
        "isolated must default to False — most tests only need an identity, and "
        "defaulting to True would skip them all whenever the limiter trips")

    client = _Client()
    headers = asyncio.run(auth_headers(client, prefix="x"))
    assert headers == {"Authorization": "Bearer demo-token"}
    assert any(p.endswith("/auth/demo") for p in client.paths)


def test_the_tests_that_assert_on_their_own_data_opt_in():
    """Named explicitly: each of these seeds rows under its own user and then
    asserts on the counts, so a shared identity corrupts the assertion."""
    from pathlib import Path
    here = Path(__file__).resolve().parent
    for name in ("test_analytics_honest.py", "test_bot_performance.py",
                 "test_bot_activity.py"):
        src = (here / name).read_text()
        assert "isolated=True" in src, f"{name} asserts on per-user data without isolated=True"
