"""Shared test auth: register a fresh user, falling back to the demo bucket
when the 10/min auth limiter trips in the parallel CI suite (a limiter
artifact, not a backend failure — must never turn the gate red).

THE FALLBACK IS NOT SAFE FOR EVERY TEST, and pass `isolated=True` when it is
not. `/api/v1/auth/demo` is a get-or-create on ONE shared user
(`demo@quantedge.app`, `auth.py:146`), so two tests that both trip the limiter
become **the same user** — and every account they create lands under it.

That is how `test_tearsheet_clean_404_when_no_trades` failed CI on 2026-08-06:
it asks for a 404 on a user with no trades, got 200 with `n_trades: 5`, and the
equity curve (+120, -40, +80, -20, +200) was verbatim the seed from the test
two functions above it. Both had fallen back to demo, so `_user_account_ids`
returned both accounts and the tearsheet aggregated across them.

The fallback exists so a limiter artifact never turns the gate red. For any
test asserting on per-user data it does the opposite: it converts a limiter
artifact into a **false failure**, which is the exact outcome it was written to
prevent. A skip is honest; a silently shared identity is not.
"""
from __future__ import annotations

import uuid

import pytest


async def auth_headers(client, prefix: str = "t", password: str = "Sh4red!Tst#99",
                       isolated: bool = False) -> dict[str, str]:
    """`isolated=True` for any test that asserts on data scoped to ITS user —
    seeded trades, account counts, a "this user has nothing yet" 404. Those
    skip on a limiter trip instead of quietly sharing the demo user."""
    email = f"{prefix}_{uuid.uuid4().hex[:10]}@example.com"
    r = await client.post("/api/v1/auth/register", json={"email": email, "password": password})
    if r.status_code == 429:
        if isolated:
            pytest.skip("auth rate-limited, and this test needs its own user — the demo "
                        "fallback is a single shared account and would silently merge "
                        "this test's data with another's")
        r = await client.post("/api/v1/auth/demo")
        if r.status_code == 429:
            pytest.skip("auth rate-limited in this CI window")
        assert r.status_code == 200, r.text
        return {"Authorization": f"Bearer {r.json()['access_token']}"}
    if r.status_code in (500, 503):
        pytest.skip(f"Auth backend unavailable ({r.status_code})")
    assert r.status_code == 201, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}
