"""The dashboard showed an empty book while the broker held one.

Measured 2026-07-28, live production:

    GET /api/v1/positions/          -> []
    GET /api/v1/positions/?account_id=<the user's alpaca account> -> []
    Alpaca account                  equity $21,752.63, cash $17,545.47
                                    i.e. ~$4.2k of OPEN positions

"Still no trades" was the correct read of the product surface, and wrong about
the system: the desks had placed three orders that same hour and two had
filled.

WHY. `.github/scripts/desk_order_placer.py` places orders straight at Alpaca
using the platform's own credentials. It never writes to the `Position` table,
and nothing else populates it. The endpoint's live branch could not cover for
that, because it required BOTH an explicit `account_id` AND a per-account
`encrypted_key` — and this deployment's Alpaca credentials live in the
environment, not on an Account row. So every default call fell through to an
empty table and reported an empty book.

Reading `settings.alpaca_api_key` here follows what analytics.py already does
in three places: in this deployment those env credentials ARE the trading
account. It is scoped to users who own an Alpaca account row so it stays an
account the caller is entitled to see.

The DB keeps priority — this is a fallback for "the table is empty", not a
replacement. A real `Position` row still wins, which keeps bot-managed
positions authoritative.
"""
from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_the_endpoint_answers_without_a_server_error(client):
    """Whatever the credential state, this must not 500."""
    r = await client.post("/api/v1/auth/demo")
    if r.status_code == 429:
        pytest.skip("auth rate-limited in this CI window")
    assert r.status_code == 200, r.text
    headers = {"Authorization": f"Bearer {r.json()['access_token']}"}

    resp = await client.get("/api/v1/positions/", headers=headers)
    assert resp.status_code < 500, resp.text
    assert isinstance(resp.json(), list)


@pytest.mark.asyncio
async def test_a_broker_outage_degrades_to_empty_not_a_500(client, monkeypatch):
    """Fail-soft is the point: an empty dashboard beats a broken one.

    Breaks the transport rather than the helper — patching the helper itself
    to raise would only prove that a raising function raises, which was the
    first draft of this test and tested nothing.
    """
    import httpx

    class _Broken:
        def __init__(self, *_a, **_k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_a):
            return False

        async def get(self, *_a, **_k):
            raise httpx.ConnectError("alpaca unreachable")

    monkeypatch.setattr(httpx, "AsyncClient", _Broken)

    r = await client.post("/api/v1/auth/demo")
    if r.status_code == 429:
        pytest.skip("auth rate-limited in this CI window")
    headers = {"Authorization": f"Bearer {r.json()['access_token']}"}
    resp = await client.get("/api/v1/positions/", headers=headers)
    assert resp.status_code == 200, resp.text
    assert resp.json() == []


@pytest.mark.asyncio
async def test_the_helper_returns_empty_without_credentials(monkeypatch):
    """No env credentials -> [] , never an exception."""
    from app.api.v1.positions import _live_platform_positions
    from app.config import settings

    monkeypatch.setattr(settings, "alpaca_api_key", "", raising=False)
    monkeypatch.setattr(settings, "alpaca_secret_key", "", raising=False)

    assert await _live_platform_positions(None, None) == []


def test_the_alpaca_shape_is_mapped_to_the_response_schema():
    """Field names differ between Alpaca and PositionOut — pin the mapping."""
    from app.api.v1.positions import PositionOut, _alpaca_position_to_out

    out = _alpaca_position_to_out({
        "asset_id": "abc", "symbol": "SHIB/USD", "qty": "1200000",
        "avg_entry_price": "0.0000271", "current_price": "0.0000283",
        "unrealized_pl": "14.40",
    })
    model = PositionOut(**out)
    assert model.symbol == "SHIB/USD"
    assert model.quantity == pytest.approx(1_200_000)
    assert model.avg_cost == pytest.approx(0.0000271)
    assert model.unrealized_pnl == pytest.approx(14.40)
    assert model.side == "long"


def test_a_short_position_is_labelled_short():
    from app.api.v1.positions import _alpaca_position_to_out

    out = _alpaca_position_to_out({"symbol": "SPY", "qty": "-5",
                                   "avg_entry_price": "600"})
    assert out["side"] == "short"
    assert out["quantity"] == -5
