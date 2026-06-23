"""Unit tests for the Kalshi public market-data endpoint (issue #131).

Calls the endpoint function directly with a mocked httpx client (no auth, no
network). Verifies the Kalshi v2 schema (`*_dollars` / `*_fp` fields) is
normalised to the same shape as /market-data/polymarket.
"""
import pytest

from app.api.v1 import market_data as md


class _FakeResp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def json(self):
        return self._payload


class _FakeClient:
    def __init__(self, resp):
        self._resp = resp

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, url, params=None):
        return self._resp


_SAMPLE = {
    "markets": [
        {
            "ticker": "POTUS-2028",
            "title": "Will X win the 2028 election?",
            "last_price_dollars": "0.4200",
            "yes_bid_dollars": "0.41",
            "yes_ask_dollars": "0.43",
            "volume_24h_fp": "1500.00",
            "liquidity_dollars": "9000.0",
            "open_interest_fp": "320.0",
            "close_time": "2028-11-07T00:00:00Z",
            "status": "active",
        },
        {
            "ticker": "BTC-100K",
            "title": "Bitcoin above $100k?",
            "last_price_dollars": "0.0000",          # no last trade → use bid/ask mid
            "yes_bid_dollars": "0.60",
            "yes_ask_dollars": "0.62",
            "volume_24h_fp": "500.0",
            "liquidity_dollars": "2000.0",
            "close_time": "2026-12-31T00:00:00Z",
            "status": "active",
        },
    ]
}


def _patch(monkeypatch, payload, status=200):
    monkeypatch.setattr(md.httpx, "AsyncClient", lambda *a, **k: _FakeClient(_FakeResp(payload, status)))


@pytest.mark.asyncio
async def test_kalshi_normalizes_dollar_fields(monkeypatch):
    _patch(monkeypatch, _SAMPLE)
    out = await md.get_kalshi_markets(filter="", sort="volume", limit=50, current_user=None)
    assert len(out) == 2
    top = out[0]  # sorted by volume desc → POTUS (1500) first
    assert top["id"] == "POTUS-2028"
    assert top["yes_price"] == 0.42                  # last_price_dollars, already 0–1
    assert top["no_price"] == round(1 - 0.42, 4)
    assert top["volume_24h"] == 1500.0
    assert top["liquidity"] == 9000.0
    assert top["category"] == "politics"
    assert top["active"] is True and top["closed"] is False


@pytest.mark.asyncio
async def test_kalshi_uses_bidask_mid_when_no_last(monkeypatch):
    _patch(monkeypatch, _SAMPLE)
    out = await md.get_kalshi_markets(filter="", sort="volume", limit=50, current_user=None)
    btc = next(m for m in out if m["id"] == "BTC-100K")
    assert btc["yes_price"] == 0.61                  # (0.60 + 0.62) / 2
    assert btc["category"] == "crypto"


@pytest.mark.asyncio
async def test_kalshi_filter_matches_title(monkeypatch):
    _patch(monkeypatch, _SAMPLE)
    out = await md.get_kalshi_markets(filter="bitcoin", sort="volume", limit=50, current_user=None)
    assert len(out) == 1 and out[0]["id"] == "BTC-100K"


@pytest.mark.asyncio
async def test_kalshi_non_200_returns_empty(monkeypatch):
    _patch(monkeypatch, {}, status=503)
    out = await md.get_kalshi_markets(filter="", sort="volume", limit=50, current_user=None)
    assert out == []


@pytest.mark.asyncio
async def test_kalshi_upstream_error_returns_empty(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("network down")
    monkeypatch.setattr(md.httpx, "AsyncClient", _boom)
    out = await md.get_kalshi_markets(filter="", sort="volume", limit=50, current_user=None)
    assert out == []
