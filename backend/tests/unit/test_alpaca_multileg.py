"""Alpaca multi-leg options order layer (IMPROVEMENTS P0: 'kills the
TradeStation dependency').

The implementation existed and was wired into the bot engine, but had ZERO
tests — the exact gap class that let the truncated-broker and 403-money-path
bugs ship. These pin: OCC symbol construction, delta-nearest contract picking,
the mleg payload shape Alpaca requires, and that unresolvable legs degrade to
None (alert) instead of sending a partial spread.
"""
from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.brokers.alpaca_orders import (
    build_occ_symbol,
    pick_contract_by_delta,
    submit_alpaca_multileg_order,
)


# ── OCC symbol construction ───────────────────────────────────────────────────

def test_occ_symbol_call():
    assert build_occ_symbol("SPY", date(2026, 7, 10), 620, "call") == "SPY260710C00620000"


def test_occ_symbol_put_fractional_strike():
    # strike*1000, zero-padded to 8 — 462.5 → 00462500
    assert build_occ_symbol("qqq", date(2026, 12, 18), 462.5, "put") == "QQQ261218P00462500"


# ── Delta-nearest contract picking ────────────────────────────────────────────

def _snap(delta):
    return {"greeks": {"delta": delta}}


def test_pick_contract_nearest_abs_delta():
    snaps = {
        "SPY..C00600000": _snap(0.62),
        "SPY..C00620000": _snap(0.48),
        "SPY..C00640000": _snap(0.31),
    }
    assert pick_contract_by_delta(snaps, 0.30, "call") == "SPY..C00640000"


def test_pick_contract_uses_abs_for_puts_and_skips_missing_greeks():
    snaps = {
        "SPY..P00580000": _snap(-0.16),
        "SPY..P00600000": _snap(-0.35),
        "SPY..P00560000": {"greeks": {}},       # no delta — must be skipped
        "SPY..P00550000": None,                  # malformed — must be skipped
    }
    assert pick_contract_by_delta(snaps, 0.16, "put") == "SPY..P00580000"


def test_pick_contract_empty_returns_none():
    assert pick_contract_by_delta({}, 0.3, "call") is None


# ── submit_alpaca_multileg_order ──────────────────────────────────────────────

def _account():
    return SimpleNamespace(id="acct-1", mode="paper", broker="alpaca")


class _Resp:
    def __init__(self, code=200, body=None):
        self.status_code = code
        self._body = body or {"id": "mleg-1", "status": "accepted"}
        self.text = str(self._body)

    def json(self):
        return self._body


@pytest.mark.asyncio
async def test_multileg_builds_mleg_payload():
    captured = {}

    async def fake_post(self, url, json=None, headers=None):
        captured["url"] = url
        captured["payload"] = json
        return _Resp(200)

    legs = [
        {"side": "sell", "option_type": "put", "dte": 30, "delta": 0.16},
        {"side": "buy", "option_type": "put", "dte": 30, "delta": 0.10, "ratio": 1},
    ]
    with patch("app.brokers.alpaca_orders.resolve_leg_symbol",
               new=AsyncMock(side_effect=["SPY260814P00580000", "SPY260814P00560000"])), \
         patch("app.brokers.alpaca_orders._headers", new=AsyncMock(return_value={})), \
         patch("app.brokers.alpaca_orders._base_url", return_value="https://paper-api.alpaca.markets"), \
         patch("httpx.AsyncClient.post", new=fake_post):
        out = await submit_alpaca_multileg_order(_account(), "SPY", legs, quantity=2)

    assert out and out["id"] == "mleg-1"
    p = captured["payload"]
    assert p["order_class"] == "mleg"
    assert p["qty"] == "2"
    assert captured["url"].endswith("/v2/orders")
    assert [l["symbol"] for l in p["legs"]] == ["SPY260814P00580000", "SPY260814P00560000"]
    assert p["legs"][0]["position_intent"] == "sell_to_open"
    assert p["legs"][1]["position_intent"] == "buy_to_open"
    assert p["legs"][0]["ratio_qty"] == "1"


@pytest.mark.asyncio
async def test_multileg_unresolvable_leg_sends_nothing():
    posted = []

    async def fake_post(self, url, json=None, headers=None):
        posted.append(url)
        return _Resp(200)

    legs = [
        {"side": "sell", "option_type": "put", "dte": 30, "delta": 0.16},
        {"side": "buy", "option_type": "put", "dte": 30, "delta": 0.10},
    ]
    with patch("app.brokers.alpaca_orders.resolve_leg_symbol",
               new=AsyncMock(side_effect=["SPY260814P00580000", None])), \
         patch("httpx.AsyncClient.post", new=fake_post):
        out = await submit_alpaca_multileg_order(_account(), "SPY", legs)

    assert out is None
    assert posted == []          # NEVER send a partial spread


@pytest.mark.asyncio
async def test_multileg_broker_rejection_returns_none():
    async def fake_post(self, url, json=None, headers=None):
        return _Resp(422, {"message": "option level too low"})

    legs = [{"side": "buy", "option_type": "call", "dte": 30, "delta": 0.5}]
    with patch("app.brokers.alpaca_orders.resolve_leg_symbol",
               new=AsyncMock(return_value="SPY260814C00620000")), \
         patch("app.brokers.alpaca_orders._headers", new=AsyncMock(return_value={})), \
         patch("app.brokers.alpaca_orders._base_url", return_value="https://paper-api.alpaca.markets"), \
         patch("httpx.AsyncClient.post", new=fake_post):
        assert await submit_alpaca_multileg_order(_account(), "SPY", legs) is None
