"""Money-path tests: _place_order / _alpaca_post_sync against a mock Alpaca.

This is the path where signals become dollars. It had ZERO direct tests, which
is exactly why the first-ever order attempt shipped blind into a 403. These pin:
order body construction (pricing, rounding, tif, client id), and that every
broker failure mode (403 w/ body, 429, timeout, malformed JSON) degrades to
None + a printed reason — never an unhandled crash that kills the desk run.
"""
from __future__ import annotations

import asyncio
import importlib.util
import io
import json
import urllib.error
from pathlib import Path

import pytest

pytest.importorskip("pandas")

_MOD = Path(__file__).parent / "desk_order_placer.py"


def _load():
    spec = importlib.util.spec_from_file_location("dop_money_test", _MOD)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)  # type: ignore[union-attr]
    return m


dop = _load()


class _Resp:
    def __init__(self, payload: dict):
        self._b = json.dumps(payload).encode()
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def read(self): return self._b


def _capture_urlopen(captured: dict, payload: dict | None = None, exc: Exception | None = None):
    def fake(req, timeout=8):
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data) if req.data else None
        captured["ua"] = req.get_header("User-agent")
        if exc:
            raise exc
        return _Resp(payload if payload is not None else {"id": "ord-1", "status": "accepted"})
    return fake


# ── happy path: body construction rules ──────────────────────────────────────

def test_crypto_limit_order_body(monkeypatch):
    cap: dict = {}
    monkeypatch.setattr("urllib.request.urlopen", _capture_urlopen(cap))
    out = asyncio.run(dop._place_order("BTC/USD", "buy", 1000.0, limit_price=50_000.0,
                                       client_order_id="qe-strategy-name-that-is-way-too-long-for-alpaca-limits"))
    assert out and out["id"] == "ord-1"
    b = cap["body"]
    assert b["symbol"] == "BTC/USD" and b["side"] == "buy"
    assert b["time_in_force"] == "gtc"                      # crypto = gtc
    assert b["type"] == "limit"
    assert float(b["limit_price"]) == pytest.approx(50_050.0)   # 0.1% through-market on buys
    assert float(b["qty"]) == pytest.approx(1000.0 / 50_050.0, rel=1e-4)
    assert len(b["client_order_id"]) <= 48                  # Alpaca cap respected
    assert "/v2/orders" in cap["url"]
    assert cap["ua"] and "Mozilla" in cap["ua"]             # not the urllib default UA


def test_equity_market_order_uses_notional_and_day_tif(monkeypatch):
    cap: dict = {}
    monkeypatch.setattr("urllib.request.urlopen", _capture_urlopen(cap))
    out = asyncio.run(dop._place_order("SPY", "sell", 500.0))
    assert out is not None
    b = cap["body"]
    assert b["time_in_force"] == "day"
    assert b["type"] == "market"
    assert b["notional"] == "500.0" and "qty" not in b


def test_sell_limit_prices_through_market_downward(monkeypatch):
    cap: dict = {}
    monkeypatch.setattr("urllib.request.urlopen", _capture_urlopen(cap))
    asyncio.run(dop._place_order("SPY", "sell", 500.0, limit_price=100.0))
    assert float(cap["body"]["limit_price"]) == pytest.approx(99.9)


def test_crypto_market_order_sizes_from_quote(monkeypatch):
    cap: dict = {}
    monkeypatch.setattr("urllib.request.urlopen", _capture_urlopen(cap))
    async def fake_get(path, params=None, data_api=False):
        assert "latest/quotes" in path
        return {"quotes": {"ETH/USD": {"ap": 2000.0}}}
    monkeypatch.setattr(dop, "_alpaca_get", fake_get)
    asyncio.run(dop._place_order("ETH/USD", "buy", 400.0))
    assert cap["body"]["type"] == "market"
    assert float(cap["body"]["qty"]) == pytest.approx(0.2)


def test_crypto_market_order_refuses_zero_quote(monkeypatch):
    cap: dict = {}
    monkeypatch.setattr("urllib.request.urlopen", _capture_urlopen(cap))
    async def fake_get(path, params=None, data_api=False):
        return {"quotes": {"ETH/USD": {"ap": 0}}}
    monkeypatch.setattr(dop, "_alpaca_get", fake_get)
    assert asyncio.run(dop._place_order("ETH/USD", "buy", 400.0)) is None
    assert "body" not in cap                                # no POST was made


# ── failure modes: degrade to None, never crash the desk ─────────────────────

def _http_error(code: int, body: bytes = b'{"message":"forbidden."}'):
    return urllib.error.HTTPError("https://paper-api.alpaca.markets/v2/orders",
                                  code, "err", {}, io.BytesIO(body))

def test_403_returns_none_and_prints_alpaca_reason(monkeypatch, capsys):
    monkeypatch.setattr("urllib.request.urlopen",
                        _capture_urlopen({}, exc=_http_error(403, b'{"message":"crypto not enabled"}')))
    out = asyncio.run(dop._place_order("BTC/USD", "buy", 100.0, limit_price=50_000.0))
    assert out is None
    printed = capsys.readouterr().out
    assert "403" in printed and "crypto not enabled" in printed   # the diagnostic that was missing


def test_429_returns_none_not_crash(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", _capture_urlopen({}, exc=_http_error(429)))
    assert asyncio.run(dop._place_order("SPY", "buy", 100.0, limit_price=500.0)) is None


def test_network_timeout_returns_none(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen",
                        _capture_urlopen({}, exc=urllib.error.URLError("timed out")))
    assert asyncio.run(dop._place_order("SPY", "buy", 100.0, limit_price=500.0)) is None


def test_malformed_json_response_returns_none(monkeypatch):
    class _Bad:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return b"<html>cloudflare says no</html>"
    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout=8: _Bad())
    assert asyncio.run(dop._place_order("SPY", "buy", 100.0, limit_price=500.0)) is None
