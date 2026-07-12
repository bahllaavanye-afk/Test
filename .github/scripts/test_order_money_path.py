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


# ── Negative-cash auto-recovery (the real first-trade blocker) ────────────────

def _reload():
    spec = importlib.util.spec_from_file_location("dop_recover_test", _MOD)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)  # type: ignore[union-attr]
    return m


def test_recovery_flattens_on_negative_cash_zero_bp(monkeypatch):
    m = _reload()
    calls = []
    def fake_delete(path):
        calls.append(path)
        return {"status": 207, "body": [{"symbol": "AAPL"}, {"symbol": "SPY"}]}
    monkeypatch.setattr(m, "_alpaca_delete_sync", fake_delete)
    acct = {"cash": "-25207.26", "non_marginable_buying_power": "0"}
    assert asyncio.run(m.recover_negative_cash(acct)) is True
    assert "/v2/orders" in calls[0]                       # cancels orders first
    assert "/v2/positions" in calls[1]                    # then closes positions


def test_recovery_never_touches_healthy_account(monkeypatch):
    m = _reload()
    monkeypatch.setattr(m, "_alpaca_delete_sync",
                        lambda p: (_ for _ in ()).throw(AssertionError("must not be called")))
    assert asyncio.run(m.recover_negative_cash({"cash": "90000", "non_marginable_buying_power": "90000"})) is False
    # negative cash but SOME buying power → not the broken state either
    assert asyncio.run(m.recover_negative_cash({"cash": "-100", "non_marginable_buying_power": "500"})) is False


def test_recovery_refuses_non_paper_endpoint(monkeypatch):
    m = _reload()
    monkeypatch.setattr(m, "ALPACA_PAPER_BASE", "https://api.alpaca.markets")
    monkeypatch.setattr(m, "_alpaca_delete_sync",
                        lambda p: (_ for _ in ()).throw(AssertionError("must not be called")))
    assert asyncio.run(m.recover_negative_cash({"cash": "-1", "non_marginable_buying_power": "0"})) is False


def test_recovery_kill_switch(monkeypatch):
    m = _reload()
    monkeypatch.setattr(m, "AUTO_FLATTEN_ON_NEGATIVE_CASH", False)
    monkeypatch.setattr(m, "_alpaca_delete_sync",
                        lambda p: (_ for _ in ()).throw(AssertionError("must not be called")))
    assert asyncio.run(m.recover_negative_cash({"cash": "-1", "non_marginable_buying_power": "0"})) is False


# ── Cancel-replace execution (_ensure_filled) ─────────────────────────────────

def test_ensure_filled_returns_fill_when_limit_fills(monkeypatch):
    m = _reload()
    monkeypatch.setattr(m, "FILL_POLL_S", 0)
    seq = iter([{"id": "o1", "status": "open"}, {"id": "o1", "status": "filled"}])
    async def fake_get(path, params=None, data_api=False): return next(seq)
    monkeypatch.setattr(m, "_alpaca_get", fake_get)
    out = asyncio.run(m._ensure_filled({"id": "o1", "type": "limit"}, "BTC/USD", "buy", 100))
    assert out["status"] == "filled"


def test_ensure_filled_replaces_stale_limit_with_market(monkeypatch):
    m = _reload()
    monkeypatch.setattr(m, "FILL_POLL_S", 0)
    monkeypatch.setattr(m, "FILL_WAIT_S", 0)   # immediately stale
    cancelled, placed = [], []
    monkeypatch.setattr(m, "_alpaca_delete_sync", lambda p: cancelled.append(p) or {"status": 204})
    async def fake_place(symbol, side, notional, limit_price=None, client_order_id=None):
        placed.append((symbol, side, notional)); return {"id": "o2", "type": "market", "status": "accepted"}
    monkeypatch.setattr(m, "_place_order", fake_place)
    out = asyncio.run(m._ensure_filled({"id": "o1", "type": "limit"}, "BTC/USD", "buy", 100))
    assert cancelled and "o1" in cancelled[0]
    assert placed == [("BTC/USD", "buy", 100)]
    assert out["id"] == "o2"


def test_ensure_filled_double_fill_guard_on_cancel_race(monkeypatch):
    m = _reload()
    monkeypatch.setattr(m, "FILL_POLL_S", 0)
    monkeypatch.setattr(m, "FILL_WAIT_S", 0)
    def cancel_fails(p): raise RuntimeError("422 order not cancelable")
    monkeypatch.setattr(m, "_alpaca_delete_sync", cancel_fails)
    async def fake_get(path, params=None, data_api=False): return {"id": "o1", "status": "filled"}
    monkeypatch.setattr(m, "_alpaca_get", fake_get)
    async def must_not_place(*a, **k): raise AssertionError("double order!")
    monkeypatch.setattr(m, "_place_order", must_not_place)
    out = asyncio.run(m._ensure_filled({"id": "o1", "type": "limit"}, "BTC/USD", "buy", 100))
    assert out["status"] == "filled"          # fill kept, no replacement sent


def test_ensure_filled_passthrough_for_market_orders(monkeypatch):
    m = _reload()
    o = {"id": "o1", "type": "market", "status": "accepted"}
    assert asyncio.run(m._ensure_filled(o, "SPY", "buy", 100)) is o
