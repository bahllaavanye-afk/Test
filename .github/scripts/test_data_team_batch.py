"""Regression test for the Data Team 429 fix.

The Data Team report showed many symbols as "fetch failed: HTTP Error 429"
because it fetched every symbol one-by-one against Alpaca's free tier. It now
batches into one request per asset class (crypto + stocks) with 429 backoff.
"""
from __future__ import annotations

import importlib.util
import sys
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

_MOD = Path(__file__).parent / "data_team.py"


def _load():
    sys.path.insert(0, str(Path(__file__).parent))
    spec = importlib.util.spec_from_file_location("data_team_under_test", _MOD)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)  # type: ignore[union-attr]
    return m


dt = _load()


def _fresh_bars(n=60):
    today = datetime.now(timezone.utc).strftime("%Y-%m-%dT00:00:00Z")
    return [{"t": today, "o": 1, "h": 2, "l": 1, "c": 1.5, "v": 100} for _ in range(n)]


def test_fetch_all_bars_batches_one_call_per_asset_class(monkeypatch):
    calls = []

    def fake_get(path, params):
        calls.append((path, params["symbols"]))
        syms = params["symbols"].split(",")
        return {"bars": {s: _fresh_bars() for s in syms}}

    monkeypatch.setattr(dt, "_get", fake_get)
    out = dt.fetch_all_bars(["BTC/USD", "ETH/USD", "AAPL", "MSFT", "SPY"])
    assert len(calls) == 2, calls
    assert {"BTC/USD", "ETH/USD", "AAPL", "MSFT", "SPY"} == set(out)
    assert all(len(v) == 60 for v in out.values())


def test_batch_failure_marks_symbols_fetch_failed_not_healthy(monkeypatch):
    def boom(path, params):
        raise urllib.error.HTTPError(path, 429, "Too Many Requests", {}, None)

    monkeypatch.setattr(dt, "_get", boom)
    out = dt.fetch_all_bars(["BTC/USD", "AAPL"])
    # a failed batch must yield None (fetch-failed), never an empty 'healthy' list
    assert out["BTC/USD"] is None and out["AAPL"] is None
    r = dt.check_symbol("AAPL", None)
    assert r["ok"] is False and "rate-limited" in r["reason"]


def test_check_symbol_flags_stale_and_short(monkeypatch):
    # fresh & long → healthy
    assert dt.check_symbol("BTC/USD", _fresh_bars(60))["ok"] is True
    # too few rows → not ok
    assert dt.check_symbol("BTC/USD", _fresh_bars(10))["ok"] is False
    # stale last bar → not ok
    old = "2026-01-01T00:00:00Z"
    stale = [{"t": old, "o": 1, "h": 2, "l": 1, "c": 1.5, "v": 100} for _ in range(60)]
    r = dt.check_symbol("BTC/USD", stale)
    assert r["ok"] is False and "stale" in r["reason"]


def test_get_retries_on_429(monkeypatch):
    attempts = {"n": 0}

    class _Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return b'{"bars": {}}'

    def fake_urlopen(req, timeout=15):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise urllib.error.HTTPError(req.full_url, 429, "Too Many Requests", {}, None)
        return _Resp()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("time.sleep", lambda *_: None)
    assert dt._get("/x", {"symbols": "AAPL"}) == {"bars": {}}
    assert attempts["n"] == 3
