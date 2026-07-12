"""FX desk (OANDA) money-path tests — same rigor as the Alpaca desk suite."""
from __future__ import annotations

import importlib.util
import io
import json
import sys
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

import pytest

pytest.importorskip("pandas")

_MOD = Path(__file__).parent / "fx_desk.py"


def _load(monkeypatch=None):
    spec = importlib.util.spec_from_file_location("fx_desk_test", _MOD)
    m = importlib.util.module_from_spec(spec)
    sys.modules["fx_desk_test"] = m
    spec.loader.exec_module(m)  # type: ignore[union-attr]
    m.OANDA_API_KEY = "test-key"
    m.OANDA_ACCOUNT_ID = "101-001-1234567-001"
    return m


fx = _load()


def _dt(wd_name: str, hour: int, minute: int = 0) -> datetime:
    # anchor week: Mon 2026-07-06 .. Sun 2026-07-12
    days = {"mon": 6, "tue": 7, "wed": 8, "thu": 9, "fri": 10, "sat": 11, "sun": 12}
    return datetime(2026, 7, days[wd_name], hour, minute, tzinfo=timezone.utc)


def test_market_hours_24_5():
    assert fx.fx_market_open(_dt("wed", 3)) is True
    assert fx.fx_market_open(_dt("fri", 20, 59)) is True
    assert fx.fx_market_open(_dt("fri", 21)) is False          # Friday close
    assert fx.fx_market_open(_dt("sat", 12)) is False
    assert fx.fx_market_open(_dt("sun", 20, 59)) is False
    assert fx.fx_market_open(_dt("sun", 21, 6)) is True        # Sunday reopen


class _Resp:
    def __init__(self, payload): self._b = json.dumps(payload).encode()
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def read(self): return self._b


def test_fetch_candles_parses_ohlcv(monkeypatch):
    candles = [{"time": f"2026-01-{i%27+1:02d}T00:00:00Z", "complete": True, "volume": 1000,
                "mid": {"o": "1.10", "h": "1.11", "l": "1.09", "c": "1.105"}} for i in range(60)]
    monkeypatch.setattr("urllib.request.urlopen",
                        lambda req, timeout=20: _Resp({"candles": candles}))
    df = fx.fetch_candles("EUR_USD")
    assert df is not None and len(df) == 60
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert float(df["close"].iloc[-1]) == pytest.approx(1.105)


def test_fetch_candles_too_short_returns_none(monkeypatch):
    candles = [{"time": "2026-01-01T00:00:00Z", "complete": True, "volume": 1,
                "mid": {"o": "1", "h": "1", "l": "1", "c": "1"}}] * 10
    monkeypatch.setattr("urllib.request.urlopen",
                        lambda req, timeout=20: _Resp({"candles": candles}))
    assert fx.fetch_candles("EUR_USD") is None


def test_place_order_body_signs_units(monkeypatch):
    cap = {}
    def fake(req, timeout=20):
        cap["url"] = req.full_url
        cap["body"] = json.loads(req.data)
        cap["auth"] = req.get_header("Authorization")
        return _Resp({"orderFillTransaction": {"id": "42"}})
    monkeypatch.setattr("urllib.request.urlopen", fake)

    assert fx.place_order("EUR_USD", "buy", 1500) is not None
    o = cap["body"]["order"]
    assert o["units"] == "1500" and o["instrument"] == "EUR_USD" and o["type"] == "MARKET"
    assert cap["auth"] == "Bearer test-key"
    assert "/v3/accounts/101-001-1234567-001/orders" in cap["url"]

    fx.place_order("USD_JPY", "sell", 800)
    assert cap["body"]["order"]["units"] == "-800"              # shorts are negative


def _http_error(code, body=b'{"errorMessage":"x"}'):
    return urllib.error.HTTPError("u", code, "e", {}, io.BytesIO(body))


def test_oanda_retries_429_then_succeeds(monkeypatch):
    calls = {"n": 0}
    def fake(req, timeout=20):
        calls["n"] += 1
        if calls["n"] < 3:
            raise _http_error(429)
        return _Resp({"ok": True})
    monkeypatch.setattr("urllib.request.urlopen", fake)
    monkeypatch.setattr("time.sleep", lambda *_: None)
    assert fx._oanda("GET", "/x") == {"ok": True}
    assert calls["n"] == 3


def test_order_failure_degrades_to_none_with_reason(monkeypatch, capsys):
    monkeypatch.setattr("urllib.request.urlopen",
                        lambda req, timeout=20: (_ for _ in ()).throw(
                            _http_error(403, b'{"errorMessage":"insufficient authorization"}')))
    assert fx.place_order("EUR_USD", "buy", 100) is None
    out = capsys.readouterr().out
    assert "403" in out and "insufficient authorization" in out   # reason surfaced


def test_run_without_keys_exits_clean(monkeypatch, capsys):
    import asyncio
    m = _load()
    m.OANDA_API_KEY = ""
    m.OANDA_ACCOUNT_ID = ""
    assert asyncio.run(m.run()) == 0
    assert "keys absent" in capsys.readouterr().out
