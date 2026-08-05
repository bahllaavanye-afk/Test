"""No crypto backtest has ever produced a result, and nothing said so.

`quick-backtest.yml`'s job is named "Run backtests across all desks". Every
persisted result reads `desks: ['equity']`.

Measured 2026-08-05: `api.binance.com/api/v3/klines` returns **HTTP 451** —
Unavailable For Legal Reasons, i.e. geo-blocked from the runner region.
`fetch_crypto_ohlcv` returned a bare `None` on any non-200, and the caller does
`if not ohlcv: continue`, so the symbol was dropped with **no log line at all**.
Only exceptions were logged, and a 451 is not an exception.

So: the crypto section runs every 15 minutes, fetches nothing, prints nothing,
contributes nothing, and the workflow reports success. That is the same
green-looking absence this repo has spent the week paying down — this instance
had the added disguise of a config block (`SYMBOLS["crypto"]`) that looked wired.

Two changes: the status is printed (a 451 must never be silent again), and there
is a yfinance fallback. yfinance already powers the equity backtests in the same
process, so it is reachable from the runner even when Binance is not. Whether it
resolves BTC-USD there is unverified from the dev container — both hosts are
blocked here — so the log line is the part that matters most: the next run will
say which path it took.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import quick_backtest_runner as Q  # noqa: E402

_SRC = Path(__file__).resolve().parent / "quick_backtest_runner.py"


@pytest.mark.parametrize("binance,expected", [
    ("BTCUSDT", "BTC-USD"),
    ("ETHUSDT", "ETH-USD"),
    ("SOLUSDT", "SOL-USD"),
])
def test_the_symbol_mapping(binance, expected, monkeypatch):
    seen = {}
    monkeypatch.setattr(Q, "fetch_ohlcv", lambda s, **k: seen.setdefault("sym", s))
    Q._crypto_via_yfinance(binance)
    assert seen["sym"] == expected, (
        f"{binance} mapped to {seen['sym']}, not {expected} — yfinance uses "
        "BASE-USD, not the Binance pair name."
    )


def test_an_empty_symbol_does_not_query_a_bare_dash(monkeypatch):
    called = []
    monkeypatch.setattr(Q, "fetch_ohlcv", lambda s, **k: called.append(s))
    assert Q._crypto_via_yfinance("USDT") is None
    assert not called, "queried yfinance for '-USD' after stripping the whole symbol"


def test_a_non_200_is_logged_not_swallowed(monkeypatch, capsys):
    """The 451 was invisible for the life of this workflow."""
    class R:
        status_code = 451
        def json(self): return []
    monkeypatch.setattr(Q.requests, "get", lambda *a, **k: R())
    monkeypatch.setattr(Q, "fetch_ohlcv", lambda s, **k: None)
    Q.fetch_crypto_ohlcv("BTCUSDT")
    out = capsys.readouterr().out
    assert "451" in out, (
        "a non-200 from Binance produced no output. The caller's `continue` "
        "then drops the symbol silently, which is how this went unnoticed."
    )
    assert "geo-blocked" in out, "451 is not explained, so the cause stays a mystery"


def test_a_non_200_falls_back_rather_than_giving_up(monkeypatch):
    class R:
        status_code = 451
        def json(self): return []
    monkeypatch.setattr(Q.requests, "get", lambda *a, **k: R())
    monkeypatch.setattr(Q, "fetch_ohlcv",
                        lambda s, **k: {"close": [1.0, 2.0], "high": [], "low": [], "volume": []})
    got = Q.fetch_crypto_ohlcv("BTCUSDT")
    assert got and got["close"] == [1.0, 2.0], (
        "the fallback did not run — crypto stays permanently empty wherever "
        "Binance is blocked."
    )


def test_an_exception_also_falls_back(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("dns")
    monkeypatch.setattr(Q.requests, "get", boom)
    monkeypatch.setattr(Q, "fetch_ohlcv", lambda s, **k: {"close": [3.0], "high": [], "low": [], "volume": []})
    assert Q.fetch_crypto_ohlcv("ETHUSDT")["close"] == [3.0]


def test_a_healthy_binance_response_is_still_preferred(monkeypatch):
    """The fallback must not displace the primary source."""
    rows = [[0, "1", "3", "0.5", "2", "10"]] * 3
    class R:
        status_code = 200
        def json(self): return rows
    monkeypatch.setattr(Q.requests, "get", lambda *a, **k: R())
    monkeypatch.setattr(Q, "fetch_ohlcv",
                        lambda s, **k: pytest.fail("fell back despite a 200 from Binance"))
    got = Q.fetch_crypto_ohlcv("BTCUSDT")
    assert got["close"] == [2.0, 2.0, 2.0] and got["high"] == [3.0, 3.0, 3.0]


def test_the_crypto_section_is_still_wired_into_the_run():
    """A fixed fetcher is worthless if nothing calls it."""
    src = _SRC.read_text()
    assert 'SYMBOLS["crypto"]' in src, "the crypto symbol loop is gone"
    assert '"desk": "crypto"' in src, (
        "crypto results are no longer tagged, so they cannot reach the "
        "persisted desks list even when the fetch succeeds"
    )
