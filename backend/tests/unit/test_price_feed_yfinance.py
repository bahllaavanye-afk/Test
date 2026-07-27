"""The yfinance fallback feed must not leak loops or fail silently.

`_yf_publish_sync` runs in an executor thread, once per symbol, every 60s,
forever. It used to hand-manage the event loop:

    loop = asyncio.new_event_loop()
    loop.run_until_complete(asyncio.gather(...))
    loop.close()                       # ordinary statement, not a finally

so any raise from `run_until_complete` skipped the close and left the loop for
the garbage collector, emitting `ResourceWarning: unclosed event loop` every
time, while the enclosing handler logged at DEBUG. And a feed publishing
nothing at all looked exactly like a healthy one.

NOTE: log assertions use `capsys`, not `caplog` — structlog does not propagate
to stdlib logging.
"""
from __future__ import annotations

import asyncio
import gc
import warnings

import pytest

from app.tasks import price_feed as mod


class _Cache:
    def __init__(self):
        self.published: list[tuple[str, dict]] = []

    async def set_price(self, source, symbol, quote):
        self.published.append((symbol, quote))


class _FastInfo:
    def __init__(self, last):
        self.last_price = last


def _install_yf(monkeypatch, last=None, raises=False):
    """Stub the `import yfinance as yf` that happens inside the function."""
    import sys
    import types

    fake = types.ModuleType("yfinance")

    class _Ticker:
        def __init__(self, sym):
            if raises:
                raise RuntimeError("yfinance is down")
            self.fast_info = _FastInfo(last)

    fake.Ticker = _Ticker
    monkeypatch.setitem(sys.modules, "yfinance", fake)


@pytest.fixture(autouse=True)
def _quiet_broadcast(monkeypatch):
    async def _noop(*_a, **_kw):
        return None

    monkeypatch.setattr(mod.manager, "broadcast", _noop, raising=False)


def test_publishes_a_quote(monkeypatch):
    _install_yf(monkeypatch, last=123.45)
    cache = _Cache()

    assert mod._yf_publish_sync("AAPL", cache) is True
    assert cache.published == [("AAPL", {"last": 123.45, "bid": 123.45, "ask": 123.45})]


def test_reports_failure_instead_of_looking_successful(monkeypatch):
    _install_yf(monkeypatch, raises=True)
    cache = _Cache()

    assert mod._yf_publish_sync("AAPL", cache) is False, (
        "a failed publish must be distinguishable from a successful one"
    )
    assert cache.published == []


def test_zero_price_is_not_published(monkeypatch):
    _install_yf(monkeypatch, last=0.0)
    cache = _Cache()

    assert mod._yf_publish_sync("AAPL", cache) is False
    assert cache.published == []


def test_a_failed_publish_closes_its_event_loop(monkeypatch):
    """The old form skipped `loop.close()` whenever the publish raised.

    That was NOT an unbounded descriptor leak — CPython's refcounting reclaims
    the loop through `BaseEventLoop.__del__` — but cleanup was left to the
    garbage collector, and every failure emitted `ResourceWarning: unclosed
    event loop` plus two unclosed sockets. `asyncio.run` closes deterministically.
    """
    _install_yf(monkeypatch, last=100.0)

    async def _boom(*_a, **_kw):
        raise RuntimeError("publish blew up")

    monkeypatch.setattr(mod, "_publish_quote", _boom)

    mod._yf_publish_sync("AAPL", _Cache())      # warm up any lazy imports
    gc.collect()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ResourceWarning)
        for _ in range(20):
            assert mod._yf_publish_sync("AAPL", _Cache()) is False
        gc.collect()

    unclosed = [w for w in caught if "unclosed event loop" in str(w.message)]
    assert not unclosed, (
        f"{len(unclosed)} event loops were left for the garbage collector to "
        "close — the loop must be closed on the error path, not finalized later"
    )


@pytest.mark.asyncio
async def test_a_wholly_dead_cycle_is_escalated(capsys, monkeypatch):
    """Per-symbol drops stay at debug; a feed publishing nothing must be loud."""
    monkeypatch.setattr(mod, "_yf_publish_sync", lambda *_a, **_kw: False)

    slept = {"n": 0}

    async def _stop_after_two_cycles(_seconds):
        slept["n"] += 1
        if slept["n"] >= 2:
            raise asyncio.CancelledError

    monkeypatch.setattr(mod.asyncio, "sleep", _stop_after_two_cycles)

    with pytest.raises(asyncio.CancelledError):
        await mod._yfinance_price_feed(["AAPL", "MSFT"])

    out = capsys.readouterr().out
    assert "published NOTHING this cycle" in out
    assert "consecutive_dead_cycles=2" in out, (
        "consecutive dead cycles must be counted, so a persistent outage escalates"
    )


@pytest.mark.asyncio
async def test_recovery_is_reported(capsys, monkeypatch):
    calls = {"n": 0}

    def _fail_then_succeed(*_a, **_kw):
        calls["n"] += 1
        return calls["n"] > 1        # first symbol of cycle 1 fails, rest publish

    monkeypatch.setattr(mod, "_yf_publish_sync", _fail_then_succeed)

    async def _stop(_seconds):
        raise asyncio.CancelledError

    monkeypatch.setattr(mod.asyncio, "sleep", _stop)

    with pytest.raises(asyncio.CancelledError):
        await mod._yfinance_price_feed(["AAPL", "MSFT"])

    # one of two published → not a dead cycle, so no error line
    assert "published NOTHING" not in capsys.readouterr().out
