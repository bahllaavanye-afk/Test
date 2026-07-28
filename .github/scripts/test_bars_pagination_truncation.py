"""A 429 mid-pagination silently deleted the back half of the alphabet.

`_get_bars_batch` requests many symbols in one call and follows
`next_page_token`. On ANY exception it used to `break` — keeping whatever
pages had already landed and abandoning the rest without saying so.

Alpaca paginates in SYMBOL ORDER, so that truncation is not random loss, it is
**alphabetical and deterministic**. Measured 2026-07-28 across three crypto
desk runs, the symbols that survived were exactly the alphabetically-first N
of the 20-symbol universe:

    09:27  bars_fetched=4   AAVE AVAX BAT BCH
    06:46  bars_fetched=5   AAVE AVAX BAT BCH BTC
    04:45  bars_fetched=11  AAVE AVAX BAT BCH BTC CRV DOGE DOT ETH GRT LINK

Each is a clean prefix — verified against the universe list printed in the
same run's failure message. So SHIB, SOL, SUSHI, UNI, XRP, XTZ, YFI, MKR and
LTC received **no bars at all**, and no ensemble ever voted on them. The desks
were not sampling the universe; they were reading the first page of it.

This was originally misdiagnosed as workflow contention (two desk workflows
racing for the rate limit). Decontending them was independently correct, but
the starvation survived it: the 09:27 run had no competing run and still kept
4 of 20. The retry below is the actual fix.
"""
from __future__ import annotations

import asyncio

import pytest

from desk_order_placer import (
    _BARS_MAX_RETRIES,
    _get_bars_batch,
    _is_rate_limited,
)

_CRYPTO = [f"{c}/USD" for c in
           ("AAVE", "AVAX", "BAT", "BCH", "BTC", "SHIB", "SOL", "UNI", "XRP", "YFI")]


# ── the throttle discriminator ───────────────────────────────────────────────

@pytest.mark.parametrize("exc", [
    Exception("HTTP Error 429: Too Many Requests"),
    Exception("429"),
    Exception("Too Many Requests"),
])
def test_a_throttle_is_recognised(exc):
    assert _is_rate_limited(exc)


@pytest.mark.parametrize("exc", [
    Exception("HTTP Error 404: Not Found"),
    Exception("HTTP Error 422: Unprocessable Entity"),
    Exception("asset MKR/USD is not active"),
    Exception("connection reset"),
])
def test_a_real_answer_is_not_retried(exc):
    """Retrying a 404/422 just burns the job's time budget for no gain."""
    assert not _is_rate_limited(exc)


# ── the regression ───────────────────────────────────────────────────────────

def _bars_page(symbols, token=None):
    return {
        "bars": {s: [{"t": "2026-07-01T00:00:00Z", "o": 1.0, "h": 1.0,
                      "l": 1.0, "c": 1.0, "v": 10.0}] for s in symbols},
        "next_page_token": token,
    }


def _run_with(responses, monkeypatch):
    """Drive _get_bars_batch against a scripted sequence of pages/exceptions."""
    import desk_order_placer as dop

    seq = list(responses)
    calls: list[dict] = []

    async def fake_get(path, params=None, data_api=False):
        calls.append(params or {})
        item = seq.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(dop, "_alpaca_get", fake_get)
    # Bind the real sleep BEFORE patching: `dop.asyncio` is the global asyncio
    # module, so a lambda calling `asyncio.sleep` would call its own patch.
    real_sleep = asyncio.sleep
    monkeypatch.setattr(dop.asyncio, "sleep", lambda *_: real_sleep(0))
    out = asyncio.run(_get_bars_batch(_CRYPTO))
    return out, calls


def test_a_throttled_second_page_no_longer_deletes_the_rest(monkeypatch):
    """THE BUG: page 2 gets a 429, and the back half of the alphabet vanishes."""
    first, rest = _CRYPTO[:4], _CRYPTO[4:]
    out, _ = _run_with([
        _bars_page(first, token="p2"),
        Exception("HTTP Error 429: Too Many Requests"),  # used to end it here
        _bars_page(rest),                                # now reached on retry
    ], monkeypatch)

    assert set(out) == set(_CRYPTO), f"missing {sorted(set(_CRYPTO) - set(out))}"


def test_the_symbols_that_used_to_be_lost_are_specifically_present(monkeypatch):
    """Named because these are the ones that never got bars in production."""
    first, rest = _CRYPTO[:4], _CRYPTO[4:]
    out, _ = _run_with([
        _bars_page(first, token="p2"),
        Exception("HTTP Error 429: Too Many Requests"),
        _bars_page(rest),
    ], monkeypatch)

    for sym in ("SHIB/USD", "SOL/USD", "UNI/USD", "XRP/USD", "YFI/USD"):
        assert sym in out, f"{sym} still starved"


def test_the_retry_re_requests_the_SAME_page(monkeypatch):
    """A retry that drops page_token would restart and loop forever."""
    first, rest = _CRYPTO[:4], _CRYPTO[4:]
    _, calls = _run_with([
        _bars_page(first, token="p2"),
        Exception("HTTP Error 429: Too Many Requests"),
        _bars_page(rest),
    ], monkeypatch)

    retried = [c for c in calls if c.get("page_token") == "p2"]
    assert len(retried) == 2, f"expected the p2 request twice, got {len(retried)}"


def test_a_non_throttle_error_still_stops_immediately(monkeypatch):
    """Only the throttle earns patience — everything else should fail fast."""
    out, calls = _run_with([
        _bars_page(_CRYPTO[:4], token="p2"),
        Exception("HTTP Error 422: Unprocessable Entity"),
    ], monkeypatch)

    assert set(out) == set(_CRYPTO[:4])
    assert len(calls) == 2, "a 422 must not be retried"


def test_the_retry_budget_is_finite(monkeypatch):
    """Persistent throttling must end the page, not hang the job."""
    out, calls = _run_with(
        [_bars_page(_CRYPTO[:4], token="p2")]
        + [Exception("HTTP Error 429: Too Many Requests")] * (_BARS_MAX_RETRIES + 1),
        monkeypatch,
    )
    assert set(out) == set(_CRYPTO[:4])
    assert len(calls) == _BARS_MAX_RETRIES + 2


def test_a_clean_multi_page_fetch_is_unaffected(monkeypatch):
    """No throttle, no retries, every symbol present."""
    out, calls = _run_with([
        _bars_page(_CRYPTO[:4], token="p2"),
        _bars_page(_CRYPTO[4:]),
    ], monkeypatch)
    assert set(out) == set(_CRYPTO)
    assert len(calls) == 2
