"""Crypto positions had NO stop-loss enforcement — silently.

Found in the live Render logs on 2026-07-28, dumped by a failing deploy:

    PositionMonitor: broker quote failed, skipping
      502: Alpaca quote failed for UNIUSD: 'UNIUSD'
      502: Alpaca quote failed for SHIBUSD: 'SHIBUSD'

Alpaca reports crypto POSITIONS slashless — `UNIUSD`, `SHIBUSD`, `AAVEUSD` —
while every price path in this system keys on the slashed pair, `UNI/USD`. So
BOTH lookups missed on every crypto position, every tick:

  Redis   `exchange_for("UNIUSD")` sees no "/" so returns "alpaca", reading
          `price:alpaca:UNIUSD` — but price_feed writes `price:crypto:UNI/USD`.
  broker  `_is_crypto("UNIUSD")` is False (no "/", and it does not end with
          BTC/ETH/SOL/DOGE), so the request went to the STOCK quote endpoint
          and raised KeyError on the missing symbol.

With no price, `_check_position_exits` returns early — so stop-loss and
take-profit were never evaluated for any crypto position. The account was
holding six of them at the time. A missing price is indistinguishable from a
cold cache, which is why this never surfaced as an error anyone read.

This is the same failure family as the `prices:{symbol}` topic-vs-key bug
already documented in app/tasks/CLAUDE.md: a silent miss that falls through to
a path that cannot work.

Candidates are DERIVED rather than matched against a hardcoded crypto
universe, so adding pairs needs no maintenance, and the original symbol is
always retained as a fallback — an equity that happens to end in "USD" still
resolves.
"""
from __future__ import annotations

import pytest

from app.tasks.position_monitor import _price_symbol_candidates


# ── the regression ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("slashless,expected", [
    ("UNIUSD", "UNI/USD"),
    ("SHIBUSD", "SHIB/USD"),
    ("AAVEUSD", "AAVE/USD"),
    ("BCHUSD", "BCH/USD"),
    ("DOGEUSD", "DOGE/USD"),
    ("ETHUSD", "ETH/USD"),
])
def test_the_live_crypto_positions_resolve_to_the_slashed_pair(slashless, expected):
    """These are the exact symbols the account held when this was found."""
    assert _price_symbol_candidates(slashless)[0] == expected


def test_the_slashed_form_is_tried_FIRST():
    """Order matters: the slashless form is what fails at the broker."""
    assert _price_symbol_candidates("UNIUSD") == ["UNI/USD", "UNIUSD"]


def test_the_original_is_still_retained_as_a_fallback():
    """Fail-safe: never make a working symbol unresolvable."""
    assert "UNIUSD" in _price_symbol_candidates("UNIUSD")


# ── equities must be untouched ───────────────────────────────────────────────

@pytest.mark.parametrize("sym", ["SPY", "AAPL", "QQQ", "IWM", "GLD", "TLT"])
def test_equities_are_returned_unchanged_and_alone(sym):
    """A single candidate means no extra broker call for the common case."""
    assert _price_symbol_candidates(sym) == [sym]


def test_an_already_slashed_symbol_is_left_alone():
    assert _price_symbol_candidates("BTC/USD") == ["BTC/USD"]


# ── shape and safety ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("junk", ["", "   ", None])
def test_empty_input_yields_no_candidates(junk):
    """The caller returns early on falsy symbols; this must not raise."""
    assert _price_symbol_candidates(junk) == []


def test_other_quote_currencies_are_handled():
    assert _price_symbol_candidates("BTCUSDT")[0] == "BTC/USDT"
    assert _price_symbol_candidates("BTCUSDC")[0] == "BTC/USDC"


def test_usdt_is_matched_before_usd():
    """USD is a suffix of USDT — matching USD first would give 'BTCUS/DT'."""
    assert _price_symbol_candidates("BTCUSDT")[0] == "BTC/USDT"


def test_a_bare_quote_currency_is_not_split_into_an_empty_base():
    """'USD' alone must not become '/USD'."""
    assert _price_symbol_candidates("USD") == ["USD"]


def test_symbols_are_upper_cased():
    assert _price_symbol_candidates("uniusd")[0] == "UNI/USD"


def test_the_redis_exchange_routing_now_agrees():
    """The derived form is what `exchange_for` needs to route to crypto.

    This is the half of the bug that the broker fix alone would not have
    covered — the Redis key was wrong too.
    """
    from app.redis_client import exchange_for, price_key

    cand = _price_symbol_candidates("UNIUSD")[0]
    assert exchange_for(cand) == "crypto"
    assert price_key(exchange_for(cand), cand) == "price:crypto:UNI/USD"
    # and the old behaviour, for contrast
    assert exchange_for("UNIUSD") == "alpaca"


# ── the gap the pricing fix EXPOSED ──────────────────────────────────────────
# Fixing the price lookup was necessary but not sufficient. `pos_exit:` configs
# are written ONLY by strategy_runner, on its own fills. The live paper orders
# are placed by the GitHub Actions desks (.github/scripts/desk_order_placer.py),
# whose workflow even sets REDIS_URL="" — so they write nothing.
#
# Every desk-placed position therefore reaches `if not exit_config: return` and
# is skipped, with NO stop-loss or take-profit enforcement. Measured live
# 2026-07-28: all 16 open positions, every sweep.
#
# That skip was a logger.debug — invisible in production, so an unmonitored
# position looked exactly like a monitored one. It is now a per-sweep warning
# carrying the count. Deliberately NOT applying a default stop: no global
# default exists here (bots carry per-template stop_loss_pct, strategy_runner
# uses signal.stop_loss), so inventing a threshold would start closing real
# positions on a number nobody chose.


class _StubBroker:
    def __init__(self, positions):
        self._positions = positions
        self.quoted: list[str] = []

    async def get_positions(self):
        return self._positions

    async def get_quote(self, symbol):
        self.quoted.append(symbol)
        if "/" in symbol or symbol.isalpha():
            class _Q:
                last = 1.0
                ask = 1.0
            return _Q()
        raise RuntimeError(f"Alpaca quote failed for {symbol}: '{symbol}'")


def _run_sweep(monkeypatch, positions):
    import asyncio

    import app.tasks.position_monitor as pm

    seen: list[tuple[str, dict]] = []

    class _Logger:
        def warning(self, msg, **kw):
            seen.append((msg, kw))

        def debug(self, *a, **kw):
            pass

        def error(self, *a, **kw):
            pass

        def info(self, *a, **kw):
            pass

    monkeypatch.setattr(pm, "logger", _Logger())
    broker = _StubBroker(positions)
    monitor = pm.PositionMonitor(broker, None, None)
    asyncio.run(monitor._check_all_positions())
    return seen, broker


def test_unmonitored_positions_are_reported_with_a_count(monkeypatch):
    seen, _ = _run_sweep(monkeypatch, [{"symbol": "UNIUSD"}, {"symbol": "SPY"}])
    warns = [(m, k) for m, k in seen if "NO exit config" in m]
    assert warns, f"no unmonitored warning emitted: {seen}"
    _, kw = warns[0]
    assert kw["unmonitored"] == 2 and kw["of_total"] == 2
    assert "UNIUSD" in kw["symbols"]


def test_the_crypto_symbols_no_longer_fail_to_price(monkeypatch):
    """The pricing fix, exercised through the real sweep.

    Before the fix these raised and produced a 'broker quote failed' warning,
    returning before the exit-config stage was ever reached.
    """
    seen, broker = _run_sweep(monkeypatch, [{"symbol": "UNIUSD"}, {"symbol": "SHIBUSD"}])
    assert not [m for m, _ in seen if "broker quote failed" in m], seen
    assert "UNI/USD" in broker.quoted and "SHIB/USD" in broker.quoted


def test_a_clean_sweep_emits_no_warning(monkeypatch):
    """No positions -> nothing to say."""
    seen, _ = _run_sweep(monkeypatch, [])
    assert not [m for m, _ in seen if "NO exit config" in m]
