"""Desk order sizing must not ask for money the account doesn't have.

Live evidence from the crypto 24x7 desk, 2026-07-27 08:40 UTC:

    403: insufficient balance for USD (requested: 134.58, available: 6.71,
         balance: -8287.81)
    422: asset MKR/USD is not active
    place_order failed SHIB/USD buy: float division by zero
    Done. 1 orders placed across 9 desks.

Three separate defects behind that:

1. `cash_capped_notional()` existed for exactly this, but `run_desk` passed
   `desk.notional_usd` RAW — so every order asked for $135 against $6.71 of
   buying power.
2. `round(limit_price * 1.001, 2)` flattens sub-cent crypto to 0.0. SHIB at
   ~$0.00001 became a divide-by-zero, swallowed as a generic failure.
3. A desk that funded nothing still ended with a tidy "✓ Place orders".
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("httpx")

_MOD = Path(__file__).resolve().parent / "desk_order_placer.py"
_spec = importlib.util.spec_from_file_location("dop_test", _MOD)
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)  # type: ignore[union-attr]


# ── sub-cent crypto precision ────────────────────────────────────────────────

def test_shib_limit_price_does_not_round_to_zero():
    """The exact live failure: SHIB ~ $0.00001 with 2dp rounding -> 0.0."""
    shib = 0.00001
    assert round(shib * 1.001, 2) == 0.0, "pinning the old behaviour"

    prec = m._price_precision(shib * 1.001, is_crypto=True)
    lp = round(shib * 1.001, prec)
    assert lp > 0, "a positive price must never round to zero"
    # and the division that used to raise now works
    assert 25.0 / lp > 0


@pytest.mark.parametrize("price,expected_nonzero", [
    (60_000.0, True),     # BTC
    (3_500.0, True),      # ETH
    (1.02, True),         # stablecoin-ish
    (0.05, True),
    (0.00042, True),
    (0.00001, True),      # SHIB
    (0.000000123, True),  # deep sub-cent
])
def test_crypto_prices_survive_rounding(price, expected_nonzero):
    lp = round(price, m._price_precision(price, is_crypto=True))
    assert (lp > 0) is expected_nonzero, f"{price} rounded to {lp}"


def test_equities_still_round_to_cents():
    """Equity quotes are in cents — precision must not drift."""
    assert m._price_precision(187.34, is_crypto=False) == 2
    assert m._price_precision(0.0001, is_crypto=False) == 2


# ── cash capping ─────────────────────────────────────────────────────────────

def test_order_is_capped_to_available_buying_power():
    account = {"buying_power": 100.0, "non_marginable_buying_power": 100.0}
    sized = m.cash_capped_notional(400.0, "AAPL", account)
    assert sized == pytest.approx(95.0), "95% of buying power"
    assert sized < 400.0


def test_exhausted_account_returns_zero_not_a_doomed_order():
    """The live case: $6.71 available against a $135 desk notional."""
    account = {"buying_power": 6.71, "non_marginable_buying_power": 6.71}
    assert m.cash_capped_notional(134.58, "AAVE/USD", account) == 0.0, (
        "an unaffordable order must be skipped, not sent to be 403'd"
    )


def test_negative_balance_account_funds_nothing():
    account = {"buying_power": -8287.81, "non_marginable_buying_power": 0.0}
    assert m.cash_capped_notional(500.0, "BTC/USD", account) == 0.0


def test_crypto_uses_non_marginable_buying_power():
    """Crypto cannot be bought on margin — the cap must read the right field."""
    account = {"buying_power": 10_000.0, "non_marginable_buying_power": 50.0}
    assert m.cash_capped_notional(400.0, "BTC/USD", account) == pytest.approx(47.5)
    assert m.cash_capped_notional(400.0, "AAPL", account) == pytest.approx(400.0)


def test_healthy_account_is_not_throttled():
    account = {"buying_power": 100_000.0, "non_marginable_buying_power": 100_000.0}
    assert m.cash_capped_notional(400.0, "BTC/USD", account) == pytest.approx(400.0)
