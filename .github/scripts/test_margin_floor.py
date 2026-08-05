"""Nothing stopped the book from levering itself to a standstill.

Measured 2026-08-05 — buying power across ten consecutive desk runs:

    00:41 $0.00   01:31 $0.00    04:26 $116.98
    00:58 $0.00   02:22 $46.35   05:15 $101.23
    01:03 $27.61  07:05 $115.79  07:56 $206.86

`cash` pinned at exactly -$33,401.86 from 00:58 through 07:56: eight runs, zero
fills. The desks were healthy the whole time — run 30986611287 generated 51
signals, 17 cleared the gate, 0 were placed. The account had simply run out of
margin, and **nothing prevented it**, because every individual order was
affordable at the moment it was sized. `cash_capped_notional` asked "can we pay
for this one?" and never "should we spend the last of it?".

`recover_negative_cash` deliberately refuses to unwind this — flattening a
levered book realises losses and trips the daily loss cap (2026-07-27) — so the
only place to prevent it is before the order that exhausts the margin.

`MARGIN_FLOOR_PCT` reserves a fraction of equity as untouchable buying power.
This does not free capital already committed; it stops the state recurring after
the account is reset, which is otherwise exactly where the book ends up again.

Crypto is exempt on purpose: it sizes against `non_marginable_buying_power`,
cannot use margin at all, and is already starved by the equity book consuming
the cash it needs. A margin floor there would compound the very problem this
guard exists to prevent — `test_crypto_is_exempt` pins that.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("SECRET_KEY", "x" * 64)
os.environ.setdefault("DATABASE_URL", "")
import desk_order_placer as dop  # noqa: E402

EQUITY = 22_000.0


def _acct(bp: float, nmbp: float = 0.0, equity: float = EQUITY) -> dict:
    return {"equity": equity, "buying_power": bp, "non_marginable_buying_power": nmbp}


def test_the_exhausted_book_places_nothing():
    """The exact account state from 2026-08-05 07:56."""
    acct = {"equity": 22_013.89, "buying_power": 206.86,
            "non_marginable_buying_power": 0.0}
    assert dop.cash_capped_notional(500, "SPY", acct) == 0.0, (
        "an equity order was still allowed with buying power at $206 against a "
        "$22k book. Without the floor this sized to ~$196 and took the account "
        "the rest of the way to zero."
    )


def test_a_healthy_book_is_untouched():
    """The guard must not tax normal operation."""
    assert dop.cash_capped_notional(500, "SPY", _acct(40_000.0)) == 500.0, (
        "a full-size order on a well-funded account was reduced; the floor is "
        "interfering with ordinary trading rather than only the endgame."
    )


def test_it_sizes_down_to_the_headroom_above_the_floor():
    """Just above the floor, spend only what is genuinely spare."""
    floor = EQUITY * dop.MARGIN_FLOOR_PCT          # 2200
    got = dop.cash_capped_notional(500, "SPY", _acct(floor + 100))
    assert got == pytest.approx(95.0), (
        f"expected (2300-2200)*0.95 = 95.0, got {got}. The order is not being "
        "capped to the headroom above the floor."
    )
    assert got < 500.0


def test_crypto_is_exempt():
    """Crypto cannot use margin, and is already starved by the equity book."""
    # nmbp must be BELOW the equity floor, or the test proves nothing: with
    # nmbp above it the crypto path clears the check even when wrongly applied.
    # $1,000 against a $2,200 floor is blocked unless crypto is genuinely exempt.
    floor = EQUITY * dop.MARGIN_FLOOR_PCT
    acct = _acct(bp=100.0, nmbp=1_000.0)
    assert acct["non_marginable_buying_power"] < floor, "fixture must sit under the floor"
    assert dop.cash_capped_notional(300, "BTC/USD", acct) == 300.0, (
        "the margin floor was applied to a crypto order. Crypto sizes against "
        "non_marginable_buying_power and would be blocked by an equity-margin "
        "condition it has no part in — compounding the starvation documented on "
        "2026-08-04."
    )


def test_the_floor_can_be_disabled():
    """0 restores pre-2026-08-05 behaviour, for a deliberate override."""
    saved = dop.MARGIN_FLOOR_PCT
    try:
        dop.MARGIN_FLOOR_PCT = 0.0
        got = dop.cash_capped_notional(500, "SPY", _acct(206.86))
        assert got == pytest.approx(196.517), (
            f"with the floor off the old behaviour should return ~196.5, got {got}"
        )
    finally:
        dop.MARGIN_FLOOR_PCT = saved


def test_a_zero_equity_account_does_not_block_everything():
    """Equity 0 means unknown, not broke — don't false-trigger on a stub payload.

    The account-unavailable path supplies {equity: 0, cash: 0, buying_power: 0}.
    Note the `equity > 0` guard in the source is defensive rather than
    load-bearing: with equity 0 the floor is 0 and `bp < 0` is unreachable for a
    non-negative buying power, so removing it is behaviourally equivalent. It is
    kept to state the intent; this test pins the OUTCOME, not that branch.
    """
    got = dop.cash_capped_notional(500, "SPY", _acct(bp=10_000.0, equity=0.0))
    assert got == 500.0, (
        "a zero-equity (unknown) account was treated as being under the floor"
    )


def test_the_min_order_guard_still_applies():
    assert dop.cash_capped_notional(500, "SPY", _acct(0.0)) == 0.0
    assert dop.MIN_ORDER_USD == 25.0


def test_the_floor_is_a_real_reservation():
    assert 0 < dop.MARGIN_FLOOR_PCT < 0.5, (
        f"MARGIN_FLOOR_PCT={dop.MARGIN_FLOOR_PCT} is either a no-op or reserves "
        "so much that most of the book cannot be deployed."
    )
