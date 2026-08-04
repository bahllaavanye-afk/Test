"""A zero-order run needs cash in the summary, or it is unexplainable.

`cash_capped_notional` rejects every order once available cash falls below
`MIN_ORDER_USD`, and the desk logs it per-signal:

    · crypto_adaptive_trend/LTC/USD skipped — insufficient available cash
      (< $25; frees as pending closes fill)

That line lives in the Actions log. The message a human actually reads —
`#pnl-daily` — carried only `equity=$…`, and equity does not move when a book
becomes fully deployed: the value is still there, it is just all in positions.

So a fully-deployed account and a genuinely quiet market produced the *same*
Discord headline: healthy equity, zero orders. On 2026-08-04 that cost real time
— the cash block was only found by reading a run log, and the account state I
needed to interpret it sits at stage 2, too early in the log to reach from the
tail.

`cash` and `buying_power` are already computed at stage 2 (`desk_order_placer`
lines ~1860). Putting them in the summary costs nothing and is the difference
between "no orders" and "no orders BECAUSE the book is fully deployed".
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent / "desk_order_placer.py"


@pytest.fixture(scope="module")
def src() -> str:
    return _SRC.read_text()


def test_the_summary_reports_cash_and_buying_power(src):
    header = re.search(r'summary\s*=\s*\((.*?)\)\n', src, re.S)
    assert header, "the *QuantEdge Desk Run* summary header could not be located"
    block = header.group(1)
    assert "cash=" in block, (
        "the run summary no longer reports cash. Without it a fully-deployed "
        "book and a quiet market look identical in Discord — both are "
        "'healthy equity, 0 orders'."
    )
    assert "buying_power=" in block, "the run summary no longer reports buying power"
    assert "equity=" in block, "the run summary lost equity"


def test_the_reported_values_are_the_ones_the_guard_uses(src):
    """Report the same variables the order path checks, not a re-fetch.

    `cash` and `buying` are set once at stage 2 from the account payload. A
    summary that recomputed or re-fetched them could disagree with the values
    that actually gated the orders, which is worse than not reporting at all.
    """
    assert 'cash   = float(account.get("cash",         0))' in src, (
        "the stage-2 cash assignment changed; the summary may now report a "
        "different value from the one cash_capped_notional gated on."
    )
    assert 'buying = float(account.get("buying_power", 0))' in src, (
        "the stage-2 buying_power assignment changed"
    )
    assert "{cash:,.2f}" in src and "{buying:,.2f}" in src, (
        "the summary interpolates something other than the stage-2 `cash` / "
        "`buying` locals"
    )


def test_the_cash_skip_still_has_its_own_drop_reason(src):
    """The funnel breakdown must keep cash distinct from other drops."""
    assert '_drop("insufficient cash"' in src, (
        "the insufficient-cash drop reason is gone, so the Discord funnel line "
        "folds a fully-deployed book into some other category and the summary "
        "numbers lose their explanation."
    )


def test_min_order_usd_is_what_the_guard_compares_against(src):
    """Pins the constant named in the skip message to the real gate."""
    assert "MIN_ORDER_USD = 25.0" in src, "MIN_ORDER_USD moved or changed value"
    assert "kelly_notional = cash_capped_notional(kelly_notional, symbol, account)" in src, (
        "the cash cap is no longer applied before placement, so the skip "
        "message and the summary would describe a guard that is not running."
    )
