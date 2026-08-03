"""Execution quality was never measured, and unmeasured must not read as free.

The desk has always had both halves of an implementation-shortfall calculation
and never compared them: `limit_price` is the decision-time bar close the limit
is derived from (the arrival price), and Alpaca returns `filled_avg_price` on
the fill. So "Done. 7 orders placed across 9 desks" told us trades happened and
nothing at all about whether they happened at a good price — and a paper P&L
with no execution cost attached is not a believable one.

The trap this file guards is the sign convention and the None handling.

SIGN: implementation shortfall is a COST, not a price delta. A buy filling ABOVE
arrival paid up; a sell filling BELOW arrival gave up edge. Both are costs and
both must come back POSITIVE, or the average across a mixed book silently cancels
buy costs against sell costs and reports roughly zero no matter how bad execution
gets.

NONE: `_ensure_filled()` replaces an unfilled limit with a market order and
returns it immediately, without waiting for that fill — so `filled_avg_price` is
legitimately absent on those. Recording them as 0.0 bps would report *perfect
execution for trades nobody measured*, which is precisely the green-looking
absence this repo keeps paying for. Unmeasured has to stay distinguishable from
measured-and-fine, in the record, in the aggregate, and in the Discord summary.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from desk_order_placer import slippage_bps

_SRC = Path(__file__).resolve().parent / "desk_order_placer.py"


# ── Sign convention ──────────────────────────────────────────────────────────

def test_buying_above_arrival_is_a_positive_cost():
    assert slippage_bps("buy", 100.0, 101.0) == pytest.approx(100.0)


def test_selling_below_arrival_is_also_a_positive_cost():
    """The one that cancels out if the sign is wrong."""
    assert slippage_bps("sell", 100.0, 99.0) == pytest.approx(100.0)


def test_price_improvement_is_negative_on_both_sides():
    assert slippage_bps("buy", 100.0, 99.5) == pytest.approx(-50.0)
    assert slippage_bps("sell", 100.0, 100.5) == pytest.approx(-50.0)


def test_a_mixed_book_of_equal_costs_does_not_average_to_zero():
    """Directly pins the failure a raw price delta would produce.

    One buy and one sell, each 100 bps worse than arrival. With a signed price
    delta these are +100 and -100 and the desk reports flawless execution.
    """
    values = [slippage_bps("buy", 100.0, 101.0), slippage_bps("sell", 100.0, 99.0)]
    assert sum(values) / len(values) == pytest.approx(100.0), (
        "equal costs on opposite sides cancelled out — the sign convention has "
        "reverted to a raw price delta and average slippage is now meaningless"
    )


def test_an_exact_fill_at_arrival_is_zero():
    assert slippage_bps("buy", 100.0, 100.0) == 0.0


# ── Unmeasured is not zero ───────────────────────────────────────────────────

@pytest.mark.parametrize("arrival,fill", [
    (100.0, None),   # market replacement: submitted, fill not awaited
    (None, 100.0),   # no arrival price captured
    (None, None),
    (0.0, 100.0),    # non-positive arrival would divide by zero
    (-5.0, 100.0),
    (100.0, 0.0),    # a zero fill price is not a real fill
])
def test_unmeasurable_returns_none_never_zero(arrival, fill):
    result = slippage_bps("buy", arrival, fill)
    assert result is None, (
        f"slippage_bps({arrival}, {fill}) returned {result!r}. Returning 0.0 here "
        f"reports PERFECT execution for a trade that was never measured — the "
        f"exact green-looking absence this check exists to prevent."
    )


def test_garbage_input_does_not_raise():
    """Alpaca returns filled_avg_price as a string, and sometimes as null."""
    assert slippage_bps("buy", 100.0, "101.0") == pytest.approx(100.0)
    assert slippage_bps("buy", 100.0, "not-a-number") is None


# ── The call site must keep the distinction ──────────────────────────────────

def test_the_aggregate_filters_none_rather_than_coercing_it():
    """An aggregate built with `or 0` would re-introduce the bug downstream."""
    src = _SRC.read_text()
    assert 'o.get("slippage_bps") is not None' in src, (
        "the run summary no longer filters unmeasured fills with an explicit "
        "`is not None`. A truthiness test would also discard genuine 0.0 bps "
        "fills, and an `or 0` would count unmeasured fills as free."
    )
    assert "unmeasured" in src, (
        "the summary no longer reports how many fills were unmeasured, so a run "
        "where nothing could be measured looks identical to one with perfect "
        "execution."
    )


def test_slippage_is_recorded_on_the_order_itself():
    """Aggregates are lossy; the per-order record is what a later TCA needs."""
    src = _SRC.read_text()
    for field in ('"arrival_price"', '"fill_price"', '"slippage_bps"'):
        assert field in src, f"order records no longer carry {field}"


def test_slippage_bps_is_a_module_level_function():
    """Keep it importable and pure, so it stays testable without a broker.

    It was written as a helper rather than inline in the 700-line order loop
    specifically so these cases could be checked at all.
    """
    tree = ast.parse(_SRC.read_text())
    names = [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    assert "slippage_bps" in names, (
        "slippage_bps was inlined back into the order loop; it can no longer be "
        "tested without a live broker, which means it will stop being tested."
    )
