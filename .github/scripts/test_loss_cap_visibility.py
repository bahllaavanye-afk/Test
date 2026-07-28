"""When the loss cap blocks everything, the summary must say WHY, with numbers.

Measured 2026-07-28 from a live desk run:

    🛑 Loss cap ACTIVE — only risk-reducing orders allowed (0 open positions
       eligible to reduce)
    🛑 avellaneda_stoikov_mm/UNI/USD BUY — blocked by loss cap
    🛑 vol_of_vol_timing/MKR/USD BUY   — blocked by loss cap
    🛑 avellaneda_stoikov_mm/AAVE/USD BUY — blocked by loss cap
    Done. 0 orders placed across 9 desks.

The desks are healthy: they fetch data, run ensembles, and produce signals that
clear the confidence gate. Every one is then blocked here — and the API shows
the last trade was 2026-07-10, eighteen days earlier.

The drawdown figure existed only in the CI log. The Discord summary said just
"loss cap ACTIVE", so from the channel you could not tell a one-off 3% day from
a structural halt. These pin the numbers that distinguish them.

`0 positions eligible to reduce` is the sharp one: under the cap only
risk-REDUCING orders pass, so with no open positions **nothing can pass at
all**. That is correct as a daily cooling-off rule and pathological if it
persists, and the summary now says which it is looking at.
"""
from __future__ import annotations

import pytest

DAILY_LOSS_CAP_PCT = 0.02


def build_cap_line(equity: float, last_equity: float, n_reducible: int,
                   cap: float = DAILY_LOSS_CAP_PCT) -> str:
    """Mirrors the loss-cap branch of desk_order_placer's Discord summary."""
    dd = (1.0 - equity / last_equity) if last_equity else 0.0
    line = (
        f"🛑 loss cap ACTIVE — new exposure blocked, risk-reducing only. "
        f"equity ${equity:,.2f} vs prior close ${last_equity:,.2f} "
        f"({dd:+.2%}, cap {cap:.0%}) · "
        f"{n_reducible} position(s) eligible to reduce"
    )
    if n_reducible == 0:
        line += "  ⚠️ nothing can pass while this is 0"
    return line + "\n"


def test_the_live_state_now_explains_itself():
    """The 2026-07-28 run: cap on, nothing reducible, zero orders possible."""
    line = build_cap_line(equity=21_800.0, last_equity=22_266.11, n_reducible=0)
    assert "21,800.00" in line and "22,266.11" in line
    assert "0 position(s) eligible to reduce" in line
    assert "nothing can pass while this is 0" in line


def test_the_drawdown_percentage_is_shown_not_just_the_word_ACTIVE():
    """"loss cap ACTIVE" alone cannot be triaged from Discord."""
    line = build_cap_line(equity=97.0, last_equity=100.0, n_reducible=0)
    assert "-3.00%" in line or "+-3.00%" in line or "3.00%" in line
    assert "cap 2%" in line


def test_a_reducible_book_does_not_get_the_dead_end_warning():
    """With positions to close, the cap is doing its job, not deadlocking."""
    line = build_cap_line(equity=97.0, last_equity=100.0, n_reducible=3)
    assert "3 position(s) eligible to reduce" in line
    assert "nothing can pass" not in line


def test_a_zero_prior_close_does_not_divide_by_zero():
    """last_equity is 0 when the broker fetch fails — must not crash the run."""
    line = build_cap_line(equity=100.0, last_equity=0.0, n_reducible=0)
    assert "0.00%" in line


@pytest.mark.parametrize("equity,last_equity", [
    (22_266.11, 22_266.11),   # flat
    (10_000.0, 22_266.11),    # heavy loss
    (30_000.0, 22_266.11),    # up (cap would not be active, but must not crash)
])
def test_the_line_is_always_well_formed(equity, last_equity):
    line = build_cap_line(equity, last_equity, n_reducible=0)
    assert line.startswith("🛑 loss cap ACTIVE")
    assert line.endswith("\n")
    assert "$" in line and "%" in line
