"""When the loss cap blocks everything, the summary must say WHY, with numbers
that carry the right SIGN.

Measured 2026-07-28 from live desk runs, and it settled a question that had
been open for a session:

    17:49 Jul 27  equity $21,819.46  cash $-10,829.50  bp $25,001.91
                  → 13 orders placed, +$9,634 notional, cap NOT active
    23:44 Jul 27  equity $21,745.65  cash $21,745.65   bp $86,982.60
                  → flat book, cap ACTIVE, 0 orders
    04:46 Jul 28  identical to the cent, still capped

So the desks ARE trading — the "nothing since 2026-07-10" reading came from
the backend DB, which is on its sqlite fallback and never sees orders the
Actions desks place directly at the broker. And the cap is firing on REAL
numbers: prior close $22,253.58 → $21,745.65 is a genuine -2.28% day against
a 2% cap. The contradiction detector below stayed correctly silent.

`0 positions eligible to reduce` is the sharp one: under the cap only
risk-REDUCING orders pass, so with no open positions **nothing can pass at
all**. That is correct as a daily cooling-off and pathological if it persists,
and the summary now says which it is looking at.

THE SIGN IS THE POINT. The first cut of this computed the drawdown magnitude
(`1 - equity/last_equity`, positive on a loss) and formatted it `{:+.2%}`, so
the run above reported its -2.28% day as **"+2.28%"** — a diagnostic asserting
the opposite of what happened. Read as a gain it makes a correctly-firing cap
look like a bug, which is exactly the wrong conclusion to hand the next
investigation. These tests pin the convention: losses render negative.
"""
from __future__ import annotations

import pytest

DAILY_LOSS_CAP_PCT = 0.02


def build_cap_line(equity: float, last_equity: float, n_reducible: int,
                   cap: float = DAILY_LOSS_CAP_PCT) -> str:
    """Mirrors the loss-cap branch of desk_order_placer's Discord summary."""
    ret = (equity / last_equity - 1.0) if last_equity else 0.0
    line = (
        f"🛑 loss cap ACTIVE — new exposure blocked, risk-reducing only. "
        f"equity ${equity:,.2f} vs prior close ${last_equity:,.2f} "
        f"({ret:+.2%}, cap -{cap:.0%}) · "
        f"{n_reducible} position(s) eligible to reduce"
    )
    if n_reducible == 0:
        line += "  ⚠️ nothing can pass while this is 0"
    return line + "\n"


def test_the_live_state_now_explains_itself():
    """The 2026-07-28 run: cap on, nothing reducible, zero orders possible."""
    line = build_cap_line(equity=21_745.65, last_equity=22_253.58, n_reducible=0)
    assert "21,745.65" in line and "22,253.58" in line
    assert "0 position(s) eligible to reduce" in line
    assert "nothing can pass while this is 0" in line


def test_a_losing_day_renders_NEGATIVE():
    """The regression this file exists for: -2.28% must not print as +2.28%."""
    line = build_cap_line(equity=21_745.65, last_equity=22_253.58, n_reducible=0)
    assert "-2.28%" in line
    assert "+2.28%" not in line


@pytest.mark.parametrize("equity,last_equity", [
    (97.0, 100.0),
    (21_745.65, 22_253.58),
    (10_000.0, 22_266.11),
])
def test_no_loss_is_ever_reported_with_a_plus_sign(equity, last_equity):
    """Generalised: whenever equity < prior close, the printed % is negative.

    A cap that fires is by definition a loss, so a '+' in this line always
    means the sign convention has been inverted again.
    """
    line = build_cap_line(equity, last_equity, n_reducible=0)
    pct = line.split("(")[1].split(",")[0]
    assert pct.startswith("-"), f"loss rendered as {pct!r}"


def test_the_cap_threshold_is_also_signed():
    """'cap 2%' next to a negative number reads as a ceiling, not a floor."""
    line = build_cap_line(equity=97.0, last_equity=100.0, n_reducible=0)
    assert "cap -2%" in line


def test_the_drawdown_percentage_is_shown_not_just_the_word_ACTIVE():
    """"loss cap ACTIVE" alone cannot be triaged from Discord."""
    line = build_cap_line(equity=97.0, last_equity=100.0, n_reducible=0)
    assert "-3.00%" in line


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
    (22_253.58, 22_253.58),   # flat
    (10_000.0, 22_253.58),    # heavy loss
    (30_000.0, 22_253.58),    # up (cap would not be active, but must not crash)
])
def test_the_line_is_always_well_formed(equity, last_equity):
    line = build_cap_line(equity, last_equity, n_reducible=0)
    assert line.startswith("🛑 loss cap ACTIVE")
    assert line.endswith("\n")
    assert "$" in line and "%" in line


# ── the contradiction detector ───────────────────────────────────────────────
# `daily_loss_cap_hit` is `equity < last_equity * (1 - cap)`. With ZERO open
# positions and no fills, equity cannot move, so equity should EQUAL prior
# close and the cap cannot legitimately be active. If it is, the inputs are
# wrong — a stale `last_equity` was the suspected candidate.
#
# It did NOT fire on 2026-07-28, and that is the finding: $21,745.65 against a
# $22,253.58 prior close is a real loss, booked while the account still held
# the positions it opened at 17:49. The freeze is a correct daily cooling-off
# and lifts when Alpaca rolls `last_equity` at the next session open — it does
# not roll at the closing bell, which is why three consecutive runs across six
# hours all read the same stale-looking prior close.

def cap_is_self_contradictory(equity: float, last_equity: float) -> bool:
    """True when the cap is active but the numbers cannot justify it."""
    return bool(last_equity) and abs(equity - last_equity) < 1e-9


def test_a_flat_account_under_an_active_cap_is_flagged_as_contradictory():
    assert cap_is_self_contradictory(22_253.58, 22_253.58)


def test_the_measured_2026_07_28_state_is_NOT_contradictory():
    """The live numbers justify the cap — this is what closed the question."""
    assert not cap_is_self_contradictory(21_745.65, 22_253.58)


def test_a_missing_prior_close_is_not_flagged_as_contradictory():
    """last_equity == 0 means the fetch failed, which is a different problem."""
    assert not cap_is_self_contradictory(22_253.58, 0.0)
