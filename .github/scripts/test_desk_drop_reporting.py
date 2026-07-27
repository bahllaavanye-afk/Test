"""A surviving signal that never becomes an order must say why.

The live Discord summary read:

    funnel: 32 generated → 15 survived gate+topK (0 exploration) → 0 placed
    💤 *Equities*: no signals fired
    💤 *Crypto*: no signals fired
    ... every desk ...
    Total orders placed: *0*

Fifteen signals survived the gate, every one was dropped, and the message
states neither that it happened nor why. Worse, "no signals fired" is emitted
whenever no ORDER was placed — so "this desk had nothing to say" and "this
desk fired fifteen signals and lost all of them" render identically.

The placement loop already prints a reason at each `continue` — market closed,
no account, loss cap, pruned by attribution, insufficient cash — but those go
to the CI log, which nobody reads. This pins the reporting shape.
"""
from __future__ import annotations

from collections import Counter

import pytest


def build_funnel_line(generated: int, survived: int, explored: int,
                      placed: int, drops: Counter) -> str:
    """Mirrors the funnel block in desk_order_placer.main()."""
    line = (f"funnel: {generated} generated → {survived} survived gate+topK "
            f"({explored} exploration) → {placed} placed\n")
    if drops:
        total = sum(drops.values())
        why = " · ".join(f"{n} {reason}" for reason, n in drops.most_common())
        line += f"⚠️ {total} dropped before placement — {why}\n"
    return line


def build_desk_line(desk: str, orders: int, signals: int,
                    desk_drops: Counter) -> str:
    """Mirrors the per-desk summary in desk_order_placer.main()."""
    if orders:
        return f"✅ *{desk}*: {orders} orders"
    if signals:
        why = ", ".join(f"{n} {r}" for r, n in desk_drops.most_common(3)) or "no reason recorded"
        return f"⚠️ *{desk}*: {signals} signal(s) fired, **0 placed** — {why}"
    return f"💤 *{desk}*: no signals fired"


# ── the funnel gap ───────────────────────────────────────────────────────────

def test_the_live_case_now_explains_itself():
    """32 → 15 → 0 must no longer be silent about the 15."""
    line = build_funnel_line(32, 15, 0, 0, Counter({"insufficient cash": 15}))
    assert "15 dropped before placement" in line
    assert "15 insufficient cash" in line


def test_multiple_reasons_are_all_listed_most_common_first():
    drops = Counter({"insufficient cash": 9, "loss cap": 4, "pruned by attribution": 2})
    line = build_funnel_line(32, 15, 0, 0, drops)
    assert line.index("9 insufficient cash") < line.index("4 loss cap") < line.index("2 pruned by attribution")


def test_a_clean_run_adds_no_noise():
    """Nothing dropped → no warning line at all."""
    line = build_funnel_line(32, 15, 0, 15, Counter())
    assert "dropped before placement" not in line
    assert "⚠️" not in line


def test_partial_placement_still_reports_the_remainder():
    line = build_funnel_line(32, 15, 0, 11, Counter({"insufficient cash": 4}))
    assert "11 placed" in line
    assert "4 dropped before placement" in line


# ── the misleading per-desk line ─────────────────────────────────────────────

def test_a_desk_that_lost_every_signal_is_not_reported_as_quiet():
    """This is the bug: 15 fired, 0 placed, rendered as 'no signals fired'."""
    line = build_desk_line("Equities", orders=0, signals=15,
                           desk_drops=Counter({"insufficient cash": 15}))
    assert "no signals fired" not in line
    assert "15 signal(s) fired" in line
    assert "0 placed" in line
    assert "insufficient cash" in line


def test_a_genuinely_quiet_desk_still_reads_as_quiet():
    line = build_desk_line("Options", orders=0, signals=0, desk_drops=Counter())
    assert line == "💤 *Options*: no signals fired"


def test_a_working_desk_is_unchanged():
    line = build_desk_line("Crypto", orders=3, signals=5, desk_drops=Counter())
    assert line == "✅ *Crypto*: 3 orders"


def test_quiet_and_lost_desks_do_not_render_identically():
    """The whole point — these two states must be distinguishable."""
    quiet = build_desk_line("A", orders=0, signals=0, desk_drops=Counter())
    lost = build_desk_line("A", orders=0, signals=15,
                           desk_drops=Counter({"insufficient cash": 15}))
    assert quiet != lost


def test_desk_reasons_are_capped_to_keep_the_message_readable():
    many = Counter({f"reason{i}": 10 - i for i in range(8)})
    line = build_desk_line("Macro", orders=0, signals=40, desk_drops=many)
    listed = [r for r in many if r in line]
    assert len(listed) == 3, f"at most three reasons per desk line, got {listed}"
    assert "reason0" in line and "reason7" not in line, "keeps the most common"


def test_drops_without_a_recorded_reason_say_so_rather_than_nothing():
    """An unattributed drop must not render as an empty explanation."""
    line = build_desk_line("StatArb", orders=0, signals=3, desk_drops=Counter())
    assert "no reason recorded" in line


@pytest.mark.parametrize("reason", [
    "market closed", "no account", "loss cap",
    "pruned by attribution", "insufficient cash",
])
def test_every_placement_loop_reason_survives_into_the_message(reason):
    """The five `continue` branches in the placement loop are all reportable."""
    line = build_funnel_line(10, 5, 0, 0, Counter({reason: 5}))
    assert reason in line
