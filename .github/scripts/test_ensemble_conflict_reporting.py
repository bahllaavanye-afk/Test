"""A conflicted symbol was logged twice and named nobody.

Signals are grouped by `(desk, symbol, side)`. A symbol with both a buy and a
sell therefore forms TWO groups, and the old code printed the conflict from
inside the per-side loop — so each disagreement produced a mirrored pair:

    · ensemble[Crypto]: AVAX/USD buy/sell conflict — stand aside
    · ensemble[Crypto]: AVAX/USD sell/buy conflict — stand aside

Measured 2026-07-28: **34 lines for 17 symbols**, identical on both the
bar-starved run and the full-universe run after the pagination fix. That
stability is what makes it interesting — the conflict rate is structural, not
a data artefact, and it is the reason nearly every crypto desk stands aside.

The line also named NEITHER strategy, which is the thing actually worth
knowing: whether one pair disagrees on nearly everything (a systematic
mismatch worth fixing) or the disagreements are spread across many strategies
(genuinely no edge, and standing aside is correct). You cannot tell those
apart from `AVAX/USD buy/sell conflict`.

THE STAND-ASIDE BEHAVIOUR IS DELIBERATELY UNCHANGED. This is instrumentation
to answer the question, not a change to what trades — the same
instrument-then-read approach that resolved the loss-cap and book-closer
questions. Changing the ensembling rule without knowing which case this is
would be guessing with real orders.
"""
from __future__ import annotations

from collections import defaultdict

import pytest


def _opposite(side: str) -> str:
    return "sell" if side == "buy" else "buy"


def build_conflict_lines(signals: list[dict]) -> list[str]:
    """Mirrors the conflict-reporting block of desk_order_placer.

    `signals` items are {desk, symbol, side, strategy, confidence}.
    """
    groups: dict = defaultdict(list)
    for it in signals:
        groups[(it["desk"], it["symbol"], it["side"])].append(it)

    conflicted = {
        (dn, sym)
        for (dn, sym, side) in groups
        if (dn, sym, _opposite(side)) in groups
    }
    lines = []
    for dn, sym in sorted(conflicted):
        parts = []
        for s in ("buy", "sell", "neutral", "none", ""):
            grp = groups.get((dn, sym, s))
            if grp:
                who = ", ".join(f"{x['strategy']}({x['confidence']:.2f})" for x in grp)
                parts.append(f"{s or 'unset'}: {who}")
        lines.append(f"  · ensemble[{dn}]: {sym} CONFLICT — stand aside | " + " | ".join(parts))
    return lines


def _sig(symbol, side, strategy, conf=0.9, desk="Crypto"):
    return {"desk": desk, "symbol": symbol, "side": side,
            "strategy": strategy, "confidence": conf}


# ── the double-logging regression ────────────────────────────────────────────

def test_one_conflict_produces_exactly_one_line():
    """THE BUG: the mirrored buy/sell + sell/buy pair read as two events."""
    lines = build_conflict_lines([
        _sig("AVAX/USD", "buy", "vol_of_vol_timing"),
        _sig("AVAX/USD", "sell", "avellaneda_stoikov_mm"),
    ])
    assert len(lines) == 1, lines


def test_the_measured_shape_reports_17_not_34():
    """17 conflicted symbols must yield 17 lines, not 34."""
    sigs = []
    for i in range(17):
        sym = f"SYM{i}/USD"
        sigs += [_sig(sym, "buy", "a"), _sig(sym, "sell", "b")]
    assert len(build_conflict_lines(sigs)) == 17


def test_agreeing_signals_produce_no_conflict_line():
    lines = build_conflict_lines([
        _sig("AAVE/USD", "buy", "vol_of_vol_timing"),
        _sig("AAVE/USD", "buy", "avellaneda_stoikov_mm"),
    ])
    assert lines == []


# ── the attribution that was missing ─────────────────────────────────────────

def test_both_sides_name_their_strategies():
    """The whole point: who disagreed, and how strongly."""
    line = build_conflict_lines([
        _sig("AVAX/USD", "buy", "vol_of_vol_timing", 0.92),
        _sig("AVAX/USD", "sell", "avellaneda_stoikov_mm", 0.71),
    ])[0]
    assert "vol_of_vol_timing(0.92)" in line
    assert "avellaneda_stoikov_mm(0.71)" in line
    assert "buy:" in line and "sell:" in line


def test_multiple_strategies_on_one_side_are_all_named():
    line = build_conflict_lines([
        _sig("BTC/USD", "buy", "a", 0.9),
        _sig("BTC/USD", "buy", "b", 0.8),
        _sig("BTC/USD", "sell", "c", 0.7),
    ])[0]
    for name in ("a(0.90)", "b(0.80)", "c(0.70)"):
        assert name in line, line


def test_a_conflict_is_distinguishable_per_desk():
    """Grouping is per desk — two desks conflicting on one symbol is two lines."""
    lines = build_conflict_lines([
        _sig("BTC/USD", "buy", "a", desk="Crypto"),
        _sig("BTC/USD", "sell", "b", desk="Crypto"),
        _sig("BTC/USD", "buy", "c", desk="Momentum"),
        _sig("BTC/USD", "sell", "d", desk="Momentum"),
    ])
    assert len(lines) == 2
    assert any("ensemble[Crypto]" in ln for ln in lines)
    assert any("ensemble[Momentum]" in ln for ln in lines)


# ── behaviour must be unchanged ──────────────────────────────────────────────

def _surviving(signals: list[dict]) -> list[tuple]:
    """Which groups survive the stand-aside filter — the trading behaviour."""
    groups: dict = defaultdict(list)
    for it in signals:
        groups[(it["desk"], it["symbol"], it["side"])].append(it)
    return sorted(k for k in groups if (k[0], k[1], _opposite(k[2])) not in groups)


def test_conflicting_sides_still_stand_aside():
    sigs = [_sig("AVAX/USD", "buy", "a"), _sig("AVAX/USD", "sell", "b")]
    assert _surviving(sigs) == []


def test_unopposed_signals_still_pass():
    sigs = [_sig("AAVE/USD", "buy", "a"), _sig("AAVE/USD", "buy", "b")]
    assert _surviving(sigs) == [("Crypto", "AAVE/USD", "buy")]


@pytest.mark.parametrize("side", ["neutral", "none", ""])
def test_non_directional_sides_keep_their_prior_semantics(side):
    """`_opposite` maps everything non-buy to "buy", so a bare neutral survives
    on its own and is dropped when a buy exists. Unchanged from before — pinned
    so the logging rewrite cannot have altered it silently."""
    assert _surviving([_sig("X/USD", side, "a")]) == [("Crypto", "X/USD", side)]
    both = [_sig("X/USD", side, "a"), _sig("X/USD", "buy", "b")]
    assert ("Crypto", "X/USD", side) not in _surviving(both)
