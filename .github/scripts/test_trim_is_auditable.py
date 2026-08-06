"""A permanent, irreversible retirement was being made on unauditable evidence.

Measured 2026-08-06 from `strategy_trims.json`, with a binomial test of each
retirement's win count against a coin flip:

    strategy                 n  wins   P(<=wins | 50%)   strength
    avellaneda              10     6            0.828    indistinguishable from chance
    vol_of_vol              10     2            0.055    weak
    realized_vol_asymmetry  10     3            0.172    weak
    options_pcr_reversal    11     2            0.033    strong
    stat_arb_e              11     0            0.000    strong

`avellaneda` won 6 of 10 and was retired **permanently** on a magnitude rule
("cumulative return -7.9% <= -5.0%"). Win rate is the wrong test for a
magnitude rule — a strategy can win often and still lose money on a few large
losers, which is a legitimate reason to retire. The problem is that
`fill_tracker` recorded **no dispersion at all**, so nothing in the record could
distinguish that from one bad trade. And nothing ever removes an entry from
`strategy_trims.json`, while a retired strategy places no orders — so the
decision can never generate the evidence that would overturn it.

These tests pin the fix: the producer records dispersion, and the trim reason
carries it. **Retirement thresholds are deliberately unchanged** — what gets
trimmed is a capital-allocation policy and belongs to a human.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import strategy_trimmer as st  # noqa: E402

SCRIPTS = Path(__file__).resolve().parent


def _stats(**kw):
    base = dict(trades=10, wins=6, win_rate=0.6,
                avg_return_pct=-0.79, total_return_pct=-7.9)
    base.update(kw)
    return base


def test_a_tail_event_is_named_as_one():
    """The `avellaneda` shape: one trade lost more than the strategy's NET loss,
    so every other trade was net positive. Reporting only '-7.9% over 10 trades'
    hides that completely."""
    why = st.evaluate_trim(_stats(stdev_return_pct=2.61, worst_trade_pct=-8.4))[1]
    assert "EXCEEDS the net loss" in why
    assert "net positive" in why


def test_a_steady_bleed_reads_differently():
    """Same total, same n, opposite diagnosis — which is the entire point."""
    why = st.evaluate_trim(_stats(wins=2, win_rate=0.2,
                                  stdev_return_pct=0.31, worst_trade_pct=-1.2))[1]
    assert "EXCEEDS" not in why
    assert "15% of the loss" in why
    assert "stdev 0.31%" in why


def test_a_record_without_dispersion_degrades_to_the_old_wording():
    """Records written before this shipped, and the first run after a deploy,
    must not invent numbers they do not have."""
    why = st.evaluate_trim(_stats())[1]
    assert "cumulative return -7.9%" in why
    assert "[" not in why, f"invented a dispersion note from nothing: {why}"


def test_this_changes_REPORTING_not_WHICH_strategies_are_trimmed():
    """The thresholds are a capital-allocation policy. Adding dispersion must
    make a decision auditable, never silently alter it."""
    for extra in ({}, {"stdev_return_pct": 2.61, "worst_trade_pct": -8.4},
                  {"stdev_return_pct": 0.01, "worst_trade_pct": -0.9}):
        assert st.evaluate_trim(_stats(**extra))[0] is True
    # …and a healthy strategy stays untrimmed whatever its dispersion.
    healthy = dict(trades=20, wins=12, win_rate=0.6,
                   avg_return_pct=0.4, total_return_pct=8.0,
                   stdev_return_pct=9.9, worst_trade_pct=-30.0)
    assert st.evaluate_trim(healthy)[0] is False


def test_the_sample_floor_still_protects_a_fresh_strategy():
    assert st.evaluate_trim(_stats(trades=9))[0] is False


def test_the_producer_records_dispersion_at_all():
    """Call-site guard. The reason string can only carry what fill_tracker
    writes, and it wrote none of this until 2026-08-06."""
    src = (SCRIPTS / "fill_tracker.py").read_text()
    for field in ("stdev_return_pct", "worst_trade_pct", "sum_sq_return_pct"):
        assert field in src, f"fill_tracker does not record {field}"


def test_dispersion_is_carried_incrementally_not_recomputed_from_history():
    """fill_tracker aggregates per run and keeps no per-trade history, so a
    running sum of squares is the only way to get stdev. If someone replaces it
    with a mean-of-batch, the number silently becomes wrong across runs."""
    src = (SCRIPTS / "fill_tracker.py").read_text()
    assert 'prev.get("sum_sq_return_pct"' in src, (
        "sum of squares is not carried forward from the previous run")
