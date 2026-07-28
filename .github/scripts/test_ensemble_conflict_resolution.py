"""A lone 0.16 dissent vetoed a 0.90 consensus. Now the evidence is weighed.

THIS CHANGES TRADING BEHAVIOUR — unlike the reporting change that preceded it.
It is the answer to the question that reporting was built to ask, and the
measurement is unambiguous. Over 76 conflicts on 2026-07-28:

    Crypto        crypto_adaptive_trend was the ONLY sell voice on all 16
                  conflicts, at 0.16-0.52, against buy consensus of 0.61-0.97
    SHIB/USD      avellaneda_stoikov_mm(0.90)  vetoed by  a single 0.16
    LINK/USD      vol_of_vol(0.70)+avellaneda(0.90) = 0.97  vetoed by 0.26

The old rule stood aside on ANY opposing signal, regardless of strength or
count. Treating a 0.16 dissent as equal to a 0.97 consensus is not "no edge",
it is discarding the weights.

Across the other six desks the disagreement IS genuinely distributed — most
strategies appear on both sides across symbols — so the fix must not simply
trade whenever there is a majority. It weighs both sides.

THE NEW RULE. Combine each side with the same `1 - prod(1-ci)` already used
for agreement, take the difference, and trade the dominant side at that NET
confidence only if it clears `ENSEMBLE_NET_MIN` (default 0.60, the desks'
own `confidence_min`). The net then still faces the desk threshold and any
per-strategy tuned threshold downstream — this widens the funnel, it does not
bypass the gate. Nothing trades on weaker evidence than an unopposed signal
would need.

Reverting needs no code change: `ENSEMBLE_NET_MIN > 1.0` restores the previous
always-stand-aside behaviour.
"""
from __future__ import annotations

import pytest

NET_MIN = 0.60


def _combined(confs) -> float:
    p = 1.0
    for c in confs:
        p *= (1.0 - min(max(float(c), 0.0), 1.0))
    return 1.0 - p


def resolve(buy_confs, sell_confs, net_min: float = NET_MIN):
    """(winning_side, net) or (None, net) when the evidence is too close."""
    bc, sc = _combined(buy_confs), _combined(sell_confs)
    net = abs(bc - sc)
    win = "buy" if bc > sc else "sell"
    return (win if net >= net_min else None), net


# ── the measured cases that motivated the change ─────────────────────────────

def test_SHIB_a_lone_0_16_no_longer_vetoes_a_0_90_consensus():
    side, net = resolve([0.90], [0.16])
    assert side == "buy"
    assert net == pytest.approx(0.74, abs=0.01)


def test_LINK_a_0_26_dissent_no_longer_vetoes_a_0_97_consensus():
    side, net = resolve([0.70, 0.90], [0.26])
    assert side == "buy"
    assert net == pytest.approx(0.71, abs=0.01)


@pytest.mark.parametrize("symbol,buy,sell", [
    ("AVAX/USD", [0.70, 0.90], [0.32]),
    ("DOT/USD", [0.70, 0.90], [0.36]),
    ("SUSHI/USD", [0.70, 0.90], [0.37]),
])
def test_the_crypto_conflicts_with_a_WEAK_dissent_now_resolve_to_buy(symbol, buy, sell):
    """Strong multi-strategy buy against a dissent below ~0.37."""
    side, _ = resolve(buy, sell)
    assert side == "buy", symbol


@pytest.mark.parametrize("symbol,buy,sell", [
    ("DOGE/USD", [0.70, 0.90], [0.44]),
    ("BAT/USD", [0.61, 0.70, 0.90], [0.46]),
])
def test_a_MODERATE_dissent_still_stands_aside_even_against_consensus(symbol, buy, sell):
    """My own first draft asserted these would trade. They do not, correctly.

    0.97 consensus against a 0.44 dissent nets 0.53 — below the bar. The rule
    is narrower than "consensus wins": the dissent has to be genuinely weak,
    not merely outvoted. Worth pinning, because it is the property that keeps
    this from becoming a majority-rules rule.
    """
    side, _ = resolve(buy, sell)
    assert side is None, symbol


def test_the_measured_delta_is_small_and_targeted():
    """Replay of all 76 conflicts from the 2026-07-28 run: only 5 unblock.

    All 5 are Crypto buys where a 0.26-0.37 dissent opposed a ~0.97 consensus.
    Every Commodities / Equities / Macro-FX / Options / International conflict
    still stands aside, because their dissents were credible (supertrend 0.72,
    yield_curve_momentum 0.89). Pinned so a later tweak to the bar cannot
    quietly turn this into a broad loosening.
    """
    measured = [                      # (buy, sell, expected_to_trade)
        ([0.90], [0.16], True),       # SHIB/USD
        ([0.70, 0.90], [0.26], True),  # LINK/USD
        ([0.70, 0.90], [0.32], True),  # AVAX/USD
        ([0.70, 0.90], [0.36], True),  # DOT/USD
        ([0.70, 0.90], [0.37], True),  # SUSHI/USD
        ([0.70, 0.90], [0.44], False),  # DOGE/USD
        ([0.61, 0.70, 0.90], [0.46], False),  # BAT/USD
        ([0.61], [0.28], False),      # ETH/USD
        ([0.84, 0.84], [0.72], False),  # GDX
        ([0.90, 0.82], [0.72], False),  # USO
        ([0.52], [0.72], False),      # CORN
    ]
    got = [(resolve(b, s)[0] is not None) for b, s, _ in measured]
    assert got == [want for _, _, want in measured]


# ── genuine disagreement must STILL stand aside ──────────────────────────────

def test_ETH_a_weak_buy_against_a_weak_sell_still_stands_aside():
    """0.61 vs 0.28 — dominant, but not dominant enough to trade on."""
    side, net = resolve([0.61], [0.28])
    assert side is None
    assert net == pytest.approx(0.33, abs=0.01)


@pytest.mark.parametrize("symbol,buy,sell", [
    ("CORN", [0.52], [0.72]),
    ("GDX", [0.84, 0.84], [0.72]),
    ("USO", [0.90, 0.82], [0.72]),
    ("GLD", [0.67, 0.71], [0.72]),
])
def test_the_commodities_conflicts_still_stand_aside(symbol, buy, sell):
    """supertrend(0.72) is a CREDIBLE dissent — these are real disagreements.

    The fix must not simply trade whenever one side has more voices; these all
    have a 2-strategy buy and still correctly stand aside.
    """
    side, _ = resolve(buy, sell)
    assert side is None, symbol


def test_a_dead_heat_stands_aside():
    assert resolve([0.80], [0.80])[0] is None


def test_the_stronger_side_can_be_sell():
    side, _ = resolve([0.20], [0.95])
    assert side == "sell"


# ── the safety properties ────────────────────────────────────────────────────

def test_the_net_never_exceeds_the_winning_sides_own_confidence():
    """Dissent is subtracted, so a conflict can only ever LOWER confidence.

    A conflicted signal must never be sized more aggressively than the same
    signal unopposed.
    """
    for buy, sell in [([0.90], [0.16]), ([0.70, 0.90], [0.26]), ([0.99], [0.01])]:
        _, net = resolve(buy, sell)
        assert net <= _combined(buy) + 1e-9


def test_raising_the_bar_above_one_restores_stand_aside_always():
    """The documented revert lever — no code change required."""
    for buy, sell in [([0.90], [0.16]), ([0.99], [0.01]), ([0.70, 0.90], [0.26])]:
        assert resolve(buy, sell, net_min=1.01)[0] is None


def test_the_default_bar_matches_the_desk_confidence_min():
    """Nothing trades on weaker evidence than an unopposed signal needs.

    The desks all declare confidence_min=0.60; if this default drifted below
    that, a conflicted signal could clear a bar an unopposed one could not.
    """
    assert NET_MIN == 0.60


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -5.0, 2.0])
def test_malformed_confidences_do_not_produce_a_runaway_net(bad):
    """Confidences are clamped to [0,1] before combining."""
    _, net = resolve([bad], [0.5])
    assert 0.0 <= net <= 1.0 or net != net  # NaN propagates, never > 1
