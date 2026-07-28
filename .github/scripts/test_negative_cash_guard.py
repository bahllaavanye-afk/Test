"""A margin debit is not distress — `recover_negative_cash` must not flatten it.

THE MEASURED LOOP (2026-07-27, reconstructed from four workflow runs):

    17:49  desk-trading        13 orders, +$9,634 notional
                               cash -$10,829.50, bp $25,001.91   ← healthy
    18:42  desk-trading-crypto 🚑 RECOVERY: flattening 25 position(s)
                               cash -$14,972.80, bp $17,247.12   ← still healthy
    18:43  broker              25 sells all filled within ~1.5s
    23:44  desk-trading        equity $21,745.65 == cash, -2.28% vs prior close
                               🛑 DAILY LOSS CAP — 0 positions, nothing can pass

Buy on margin → get liquidated → realise the loss → trip the daily cap → sit
frozen until the next session rollover. Self-inflicted, and it ran for a full
cycle without anything reporting it: the recovery fires from the **crypto**
workflow while the positions it destroys were opened by the **equity** desks,
so reading either workflow's log alone shows only half the story.

THE GUARD BUG: buying marginable equities drives cash negative and
non-marginable buying power to zero *by construction*. So the original
condition — `cash < 0 and nmbp <= 0` — reduces to "this account used margin"
and matches every healthy long book the equity desks open.

The pathology the function is genuinely for is "$0 available": orphaned
notional buys that leave the account unable to place ANY order (Alpaca then
403s with 'insufficient balance for USD'). That state has **no buying power
left**. Healthy margin use does. `buying_power > 0` is the discriminator.
"""
from __future__ import annotations

import asyncio

import pytest

from desk_order_placer import recover_negative_cash


def _attempted(account: dict) -> bool:
    """Did the function try to FLATTEN, i.e. issue the destructive DELETEs?

    The return value cannot answer this. Without broker credentials the
    flatten attempt raises, the broad `except` swallows it, and the function
    returns False — identical to declining. The first draft of this file
    asserted `is False` and therefore **passed against the unfixed code**,
    which is the whole failure it was written to catch. Patching the delete
    and watching for the call is the only honest discriminator.
    """
    import desk_order_placer as dop

    calls: list[str] = []

    def _fake_delete(path):
        calls.append(path)
        return {"body": []}

    orig = dop._alpaca_delete_sync
    dop._alpaca_delete_sync = _fake_delete
    try:
        asyncio.run(recover_negative_cash(account))
    finally:
        dop._alpaca_delete_sync = orig
    return any("positions" in c for c in calls)


def _acct(cash: float, bp: float, nmbp: float = 0.0) -> dict:
    return {"cash": cash, "buying_power": bp, "non_marginable_buying_power": nmbp}


# ── the regression ───────────────────────────────────────────────────────────

def test_the_2026_07_27_liquidation_would_not_happen_now():
    """The exact account state that flattened 25 healthy positions."""
    assert not _attempted(_acct(cash=-14_972.80, bp=17_247.12))


def test_the_earlier_snapshot_in_the_same_cycle_is_also_spared():
    """Right after the 13 orders filled — the state the loop starts from."""
    assert not _attempted(_acct(cash=-10_829.50, bp=25_001.91))


@pytest.mark.parametrize("cash,bp", [
    (-1.0, 0.01),          # barely negative, barely any power — still tradeable
    (-50_000.0, 100.0),    # deep debit, small power
    (-14_972.80, 17_247.12),
])
def test_any_remaining_buying_power_blocks_the_flatten(cash, bp):
    """`buying_power > 0` means the account can still trade. Never flatten it.

    Flattening is destructive and unrecoverable — it realises every open P&L
    at market. It has to be reserved for the state where nothing else works.
    """
    assert not _attempted(_acct(cash=cash, bp=bp))


# ── the pathology it still has to catch ──────────────────────────────────────
# Guarded so a fix for the false positive doesn't quietly disable the feature.

def test_a_genuinely_stuck_account_still_qualifies():
    """cash < 0, nothing available anywhere — the orphaned-notional state.

    Must still reach the flatten. Without this, "stop the bad liquidation"
    and "disable the feature" are indistinguishable.
    """
    assert _attempted(_acct(cash=-14_972.80, bp=0.0))


def test_positive_cash_is_never_a_flatten_regardless_of_buying_power():
    assert not _attempted(_acct(cash=1_000.0, bp=0.0))


def test_available_non_marginable_cash_is_never_a_flatten():
    """nmbp > 0 means crypto can still trade — the original short-circuit."""
    assert not _attempted(_acct(cash=-100.0, bp=0.0, nmbp=500.0))
