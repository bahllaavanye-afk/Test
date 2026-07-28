"""Who closed the book? `client_order_id` is the only witness that survives.

On 2026-07-27 a book of 13 orders (+$9,634 notional, placed 17:49) was fully
flat by 23:44, having realised enough loss to trip the daily cap and freeze
trading. Nothing in this repo could say what closed it:

  - the orders carried no bracket/OCO legs, so Alpaca did not close them
  - `recover_negative_cash` never fired (no 🚑 line in any run's log)
  - no other Actions script closes positions — `live_trading_reporter` and
    `research_to_trade` only GET /v2/positions
  - the backend DB is on its sqlite fallback and returned **zero** order rows,
    so it held no record either

That leaves the broker's own `client_order_id`, which every order this repo
places is tagged with (`qe-<strategy>-<sym>-<ts>`). Anything without that
prefix came from somewhere we are not watching — most likely the backend's
PositionMonitor exit loop, which was wired up two sessions ago and would be
doing exactly its job. The point of these tests is that the NEXT time a flat
book shows up under an active cap, the run names the closer instead of
leaving it to inference — the same play that resolved the loss-cap question.
"""
from __future__ import annotations

import pytest

from desk_order_placer import _order_origin

OURS = "this desk placer"


def test_a_desk_placed_order_is_recognised_as_ours():
    """The exact shape built at the `coid = ...` line in the placement loop."""
    assert _order_origin("qe-stat_arb_e-SPY-1785200000") == OURS


@pytest.mark.parametrize("coid", [
    "qe-avellaneda-SHIB-1785216000",
    "qe-time_serie-EWT-1785216000",
    "qe-x-A-0",
])
def test_every_qe_prefixed_form_is_ours(coid):
    assert _order_origin(coid) == OURS


@pytest.mark.parametrize("coid", [
    None,                       # broker-generated: Alpaca auto-liquidation
    "",                         # ditto
    "exit-SPY-stop-loss",       # backend PositionMonitor shape
    "bot_exit_checker-123",
    "manual",
])
def test_anything_else_is_flagged_external(coid):
    """The whole point: a non-`qe-` order is the signal we were missing."""
    assert _order_origin(coid) != OURS
    assert "EXTERNAL" in _order_origin(coid)


def test_a_lookalike_prefix_is_not_claimed_as_ours():
    """`qe` without the hyphen is a different namespace — do not absorb it.

    Claiming someone else's order as ours is the failure that matters here:
    it would report the desk placer closed a book it never touched, and send
    the next investigation back to the code that is working.
    """
    assert _order_origin("qexit-SPY-1") != OURS
    assert _order_origin("QE-SPY-1") != OURS


def test_the_origin_label_is_actionable_not_just_a_boolean():
    """The log line has to tell a reader where to look next."""
    label = _order_origin("exit-SPY-stop-loss")
    assert "backend" in label or "broker" in label
