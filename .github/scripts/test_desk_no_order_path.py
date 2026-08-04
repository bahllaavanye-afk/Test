"""A desk with no venue route must not report its signals as "market closed".

The Polymarket desk produces `conf=1.00` ensembles on live prediction markets and
has never been able to place a single order: `desk_order_placer` references
`brokers` ZERO times — every order is a POST to Alpaca `/v2/orders` — while
`backend/app/brokers/polymarket.py` exists and is never imported here. The CLOB
signing path is unwired (IMPROVEMENTS: "[P1] Polymarket desk is signal-only").

Those signals were being dropped with `(Polymarket closed)`, which is wrong
twice: prediction markets never close, and even open there is nowhere to send the
order. The message describes a temporary condition while hiding a permanent one.

That cost real diagnostic time on 2026-08-03. The obvious reading of "closed" is
"set always_open=True", and doing that would have shipped
`PM:Will Tucker Carlson win the 2028 Republi…` to Alpaca, where it fails. The
guard has to distinguish the two reasons, and the check has to happen BEFORE the
clock or the clock's message wins.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from desk_order_placer import DESKS, DeskConfig

_SRC = Path(__file__).resolve().parent / "desk_order_placer.py"


def test_deskconfig_has_an_executable_flag():
    assert "executable" in DeskConfig._fields, (
        "DeskConfig lost its `executable` flag, so a desk with no venue route "
        "can no longer be distinguished from one whose market is shut."
    )
    assert DeskConfig._field_defaults.get("executable") is True, (
        "`executable` must default True — a new desk is assumed routable, and "
        "only a desk known to have no path opts out."
    )


def test_polymarket_is_marked_unexecutable():
    poly = [d for d in DESKS if d.name == "Polymarket"]
    assert poly, "the Polymarket desk vanished from DESKS"
    assert poly[0].executable is False, (
        "Polymarket is marked executable, but desk_order_placer has no CLOB "
        "route — it POSTs every order to Alpaca /v2/orders, and Alpaca cannot "
        "trade PM: symbols. Marking it executable resurrects the misleading "
        "'(Polymarket closed)' drop."
    )


def test_every_other_desk_is_executable():
    """Guard against the flag being used to quietly mute a working desk."""
    unroutable = [d.name for d in DESKS if not d.executable]
    assert unroutable == ["Polymarket"], (
        f"desks marked unexecutable changed to {unroutable}. Every other desk "
        f"routes to Alpaca; muting one here would silently stop its trading "
        f"while the run still reported success."
    )


def test_the_no_path_check_runs_before_the_market_clock():
    """Order matters: the clock's message would otherwise win and mislead again.

    Both branches `continue`, so whichever is evaluated first owns the reported
    reason. Polymarket is not `always_open`, so with the clock first every
    outside-RTH run would still say "closed".
    """
    src = _SRC.read_text()
    no_path = src.index("if not desk.executable:")
    clock = src.index("desk_open = is_open or desk.always_open", no_path - 4000)
    assert no_path < clock, (
        "the market-clock branch is evaluated before the no-order-path branch, "
        "so an unroutable desk is reported as 'closed' again — the exact "
        "misdiagnosis this guard exists to prevent."
    )


def test_the_drop_reason_is_distinct_from_market_closed():
    src = _SRC.read_text()
    assert '_drop("no order path"' in src, (
        "the no-order-path drop no longer has its own reason string, so the "
        "Discord funnel line folds it into 'market closed' and the permanent "
        "gap disappears into a transient-sounding one."
    )
    assert '_drop("market closed"' in src, "the market-closed reason was removed"


def test_the_no_path_branch_actually_skips_placement():
    """A branch that logs and falls through would still attempt the order."""
    tree = ast.parse(_SRC.read_text())
    found = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if (isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not)
                and isinstance(test.operand, ast.Attribute)
                and test.operand.attr == "executable"):
            assert any(isinstance(n, ast.Continue) for n in ast.walk(node)), (
                "the `not desk.executable` branch does not `continue`, so an "
                "unroutable desk's signal falls through to order placement."
            )
            found = True
    assert found, "no `if not desk.executable:` guard found in the order loop"
