"""The field that gates crypto was the one field never reported.

`cash_capped_notional` sizes against a different account field per asset class,
correctly, because Alpaca crypto is cash-only and cannot use margin:

    field = "non_marginable_buying_power" if is_crypto else "buying_power"
    avail = float(account.get(field, 0) or 0) * 0.95
    if avail < MIN_ORDER_USD:      # $25
        return 0.0                 # -> "insufficient available cash"

Buying marginable equities drives cash negative and
`non_marginable_buying_power` to ~0 *by construction*. So the crypto desk starves
whenever the equity book is levered — regardless of signal quality — while
equities keep trading on ample `buying_power`.

Run 30930093709 (2026-08-04 16:45) showed exactly that: `equity=$21,968.66`,
`cash=$-2,554.56`, `buying_power=$49,855.63`, 12 equity-side orders placed, and
`⚠️ *Crypto*: 2 signal(s) fired, 0 placed — 2 insufficient cash` — with those two
signals having already cleared the 0.60 confidence gate.

Diagnosing it required **inferring** `non_marginable_buying_power`, because it
was the one input to that decision the desk never printed. Reporting equity, cash
and buying_power while withholding the field that actually decides crypto sizing
shows a book that looks perfectly funded next to a desk that cannot place a $25
order.
"""
from __future__ import annotations

from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent / "desk_order_placer.py"


@pytest.fixture(scope="module")
def src() -> str:
    return _SRC.read_text()


def test_non_marginable_bp_is_captured(src):
    assert 'nmbp   = float(account.get("non_marginable_buying_power", 0) or 0)' in src, (
        "the desk no longer reads non_marginable_buying_power into a local, so "
        "the field that gates every crypto order cannot be reported."
    )


def test_it_is_initialised_so_the_no_account_path_cannot_crash(src):
    """`account` may be the {equity:0, cash:0, buying_power:0} stub.

    On the account-unavailable path the assignment branch is skipped entirely, so
    the summary would raise NameError building its own header — turning a
    degraded run into a crashed one.
    """
    assert "nmbp   = 0.0" in src, (
        "nmbp has no default. On the 'Account unavailable — signal-only mode' "
        "path the assignment is skipped and the summary would NameError."
    )
    init = src.index("nmbp   = 0.0")
    assign = src.index('nmbp   = float(account.get("non_marginable_buying_power"')
    assert init < assign, "the default must be set before the conditional assignment"


def test_the_stage_two_account_line_reports_it(src):
    assert "non_marginable_bp=${nmbp:.2f}" in src, (
        "the stage-2 Account line no longer reports non-marginable buying power."
    )


def test_the_run_summary_reports_it(src):
    """The summary is what reaches Discord and (since #1403) the log."""
    assert "crypto_bp=${nmbp:,.2f}" in src, (
        "the run summary dropped crypto buying power. Without it the header "
        "shows a well-funded book next to a crypto desk that cannot place a $25 "
        "order, with nothing connecting the two."
    )


def test_the_gate_this_reports_on_still_exists(src):
    """A report about a guard that no longer runs is worse than no report."""
    assert 'field = "non_marginable_buying_power" if is_crypto else "buying_power"' in src, (
        "cash_capped_notional no longer switches fields by asset class, so "
        "crypto_bp in the summary would describe a decision that is not made."
    )
    assert "MIN_ORDER_USD = 25.0" in src, "MIN_ORDER_USD moved or changed"
