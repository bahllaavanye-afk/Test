"""The run summary must reach stdout, not only Discord.

`summary` is the densest artifact the desk produces: funnel counts (generated →
survived → placed), per-desk drop reasons, execution slippage, and
equity/cash/buying_power. It was built and handed straight to
`_post_chat("#pnl-daily", summary)` — never printed.

So the whole synthesised picture lived in exactly one place. A `notify` failure,
a rotated bot token, a rate limit, or simply nobody reading `#pnl-daily`, and it
is gone; the Actions log retains only the raw per-signal lines it was derived
from.

This was found the way these things usually are: trying to verify the
`cash=`/`buying_power=` fields added to that same summary earlier on 2026-08-04.
They could not be confirmed from any run log, because the line carrying them is
never written to stdout. **A change whose output cannot be observed is
indistinguishable from one that never shipped** — which is the failure mode this
repo keeps rediscovering, this time in my own work.

The print is ordered BEFORE `_post_chat` deliberately: if Discord raises, the
record still exists.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent / "desk_order_placer.py"


@pytest.fixture(scope="module")
def src() -> str:
    return _SRC.read_text()


def test_the_summary_is_printed(src):
    assert "print(summary, flush=True)" in src, (
        "the run summary is no longer printed to stdout. It would exist only in "
        "Discord again — one delivery failure from being lost entirely, and "
        "unverifiable from any run log."
    )


def test_the_print_precedes_the_discord_post(src):
    """Order matters: a raising _post_chat must not take the record with it."""
    idx_print = src.index("print(summary, flush=True)")
    idx_post = src.index('_post_chat("#pnl-daily", summary)')
    assert idx_print < idx_post, (
        "the summary is printed AFTER _post_chat. If the Discord call raises, "
        "the log keeps nothing — which is the single-point-of-failure this guard "
        "exists to remove."
    )


def test_the_printed_value_is_the_posted_value(src):
    """Print the same object, not a re-rendered approximation of it.

    A separately-formatted log line would drift from what Discord shows, and the
    two would disagree exactly when someone is trying to reconcile them.
    """
    tree = ast.parse(src)
    printed_summary = False
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "print"):
            continue
        for arg in node.args:
            if isinstance(arg, ast.Name) and arg.id == "summary":
                printed_summary = True
    assert printed_summary, (
        "no `print(summary)` found — the log line is being built separately from "
        "the Discord payload and can drift from it."
    )


def test_the_summary_still_carries_its_four_signals(src):
    """Guard the content the print exists to preserve."""
    for token, why in [
        ("funnel:", "the generated→survived→placed funnel"),
        ("execution:", "the slippage/TCA line"),
        ("cash=$", "available cash — distinguishes a deployed book from a quiet market"),
        ("equity=$", "account equity"),
    ]:
        assert token in src, f"the summary no longer includes {why} ({token!r})"
