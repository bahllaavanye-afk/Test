"""A desk channel that only speaks on success cannot report a stall.

Per-desk Discord posts happened in exactly one branch: `if desk_order_list:`.
So `#desk-equities`, `#desk-crypto` and the other seven said something only when
the desk **placed an order**. A desk that fired signals and had every one
dropped produced the same output as a desk that never ran: nothing.

Measured 2026-08-05 during the deep review — buying power across ten desk runs:

    00:41 $0.00   01:31 $0.00    04:26 $116.98
    00:58 $0.00   02:22 $46.35   05:15 $101.23
    01:03 $27.61  07:05 $115.79  07:56 $206.86

`cash` pinned at exactly -$33,401.86 from 00:58 to 07:56 — eight consecutive
runs, zero fills. Run 30986611287 generated 51 signals and placed 0. **Every
`#desk-*` channel was silent for those seven hours** while the desks were
working perfectly and the account was simply out of margin.

Silence is the one thing a monitoring channel must never mean, and this repo has
now paid for that lesson in `#pnl-daily` (the run summary), in the desk funnel
line, and here.

The post is gated on `n_sig` deliberately: a genuinely quiet market — no signals
at all — still says nothing, so a closed-market weekend cannot turn nine desks
into a noise generator.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent / "desk_order_placer.py"


@pytest.fixture(scope="module")
def src() -> str:
    return _SRC.read_text()


@pytest.fixture(scope="module")
def zero_order_branch(src: str) -> str:
    """The `else:` arm taken when a desk placed nothing."""
    i = src.index("n_sig = desk_signals.get(desk.name, 0)")
    j = src.index("tracker.set_output(orders_placed=", i)
    return src[i:j]


def test_a_stalled_desk_posts_to_its_own_channel(zero_order_branch):
    assert "_post_chat(" in zero_order_branch, (
        "a desk that fired signals and placed nothing still posts only to the "
        "aggregate summary. Its own channel stays silent, which is "
        "indistinguishable from the desk never having run."
    )
    assert "desk.chat_channel" in zero_order_branch, (
        "the zero-order post does not target the desk's own channel"
    )


def test_the_post_names_the_drop_reason(zero_order_branch):
    """'0 orders' without a reason just moves the ambiguity."""
    # Scope to the _post_chat CALL, not the whole branch. `why` also appears in
    # the aggregate desk_summaries line above, so a branch-wide match passes
    # even when the channel post hardcodes "reason: unknown".
    call = zero_order_branch[zero_order_branch.index("_post_chat("):]
    call = call[:call.index(")\n") + 1]
    assert "{why}" in call, (
        "the zero-order post no longer interpolates the drop reason, so the "
        "channel says a desk stalled without saying why — insufficient cash, "
        "market closed and no order path would all look identical."
    )
    assert "{n_sig}" in call, "the post no longer reports how many signals fired"


def test_a_silent_desk_stays_silent(zero_order_branch):
    """No signals at all must NOT post — otherwise weekends become spam.

    Nine desks x ~29 runs/day of "nothing happened" would train people to mute
    the channels, which costs more than the ambiguity being fixed.
    """
    post_idx = zero_order_branch.index("_post_chat(")
    guard = zero_order_branch[:post_idx]
    assert "if n_sig:" in guard, (
        "the zero-order post is outside the `if n_sig:` guard, so a desk with "
        "no signals — a closed market — posts too."
    )
    # The `else` arm for n_sig == 0 must remain post-free.
    tail = zero_order_branch[zero_order_branch.index("else:", post_idx):]
    assert "_post_chat(" not in tail, (
        "the no-signals branch now posts as well; a quiet market will flood "
        "nine desk channels every run."
    )


def test_the_success_path_still_posts(src):
    """The fix must not disturb the branch that already worked."""
    assert 'lines = [f"*{desk.name} Desk* — {len(desk_order_list)} order(s) placed"]' in src, (
        "the order-placed summary changed or disappeared"
    )
    assert src.count("_post_chat(desk.chat_channel") >= 1


def test_both_desk_posts_are_distinguishable(src):
    """A reader scanning the channel must tell a fill from a stall at a glance."""
    assert "order(s) placed" in src and "0 orders placed" in src, (
        "the placed and stalled messages are no longer distinct strings"
    )


def test_the_aggregate_summary_is_unchanged(zero_order_branch):
    """#pnl-daily keeps its funnel line — this adds a channel, removes nothing."""
    assert "desk_summaries.append(" in zero_order_branch, (
        "the aggregate per-desk summary line was dropped in favour of the "
        "channel post; #pnl-daily loses its funnel breakdown."
    )


def test_every_desk_has_a_channel_to_post_to(src):
    """A desk without a channel would raise inside the new post call."""
    names = re.findall(r'DeskConfig\(\s*\n\s*name="([^"]+)"', src)
    channels = re.findall(r'chat_channel="(#[a-z0-9-]+)"', src)
    assert len(names) == len(channels) >= 9, (
        f"{len(names)} desks but {len(channels)} channels — a desk without a "
        "chat_channel now hits _post_chat on the stalled path."
    )
