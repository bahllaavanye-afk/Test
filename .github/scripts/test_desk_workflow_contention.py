"""Guard: the two desk workflows must not be ignited by the same event.

THE MEASUREMENT. Over the last 60 runs of each workflow (2026-07-28), 22 fired
at the identical timestamp as their twin:

    (workflow_run, workflow_run) -> 15
    (push,         push)         -> 7

One of the push pairs, 06:46:05, both on sha 49e46ded:

    desk-trading.yml          bars_fetched=70   all 20 crypto symbols
    desk-trading-crypto-24x7  bars_fetched=5    ← lost the race, HTTP 429

They share BOTH triggers: `workflow_run: ["CI"] completed`, and a `push` path
filter that both list (`.github/scripts/desk_order_placer.py`). Neither has a
job-level market-hours gate, and they use *different* concurrency groups
(`desk-trading` vs `desk-crypto`) so nothing serialises them — two runs in
parallel hammering the same free-tier Alpaca data API, while
`desk-trading.yml` already runs EVERY desk, crypto included, 24/7.

FIRST FIX WAS HALF A FIX. It guarded `github.event_name != 'workflow_run'`,
which covers 15 of the 22 — but the pair actually measured above was a *push*
collision, so the headline evidence was not even an instance of what that
guard blocked. Now a whitelist (`schedule` or `workflow_dispatch`): this
workflow exists for its own cron, so anything it shares with the equity
workflow it should cede.

So the crypto-only run was doing duplicate work, doing it worse (5 of 20
symbols), and degrading the twin that was doing it properly. Thin data is not
cosmetic here: it feeds the ensembles, and the crypto desks had been logging
`sell/buy conflict — stand aside` on nearly every symbol.

The crypto workflow's real value is its own cron, for stretches when CI is not
completing. On `workflow_run` it adds nothing, so it now skips.

Text-scan based, like `test_workflow_discord_env.py` — no yaml dependency.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_WORKFLOWS = Path(__file__).resolve().parents[1] / "workflows"
_EQUITY = _WORKFLOWS / "desk-trading.yml"
_CRYPTO = _WORKFLOWS / "desk-trading-crypto-24x7.yml"


def _text(p: Path) -> str:
    assert p.is_file(), f"missing workflow {p}"
    return p.read_text()


def test_both_desk_workflows_exist():
    assert _EQUITY.is_file() and _CRYPTO.is_file()


def test_the_equity_desk_still_rides_ci_completions():
    """It is the one that should — it covers every desk, crypto included.

    If this ever stops being true the skip below becomes a coverage hole, so
    the two assertions are deliberately coupled.
    """
    assert "workflow_run:" in _text(_EQUITY)


_WHITELIST = re.compile(
    r"if:\s*github\.event_name\s*==\s*'schedule'\s*\|\|\s*"
    r"github\.event_name\s*==\s*'workflow_dispatch'"
)


def test_the_crypto_desk_only_runs_on_its_own_schedule():
    """The regression: two parallel runs competing for one rate limit.

    A whitelist, not a blacklist. Blacklisting `workflow_run` alone left the
    `push` collision live — 7 of the 22 measured pairs, including the one the
    original fix cited as its evidence.
    """
    assert _WHITELIST.search(_text(_CRYPTO)), (
        "desk-trading-crypto-24x7.yml must run ONLY on schedule/workflow_dispatch — "
        "it shares both workflow_run and the desk_order_placer.py push path with "
        "desk-trading.yml, and loses the Alpaca rate-limit race when they collide "
        "(measured: 5 of 20 crypto symbols vs 20 of 20)"
    )


@pytest.mark.parametrize("shared_trigger", ["workflow_run", "push"])
def test_every_trigger_shared_with_the_equity_desk_is_ceded(shared_trigger):
    """Both shared ignition paths must be covered, not just the first one found.

    `workflow_run` was fixed first and `push` was missed. Enumerating the
    overlap makes a partial fix fail rather than look complete.
    """
    crypto, equity = _text(_CRYPTO), _text(_EQUITY)
    if shared_trigger not in crypto or shared_trigger not in equity:
        return  # no longer shared — nothing to cede
    assert _WHITELIST.search(crypto), (
        f"both workflows declare `{shared_trigger}`, so the crypto job must be "
        f"whitelisted to schedule/workflow_dispatch only"
    )


def test_the_shared_push_path_is_still_the_reason_this_guard_exists():
    """Documents the overlap, so removing it later is a deliberate act."""
    shared = ".github/scripts/desk_order_placer.py"
    assert shared in _text(_CRYPTO) and shared in _text(_EQUITY)


def test_the_crypto_desk_keeps_its_own_schedule():
    """The skip must not leave crypto with no independent trigger at all.

    Without this, 'stop the duplicate run' and 'delete round-the-clock crypto
    coverage' look identical in the diff.
    """
    crypto = _text(_CRYPTO)
    assert "schedule:" in crypto and "cron:" in crypto
    assert "workflow_dispatch" in crypto


def test_the_two_desks_keep_separate_concurrency_groups():
    """Documents WHY they can collide: nothing serialises them.

    If someone ever puts them in one group the contention disappears for a
    different reason, and the guard above can be revisited.
    """
    assert "group: desk-trading" in _text(_EQUITY)
    assert "group: desk-crypto" in _text(_CRYPTO)
