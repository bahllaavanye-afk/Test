"""Guard: the two desk workflows must not be ignited by the same event.

THE MEASUREMENT (2026-07-28 06:46, both runs ignited by one CI completion):

    desk-trading.yml          bars_fetched=70   all 20 crypto symbols
    desk-trading-crypto-24x7  bars_fetched=5    ← lost the race, HTTP 429

Both rode `workflow_run: workflows: ["CI"] types: [completed]`, neither has a
job-level market-hours gate, and they use *different* concurrency groups
(`desk-trading` vs `desk-crypto`) so nothing serialises them. Every CI
completion therefore launched two runs in parallel that hammered the same
free-tier Alpaca data API — and `desk-trading.yml` already runs EVERY desk,
crypto included, 24/7.

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


def test_the_crypto_desk_does_not_also_run_on_workflow_run():
    """The regression: two parallel runs competing for one rate limit."""
    crypto = _text(_CRYPTO)
    if "workflow_run:" not in crypto:
        return  # trigger removed outright — also fine
    assert re.search(r"if:\s*github\.event_name\s*!=\s*'workflow_run'", crypto), (
        "desk-trading-crypto-24x7.yml still rides workflow_run with no guard — "
        "it will run in parallel with desk-trading.yml and lose the Alpaca "
        "rate-limit race (measured: 5 of 20 crypto symbols vs 20 of 20)"
    )


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
