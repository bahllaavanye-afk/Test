"""The merge gate had no trigger that a bot-authored PR could ever fire.

Measured 2026-08-03. `auto-merge.yml`'s last run was **2026-07-29 23:45**, on a
human-pushed branch. In the three days since, the improver opened ~90 PRs and
the gate never woke once. `#1341` sat green — `test`, `test-agents`,
`frontend-build` all success — labelled `automerge`, not a draft, unmerged.

Every declared trigger is dead for exactly the PRs this gate exists to land,
because GitHub suppresses workflow runs from events attributed to GITHUB_TOKEN
and every step of the improver's loop uses it:

    pull_request_target: labeled     the bot applies `automerge`   -> suppressed
    check_suite: completed           checks from the bot's CI      -> suppressed
    workflow_run: [CI] completed     CI dispatched by the bot      -> suppressed

Every historical auto-merge run confirms it: the event is `pull_request_target`
or `check_suite`, and each traces back to a human push.

This is the next link in the same chain as the missing `actions: write` (#1245).
That fix made CI actually run on improver PRs. This one makes something read the
result. Fixing one stage keeps exposing the next — worth expecting rather than
being surprised by.

`workflow_dispatch` is the documented exception to the recursion guard and was
already declared here, but nothing ever called it. The fix is a `schedule`: a
heartbeat the gate owns, independent of any bot event.

**The schedule is only meaningful because of the zero-candidate fallback.** A
scheduled run has no `pull_request`, `check_suite` or `workflow_run` payload, so
candidate collection yields nothing; without the `candidates.size === 0` branch
that scans all open PRs, the run would succeed having examined zero PRs — a
green no-op, the exact failure class this repo keeps paying for. The two must
travel together, which is what `test_the_schedule_is_not_decorative` pins.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_WF = Path(__file__).resolve().parents[1] / "workflows" / "auto-merge.yml"


@pytest.fixture(scope="module")
def text() -> str:
    assert _WF.exists(), f"{_WF} is gone — the merge gate moved"
    return _WF.read_text()


def test_the_gate_has_a_trigger_no_bot_event_is_needed_for(text):
    """A schedule (or another self-owned heartbeat) must be declared."""
    assert re.search(r"^\s*schedule:", text, re.M), (
        "auto-merge.yml declares only bot-suppressed triggers again. "
        "pull_request_target/check_suite/workflow_run are all attributed to "
        "GITHUB_TOKEN for improver PRs, so none of them fires and green PRs sit "
        "unmerged indefinitely."
    )
    assert re.search(r"^\s*-\s*cron:", text, re.M), "the schedule has no cron entry"


def test_the_schedule_is_not_decorative(text):
    """Without the zero-candidate fallback, a scheduled run inspects nothing.

    This is the coupling that matters: a scheduled trigger plus payload-only
    candidate collection is a workflow that succeeds having done nothing at all.
    """
    assert "candidates.size === 0" in text, (
        "the fallback that scans all open PRs is gone. A scheduled run has no "
        "event payload, so it would collect zero candidates and exit green "
        "having merged nothing — the schedule would be pure decoration."
    )
    fallback = text.index("candidates.size === 0")
    scan = text.index("state: 'open'", fallback)
    assert scan > fallback, "the fallback must actually list open PRs"


def test_the_required_checks_gate_survives(text):
    """The heartbeat must not become a way to merge unvalidated PRs.

    Adding a trigger that fires without any CI event makes it more important,
    not less, that the gate still refuses a PR whose checks never ran.
    """
    assert "REQUIRED_CHECKS" in text, (
        "the required-checks list is gone. With a schedule trigger the gate now "
        "runs when no CI event has occurred at all, so 'checks must be PRESENT, "
        "not merely non-failing' is load-bearing."
    )
    for name in ("test", "test-agents", "frontend-build"):
        assert f"'{name}'" in text, f"required check {name!r} no longer listed"


def test_the_label_gate_survives(text):
    """Opt-in by label is what keeps a periodic sweep from merging by accident."""
    assert "REQUIRED_LABEL = 'automerge'" in text, (
        "the opt-in label gate is gone. A scheduled sweep over ALL open PRs "
        "without it would auto-merge anything that happens to be green."
    )
