"""The merge gate had no trigger that a bot-authored PR could ever fire.

Measured 2026-08-03. `auto-merge.yml`'s last run was **2026-07-29 23:45**, on a
human-pushed branch. In the three days since, the improver opened ~90 PRs and
the gate never woke once. `#1341` sat green — `test`, `test-agents`,
`frontend-build` all success — labelled `automerge`, not a draft, unmerged.

CORRECTED 2026-08-03 04:40. My first explanation here was that every declared
trigger is *suppressed* for bot PRs. That is wrong, and the run log says so:
auto-merge DID fire at 03:38 (`30782308923`, `event=pull_request_target`). Its
entire output was one line —

    #232: base claude/advanced-trading-bot-d5Lmw != main

— one PR evaluated, skipped, done.

The real mechanism is narrower and worse. A `pull_request_target` payload sets
`context.payload.pull_request`, so `candidates` gets exactly ONE entry and the
`candidates.size === 0` branch — the only path that scans all open PRs — never
runs. **The gate wakes, but only ever for the single PR whose event woke it.**
Nothing sweeps the backlog, so a PR that goes green after its own event has
passed is never revisited.

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


def test_the_pacemaker_dispatches_the_gate_too():
    """The schedule alone was not enough — it produced zero runs in 2h47m.

    `auto-merge.yml`'s own `schedule` is subject to exactly the starvation the
    pacemaker exists to route around; its header says so outright ("GitHub
    starves free-tier schedules under load"). So the gate gets dispatched from
    the pacemaker's heartbeat as well, which does not depend on cron.

    The two are deliberately kept BOTH: they show up distinguishably in the run
    log (`event=schedule` vs `event=workflow_dispatch`), so whichever actually
    delivers can be identified rather than guessed at.
    """
    pm = _WF.parent / "pacemaker.yml"
    assert pm.exists(), "pacemaker.yml is gone — the fleet heartbeat moved"
    text = pm.read_text()
    assert "auto-merge.yml/dispatches" in text, (
        "the pacemaker no longer dispatches the merge gate. Its own schedule is "
        "starved (zero runs in 2h47m when measured), so without this the gate "
        "only ever evaluates the single PR whose event woke it and green PRs "
        "accumulate indefinitely."
    )
    # A dispatch that cannot report its own failure is the 403 bug again.
    block = text[text.index("auto-merge.yml/dispatches"):]
    assert "::error::" in block[:900], (
        "the auto-merge dispatch swallows its failure. A silent skip here is "
        "the same class as the permanent 403 that hid in "
        "continuous-improvement.yml for its entire lifetime."
    )


def test_the_merge_dispatch_does_not_kill_the_fleet_heartbeat():
    """Losing the merge sweep must not take the whole pacemaker down with it."""
    text = (_WF.parent / "pacemaker.yml").read_text()
    block = text[text.index("auto-merge.yml/dispatches"):]
    assert "exit 1" not in block[:900], (
        "the auto-merge dispatch exits non-zero on failure. The CI dispatch "
        "above it is the load-bearing heartbeat for 36 downstream workflows; a "
        "failed merge sweep must not abort the job before/around it."
    )
