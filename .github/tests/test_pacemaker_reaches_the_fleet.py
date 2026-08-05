"""The pacemaker's CI cascade never reached the 36 workflows chained off it.

`pacemaker.yml`'s header states the design: 36 workflows are chained off
`workflow_run: workflows: ["CI"]`, so dispatching CI revives the whole fleet.
Measured 2026-08-05, it does not.

    CI on main, dispatched by the pacemaker, succeeded at 01:26, 02:16, 04:21
    and 05:11 — triggering_actor `github-actions[bot]`.
    Downstream workflow_run events from those four completions: ZERO.

    The last real cascade was 01:01 — Peer Review, Desk Trading, Fill Tracker,
    employee-conversations and multi-agent-discussion all firing together. Its
    triggering_actor was `bahllaavanye-afk (User)`: a PR merge.

GitHub's recursion guard is the cause — "events triggered by the GITHUB_TOKEN
will not create a new workflow run". `workflow_dispatch` is the documented
exception that lets the pacemaker START CI, and that part demonstrably works.
But the CI run it starts is itself GITHUB_TOKEN-triggered, so *its* `completed`
event cascades to nothing. The exception covers the dispatch, not the
descendants.

The symptom is precisely what the pacemaker was built to cure: the fleet wakes
when a human merges a PR and is otherwise left to free-tier cron.
`employee-conversations` is on `cron: '5 * * * *'` and ran four times in eight
hours (20:36, 22:16, 00:09, 04:14).

These tests pin the direct-dispatch workaround. They are deliberately about
WIRING, not about the count of dispatched workflows: the list is a judgement
call (dispatching all 36 every 50 minutes is an unmeasured change in free-tier
minutes) and is expected to grow.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

import yaml

_WF_DIR = Path(__file__).resolve().parents[1] / "workflows"
_PACEMAKER = _WF_DIR / "pacemaker.yml"

FLEET = ["employee-conversations.yml", "multi-agent-discussion.yml"]


@pytest.fixture(scope="module")
def pacemaker_src() -> str:
    return _PACEMAKER.read_text()


@pytest.fixture(scope="module")
def pacemaker_doc() -> dict:
    return yaml.safe_load(_PACEMAKER.read_text())


def test_the_pacemaker_dispatches_the_fleet_directly(pacemaker_src):
    for wf in FLEET:
        assert wf in pacemaker_src, (
            f"{wf} is no longer dispatched by the pacemaker. It is chained off "
            "workflow_run: [CI], and a pacemaker-dispatched CI run does not "
            "cascade — so it falls back to free-tier cron, which delivered 4 "
            "runs in 8 hours."
        )


@pytest.mark.parametrize("wf", FLEET)
def test_each_dispatched_workflow_exists(wf):
    assert (_WF_DIR / wf).exists(), (
        f"the pacemaker dispatches {wf}, which does not exist — the POST 404s "
        "and the step reports a failure every 50 minutes."
    )


@pytest.mark.parametrize("wf", FLEET)
def test_each_dispatched_workflow_accepts_workflow_dispatch(wf):
    """A dispatch to a workflow without the trigger is a silent 404."""
    doc = yaml.safe_load((_WF_DIR / wf).read_text())
    # PyYAML parses the bare key `on:` as the boolean True.
    triggers = doc.get("on", doc.get(True))
    assert isinstance(triggers, dict) and "workflow_dispatch" in triggers, (
        f"{wf} does not declare workflow_dispatch, so the pacemaker's POST "
        "cannot start it."
    )


@pytest.mark.parametrize("wf", FLEET)
def test_each_dispatched_workflow_queues_rather_than_clobbers(wf):
    """A dispatch overlapping a cron run must not cancel it mid-flight.

    These write agent_memory.json; cancelling a run between load and save loses
    whatever it had accumulated.
    """
    doc = yaml.safe_load((_WF_DIR / wf).read_text())
    conc = doc.get("concurrency")
    if conc is None:
        pytest.skip(f"{wf} declares no concurrency group — nothing to clobber")
    assert conc.get("cancel-in-progress") in (False, None), (
        f"{wf} sets cancel-in-progress: true. A pacemaker dispatch would kill "
        "an in-flight cron run of the same workflow."
    )


def test_the_dispatch_step_is_gated_on_the_killswitch(pacemaker_doc):
    """Every other dispatch honours it; a new one that ignores it is a hole."""
    # Derive the job key rather than hardcoding it — an earlier version guessed
    # "beat" (it is "heartbeat") and the KeyError read as a wiring failure.
    (job,) = pacemaker_doc["jobs"].values()
    steps = job["steps"]
    fleet_steps = [s for s in steps if "agent fleet" in str(s.get("name", "")).lower()]
    assert fleet_steps, "the fleet dispatch step disappeared"
    for s in fleet_steps:
        assert "killswitch" in str(s.get("if", "")), (
            "the fleet dispatch ignores the killswitch, so it keeps firing "
            "when the pacemaker is meant to be off."
        )


def test_a_failed_dispatch_is_reported_not_swallowed(pacemaker_src):
    """The failure this repo keeps hitting: a dead step that looks fine."""
    block = pacemaker_src[pacemaker_src.index("Dispatch the agent fleet"):]
    block = block[:block.index("Alert if the heartbeat broke")]
    assert "::error::" in block, (
        "a failed fleet dispatch no longer emits ::error::, so the fleet going "
        "quiet again would be invisible in the run summary."
    )
    assert "GITHUB_STEP_SUMMARY" in block, "the dispatch result is not surfaced"


def test_it_does_not_dispatch_the_crypto_desk(pacemaker_src):
    """Guards a documented, measured race — see the desk-trading comment.

    desk-trading.yml already runs all nine desks including the always-open
    crypto one; dispatching desk-trading-crypto-24x7.yml too makes the pair run
    in parallel and compete for Alpaca's free-tier data limit (22 collisions in
    60 runs, 2026-07-28).
    """
    # Executable lines only. The filename appears in the comment that explains
    # why it is NOT dispatched, so a whole-file match fails on its own rationale
    # — the same false positive that bit the conversations-producer scan.
    live = [ln for ln in pacemaker_src.splitlines() if not ln.lstrip().startswith("#")]
    assert not any("desk-trading-crypto-24x7.yml" in ln for ln in live), (
        "the pacemaker dispatches the crypto desk as well as desk-trading — "
        "that recreates the measured data-limit collision."
    )


def test_the_ci_dispatch_still_happens(pacemaker_src):
    """The direct dispatch supplements the heartbeat, it does not replace it.

    CI on main is still what keeps the merge gate and deploy path moving, and
    still what a HUMAN merge cascades from.
    """
    assert "test.yml/dispatches" in pacemaker_src, (
        "the pacemaker no longer dispatches CI — the heartbeat that re-triggers "
        "this workflow is gone and the whole chain stops."
    )


def test_the_finding_is_recorded_where_the_next_reader_looks(pacemaker_src):
    """The header claims the CI cascade revives the fleet. It must not still.

    Three findings this session were independently re-derived because the
    correction lived somewhere other than the claim.
    """
    idx = pacemaker_src.index("Dispatch the agent fleet")
    preamble = pacemaker_src[:idx]
    assert "recursion guard" in preamble or "GITHUB_TOKEN will not create" in preamble, (
        "the reason the CI cascade cannot reach the fleet is no longer "
        "explained in pacemaker.yml. The header still claims dispatching CI "
        "revives 36 workflows, and the next reader will believe it."
    )
