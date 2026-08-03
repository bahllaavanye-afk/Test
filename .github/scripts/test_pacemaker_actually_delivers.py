"""The pacemaker's value is entirely in its LAST step, so anything that stops
the job early is a total loss — and for most of its life something did.

`pacemaker.yml` is shaped "sleep 3000s, THEN dispatch". It carried
`concurrency: cancel-in-progress: true`, justified in a comment as collapsing a
burst of PRs into a single heartbeat. But its five ignition workflows plus CI
are independently phased and arrive far more often than every 50 minutes, so in
practice each arrival cancelled the sleeper before it reached the dispatch.

Measured 2026-08-03 over the workflow's last 30 runs:

    cancelled  25      durations 0.1 .. 47.9 min   (sleep step is 50.0 min)
    success     4      durations 50.2 .. 50.4 min  — these are the only ones
    running     1                                     that dispatched anything

Not one cancelled run survived to 50 minutes. The heartbeat ran at roughly half
its intended rate, and the losses were recorded as `cancelled` — a conclusion no
alert path in this repo watches, and which reads as green at a glance. Same
family as the 403 that hid under `continue-on-error` in
continuous-improvement.yml: the failure was visible in principle and invisible
in practice.

The second thing pinned here is who gets dispatched. Both desk workflows declare
`workflow_run: workflows: ["CI"]` so they can ride CI completions instead of
cron. Across their last 30 runs each, desk-trading fired 28x schedule + 2x push
and the crypto desk 30x schedule — ZERO workflow_run events on either. That
trigger has never delivered, leaving the desks on cron that GitHub starves: 12
of desk-trading's last 30 runs landed inside US regular trading hours, against a
nominal 26 per weekday. Every out-of-hours run generates signals and throws them
all away (run 30673525449: conf=1.00 SLV, 1.00 EPOL, 0.98 EIDO, 0 orders).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

_WF = Path(__file__).resolve().parents[1] / "workflows"
_PACEMAKER = _WF / "pacemaker.yml"

# Every workflow the pacemaker POSTs a /dispatches call to.
_DISPATCH_TARGETS = (
    "test.yml",          # CI — drives the workflow_run fleet
    "auto-merge.yml",    # merge gate, own schedule is starved
    "desk-trading.yml",  # ALL NINE desks, crypto included (it is always_open)
)

# Deliberately NOT dispatched. See test_the_crypto_desk_is_not_dispatched_too.
_MUST_NOT_DISPATCH = ("desk-trading-crypto-24x7.yml",)


@pytest.fixture(scope="module")
def text() -> str:
    assert _PACEMAKER.exists(), f"{_PACEMAKER} is gone — the fleet heartbeat moved"
    return _PACEMAKER.read_text()


@pytest.fixture(scope="module")
def parsed() -> dict:
    return yaml.safe_load(_PACEMAKER.read_text())


def test_the_pacemaker_does_not_cancel_itself(parsed):
    """`cancel-in-progress: true` on a sleep-then-act job discards the act.

    This is the regression that cost 25 of 30 heartbeats. Keep it false so the
    in-progress sleeper always reaches its dispatch step; GitHub parks the newest
    trigger as pending (superseding any older pending run), which preserves the
    "exactly one pacemaker" property the original comment wanted without
    destroying the run that is about to do the work.
    """
    concurrency = parsed.get("concurrency")
    assert concurrency, "pacemaker lost its concurrency group — sleepers will stack"
    assert concurrency.get("cancel-in-progress") is not True, (
        "pacemaker.yml is back to cancel-in-progress: true. Its upstream "
        "triggers arrive every ~18 min while the job sleeps 50 min before "
        "dispatching, so this cancels the heartbeat before it fires: measured "
        "25 cancelled / 4 success over 30 runs, with every cancelled run dying "
        "short of the sleep (max 47.9 min)."
    )


def test_the_sleep_is_shorter_than_the_job_timeout(parsed):
    """A sleep that outlives the timeout is the same bug by another route."""
    job = parsed["jobs"]["heartbeat"]
    timeout_s = int(job["timeout-minutes"]) * 60
    sleeps = [
        int(m)
        for step in job["steps"]
        for m in re.findall(r"sleep\s+(\d+)", str(step.get("run", "")))
    ]
    assert sleeps, "the pacemaker no longer sleeps — it would dispatch in a tight loop"
    assert max(sleeps) < timeout_s, (
        f"sleep {max(sleeps)}s >= timeout {timeout_s}s: the job would be killed "
        f"before dispatching, exactly like the cancel-in-progress regression."
    )


def _dispatched_workflows(text: str) -> set[str]:
    """Every workflow file the pacemaker can actually POST /dispatches for.

    Three forms are in use and all must be recognised, or the test measures its
    own regex instead of the workflow:
      1. a literal URL      — `.../workflows/test.yml/dispatches`
      2. a shell loop       — `for wf in a.yml b.yml; do ... /workflows/$wf/...`
      3. a plain assignment — `wf=a.yml` ... `/workflows/$wf/dispatches`

    Matching only form 1 would silently report the desks as "not dispatched"
    while they were being dispatched perfectly well; form 3 was added when the
    desk loop collapsed to a single target and immediately broke this helper,
    which is exactly what `test_the_loop_resolver_is_not_vacuous` guards against.
    """
    found = set(re.findall(r"workflows/([A-Za-z0-9._-]+\.yml)/dispatches", text))
    if re.search(r"workflows/\$\{?wf\}?/dispatches", text):
        for loop in re.findall(r"for\s+wf\s+in\s+([^\n;]+)", text):
            found.update(w for w in loop.split() if w.endswith(".yml"))
        found.update(re.findall(r"^\s*wf=([A-Za-z0-9._-]+\.yml)\s*$", text, re.M))
    return found


@pytest.mark.parametrize("target", _DISPATCH_TARGETS)
def test_the_pacemaker_dispatches_it(text, target):
    assert target in _dispatched_workflows(text), (
        f"the pacemaker no longer dispatches {target}. Cron cannot be relied on "
        f"here — free-tier schedules are starved (measured 1h20m-3h12m late), "
        f"and for the desks the workflow_run trigger has never fired at all."
    )


def test_the_loop_resolver_is_not_vacuous(text):
    """Guard the helper above: a resolver that finds nothing passes nothing."""
    found = _dispatched_workflows(text)
    assert len(found) >= len(_DISPATCH_TARGETS), (
        f"_dispatched_workflows resolved only {sorted(found)} — if the workflow "
        f"changed how it spells the dispatch URL, the parametrised tests above "
        f"would all fail for the wrong reason."
    )


@pytest.mark.parametrize("target", _DISPATCH_TARGETS)
def test_dispatch_targets_accept_workflow_dispatch(target):
    """A POST to /dispatches 404s unless the target declares workflow_dispatch.

    Without this the pacemaker would keep reporting a failed dispatch forever
    while every downstream workflow quietly went idle.
    """
    wf = _WF / target
    assert wf.exists(), f"{target} is gone but the pacemaker still dispatches it"
    triggers = yaml.safe_load(wf.read_text())
    # PyYAML parses the bare key `on:` as the boolean True.
    on = triggers.get("on", triggers.get(True))
    assert "workflow_dispatch" in on, (
        f"{target} does not declare workflow_dispatch, so the pacemaker's POST "
        f"to its /dispatches endpoint returns 404 and the workflow is never "
        f"driven."
    )


@pytest.mark.parametrize("target", _MUST_NOT_DISPATCH)
def test_the_crypto_desk_is_not_dispatched_too(text, target):
    """Dispatching both desk workflows recreates a measured collision.

    `desk-trading.yml` runs ALL NINE desks and crypto is `always_open=True`, so
    one dispatch already covers the 24/7 desk. `desk-trading-crypto-24x7.yml`
    exists only to fill the hours the equity cron cannot reach, and its job
    carries

        if: github.event_name == 'schedule' || github.event_name == 'workflow_dispatch'

    specifically so it CEDES every trigger it shares with the equity workflow —
    the two use different concurrency groups, so on a shared trigger they run in
    PARALLEL and compete for Alpaca's free-tier data limit. Measured over 60 runs
    (2026-07-28): 22 collided. One pair on sha 49e46ded had desk-trading fetch 70
    bars while the crypto-only run got 5 and 429s — the crypto run was strictly
    worse AND degraded its own twin.

    `workflow_dispatch` is on that allowlist, so a dispatch from here is the one
    route the cede-rule cannot block. This test exists because the first version
    of the pacemaker step dispatched both and would have reintroduced the bug.
    """
    assert target not in _dispatched_workflows(text), (
        f"the pacemaker dispatches {target} as well as desk-trading.yml. Both "
        f"run the crypto desk, in different concurrency groups, so they execute "
        f"in parallel and race for Alpaca bars — 22 of 60 runs collided when "
        f"this last happened. desk-trading.yml alone already covers crypto."
    )


def test_losing_a_desk_dispatch_does_not_kill_the_ci_heartbeat(text):
    """CI is the load-bearing dispatch; the desks must not be able to abort it."""
    # Bound the search to the desk step rather than the whole remaining file.
    step = text[text.index("wf=desk-trading.yml"):].split("\n      - name:")[0]
    assert "exit 1" not in step, (
        "the desk dispatch exits non-zero on failure. The CI dispatch is the "
        "heartbeat for 36 downstream workflows and must survive a desk dispatch "
        "failing."
    )
    assert "::error::" in step, (
        "the desk dispatch swallows its failure. A silent skip is the bug class "
        "this repo keeps paying for — see the 403 under continue-on-error in "
        "continuous-improvement.yml."
    )
