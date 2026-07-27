"""The fleet must not be downstream of a human pushing code.

Measured 2026-07-27: 36 workflows — every desk, company-brain, error-triage,
continuous-improvement, employee-conversations — are chained off
`workflow_run: workflows: ["CI"]`. CI (test.yml) triggers ONLY on
`pull_request` and `workflow_dispatch`.

So the entire fleet was downstream of somebody opening a PR, and cron could not
cover the gap. desk-trading-crypto is configured `7,27,47 * * * *` (3x/hour)
and actually fired at 23:25, 01:05, 04:47, 08:39, 11:57 — gaps of 3.5 hours
overnight, ~10% of its intended rate, while running near-hourly during the day
when PRs were landing.

The reported symptom was exact: "Discord is only active when this chat
resumes."

`pacemaker.yml` closes the loop: sleep → dispatch CI → CI completion
re-triggers the pacemaker → sleep → dispatch. These tests pin the properties
that make that loop safe and self-sustaining, because every one of them is a
way the loop could quietly die or run away.
"""
from __future__ import annotations

import pathlib

import pytest

yaml = pytest.importorskip("yaml")

WORKFLOWS = pathlib.Path(__file__).resolve().parents[1] / "workflows"
PACEMAKER = WORKFLOWS / "pacemaker.yml"


def _load(path: pathlib.Path) -> dict:
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    # `on:` parses as the boolean True under YAML 1.1
    doc["_on"] = doc.get("on", doc.get(True))
    return doc


def test_pacemaker_exists():
    assert PACEMAKER.exists(), (
        "without a pacemaker the fleet only runs when a human pushes code"
    )


def test_pacemaker_is_retriggered_by_a_workflow_chain_not_only_cron():
    """Cron is uneven on the free tier — the chain is the real mechanism."""
    on = _load(PACEMAKER)["_on"]
    assert "workflow_run" in on, (
        "schedule alone was measured at ~10% of configured rate overnight; the "
        "pacemaker must be re-armed by the chain it drives"
    )
    assert "CI" in on["workflow_run"]["workflows"]


def test_pacemaker_can_ignite_without_a_pull_request():
    """CI cannot be the only ignition source — CI needs a PR to run at all.

    Verified the hard way: after the first deploy the pacemaker had 0 runs,
    because the only CI completion since the merge was an `action_required`
    run that never executed. A pacemaker keyed solely to CI reintroduces the
    very dependency it exists to remove.
    """
    on = _load(PACEMAKER)["_on"]
    sources = on["workflow_run"]["workflows"]
    non_ci = [w for w in sources if w != "CI"]
    assert non_ci, (
        "the pacemaker must also chain off workflows that fire on cron, or it "
        "can never start from cold without someone opening a PR"
    )

    # Every named source must exist, or the trigger silently never fires.
    names = set()
    for path in WORKFLOWS.glob("*.yml"):
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(doc, dict) and doc.get("name"):
            names.add(doc["name"])
    missing = [w for w in sources if w not in names]
    assert not missing, (
        f"workflow_run names must match a real workflow's `name:` exactly; "
        f"these match nothing and will never trigger: {missing}"
    )


def test_pacemaker_can_dispatch_workflows():
    """actions: write is required; without it the dispatch 403s every time."""
    doc = _load(PACEMAKER)
    assert doc.get("permissions", {}).get("actions") == "write"


def test_only_one_pacemaker_runs_at_a_time():
    """A burst of PRs must collapse to one heartbeat, not stack dozens of sleepers."""
    doc = _load(PACEMAKER)
    conc = doc.get("concurrency", {})
    assert conc.get("group") == "pacemaker"
    assert conc.get("cancel-in-progress") is True


def test_sleep_fits_inside_the_job_timeout():
    """A sleep longer than the timeout orphans the chain instead of extending it."""
    doc = _load(PACEMAKER)
    job = doc["jobs"]["heartbeat"]
    timeout_s = int(job["timeout-minutes"]) * 60

    sleeps = [
        int(step["run"].split("sleep", 1)[1].split()[0])
        for step in job["steps"]
        if isinstance(step.get("run"), str) and "sleep " in step["run"]
    ]
    assert sleeps, "the pacemaker must actually wait between beats"
    assert max(sleeps) < timeout_s, (
        f"sleep {max(sleeps)}s must leave room inside the {timeout_s}s job timeout "
        "for the dispatch itself"
    )


def test_there_is_a_kill_switch():
    """A self-sustaining loop needs an off switch that is not 'delete the file'."""
    src = PACEMAKER.read_text(encoding="utf-8")
    assert "PACEMAKER_DISABLED" in src


def test_a_failed_heartbeat_is_not_silent():
    """A dispatch that fails silently puts the fleet to sleep with no signal."""
    doc = _load(PACEMAKER)
    steps = doc["jobs"]["heartbeat"]["steps"]
    assert any(s.get("if") == "failure()" for s in steps), (
        "a broken heartbeat must alert — that failure mode is invisible otherwise"
    )


def test_ci_is_still_dispatchable():
    """The pacemaker drives test.yml; it must accept workflow_dispatch."""
    ci = _load(WORKFLOWS / "test.yml")["_on"]
    assert "workflow_dispatch" in ci, (
        "CI must remain dispatchable or the whole chain has no entry point"
    )


def test_the_downstream_chain_is_real():
    """Guards the premise: if nothing chains off CI, the pacemaker is pointless."""
    chained = [
        p.name for p in WORKFLOWS.glob("*.yml")
        if 'workflows: ["CI"]' in p.read_text(encoding="utf-8")
    ]
    assert len(chained) >= 10, (
        f"expected many workflows chained off CI, found {len(chained)} — if this "
        "dropped, the fleet's trigger architecture changed and the pacemaker "
        "assumptions need revisiting"
    )
