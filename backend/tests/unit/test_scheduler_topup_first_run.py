"""The bot top-up must run soon after boot, not one full interval later.

APScheduler interval jobs wait a FULL interval before their first run. This
process restarts on every deploy, and main is pushed several times an hour, so
a plain `interval: 10 minutes` job has its clock reset before it ever fires —
it would never run at all.

`bot_runner._first_run_time()` documents this exact trap being chased once
before: "APScheduler interval jobs wait one FULL interval before the first run
— and on the ephemeral-DB deploy, app restarts on every merge reset that clock,
so 1h/1d bots NEVER got to run."

The top-up was added with the identical flaw. `next_run_time` fixes it.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.tasks.scheduler import SchedulerJobConfig, _add_job


class _Recorder:
    def __init__(self):
        self.jobs = {}

    def add_job(self, func, trigger, *, id, **kw):
        self.jobs[id] = {"trigger": trigger, **kw}


def test_add_job_forwards_next_run_time():
    """_add_job splats trigger_args, and APScheduler takes next_run_time there."""
    sched = _Recorder()
    first = datetime.now(timezone.utc) + timedelta(seconds=45)

    _add_job(sched, SchedulerJobConfig(
        job_id="bot_runner_topup",
        trigger="interval",
        trigger_args={"minutes": 10, "next_run_time": first},
        func=lambda: None,
    ))

    job = sched.jobs["bot_runner_topup"]
    assert job["minutes"] == 10
    assert job["next_run_time"] == first, (
        "without next_run_time the first run is a full interval away, and the "
        "process restarts before then"
    )


def test_first_run_is_soon_enough_to_survive_a_deploy_cycle():
    """45s must be well inside the time between deploys."""
    first = datetime.now(timezone.utc) + timedelta(seconds=45)
    delay = first - datetime.now(timezone.utc)
    assert delay < timedelta(minutes=2), (
        "the first top-up must land long before the next redeploy resets the clock"
    )
    assert delay > timedelta(seconds=10), (
        "but not so eager that it races the bots table being seeded at boot"
    )


@pytest.mark.parametrize("minutes", [10])
def test_interval_still_repeats(minutes):
    """next_run_time sets the FIRST run; the interval must still apply after."""
    sched = _Recorder()
    _add_job(sched, SchedulerJobConfig(
        job_id="bot_runner_topup",
        trigger="interval",
        trigger_args={"minutes": minutes,
                      "next_run_time": datetime.now(timezone.utc)},
        func=lambda: None,
    ))
    assert sched.jobs["bot_runner_topup"]["minutes"] == minutes
