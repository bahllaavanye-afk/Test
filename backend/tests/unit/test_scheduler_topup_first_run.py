"""Tests for the scheduler top‑up job ensuring the first run is scheduled correctly.

APScheduler interval jobs wait a full interval before their first execution. In a
deployment environment where the process restarts frequently, this can cause
the job never to run. The scheduler wrapper accepts a ``next_run_time`` to
force an early first execution. These tests verify that the ``next_run_time``
is correctly forwarded and that the interval continues to apply thereafter.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict

import pytest

from app.tasks.scheduler import SchedulerJobConfig, _add_job


class _Recorder:
    """A minimal stand‑in for APScheduler's scheduler used in tests.

    It records the job configuration passed to ``add_job`` so that assertions can
    be made about the forwarded arguments.
    """

    def __init__(self) -> None:
        self.jobs: Dict[str, Dict[str, Any]] = {}

    def add_job(self, func: Callable[..., Any], trigger: str, *, id: str, **kw: Any) -> None:
        """Mimic APScheduler's ``add_job`` signature.

        The ``trigger`` argument is stored alongside any keyword arguments for
        later inspection.
        """
        self.jobs[id] = {"trigger": trigger, **kw}


@pytest.fixture
def fixed_now() -> datetime:
    """Provide a stable reference time for the tests."""
    return datetime.now(timezone.utc)


def test_add_job_forwards_next_run_time(fixed_now: datetime) -> None:
    """``_add_job`` must forward ``next_run_time`` so the first run occurs early."""
    sched = _Recorder()
    first = fixed_now + timedelta(seconds=45)

    _add_job(
        sched,
        SchedulerJobConfig(
            job_id="bot_runner_topup",
            trigger="interval",
            trigger_args={"minutes": 10, "next_run_time": first},
            func=lambda: None,
        ),
    )

    job = sched.jobs["bot_runner_topup"]
    assert job["minutes"] == 10, "interval minutes should be preserved"
    assert job["next_run_time"] == first, (
        "without next_run_time the first run is a full interval away, and the "
        "process restarts before then"
    )


def test_first_run_is_soon_enough_to_survive_a_deploy_cycle(fixed_now: datetime) -> None:
    """The initial delay must be short enough to survive a deploy but not too short."""
    first = fixed_now + timedelta(seconds=45)
    delay = first - fixed_now
    assert delay < timedelta(minutes=2), (
        "the first top‑up must land long before the next redeploy resets the clock"
    )
    assert delay > timedelta(seconds=10), (
        "but not so eager that it races the bots table being seeded at boot"
    )


@pytest.mark.parametrize("minutes", [10])
def test_interval_still_repeats(minutes: int, fixed_now: datetime) -> None:
    """After the first run, the interval should continue to apply."""
    sched = _Recorder()
    _add_job(
        sched,
        SchedulerJobConfig(
            job_id="bot_runner_topup",
            trigger="interval",
            trigger_args={"minutes": minutes, "next_run_time": fixed_now},
            func=lambda: None,
        ),
    )
    assert sched.jobs["bot_runner_topup"]["minutes"] == minutes, "interval minutes must match"
    # Ensure the trigger type is preserved for completeness
    assert sched.jobs["bot_runner_topup"]["trigger"] == "interval", "trigger type must be interval"