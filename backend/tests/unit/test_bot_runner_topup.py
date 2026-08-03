"""Enabled bots must end up with a scheduler job, or say so loudly.

Ignition was a ONE-SHOT job at boot. On the ephemeral SQLite fallback the bots
table is empty at that moment, so it scheduled zero bots, logged `count=0`, and
never looked again — the bots seeded afterwards sat enabled-but-unscheduled
forever. Observed live on 2026-07-27:

    61 enabled bots · jobs_total=11 · bot_jobs=2 (exit-checker + lifecycle)
    every bot at last_run_at=None · 0 orders · 0 trades

The API reported a healthy fleet the whole time.

NOTE: log assertions use `capsys`, not `caplog` — structlog does not propagate
to stdlib logging.
"""
from __future__ import annotations

import pytest

from app.tasks.bot_runner import BotRunner


class _Job:
    def __init__(self, jid):
        self.id = jid


class _Scheduler:
    """Records add_job calls and answers get_job like APScheduler does."""

    def __init__(self):
        self.jobs: dict[str, dict] = {}

    def add_job(self, func, trigger, *, id, kwargs=None, **kw):
        self.jobs[id] = {"trigger": trigger, "kwargs": kwargs, **kw}

    def get_job(self, jid):
        return _Job(jid) if jid in self.jobs else None

    def remove_job(self, jid):
        self.jobs.pop(jid, None)


class _Bot:
    def __init__(self, bid, interval="1h"):
        self.id = bid
        self.name = f"bot-{bid}"
        self.is_enabled = True
        self.is_archived = False
        self.trigger = {"type": "schedule", "interval": interval}


def _patch_db(monkeypatch, bots):
    """Stand in for the AsyncSessionLocal + select(Bot) query in start().

    Patches the ATTRIBUTE on the real app.database module. Replacing the whole
    module in sys.modules breaks `from app.database import Base`, which
    app.models.bot needs — and start() imports Bot.
    """
    import app.database as db_mod

    class _Result:
        def scalars(self):
            class _S:
                def all(_self):
                    return bots
            return _S()

    class _Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def execute(self, _q):
            return _Result()

    monkeypatch.setattr(db_mod, "AsyncSessionLocal", lambda: _Session())


def _patch_db_none(monkeypatch):
    """Patch the DB layer to return None instead of a list."""
    import app.database as db_mod

    class _Result:
        def scalars(self):
            class _S:
                def all(_self):
                    return None
            return _S()

    class _Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def execute(self, _q):
            return _Result()

    monkeypatch.setattr(db_mod, "AsyncSessionLocal", lambda: _Session())


@pytest.mark.asyncio
async def test_enabled_bots_get_scheduled(monkeypatch):
    _patch_db(monkeypatch, [_Bot("a"), _Bot("b")])
    sched = _Scheduler()

    n = await BotRunner(sched).start()

    assert n == 2
    assert set(sched.jobs) == {"bot_a", "bot_b"}


@pytest.mark.asyncio
async def test_top_up_only_schedules_missing_bots(monkeypatch):
    """The pass must be idempotent — re-running must not touch existing jobs."""
    _patch_db(monkeypatch, [_Bot("a"), _Bot("b")])
    sched = _Scheduler()
    runner = BotRunner(sched)

    await runner.start()
    sched.jobs["bot_a"]["sentinel"] = "original"

    # a new bot appears after boot — exactly the ephemeral-DB case
    _patch_db(monkeypatch, [_Bot("a"), _Bot("b"), _Bot("c")])
    n = await runner.start(only_missing=True)

    assert n == 1, "only the new bot should be scheduled"
    assert set(sched.jobs) == {"bot_a", "bot_b", "bot_c"}
    assert sched.jobs["bot_a"].get("sentinel") == "original", (
        "an existing bot's job must not be replaced — that would reset its "
        "next_run_time on every pass and it would never fire"
    )


@pytest.mark.asyncio
async def test_bots_left_unscheduled_are_escalated(monkeypatch, capsys):
    """The live failure: enabled bots, no jobs, and nothing said a word."""
    _patch_db(monkeypatch, [_Bot("a"), _Bot("b")])

    class _RefusingScheduler(_Scheduler):
        def add_job(self, *a, **kw):
            return None          # silently drops the job, as if never scheduled

    await BotRunner(_RefusingScheduler()).start()

    out = capsys.readouterr().out
    assert "NO scheduler job" in out
    assert "no orders will be placed" in out


@pytest.mark.asyncio
async def test_empty_db_at_boot_reports_nothing_scheduled(monkeypatch):
    """Ignition on a fresh SQLite DB — must return 0, not appear successful."""
    _patch_db(monkeypatch, [])
    sched = _Scheduler()

    n = await BotRunner(sched).start()

    assert n == 0
    assert sched.jobs == {}


@pytest.mark.asyncio
async def test_a_later_top_up_recovers_from_the_empty_boot(monkeypatch):
    """The whole point: bots seeded after boot still get scheduled."""
    _patch_db(monkeypatch, [])
    sched = _Scheduler()
    runner = BotRunner(sched)

    assert await runner.start() == 0          # boot: DB empty

    _patch_db(monkeypatch, [_Bot("a"), _Bot("b"), _Bot("c")])
    assert await runner.start(only_missing=True) == 3

    assert set(sched.jobs) == {"bot_a", "bot_b", "bot_c"}


@pytest.mark.asyncio
async def test_start_failure_returns_zero_rather_than_raising(monkeypatch, capsys):
    """A scheduler task must not die on a DB hiccup."""
    import app.database as db_mod

    def _boom():
        raise RuntimeError("db down")

    monkeypatch.setattr(db_mod, "AsyncSessionLocal", _boom)

    assert await BotRunner(_Scheduler()).start() == 0
    assert "BotRunner.start failed" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_none_db_returns_zero_and_no_jobs(monkeypatch):
    """When the DB layer returns None, BotRunner should treat it as empty."""
    _patch_db_none(monkeypatch)
    sched = _Scheduler()

    n = await BotRunner(sched).start()

    assert n == 0
    assert sched.jobs == {}


@pytest.mark.asyncio
async def test_only_missing_without_new_bots_returns_zero(monkeypatch):
    """Calling start with only_missing=True should be a no‑op when no bots are new."""
    _patch_db(monkeypatch, [_Bot("a"), _Bot("b")])
    sched = _Scheduler()
    runner = BotRunner(sched)

    # Initial schedule
    assert await runner.start() == 2
    assert set(sched.jobs) == {"bot_a", "bot_b"}

    # Patch DB with the same bots again
    _patch_db(monkeypatch, [_Bot("a"), _Bot("b")])
    n = await runner.start(only_missing=True)

    assert n == 0, "no new bots means nothing should be scheduled"
    assert set(sched.jobs) == {"bot_a", "bot_b"}