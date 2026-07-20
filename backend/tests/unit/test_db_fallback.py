"""SQLite fallback when the primary DB is dead (the Supabase-pause outage).

2026-07-20: the live keeper backend 500'd on every DB-touching endpoint because
the Supabase free-tier project auto-paused — /health stayed green while login,
demo, trades and bots were all broken. ensure_database_alive() now probes the
primary at boot and rebinds AsyncSessionLocal in place to a local SQLite file,
so the platform stays functional (bots reseed, desk trades resync from Alpaca).

These tests pin: the fallback engages on an unreachable primary, sessions work
afterwards (schema created), the failure is recorded for /health/detailed, and
the fallback does NOT engage when disabled or already on SQLite.
"""
from __future__ import annotations

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

import app.database as db_mod
from app.config import settings

UNREACHABLE_PG = "postgresql+asyncpg://nouser:nopass@127.0.0.1:9/nodb"


@pytest.fixture
def _restore_db_module(tmp_path):
    """Snapshot and restore every module global the fallback path mutates, so the
    rest of the suite keeps its working test-DB binding."""
    saved = (
        db_mod.engine,
        db_mod._is_sqlite,
        db_mod.db_fallback_active,
        db_mod.db_primary_error,
        db_mod.FALLBACK_SQLITE_URL,
        settings.db_fallback_to_sqlite,
    )
    db_mod.FALLBACK_SQLITE_URL = f"sqlite+aiosqlite:///{tmp_path}/fallback.db"
    yield
    (
        db_mod.engine,
        db_mod._is_sqlite,
        db_mod.db_fallback_active,
        db_mod.db_primary_error,
        db_mod.FALLBACK_SQLITE_URL,
        settings.db_fallback_to_sqlite,
    ) = saved
    db_mod.AsyncSessionLocal.configure(bind=saved[0])


async def test_fallback_engages_when_primary_unreachable(_restore_db_module):
    db_mod.engine = create_async_engine(UNREACHABLE_PG)
    db_mod._is_sqlite = False
    settings.db_fallback_to_sqlite = True
    db_mod.db_fallback_active = False

    live = await db_mod.ensure_database_alive(probe_timeout=5.0)

    assert db_mod.db_fallback_active is True
    assert db_mod.db_primary_error  # recorded for /health/detailed
    assert "sqlite" in str(live.url)

    # The rebound sessionmaker must work AND have the schema (create_all ran).
    async with db_mod.AsyncSessionLocal() as s:
        assert (await s.execute(text("SELECT 1"))).scalar() == 1
        n = (await s.execute(text("SELECT COUNT(*) FROM users"))).scalar()
        assert n == 0  # table exists, empty

    await live.dispose()


async def test_fallback_disabled_keeps_dead_engine(_restore_db_module):
    dead = create_async_engine(UNREACHABLE_PG)
    db_mod.engine = dead
    db_mod._is_sqlite = False
    settings.db_fallback_to_sqlite = False
    db_mod.db_fallback_active = False

    live = await db_mod.ensure_database_alive(probe_timeout=5.0)

    assert live is dead                       # unchanged: operator opted out
    assert db_mod.db_fallback_active is False
    assert db_mod.db_primary_error            # but the failure is still recorded
    await dead.dispose()


async def test_healthy_primary_is_untouched(_restore_db_module):
    # The suite's own (working) engine: the probe succeeds, nothing is rebound.
    db_mod.db_fallback_active = False
    before = db_mod.engine
    live = await db_mod.ensure_database_alive(probe_timeout=10.0)
    assert live is before
    assert db_mod.db_fallback_active is False


async def test_already_sqlite_never_falls_back(_restore_db_module):
    # A dead *sqlite* URL must not trigger the fallback path (nothing better to
    # switch to) — _is_sqlite guards it.
    db_mod._is_sqlite = True
    settings.db_fallback_to_sqlite = True
    db_mod.db_fallback_active = False
    live = await db_mod.ensure_database_alive(probe_timeout=10.0)
    assert db_mod.db_fallback_active is False
    assert live is db_mod.engine
