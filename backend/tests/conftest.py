"""Shared test fixtures for all tests."""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import pytest_asyncio

# Resolve test DB path relative to this file so it works in any environment.
# Under pytest-xdist each worker is a separate process; give every worker its own
# DB file (via PYTEST_XDIST_WORKER) so one worker's session-teardown drop_all can't
# pull the tables out from under another worker (the "no such table: users" race).
_WORKER = os.environ.get("PYTEST_XDIST_WORKER", "")
_DB_FILE = f"test_{_WORKER}.db" if _WORKER else "test.db"
_TEST_DB = (Path(__file__).resolve().parent.parent / _DB_FILE).as_posix()

# Force test DB — must override parent env to prevent tests from wiping the dev DB
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{_TEST_DB}"
os.environ["ALEMBIC_DATABASE_URL"] = f"sqlite:///{_TEST_DB}"
os.environ.setdefault("REDIS_URL", "")
os.environ.setdefault("SECRET_KEY", "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2")
os.environ.setdefault("TRADING_MODE", "test")
os.environ.setdefault("DEBUG", "false")
os.environ.setdefault("ALPACA_API_KEY", "test-key")
os.environ.setdefault("ALPACA_SECRET_KEY", "test-secret")
os.environ.setdefault("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")


@pytest_asyncio.fixture(scope="session")
async def _create_tables():
    """Create all DB tables once per test session (no background tasks)."""
    import app.models  # noqa: F401 — side-effect: registers all ORM models with Base.metadata
    from app.database import engine, Base
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    # teardown: drop all tables to keep things clean
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture(scope="module", autouse=True)
async def _isolate_each_file(_create_tables):
    """Give every test FILE a clean database.

    `_create_tables` is session-scoped, so all files landing on the same xdist
    worker previously shared not just the schema but every row. Under
    `--dist loadfile` which files share a worker is a scheduler detail, so
    tests passed or failed on packing order — the suite was deterministic only
    by luck.

    Two concrete failures this caused, both real:
      * `test_seed_additive` asserted `n_first == len(BOT_TEMPLATES)`, silently
        requiring it to be the first thing to seed on its worker. It went red
        as `assert 0 == 61` when an unrelated speed-up reshuffled the packing.
      * Any file adding a second Account for the demo user broke seeding for
        every later file on that worker — which is how the `scalar_one_or_none`
        production bug in `bots/seed.py` was finally surfaced.

    Wiping rows (not tables) between modules is cheap: the schema is built once
    per session, and this is a handful of DELETEs per file on SQLite.

    Deliberately MODULE scope, not function: within one file, tests building on
    each other is normal and intended (fixtures that create a user then act as
    it). The coupling that has to go is the ACCIDENTAL kind, between files that
    never meant to share anything.
    """
    from app.database import engine, Base

    async with engine.begin() as conn:
        for table in reversed(Base.metadata.sorted_tables):
            await conn.execute(table.delete())
    yield


@pytest_asyncio.fixture
async def client(_create_tables):
    """Async HTTP test client — tables are pre-created, no background agents."""
    from httpx import AsyncClient, ASGITransport
    from app.main import app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
