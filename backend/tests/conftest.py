"""Shared test fixtures for all tests."""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest
import pytest_asyncio

# Resolve test DB path relative to this file so it works in any environment
_TEST_DB = (Path(__file__).resolve().parent.parent / "test.db").as_posix()

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


@pytest.fixture(scope="session")
def event_loop_policy():
    return asyncio.DefaultEventLoopPolicy()


@pytest_asyncio.fixture(scope="session")
async def _create_tables():
    """Create all DB tables once per test session (no background tasks)."""
    from app.database import engine, Base
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    # teardown: drop all tables to keep things clean
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def client(_create_tables):
    """Async HTTP test client — tables are pre-created, no background agents."""
    from httpx import AsyncClient, ASGITransport
    from app.main import app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
