from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase
from app.config import settings

_is_sqlite = settings.database_url.startswith("sqlite")

if _is_sqlite:
    # NullPool: each session gets a fresh connection — avoids cross-connection
    # visibility issues where pooled connections cache an empty schema.
    from sqlalchemy.pool import NullPool as _NullPool
    _engine_kwargs: dict = {
        "poolclass": _NullPool,
        "connect_args": {"check_same_thread": False},
    }
else:
    _engine_kwargs = {
        "connect_args": {
            "server_settings": {"jit": "off"},
            "command_timeout": 60,
        },
        "pool_size": 5,
        "max_overflow": 10,
        "pool_pre_ping": True,
        "pool_recycle": 1800,
        "pool_timeout": 30,
    }

engine = create_async_engine(
    settings.database_url,
    echo=settings.debug,
    **_engine_kwargs,
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    pass


# Set by ensure_database_alive() when the primary DB was unreachable at boot and
# the app switched to the local SQLite fallback. Read by /health/detailed.
db_fallback_active: bool = False
db_primary_error: str | None = None

FALLBACK_SQLITE_URL = "sqlite+aiosqlite:///./fallback.db"


async def ensure_database_alive(probe_timeout: float = 10.0):
    """Probe the configured DATABASE_URL; fall back to local SQLite if it's dead.

    The Supabase free tier auto-pauses after 7 idle days, which made every
    DB-touching endpoint 500 while /health stayed green — the whole site looked
    broken. When the probe fails and ``settings.db_fallback_to_sqlite`` is on,
    this rebinds ``AsyncSessionLocal`` IN PLACE (``.configure(bind=...)``) so
    every module holding a reference switches automatically, creates the schema,
    and records the failure for /health/detailed. Bots reseed from templates and
    desk trades resync from Alpaca's 30-day order history, so the platform is
    functional (not durable) until the primary DB is restored.

    Returns the live engine. Never raises.
    """
    global engine, db_fallback_active, db_primary_error
    import asyncio as _asyncio

    from sqlalchemy import text as _text

    try:
        async def _probe():
            async with engine.connect() as conn:
                await conn.execute(_text("SELECT 1"))

        await _asyncio.wait_for(_probe(), timeout=probe_timeout)
        return engine
    except Exception as exc:  # noqa: BLE001 — any connect failure means "dead"
        db_primary_error = str(exc)[:300]

    if _is_sqlite or not settings.db_fallback_to_sqlite:
        # Already on SQLite (nothing better to fall back to) or fallback disabled.
        return engine

    from app.utils.logging import logger

    logger.error(
        "PRIMARY DATABASE UNREACHABLE — falling back to local SQLite. "
        "Data is ephemeral until the primary is restored (Supabase: unpause the project).",
        error=db_primary_error,
    )

    from sqlalchemy.pool import NullPool as _NullPool

    old_engine = engine
    engine = create_async_engine(
        FALLBACK_SQLITE_URL,
        poolclass=_NullPool,
        connect_args={"check_same_thread": False},
    )
    AsyncSessionLocal.configure(bind=engine)
    db_fallback_active = True
    try:
        await old_engine.dispose()
    except Exception:  # noqa: BLE001
        pass

    # Alembic never ran against this file — create the schema right now.
    import app.models  # noqa: F401 — registers all ORM models with Base.metadata

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return engine


async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# --------------------------------------------------------------------------- #
# Unit tests for edge cases (run with pytest --asyncio)
# --------------------------------------------------------------------------- #

import pytest
import asyncio
from unittest import mock


@pytest.mark.asyncio
async def test_ensure_database_alive_immediate_timeout(monkeypatch):
    """If the probe times out immediately, fallback should be triggered when allowed."""
    # Force non‑sqlite URL and enable fallback.
    monkeypatch.setattr(settings, "database_url", "postgresql+asyncpg://test")
    monkeypatch.setattr(settings, "db_fallback_to_sqlite", True)

    # Mock engine to raise on connect (simulating timeout).
    mock_engine = mock.AsyncMock()
    mock_conn_ctx = mock.AsyncMock()
    mock_conn_ctx.__aenter__.return_value = mock_conn_ctx
    mock_conn_ctx.execute.return_value = None
    mock_engine.connect.return_value = mock_conn_ctx
    # Make the probe raise a timeout.
    async def raise_timeout(*args, **kwargs):
        raise asyncio.TimeoutError()
    mock_conn_ctx.__aenter__.side_effect = raise_timeout

    monkeypatch.setattr(
        "backend.app.database.engine", mock_engine, raising=False
    )
    # Preserve reference to original engine for later comparison.
    original_engine = mock_engine

    # Run with a tiny timeout to force immediate failure.
    new_engine = await ensure_database_alive(probe_timeout=0.01)

    # Verify fallback was performed.
    assert new_engine != original_engine
    assert db_fallback_active is True
    assert db_primary_error is not None


@pytest.mark.asyncio
async def test_ensure_database_alive_sqlite_no_fallback(monkeypatch):
    """When using SQLite, fallback should not be attempted even if enabled."""
    # Force SQLite URL.
    monkeypatch.setattr(settings, "database_url", "sqlite+aiosqlite:///test.db")
    monkeypatch.setattr(settings, "db_fallback_to_sqlite", True)

    # Re‑evaluate the _is_sqlite flag.
    monkeypatch.setattr("backend.app.database._is_sqlite", True, raising=False)

    # Mock engine that succeeds on probe.
    mock_engine = mock.AsyncMock()
    mock_conn_ctx = mock.AsyncMock()
    mock_conn_ctx.__aenter__.return_value = mock_conn_ctx
    mock_conn_ctx.execute.return_value = None
    mock_engine.connect.return_value = mock_conn_ctx
    monkeypatch.setattr(
        "backend.app.database.engine", mock_engine, raising=False
    )

    result_engine = await ensure_database_alive(probe_timeout=0.1)

    # Engine should remain unchanged and no fallback flag set.
    assert result_engine is mock_engine
    assert db_fallback_active is False
    assert db_primary_error is None


@pytest.mark.asyncio
async def test_get_db_rollback_on_exception(monkeypatch):
    """Ensure that a session rollback is called when an exception occurs inside get_db."""
    # Create a mock session with rollback tracking.
    mock_session = mock.AsyncMock()
    mock_session.rollback = mock.AsyncMock()
    mock_session.close = mock.AsyncMock()

    # Mock AsyncSessionLocal to return a context manager yielding the mock session.
    async def async_cm():
        async with mock.AsyncMock() as _:
            yield mock_session

    class MockAsyncSessionLocal:
        async def __aenter__(self):
            return mock_session

        async def __aexit__(self, exc_type, exc, tb):
            await mock_session.close()
            return False

    monkeypatch.setattr(
        "backend.app.database.AsyncSessionLocal",
        MockAsyncSessionLocal(),
        raising=False,
    )

    # Use the generator directly to simulate an exception inside the context.
    gen = get_db()
    session = await gen.__anext__()
    assert session is mock_session

    # Simulate an error occurring in the user code.
    with pytest.raises(RuntimeError):
        raise RuntimeError("test error")

    # Close the generator to trigger finally block.
    await gen.aclose()

    # Verify rollback was called.
    mock_session.rollback.assert_awaited_once()
    mock_session.close.assert_awaited_once()