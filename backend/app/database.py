from __future__ import annotations

import asyncio as _asyncio
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase
from app.config import settings
from app.utils.logging import logger

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
    """Base class for all ORM models."""
    pass


# Set by ensure_database_alive() when the primary DB was unreachable at boot and
# the app switched to the local SQLite fallback. Read by /health/detailed.
db_fallback_active: bool = False
db_primary_error: str | None = None

FALLBACK_SQLITE_URL = "sqlite+aiosqlite:///./fallback.db"


async def ensure_database_alive(probe_timeout: float = 10.0):
    """Probe the configured DATABASE_URL; fall back to local SQLite if it's dead.

    The Supabase free tier auto-pauses after 7 idle days, which made every
    DB‑touching endpoint 500 while /health stayed green — the whole site looked
    broken. When the probe fails and ``settings.db_fallback_to_sqlite`` is on,
    this rebinds ``AsyncSessionLocal`` IN PLACE (``.configure(bind=…)``) so
    every module holding a reference switches automatically, creates the schema,
    and records the failure for /health/detailed. Bots reseed from templates and
    desk trades resync from Alpaca's 30‑day order history, so the platform is
    functional (not durable) until the primary DB is restored.

    Returns the live engine. Never raises.
    """
    global engine, db_fallback_active, db_primary_error
    from sqlalchemy import text as _text

    try:
        async def _probe() -> None:
            async with engine.connect() as conn:
                await conn.execute(_text("SELECT 1"))

        await _asyncio.wait_for(_probe(), timeout=probe_timeout)
        logger.debug("Primary database probe succeeded.")
        return engine
    except Exception as exc:  # noqa: BLE001 — any connect failure means "dead"
        db_primary_error = str(exc)[:300]
        logger.warning(
            "Primary database probe failed; attempting fallback.",
            error=db_primary_error,
        )

    if _is_sqlite or not settings.db_fallback_to_sqlite:
        # Already on SQLite (nothing better to fall back to) or fallback disabled.
        return engine

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
        logger.debug("Old engine disposal raised an exception; ignored.", exc_info=True)

    # Alembic never ran against this file — create the schema right now.
    import app.models  # noqa: F401 — registers all ORM models with Base.metadata

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    return engine


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Provide a transactional async session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            logger.exception("Session rollback due to exception.")
            raise
        finally:
            await session.close()