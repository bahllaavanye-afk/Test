from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import NullPool
from app.config import settings

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
SQLITE_PREFIX: str = "sqlite"
SQLITE_CHECK_SAME_THREAD: bool = False
SQLITE_POOL_CLASS = NullPool

SERVER_SETTINGS_JIT: str = "off"
COMMAND_TIMEOUT: int = 60
POOL_SIZE: int = 5
MAX_OVERFLOW: int = 10
POOL_PRE_PING: bool = True
POOL_RECYCLE: int = 1800
POOL_TIMEOUT: int = 30

PROBE_TIMEOUT_DEFAULT: float = 10.0
MAX_ERROR_LENGTH: int = 300

FALLBACK_MESSAGE: str = (
    "PRIMARY DATABASE UNREACHABLE — falling back to local SQLite. "
    "Data is epoxy until the primary is restored (Supabase: unpause the project)."
)

FALLBACK_SQLITE_URL: str = "sqlite+aiosqlite:///./fallback.db"

# ----------------------------------------------------------------------
# Engine configuration
# ----------------------------------------------------------------------
_is_sqlite: bool = settings.database_url.startswith(SQLITE_PREFIX)

if _is_sqlite:
    # NullPool: each session gets a fresh connection — avoids cross-connection
    # visibility issues where pooled connections cache an empty schema.
    _engine_kwargs: dict = {
        "poolclass": SQLITE_POOL_CLASS,
        "connect_args": {"check_same_thread": SQLITE_CHECK_SAME_THREAD},
    }
else:
    _engine_kwargs = {
        "connect_args": {
            "server_settings": {"jit": SERVER_SETTINGS_JIT},
            "command_timeout": COMMAND_TIMEOUT,
        },
        "pool_size": POOL_SIZE,
        "max_overflow": MAX_OVERFLOW,
        "pool_pre_ping": POOL_PRE_PING,
        "pool_recycle": POOL_RECYCLE,
        "pool_timeout": POOL_TIMEOUT,
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


async def ensure_database_alive(probe_timeout: float = PROBE_TIMEOUT_DEFAULT):
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
        db_primary_error = str(exc)[:MAX_ERROR_LENGTH]

    if _is_sqlite or not settings.db_fallback_to_sqlite:
        # Already on SQLite (nothing better to fall back to) or fallback disabled.
        return engine

    from app.utils.logging import logger

    logger.error(FALLBACK_MESSAGE, error=db_primary_error)

    old_engine = engine
    engine = create_async_engine(
        FALLBACK_SQLITE_URL,
        poolclass=SQLITE_POOL_CLASS,
        connect_args={"check_same_thread": SQLITE_CHECK_SAME_THREAD},
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