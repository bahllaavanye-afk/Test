import os
from logging.config import fileConfig
from sqlalchemy import pool, create_engine
from alembic import context
from app.models.base import Base
from app.models import (  # noqa: F401
    account, order, position, trade, strategy, backtest,
    experiment, ml_model, market_data, risk, slippage,
    comparison, user, model_release, inference_log,
)

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Resolve sync URL (psycopg2) — alembic needs a synchronous driver
_raw = os.getenv("ALEMBIC_DATABASE_URL") or os.getenv("DATABASE_URL", "")


def _to_sync_url(url: str) -> str:
    """Convert any postgres URL variant to a psycopg2 sync URL."""
    url = url.replace("+asyncpg", "").replace("+aiosqlite", "")
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    return url


_sync_url = _to_sync_url(_raw) if _raw else ""

# Read DB URL from environment — prefer explicit ALEMBIC_DATABASE_URL, fall back to DATABASE_URL
_raw_url = os.getenv("ALEMBIC_DATABASE_URL") or os.getenv("DATABASE_URL", "")


def _to_sync(url: str) -> str:
    """Convert any postgres URL variant to postgresql+psycopg2:// for sync migrations."""
    if url.startswith("postgres://"):
        return "postgresql+psycopg2://" + url[len("postgres://"):]
    if url.startswith("postgresql://"):
        return "postgresql+psycopg2://" + url[len("postgresql://"):]
    if "+asyncpg" in url:
        return url.replace("+asyncpg", "+psycopg2")
    if "+aiosqlite" in url:
        # SQLite for local dev — strip async prefix
        return url.replace("+aiosqlite", "")
    if "sqlite" in url and "+aiosqlite" not in url:
        return url  # already sync sqlite
    return url


_sync_url = _to_sync(_raw_url)


def run_migrations_offline() -> None:
    url = _sync_url or config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def _do_run_migrations(connection):
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    url = _sync_url or config.get_main_option("sqlalchemy.url")
    # Sync engine (psycopg2) — avoids asyncio IPv6 connection issues on Render
    connectable = create_engine(url, poolclass=pool.NullPool, connect_args={"connect_timeout": 30})
    with connectable.connect() as connection:
        _do_run_migrations(connection)
    connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
