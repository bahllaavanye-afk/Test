import os
from logging.config import fileConfig
from sqlalchemy import pool, create_engine
from alembic import context

# Import all models so Alembic sees the full schema
try:
    from app.models.base import Base
    from app.models import (
        account, order, position, trade, strategy, backtest,
        experiment, ml_model, market_data, risk, slippage, comparison, user,
    )
    target_metadata = Base.metadata
except ImportError:
    target_metadata = None

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
    connectable = create_engine(
        url,
        poolclass=pool.NullPool,
        connect_args={"connect_timeout": 30},
    )
    with connectable.connect() as connection:
        _do_run_migrations(connection)
    connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
