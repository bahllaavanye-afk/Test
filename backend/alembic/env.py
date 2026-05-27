import asyncio
import os
from logging.config import fileConfig
from sqlalchemy import pool, create_engine
from sqlalchemy.ext.asyncio import async_engine_from_config, create_async_engine
from alembic import context
from app.models.base import Base
from app.models import account, order, position, trade, strategy, backtest, experiment, ml_model, market_data, risk, slippage, comparison, user

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

# Read DB URL from environment (overrides alembic.ini)
_db_url = os.getenv("ALEMBIC_DATABASE_URL") or os.getenv("DATABASE_URL", "")
# Alembic offline needs sync driver; convert asyncpg → psycopg2 if needed
_sync_url = _db_url.replace("+asyncpg", "+psycopg2").replace("+aiosqlite", "")
# Alembic online needs async driver; convert psycopg2 → asyncpg if needed
_async_url = _db_url.replace("+psycopg2", "+asyncpg")
if _async_url.startswith("postgresql://"):
    _async_url = _async_url.replace("postgresql://", "postgresql+asyncpg://", 1)
if _async_url.startswith("sqlite:///"):
    _async_url = _async_url.replace("sqlite:///", "sqlite+aiosqlite:///", 1)


def run_migrations_offline() -> None:
    url = _sync_url or config.get_main_option("sqlalchemy.url")
    context.configure(url=url, target_metadata=target_metadata, literal_binds=True, dialect_opts={"paramstyle": "named"})
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection):
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    url = _async_url or config.get_section(context.config.config_ini_section, {}).get("sqlalchemy.url", "")
    connectable = create_async_engine(url, poolclass=pool.NullPool)
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
