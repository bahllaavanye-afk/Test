import asyncio
import os
from logging.config import fileConfig
from sqlalchemy import pool, create_engine
from sqlalchemy.ext.asyncio import async_engine_from_config, create_async_engine
from alembic import context
from app.models.base import Base
from app.models import account, order, position, trade, strategy, backtest, experiment, ml_model, market_data, risk, slippage, comparison, user, model_release, inference_log  # noqa: F401

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

# Read DB URL from environment (overrides alembic.ini)
_db_url = os.getenv("ALEMBIC_DATABASE_URL") or os.getenv("DATABASE_URL", "")

# Normalise the raw URL — Render/Supabase use postgres://, SQLAlchemy needs full scheme
def _to_async(url: str) -> str:
    """Convert any postgres URL variant to postgresql+asyncpg://"""
    if url.startswith("postgres://"):
        return "postgresql+asyncpg://" + url[len("postgres://"):]
    if url.startswith("postgresql://"):
        return "postgresql+asyncpg://" + url[len("postgresql://"):]
    if url.startswith("sqlite:///"):
        return "sqlite+aiosqlite:///" + url[len("sqlite:///"):]
    return url.replace("+psycopg2", "+asyncpg")

def _to_sync(url: str) -> str:
    """Convert any postgres URL variant to postgresql+psycopg2://"""
    if url.startswith("postgres://"):
        return "postgresql+psycopg2://" + url[len("postgres://"):]
    if url.startswith("postgresql://"):
        return "postgresql+psycopg2://" + url[len("postgresql://"):]
    if url.startswith("postgresql+asyncpg://"):
        return url.replace("+asyncpg", "+psycopg2")
    return url.replace("+asyncpg", "+psycopg2").replace("+aiosqlite", "")

_async_url = _to_async(_db_url)
_sync_url  = _to_sync(_db_url)


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
    connect_args = {"command_timeout": 30} if "postgresql" in url else {}
    connectable = create_async_engine(url, poolclass=pool.NullPool, connect_args=connect_args)
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
