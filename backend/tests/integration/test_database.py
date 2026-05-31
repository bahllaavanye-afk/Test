"""
Integration tests: database layer — ORM models, migrations, CRUD operations.
Uses SQLite in-memory via conftest DATABASE_URL env var.
"""
from __future__ import annotations

import pytest
from sqlalchemy import text


@pytest.mark.asyncio
async def test_database_engine_connects(_create_tables):
    """Engine must be able to open a connection."""
    from app.database import engine
    async with engine.connect() as conn:
        result = await conn.execute(text("SELECT 1"))
        row = result.fetchone()
        assert row[0] == 1


@pytest.mark.asyncio
async def test_all_tables_created(_create_tables):
    """After create_all, the major ORM tables must exist."""
    from app.database import engine
    from sqlalchemy import inspect

    expected_tables = [
        "users", "accounts", "orders", "positions", "trades",
        "strategies", "experiments", "ml_models",
    ]
    async with engine.connect() as conn:
        tables = await conn.run_sync(lambda sync_conn: inspect(sync_conn).get_table_names())

    missing = [t for t in expected_tables if t not in tables]
    assert not missing, (
        f"Missing DB tables: {missing}\nExisting: {tables}"
    )


@pytest.mark.asyncio
async def test_user_crud(_create_tables):
    """Create a user, read it back, delete it."""
    from app.database import AsyncSessionLocal
    from app.models.user import User
    from sqlalchemy import select

    async with AsyncSessionLocal() as session:
        # Create
        user = User(email="dbtest@quantedge.ai", hashed_password="hashed_x")
        session.add(user)
        await session.commit()
        await session.refresh(user)
        user_id = user.id
        assert user_id is not None

        # Read
        result = await session.execute(select(User).where(User.id == user_id))
        fetched = result.scalar_one_or_none()
        assert fetched is not None
        assert fetched.email == "dbtest@quantedge.ai"

        # Delete
        await session.delete(fetched)
        await session.commit()

        result2 = await session.execute(select(User).where(User.id == user_id))
        assert result2.scalar_one_or_none() is None


@pytest.mark.asyncio
async def test_session_factory_yields_session(_create_tables):
    """AsyncSessionLocal must yield a usable session."""
    from app.database import AsyncSessionLocal
    async with AsyncSessionLocal() as session:
        assert session is not None
        result = await session.execute(text("SELECT 42"))
        row = result.fetchone()
        assert row[0] == 42
