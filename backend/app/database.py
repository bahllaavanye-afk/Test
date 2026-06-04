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


async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
