"""Redis client with graceful no-op behavior when REDIS_URL is empty.

For the Render free tier and CI environments where Redis isn't configured,
this module returns a stub PriceCache that no-ops all operations rather than
crashing at import time.
"""
from __future__ import annotations

import json
from typing import Any

import redis.asyncio as aioredis

from app.config import settings

_pool: aioredis.ConnectionPool | None = None


def _redis_enabled() -> bool:
    """Redis is enabled only when REDIS_URL is set to a valid scheme."""
    url = (settings.redis_url or "").strip()
    return bool(url) and url.startswith(("redis://", "rediss://", "unix://"))


def get_pool() -> aioredis.ConnectionPool | None:
    """Return the connection pool, or None when Redis isn't configured."""
    global _pool
    if not _redis_enabled():
        return None
    if _pool is None:
        _pool = aioredis.ConnectionPool.from_url(
            settings.redis_url,
            max_connections=10,
            decode_responses=True,
        )
    return _pool


def get_redis() -> aioredis.Redis | None:
    """Return a Redis client, or None when REDIS_URL isn't configured."""
    pool = get_pool()
    if pool is None:
        return None
    return aioredis.Redis(connection_pool=pool)


class _NoopPriceCache:
    """No-op fallback used when Redis isn't configured.

    All writes are dropped silently; reads return None. This lets the API
    server run on Render's free tier without a Redis instance.
    """

    enabled = False

    async def set_price(self, *_, **__) -> None: ...
    async def get_price(self, *_, **__) -> None: return None
    async def set_ohlcv(self, *_, **__) -> None: ...
    async def get_ohlcv(self, *_, **__) -> None: return None
    async def set_arb_opportunity(self, *_, **__) -> None: ...
    async def publish(self, *_, **__) -> None: ...
    async def cache_prediction(self, *_, **__) -> None: ...


class PriceCache:
    """Fast price read/write via Redis. TTL-based to stay fresh."""

    enabled = True

    def __init__(self) -> None:
        client = get_redis()
        if client is None:
            raise RuntimeError("PriceCache requires REDIS_URL to be set")
        self.r = client

    async def set_price(self, exchange: str, symbol: str, data: dict, ttl: int = 5) -> None:
        key = f"price:{exchange}:{symbol}"
        await self.r.setex(key, ttl, json.dumps(data))

    async def get_price(self, exchange: str, symbol: str) -> dict | None:
        key = f"price:{exchange}:{symbol}"
        raw = await self.r.get(key)
        return json.loads(raw) if raw else None

    async def set_ohlcv(self, exchange: str, symbol: str, interval: str, data: list, ttl: int = 60) -> None:
        key = f"ohlcv:{exchange}:{symbol}:{interval}"
        await self.r.setex(key, ttl, json.dumps(data))

    async def get_ohlcv(self, exchange: str, symbol: str, interval: str) -> list | None:
        key = f"ohlcv:{exchange}:{symbol}:{interval}"
        raw = await self.r.get(key)
        return json.loads(raw) if raw else None

    async def set_arb_opportunity(self, key: str, data: dict, ttl: int = 2) -> None:
        await self.r.setex(f"arb:{key}", ttl, json.dumps(data))

    async def publish(self, channel: str, message: Any) -> None:
        await self.r.publish(channel, json.dumps(message))

    async def cache_prediction(self, symbol: str, model_id: str, data: dict, ttl: int = 60) -> None:
        key = f"ml:prediction:{symbol}:{model_id}"
        await self.r.setex(key, ttl, json.dumps(data))


# Module-level singleton — falls back to the no-op cache when Redis is disabled
if _redis_enabled():
    price_cache: PriceCache | _NoopPriceCache = PriceCache()
else:
    price_cache = _NoopPriceCache()
