"""
Redis client with no-op fallback when REDIS_URL is empty.

If REDIS_URL is not set, all cache operations silently return None/no-op.
This lets the app run on Render/local without a Redis instance — strategies
and the API still work; only real-time price caching and pub/sub are skipped.
"""
import json
from typing import Any
import redis.asyncio as aioredis
from app.config import settings

_pool: aioredis.ConnectionPool | None = None
_redis_disabled = not settings.redis_url or settings.redis_url.strip() == ""


def get_pool() -> aioredis.ConnectionPool | None:
    if _redis_disabled:
        return None
    global _pool
    if _pool is None:
        _pool = aioredis.ConnectionPool.from_url(
            settings.redis_url,
            max_connections=10,
            decode_responses=True,
        )
    return _pool


def get_redis() -> aioredis.Redis | None:
    pool = get_pool()
    if pool is None:
        return None
    return aioredis.Redis(connection_pool=pool)


class PriceCache:
    """Redis price cache. No-ops gracefully when Redis is unavailable."""

    def __init__(self):
        self._r: aioredis.Redis | None = get_redis()

    async def set_price(self, exchange: str, symbol: str, data: dict, ttl: int = 5) -> None:
        if self._r is None:
            return
        try:
            await self._r.setex(f"price:{exchange}:{symbol}", ttl, json.dumps(data))
        except Exception:
            pass

    async def get_price(self, exchange: str, symbol: str) -> dict | None:
        if self._r is None:
            return None
        try:
            raw = await self._r.get(f"price:{exchange}:{symbol}")
            return json.loads(raw) if raw else None
        except Exception:
            return None

    async def set_ohlcv(self, exchange: str, symbol: str, interval: str, data: list, ttl: int = 60) -> None:
        if self._r is None:
            return
        try:
            await self._r.setex(f"ohlcv:{exchange}:{symbol}:{interval}", ttl, json.dumps(data))
        except Exception:
            pass

    async def get_ohlcv(self, exchange: str, symbol: str, interval: str) -> list | None:
        if self._r is None:
            return None
        try:
            raw = await self._r.get(f"ohlcv:{exchange}:{symbol}:{interval}")
            return json.loads(raw) if raw else None
        except Exception:
            return None

    async def set_arb_opportunity(self, key: str, data: dict, ttl: int = 2) -> None:
        if self._r is None:
            return
        try:
            await self._r.setex(f"arb:{key}", ttl, json.dumps(data))
        except Exception:
            pass

    async def publish(self, channel: str, message: Any) -> None:
        if self._r is None:
            return
        try:
            await self._r.publish(channel, json.dumps(message))
        except Exception:
            pass

    async def cache_prediction(self, symbol: str, model_id: str, data: dict, ttl: int = 60) -> None:
        if self._r is None:
            return
        try:
            await self._r.setex(f"ml:prediction:{symbol}:{model_id}", ttl, json.dumps(data))
        except Exception:
            pass

    async def get(self, key: str) -> str | None:
        if self._r is None:
            return None
        try:
            return await self._r.get(key)
        except Exception:
            return None

    async def set(self, key: str, value: str, ttl: int = 300) -> None:
        if self._r is None:
            return
        try:
            await self._r.setex(key, ttl, value)
        except Exception:
            pass


price_cache = PriceCache()
