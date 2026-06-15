"""
Redis client with no-op fallback when REDIS_URL is empty.

If REDIS_URL is not set, all cache operations silently return None/no-op.
This lets the app run on Render/local without a Redis instance — strategies
and the API still work; only real-time price caching and pub/sub are skipped.
"""
import json
import time as _time
from typing import Any

import redis.asyncio as aioredis

from app.config import settings
from app.utils.logging import logger

_pool: aioredis.ConnectionPool | None = None

# Treat empty OR placeholder URLs as "no Redis" so we fall back to the
# in-process memory cache instead of hammering a bogus host on every call.
_PLACEHOLDER_HINTS = ("your-upstash", "your-redis", "example.com", "changeme", "<", "placeholder")


def _is_placeholder(url: str) -> bool:
    u = (url or "").strip().lower()
    return (not u) or any(h in u for h in _PLACEHOLDER_HINTS)


_redis_disabled = _is_placeholder(settings.redis_url)


def _redis_enabled() -> bool:
    """Return True if a real Redis URL is configured."""
    return not _redis_disabled


class _MemoryPriceCache:
    """In-process cache used when Redis is not configured (no Upstash needed).

    Stores real data fetched by the price feed in plain dicts with lazy TTL
    expiry, so the platform is fully functional with zero external Redis —
    real prices/OHLCV still flow to strategies and the dashboard. This is a
    cache layer only; it never fabricates data.
    """

    def __init__(self) -> None:
        self._kv: dict[str, tuple[float, Any]] = {}  # key -> (expiry_ts, value)

    def _get(self, key: str) -> Any:
        item = self._kv.get(key)
        if item is None:
            return None
        expiry, value = item
        if expiry and expiry < _time.time():
            self._kv.pop(key, None)
            return None
        return value

    def _set(self, key: str, value: Any, ttl: int) -> None:
        self._kv[key] = (_time.time() + ttl if ttl else 0.0, value)

    async def set_price(self, exchange: str, symbol: str, data: dict, ttl: int = 5) -> None:
        self._set(f"price:{exchange}:{symbol}", data, ttl)

    async def get_price(self, exchange: str, symbol: str):
        return self._get(f"price:{exchange}:{symbol}")

    async def set_ohlcv(self, exchange: str, symbol: str, interval: str, data: list, ttl: int = 60) -> None:
        # OHLCV history is long-lived for strategy consumption — keep generously.
        self._set(f"ohlcv:{exchange}:{symbol}:{interval}", data, max(ttl, 3600))

    async def get_ohlcv(self, exchange: str, symbol: str, interval: str):
        return self._get(f"ohlcv:{exchange}:{symbol}:{interval}")

    async def set_arb_opportunity(self, key: str, data: dict, ttl: int = 2) -> None:
        self._set(f"arb:{key}", data, ttl)

    async def publish(self, *args, **kwargs) -> None:
        pass  # no pub/sub in memory mode — WS broadcast handled separately

    async def cache_prediction(self, symbol: str, model_id: str, data: dict, ttl: int = 60) -> None:
        self._set(f"ml:prediction:{symbol}:{model_id}", data, ttl)

    async def get(self, key: str):
        return self._get(key)

    async def set(self, key: str, value: str, ttl: int = 300) -> None:
        self._set(key, value, ttl)

    async def ping(self) -> None:
        pass


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
        key = f"price:{exchange}:{symbol}"
        try:
            await self._r.setex(key, ttl, json.dumps(data))
        except Exception as exc:
            logger.warning("redis.set_price failed", key=key, error=str(exc))

    async def get_price(self, exchange: str, symbol: str) -> dict | None:
        if self._r is None:
            return None
        key = f"price:{exchange}:{symbol}"
        try:
            raw = await self._r.get(key)
            return json.loads(raw) if raw else None
        except Exception as exc:
            logger.warning("redis.get_price failed", key=key, error=str(exc))
            return None

    async def set_ohlcv(self, exchange: str, symbol: str, interval: str, data: list, ttl: int = 60) -> None:
        if self._r is None:
            return
        key = f"ohlcv:{exchange}:{symbol}:{interval}"
        try:
            await self._r.setex(key, ttl, json.dumps(data))
        except Exception as exc:
            logger.warning("redis.set_ohlcv failed", key=key, error=str(exc))

    async def get_ohlcv(self, exchange: str, symbol: str, interval: str) -> list | None:
        if self._r is None:
            return None
        key = f"ohlcv:{exchange}:{symbol}:{interval}"
        try:
            raw = await self._r.get(key)
            return json.loads(raw) if raw else None
        except Exception as exc:
            logger.warning("redis.get_ohlcv failed", key=key, error=str(exc))
            return None

    async def set_arb_opportunity(self, key: str, data: dict, ttl: int = 2) -> None:
        if self._r is None:
            return
        redis_key = f"arb:{key}"
        try:
            await self._r.setex(redis_key, ttl, json.dumps(data))
        except Exception as exc:
            logger.warning("redis.set_arb failed", key=redis_key, error=str(exc))

    async def publish(self, channel: str, message: Any) -> None:
        if self._r is None:
            return
        try:
            await self._r.publish(channel, json.dumps(message))
        except Exception as exc:
            logger.warning("redis.publish failed", channel=channel, error=str(exc))

    async def cache_prediction(self, symbol: str, model_id: str, data: dict, ttl: int = 60) -> None:
        if self._r is None:
            return
        key = f"ml:prediction:{symbol}:{model_id}"
        try:
            await self._r.setex(key, ttl, json.dumps(data))
        except Exception as exc:
            logger.warning("redis.cache_prediction failed", key=key, error=str(exc))

    async def get(self, key: str) -> str | None:
        if self._r is None:
            return None
        try:
            return await self._r.get(key)
        except Exception as exc:
            logger.warning("redis.get failed", key=key, error=str(exc))
            return None

    async def set(self, key: str, value: str, ttl: int = 300) -> None:
        if self._r is None:
            return
        try:
            await self._r.setex(key, ttl, value)
        except Exception as exc:
            logger.warning("redis.set failed", key=key, error=str(exc))


# Module-level singleton — falls back to the in-memory cache when Redis is
# disabled or configured with a placeholder URL (so real data still flows).
if _redis_enabled():
    price_cache: PriceCache | _MemoryPriceCache = PriceCache()
else:
    logger.info("Redis not configured — using in-process memory cache (real data still flows)")
    price_cache = _MemoryPriceCache()
