"""
Redis client with no-op fallback when REDIS_URL is empty.

If REDIS_URL is not set, all cache operations silently return None/no-op.
This lets the app run on Render/local without a Redis instance — strategies
and the API still work; only real-time price caching and pub/sub are skipped.
"""
import json
from typing import Any

import redis.asyncio as aioredis
from redis.exceptions import ConnectionError as RedisConnectionError, TimeoutError as RedisTimeoutError

from app.config import settings
from app.utils.logging import logger

_pool: aioredis.ConnectionPool | None = None
_redis_disabled = not settings.redis_url or settings.redis_url.strip() == ""

# Circuit breaker: an unreachable-but-configured Redis (e.g. the bare default
# redis://localhost:6379 on a host with no Redis) used to retry — and log — on
# *every* call, spamming connection-refused. After the first connection failure
# we trip this breaker, log once, and no-op all cache ops just like an empty URL.
_redis_tripped = False

# Connection-level failures that should trip the breaker (vs. a one-off op error).
_CONN_ERRORS = (RedisConnectionError, RedisTimeoutError, ConnectionError, OSError)


def _redis_enabled() -> bool:
    """Return True if Redis is configured and the breaker has not tripped."""
    return not _redis_disabled and not _redis_tripped


def _note_redis_error(op: str, exc: Exception, **fields) -> None:
    """Log a Redis error; trip the breaker (log once) on connection failures."""
    global _redis_tripped
    if isinstance(exc, _CONN_ERRORS):
        if not _redis_tripped:
            _redis_tripped = True
            logger.warning(
                "Redis unreachable — falling back to in-memory no-op cache for the "
                "rest of this process (further Redis errors suppressed)",
                op=op, error=str(exc), **fields,
            )
        return
    logger.warning(f"redis.{op} failed", error=str(exc), **fields)


class _NoopPriceCache:
    """Drop-in replacement for PriceCache when Redis is not configured.

    All methods are async no-ops that return None/empty so that callers
    need no conditional logic.
    """

    async def set_price(self, *args, **kwargs) -> None:
        pass

    async def get_price(self, *args, **kwargs):
        return None

    async def set_ohlcv(self, *args, **kwargs) -> None:
        pass

    async def get_ohlcv(self, *args, **kwargs):
        return None

    async def set_arb_opportunity(self, *args, **kwargs) -> None:
        pass

    async def publish(self, *args, **kwargs) -> None:
        pass

    async def cache_prediction(self, *args, **kwargs) -> None:
        pass

    async def get(self, key: str):
        return None

    async def set(self, key: str, value: str, ttl: int = 300) -> None:
        pass

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
    """Redis price cache. No-ops gracefully when Redis is unavailable.

    Every op first checks ``_client()`` which returns ``None`` once the breaker
    trips, so a Redis outage degrades to the in-memory no-op behaviour instead of
    retrying (and logging) on every single call.
    """

    def __init__(self):
        self._r: aioredis.Redis | None = get_redis()

    def _client(self) -> aioredis.Redis | None:
        """Live Redis client, or None if disabled / the breaker has tripped."""
        if self._r is None or _redis_tripped:
            return None
        return self._r

    async def set_price(self, exchange: str, symbol: str, data: dict, ttl: int = 5) -> None:
        r = self._client()
        if r is None:
            return
        key = f"price:{exchange}:{symbol}"
        try:
            await r.setex(key, ttl, json.dumps(data))
        except Exception as exc:
            _note_redis_error("set_price", exc, key=key)

    async def get_price(self, exchange: str, symbol: str) -> dict | None:
        r = self._client()
        if r is None:
            return None
        key = f"price:{exchange}:{symbol}"
        try:
            raw = await r.get(key)
            return json.loads(raw) if raw else None
        except Exception as exc:
            _note_redis_error("get_price", exc, key=key)
            return None

    async def set_ohlcv(self, exchange: str, symbol: str, interval: str, data: list, ttl: int = 60) -> None:
        r = self._client()
        if r is None:
            return
        key = f"ohlcv:{exchange}:{symbol}:{interval}"
        try:
            await r.setex(key, ttl, json.dumps(data))
        except Exception as exc:
            _note_redis_error("set_ohlcv", exc, key=key)

    async def get_ohlcv(self, exchange: str, symbol: str, interval: str) -> list | None:
        r = self._client()
        if r is None:
            return None
        key = f"ohlcv:{exchange}:{symbol}:{interval}"
        try:
            raw = await r.get(key)
            return json.loads(raw) if raw else None
        except Exception as exc:
            _note_redis_error("get_ohlcv", exc, key=key)
            return None

    async def set_arb_opportunity(self, key: str, data: dict, ttl: int = 2) -> None:
        r = self._client()
        if r is None:
            return
        redis_key = f"arb:{key}"
        try:
            await r.setex(redis_key, ttl, json.dumps(data))
        except Exception as exc:
            _note_redis_error("set_arb", exc, key=redis_key)

    async def publish(self, channel: str, message: Any) -> None:
        r = self._client()
        if r is None:
            return
        try:
            await r.publish(channel, json.dumps(message))
        except Exception as exc:
            _note_redis_error("publish", exc, channel=channel)

    async def cache_prediction(self, symbol: str, model_id: str, data: dict, ttl: int = 60) -> None:
        r = self._client()
        if r is None:
            return
        key = f"ml:prediction:{symbol}:{model_id}"
        try:
            await r.setex(key, ttl, json.dumps(data))
        except Exception as exc:
            _note_redis_error("cache_prediction", exc, key=key)

    async def get(self, key: str) -> str | None:
        r = self._client()
        if r is None:
            return None
        try:
            return await r.get(key)
        except Exception as exc:
            _note_redis_error("get", exc, key=key)
            return None

    async def set(self, key: str, value: str, ttl: int = 300) -> None:
        r = self._client()
        if r is None:
            return
        try:
            await r.setex(key, ttl, value)
        except Exception as exc:
            _note_redis_error("set", exc, key=key)


# Module-level singleton — falls back to the no-op cache when Redis is disabled
if _redis_enabled():
    price_cache: PriceCache | _NoopPriceCache = PriceCache()
else:
    price_cache = _NoopPriceCache()
