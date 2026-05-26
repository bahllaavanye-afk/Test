import json
from typing import Any
import redis.asyncio as aioredis
from app.config import settings

_pool: aioredis.ConnectionPool | None = None


def get_pool() -> aioredis.ConnectionPool:
    global _pool
    if _pool is None:
        _pool = aioredis.ConnectionPool.from_url(
            settings.redis_url,
            max_connections=10,
            decode_responses=True,
        )
    return _pool


def get_redis() -> aioredis.Redis:
    return aioredis.Redis(connection_pool=get_pool())


class PriceCache:
    """Fast price read/write via Redis. TTL-based to stay fresh."""

    def __init__(self):
        self.r = get_redis()

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


price_cache = PriceCache()
