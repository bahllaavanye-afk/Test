"""Continuous price ingestion from brokers → Redis cache → WebSocket broadcast."""
from __future__ import annotations
import asyncio
from app.redis_client import get_redis
from app.ws.manager import manager
from app.utils.logging import logger

POLL_INTERVAL = 2  # seconds


async def run_price_feed(broker, symbols: list[str]) -> None:
    """Polls broker for quotes every POLL_INTERVAL seconds and fans out to Redis + WebSocket."""
    cache = await get_redis()
    logger.info("Price feed started", symbols=symbols)
    while True:
        for symbol in symbols:
            try:
                quote = await broker.get_quote(symbol)
                price_data = {
                    "symbol": symbol,
                    "bid": quote.bid,
                    "ask": quote.ask,
                    "last": quote.last,
                    "volume": quote.volume,
                }
                await cache.set_price(symbol, quote.last)
                await manager.broadcast(f"prices:{symbol}", {"type": "quote", **price_data})
            except Exception as e:
                logger.debug("Price feed error", symbol=symbol, error=str(e))
        await asyncio.sleep(POLL_INTERVAL)
