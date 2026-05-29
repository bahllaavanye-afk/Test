"""Continuous price ingestion from brokers → Redis cache → WebSocket broadcast."""
from __future__ import annotations
import asyncio
from app.redis_client import get_redis
from app.ws.manager import manager
from app.utils.logging import logger

POLL_INTERVAL = 2      # seconds between full cycles
BATCH_SIZE    = 20     # max concurrent quote fetches per cycle


async def _fetch_and_publish(broker, symbol: str, cache) -> None:
    try:
        quote = await broker.get_quote(symbol)
        price_data = {
            "symbol":  symbol,
            "bid":     quote.bid,
            "ask":     quote.ask,
            "last":    quote.last,
            "volume":  quote.volume,
        }
        # Fire-and-forget: write to Redis and broadcast concurrently
        await asyncio.gather(
            cache.set_price(symbol, quote.last),
            manager.broadcast(f"prices:{symbol}", {"type": "quote", **price_data}),
            return_exceptions=True,
        )
    except Exception as e:
        logger.debug("Price feed error", symbol=symbol, error=str(e))


async def run_price_feed(broker, symbols: list[str]) -> None:
    """
    Polls all symbols concurrently in batches of BATCH_SIZE every POLL_INTERVAL seconds.
    Concurrent fetches reduce end-to-end latency from O(N) to O(ceil(N/BATCH_SIZE)).
    """
    cache = await get_redis()
    logger.info("Price feed started", symbols=len(symbols), batch_size=BATCH_SIZE)
    while True:
        # Process symbols in parallel batches
        for i in range(0, len(symbols), BATCH_SIZE):
            batch = symbols[i : i + BATCH_SIZE]
            await asyncio.gather(
                *[_fetch_and_publish(broker, sym, cache) for sym in batch],
                return_exceptions=True,
            )
        await asyncio.sleep(POLL_INTERVAL)
