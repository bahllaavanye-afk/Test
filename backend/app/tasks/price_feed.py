"""Continuous price ingestion from brokers → Redis cache → WebSocket broadcast."""
from __future__ import annotations
import asyncio
from app.redis_client import get_redis, price_cache
from app.ws.manager import manager
from app.utils.logging import logger

POLL_INTERVAL = 2      # seconds between full cycles
BATCH_SIZE    = 20     # max concurrent quote fetches per cycle

# Default symbols to track in paper/stub mode (when no broker keys are available)
DEFAULT_EQUITY_SYMBOLS = [
    "SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "BRK.B",
]
DEFAULT_CRYPTO_SYMBOLS = [
    "BTC/USD", "ETH/USD", "SOL/USD", "BNB/USD", "DOGE/USD",
]


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
        exchange = "crypto" if "/" in symbol else "alpaca"
        # Fire-and-forget: write to Redis and broadcast concurrently
        await asyncio.gather(
            cache.set_price(exchange, symbol, {"last": quote.last, "bid": quote.bid, "ask": quote.ask}),
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
    # get_redis() is synchronous — do NOT await it; use the module-level price_cache singleton
    cache = price_cache
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


async def start_price_feed() -> None:
    """
    Factory coroutine registered as a supervised background task in main.py.

    Creates the Alpaca broker from settings (if keys are available) and starts
    the price feed loop.  When ALPACA_API_KEY is missing, logs a warning and
    parks in stub mode (strategies fall back to broker REST calls or skip on
    missing data) — the process does NOT crash.
    """
    from app.config import settings

    broker = None
    if settings.alpaca_api_key and settings.alpaca_secret_key:
        try:
            from app.brokers.alpaca import AlpacaBroker
            broker = AlpacaBroker(
                api_key=settings.alpaca_api_key,
                secret_key=settings.alpaca_secret_key,
                paper=(settings.trading_mode != "live"),
            )
            logger.info(
                "Price feed: Alpaca broker connected",
                paper=(settings.trading_mode != "live"),
            )
        except Exception as exc:
            logger.warning(
                "Price feed: failed to create Alpaca broker — running without live quotes",
                error=str(exc),
            )
    else:
        logger.warning(
            "Price feed: ALPACA_API_KEY not set — running in stub mode (no live quotes). "
            "Set ALPACA_API_KEY + ALPACA_SECRET_KEY to enable real-time price ingestion."
        )

    if broker is None:
        # No broker available — park the task indefinitely in stub mode.
        # The _supervised wrapper in main.py will restart if this ever raises.
        logger.info("Price feed: stub mode active — no quotes will be polled")
        while True:
            await asyncio.sleep(60)
        return  # unreachable, satisfies type checkers

    symbols = DEFAULT_EQUITY_SYMBOLS + DEFAULT_CRYPTO_SYMBOLS
    await run_price_feed(broker, symbols)
