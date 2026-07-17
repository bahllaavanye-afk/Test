"""Continuous price ingestion from brokers → Redis cache → WebSocket broadcast."""
from __future__ import annotations

import asyncio
from app.redis_client import price_cache
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
    except (asyncio.TimeoutError, ConnectionError) as exc:
        logger.error(
            "Timeout or connection error while fetching quote",
            symbol=symbol,
            error=str(exc),
        )
        return
    except Exception as exc:
        logger.exception(
            "Unexpected error while fetching quote",
            symbol=symbol,
        )
        return

    price_data = {
        "symbol": symbol,
        "bid":    quote.bid,
        "ask":    quote.ask,
        "last":   quote.last,
        "volume": quote.volume,
    }
    exchange = "crypto" if "/" in symbol else "alpaca"

    try:
        results = await asyncio.gather(
            cache.set_price(
                exchange,
                symbol,
                {"last": quote.last, "bid": quote.bid, "ask": quote.ask},
            ),
            manager.broadcast(
                f"prices:{symbol}",
                {"type": "quote", **price_data},
            ),
            return_exceptions=True,
        )
        for res in results:
            if isinstance(res, Exception):
                logger.error(
                    "Error during price publish",
                    symbol=symbol,
                    error=str(res),
                )
    except Exception as exc:
        logger.exception(
            "Failed to publish price data",
            symbol=symbol,
        )


async def run_price_feed(broker, symbols: list[str]) -> None:
    """
    Polls all symbols concurrently in batches of BATCH_SIZE every POLL_INTERVAL seconds.
    Concurrent fetches reduce end-to-end latency from O(N) to O(ceil(N/BATCH_SIZE)).
    """
    cache = price_cache
    logger.info(
        "Price feed started",
        symbols=len(symbols),
        batch_size=BATCH_SIZE,
    )
    while True:
        for i in range(0, len(symbols), BATCH_SIZE):
            batch = symbols[i : i + BATCH_SIZE]
            results = await asyncio.gather(
                *[_fetch_and_publish(broker, sym, cache) for sym in batch],
                return_exceptions=True,
            )
            for res in results:
                if isinstance(res, Exception):
                    logger.error(
                        "Batch fetch error",
                        error=str(res),
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
        logger.info(
            "Price feed: no Alpaca broker — using yfinance fallback (60s cadence)"
        )
        await _yfinance_price_feed(DEFAULT_EQUITY_SYMBOLS + DEFAULT_CRYPTO_SYMBOLS)
        return

    symbols = DEFAULT_EQUITY_SYMBOLS + DEFAULT_CRYPTO_SYMBOLS
    await run_price_feed(broker, symbols)


async def _yfinance_price_feed(symbols: list[str]) -> None:
    """Poll yfinance every 60 s and publish last-close prices to Redis + WebSocket."""
    cache = price_cache
    while True:
        for sym in symbols:
            try:
                await asyncio.get_running_loop().run_in_executor(
                    None, _yf_publish_sync, sym, cache
                )
            except (RuntimeError, OSError) as exc:
                logger.error(
                    "Executor error during yfinance price fetch",
                    symbol=sym,
                    error=str(exc),
                )
            except Exception as exc:
                logger.exception(
                    "Unexpected error during yfinance price fetch",
                    symbol=sym,
                )
        await asyncio.sleep(60)


def _yf_publish_sync(symbol: str, cache) -> None:
    try:
        import yfinance as yf
    except ImportError as exc:
        logger.error(
            "yfinance library not installed",
            error=str(exc),
        )
        return

    try:
        yf_sym = (
            symbol.replace("/USD", "-USD")
            .replace("/USDT", "-USD")
        )
        info = yf.Ticker(yf_sym).fast_info
        last = float(
            getattr(info, "last_price", 0)
            or getattr(info, "regularMarketPrice", 0)
            or 0
        )
        if last <= 0:
            return
    except Exception as exc:
        logger.exception(
            "Failed to retrieve yfinance data",
            symbol=symbol,
        )
        return

    try:
        loop = asyncio.new_event_loop()
        results = loop.run_until_complete(
            asyncio.gather(
                cache.set_price(
                    "yfinance",
                    symbol,
                    {"last": last, "bid": last, "ask": last},
                ),
                manager.broadcast(
                    f"prices:{symbol}",
                    {"type": "quote", "symbol": symbol, "last": last, "bid": last, "ask": last},
                ),
                return_exceptions=True,
            )
        )
        for res in results:
            if isinstance(res, Exception):
                logger.error(
                    "Error publishing yfinance price",
                    symbol=symbol,
                    error=str(res),
                )
    finally:
        loop.close()