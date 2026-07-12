"""Continuous price ingestion from brokers → Redis cache → WebSocket broadcast."""
from __future__ import annotations

import asyncio
from typing import List, Optional

from app.redis_client import get_redis, price_cache
from app.ws.manager import manager
from app.utils.logging import logger
from pydantic import BaseModel, Field, validator

POLL_INTERVAL = 2  # seconds between full cycles
BATCH_SIZE = 20  # max concurrent quote fetches per cycle

# Default symbols to track in paper/stub mode (when no broker keys are available)
DEFAULT_EQUITY_SYMBOLS = [
    "SPY",
    "QQQ",
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "TSLA",
    "NVDA",
    "META",
    "BRK.B",
]
DEFAULT_CRYPTO_SYMBOLS = [
    "BTC/USD",
    "ETH/USD",
    "SOL/USD",
    "BNB/USD",
    "DOGE/USD",
]


class PriceQuote(BaseModel):
    """Schema representing a market quote for a single symbol."""

    symbol: str = Field(
        ...,
        description="Ticker symbol or crypto pair",
        example="AAPL",
    )
    bid: float = Field(
        ...,
        ge=0,
        description="Best bid price",
        example=150.25,
    )
    ask: float = Field(
        ...,
        ge=0,
        description="Best ask price",
        example=150.30,
    )
    last: float = Field(
        ...,
        ge=0,
        description="Last traded price",
        example=150.27,
    )
    volume: Optional[float] = Field(
        None,
        ge=0,
        description="Trade volume for the last price",
        example=1_000_000,
    )

    @validator("ask")
    def ask_must_be_ge_bid(cls, v, values):
        """Ensure ask price is not lower than bid."""
        bid = values.get("bid")
        if bid is not None and v < bid:
            raise ValueError("ask price must be greater than or equal to bid price")
        return v


async def _fetch_and_publish(broker, symbol: str, cache) -> None:
    try:
        quote = await broker.get_quote(symbol)
        price = PriceQuote(
            symbol=symbol,
            bid=quote.bid,
            ask=quote.ask,
            last=quote.last,
            volume=getattr(quote, "volume", None),
        )
        exchange = "crypto" if "/" in symbol else "alpaca"
        await asyncio.gather(
            cache.set_price(
                exchange,
                symbol,
                {"last": price.last, "bid": price.bid, "ask": price.ask},
            ),
            manager.broadcast(
                f"prices:{symbol}",
                {"type": "quote", **price.dict()},
            ),
            return_exceptions=True,
        )
    except Exception as e:
        logger.debug("Price feed error", symbol=symbol, error=str(e))


async def run_price_feed(broker, symbols: List[str]) -> None:
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
        # No Alpaca keys — fall back to yfinance polling (free, no auth needed).
        # 60-second cadence is fine for daily-resolution strategies.
        logger.info("Price feed: no Alpaca broker — using yfinance fallback (60s cadence)")
        await _yfinance_price_feed(DEFAULT_EQUITY_SYMBOLS + DEFAULT_CRYPTO_SYMBOLS)
        return

    symbols = DEFAULT_EQUITY_SYMBOLS + DEFAULT_CRYPTO_SYMBOLS
    await run_price_feed(broker, symbols)


async def _yfinance_price_feed(symbols: List[str]) -> None:
    """Poll yfinance every 60 s and publish last-close prices to Redis + WebSocket."""
    cache = price_cache
    while True:
        for sym in symbols:
            try:
                await asyncio.get_running_loop().run_in_executor(
                    None, _yf_publish_sync, sym, cache
                )
            except Exception as exc:
                logger.debug("yfinance price feed error", symbol=sym, error=str(exc))
        await asyncio.sleep(60)


def _yf_publish_sync(symbol: str, cache) -> None:
    try:
        import yfinance as yf

        yf_sym = symbol.replace("/USD", "-USD").replace("/USDT", "-USD")
        info = yf.Ticker(yf_sym).fast_info
        last = float(
            getattr(info, "last_price", 0)
            or getattr(info, "regularMarketPrice", 0)
            or 0
        )
        if last <= 0:
            return

        price = PriceQuote(
            symbol=symbol,
            bid=last,
            ask=last,
            last=last,
            volume=None,
        )
        import asyncio

        loop = asyncio.new_event_loop()
        loop.run_until_complete(
            asyncio.gather(
                cache.set_price(
                    "yfinance",
                    symbol,
                    {"last": price.last, "bid": price.bid, "ask": price.ask},
                ),
                manager.broadcast(
                    f"prices:{symbol}",
                    {"type": "quote", **price.dict()},
                ),
                return_exceptions=True,
            )
        )
        loop.close()
    except Exception:
        pass