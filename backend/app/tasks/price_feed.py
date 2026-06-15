"""Continuous price ingestion from brokers → Redis cache → WebSocket broadcast."""
from __future__ import annotations

import asyncio
import json

from app.redis_client import price_cache
from app.services.agent_logger import agent_logger
from app.utils.logging import logger
from app.ws.manager import manager

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
    # Log feed health once per minute (every ~30 cycles at POLL_INTERVAL=2s)
    _log_every_n = max(1, 60 // POLL_INTERVAL)
    _cycle = 0
    while True:
        # Process symbols in parallel batches
        for i in range(0, len(symbols), BATCH_SIZE):
            batch = symbols[i : i + BATCH_SIZE]
            await asyncio.gather(
                *[_fetch_and_publish(broker, sym, cache) for sym in batch],
                return_exceptions=True,
            )
        _cycle += 1
        if _cycle % _log_every_n == 0:
            agent_logger.log_action_fire_and_forget(
                action="price_feed_tick",
                employee_id="price_feed",
                agent_type="system",
                tool_used="alpaca_api",
                input_summary=f"{len(symbols)} symbols polled",
                output_summary=f"cycle={_cycle}",
                status="ok",
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
        # No Alpaca keys — use FREE real-data sources (no key required):
        #   • Crypto  → Binance public REST klines (real OHLCV, very reliable)
        #   • Equity  → yfinance / Stooq (real OHLCV)
        # Seeds real OHLCV history so strategies get their bars, then refreshes.
        logger.info("Price feed: no Alpaca broker — using free real-data feed (Binance + yfinance/Stooq)")
        await _free_data_feed(DEFAULT_EQUITY_SYMBOLS + DEFAULT_CRYPTO_SYMBOLS)
        return

    symbols = DEFAULT_EQUITY_SYMBOLS + DEFAULT_CRYPTO_SYMBOLS
    await run_price_feed(broker, symbols)


def _binance_symbol(symbol: str) -> str:
    """Map a unified symbol like 'BTC/USD' to a Binance pair 'BTCUSDT'."""
    base = symbol.split("/")[0].upper()
    return f"{base}USDT"


def _binance_klines_sync(symbol: str, interval: str, limit: int = 200) -> list[dict]:
    """Fetch real OHLCV klines from Binance public REST (no API key required).

    Binance is free, keyless, and very reliable. Returns a list of bar dicts.
    Raises on network/HTTP error so the caller can fall through to other sources.
    """
    import urllib.request

    pair = _binance_symbol(symbol)
    url = f"https://api.binance.com/api/v3/klines?symbol={pair}&interval={interval}&limit={limit}"
    with urllib.request.urlopen(url, timeout=10) as resp:  # noqa: S310 (trusted host)
        raw = json.loads(resp.read().decode())
    bars: list[dict] = []
    for k in raw:
        bars.append({
            "timestamp": int(k[0]),
            "open":  float(k[1]),
            "high":  float(k[2]),
            "low":   float(k[3]),
            "close": float(k[4]),
            "volume": float(k[5]),
        })
    return bars


def _yf_history_sync(symbol: str, period: str, interval: str) -> list[dict]:
    """Fetch real OHLCV history for an equity from yfinance (free, keyless)."""
    import yfinance as yf

    yf_sym = symbol.replace("/USD", "-USD").replace("/USDT", "-USD")
    df = yf.Ticker(yf_sym).history(period=period, interval=interval)
    bars: list[dict] = []
    for ts, row in df.iterrows():
        bars.append({
            "timestamp": int(ts.timestamp() * 1000),
            "open":  float(row["Open"]),
            "high":  float(row["High"]),
            "low":   float(row["Low"]),
            "close": float(row["Close"]),
            "volume": float(row.get("Volume", 0) or 0),
        })
    return bars


def _stooq_history_sync(symbol: str) -> list[dict]:
    """Fetch real daily OHLCV for an equity from Stooq CSV (free, keyless)."""
    import csv
    import io
    import urllib.request

    s = symbol.lower().replace("/", "").replace(".", "-")
    if "-usd" not in s and not s.endswith(".us"):
        s = f"{s}.us"
    url = f"https://stooq.com/q/d/l/?s={s}&i=d"
    with urllib.request.urlopen(url, timeout=10) as resp:  # noqa: S310
        text = resp.read().decode()
    bars: list[dict] = []
    for row in csv.DictReader(io.StringIO(text)):
        try:
            bars.append({
                "timestamp": row["Date"],
                "open":  float(row["Open"]),
                "high":  float(row["High"]),
                "low":   float(row["Low"]),
                "close": float(row["Close"]),
                "volume": float(row.get("Volume", 0) or 0),
            })
        except (KeyError, ValueError):
            continue
    return bars


async def _seed_symbol_history(symbol: str, cache) -> bool:
    """Seed real OHLCV history (1h + 1d) for one symbol from the best free source.

    Returns True if at least one interval was successfully populated.
    """
    loop = asyncio.get_running_loop()
    is_crypto = "/" in symbol
    exchange = "crypto" if is_crypto else "alpaca"
    ok = False

    for interval in ("1h", "1d"):
        bars: list[dict] = []
        try:
            if is_crypto:
                bars = await loop.run_in_executor(None, _binance_klines_sync, symbol, interval, 200)
            else:
                period = "60d" if interval == "1h" else "2y"
                bars = await loop.run_in_executor(None, _yf_history_sync, symbol, period, interval)
                if not bars and interval == "1d":
                    bars = await loop.run_in_executor(None, _stooq_history_sync, symbol)
        except Exception as exc:
            logger.debug("history seed failed", symbol=symbol, interval=interval, error=str(exc))
            bars = []

        if bars and len(bars) >= 30:
            await cache.set_ohlcv(exchange, symbol, interval, bars)
            last = bars[-1]["close"]
            await asyncio.gather(
                cache.set_price(exchange, symbol, {"last": last, "bid": last, "ask": last}),
                manager.broadcast(f"prices:{symbol}", {"type": "quote", "symbol": symbol, "last": last, "bid": last, "ask": last}),
                return_exceptions=True,
            )
            ok = True
    return ok


async def _free_data_feed(symbols: list[str]) -> None:
    """Continuous REAL market-data feed using free, keyless sources.

    Seeds OHLCV history for every symbol, then refreshes on a loop. Uses
    Binance public REST for crypto and yfinance/Stooq for equities. Never
    fabricates data — on total network failure a symbol is simply skipped.
    """
    cache = price_cache
    # Initial seed
    seeded = 0
    for sym in symbols:
        if await _seed_symbol_history(sym, cache):
            seeded += 1
    logger.info("Free data feed seeded", seeded=seeded, total=len(symbols))
    if seeded == 0:
        logger.warning(
            "Free data feed: no symbols seeded — all free data hosts unreachable. "
            "Allowlist api.binance.com + query1.finance.yahoo.com (or set ALPACA paper keys)."
        )

    # Refresh loop — re-pull latest bars every 60s
    while True:
        await asyncio.sleep(60)
        for sym in symbols:
            try:
                await _seed_symbol_history(sym, cache)
            except Exception as exc:
                logger.debug("free feed refresh error", symbol=sym, error=str(exc))


async def _yfinance_price_feed(symbols: list[str]) -> None:
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
        last = float(getattr(info, "last_price", 0) or getattr(info, "regularMarketPrice", 0) or 0)
        if last <= 0:
            return
        import asyncio
        loop = asyncio.new_event_loop()
        loop.run_until_complete(asyncio.gather(
            cache.set_price("yfinance", symbol, {"last": last, "bid": last, "ask": last}),
            manager.broadcast(f"prices:{symbol}", {"type": "quote", "symbol": symbol, "last": last, "bid": last, "ask": last}),
            return_exceptions=True,
        ))
        loop.close()
    except Exception:
        pass
