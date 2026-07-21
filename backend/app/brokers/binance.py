import asyncio
import time
import psutil
import os
from app.brokers.base import AbstractBroker, OrderRequest, OrderResult, QuoteResult
from app.utils.exceptions import BrokerError
from app.utils.logging import logger

try:
    import ccxt.async_support as ccxt
    CCXT_AVAILABLE = True
except ImportError:
    ccxt = None  # type: ignore
    CCXT_AVAILABLE = False
    logger.info("ccxt not installed — Binance broker disabled")


INTERVAL_MAP = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
}


class BinanceBroker(AbstractBroker):
    def __init__(self, api_key: str, secret: str, testnet: bool = True):
        self.exchange = ccxt.binance(
            {
                "apiKey": api_key,
                "secret": secret,
                "options": {"defaultType": "spot"},
                "enableRateLimit": True,
                "timeout": 30000,
            }
        )
        if testnet:
            self.exchange.set_sandbox_mode(True)

        # Cache for expensive calls
        self._ticker_cache = {"data": None, "timestamp": 0.0}
        self._ticker_lock = asyncio.Lock()
        
        # Metrics tracking
        self._signal_count = 0
        self._pnl_total = 0.0

    def _log_metrics(self, execution_time: float, signal_count: int = 0, pnl: float = 0.0):
        """Log structured metrics at INFO level."""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        self._signal_count += signal_count
        self._pnl_total += pnl
        
        logger.info(
            "broker_metrics",
            execution_time_ms=round(execution_time * 1000, 2),
            signal_count=self._signal_count,
            pnl_total=round(self._pnl_total, 4),
            memory_rss_mb=round(memory_info.rss / 1024 / 1024, 2),
            memory_vms_mb=round(memory_info.vms / 1024 / 1024, 2),
        )

    async def close(self):
        await self.exchange.close()

    async def place_order(self, request: OrderRequest) -> OrderResult:
        start_time = time.monotonic()
        try:
            if request.order_type == "market":
                order = await self.exchange.create_market_order(
                    request.symbol, request.side, request.quantity
                )
            elif request.order_type == "limit" and request.limit_price:
                order = await self.exchange.create_limit_order(
                    request.symbol,
                    request.side,
                    request.quantity,
                    request.limit_price,
                )
            else:
                order = await self.exchange.create_market_order(
                    request.symbol, request.side, request.quantity
                )

            execution_time = time.monotonic() - start_time
            logger.info(
                "order_placed",
                symbol=request.symbol,
                side=request.side,
                quantity=request.quantity,
                order_id=str(order["id"]),
                status=order["status"],
                execution_time_ms=round(execution_time * 1000, 2),
            )
            self._log_metrics(execution_time, signal_count=1)

            return OrderResult(
                broker_order_id=str(order["id"]),
                status=order["status"],
                filled_qty=float(order.get("filled", 0)),
                avg_fill_price=float(order["average"])
                if order.get("average")
                else None,
                raw_payload=order,
            )
        except Exception as e:
            execution_time = time.monotonic() - start_time
            logger.error(
                "order_placement_failed",
                symbol=request.symbol,
                error=str(e),
                execution_time_ms=round(execution_time * 1000, 2),
            )
            raise BrokerError(f"Binance: {e}")

    async def cancel_order(self, broker_order_id: str, symbol: str = "") -> bool:
        start_time = time.monotonic()
        try:
            await self.exchange.cancel_order(broker_order_id, symbol)
            execution_time = time.monotonic() - start_time
            logger.info(
                "order_cancelled",
                order_id=broker_order_id,
                symbol=symbol,
                execution_time_ms=round(execution_time * 1000, 2),
            )
            return True
        except Exception as e:
            execution_time = time.monotonic() - start_time
            logger.warning(
                "order_cancel_failed",
                order_id=broker_order_id,
                symbol=symbol,
                error=str(e),
                execution_time_ms=round(execution_time * 1000, 2),
            )
            return False

    async def get_order(self, broker_order_id: str, symbol: str = "") -> dict:
        return await self.exchange.fetch_order(broker_order_id, symbol)

    async def get_positions(self) -> list[dict]:
        start_time = time.monotonic()
        balance = await self.exchange.fetch_balance()
        positions = []
        for asset, info in balance["total"].items():
            if info > 0 and asset != "USDT":
                positions.append({"symbol": f"{asset}/USDT", "qty": info, "side": "long"})
        
        execution_time = time.monotonic() - start_time
        logger.info(
            "positions_fetched",
            position_count=len(positions),
            execution_time_ms=round(execution_time * 1000, 2),
        )
        
        return positions

    async def get_account(self) -> dict:
        start_time = time.monotonic()
        balance = await self.exchange.fetch_balance()
        usdt = balance["total"].get("USDT", 0)
        
        execution_time = time.monotonic() - start_time
        logger.info(
            "account_fetched",
            equity=round(usdt, 2),
            execution_time_ms=round(execution_time * 1000, 2),
        )
        
        return {
            "equity": usdt,
            "cash": usdt,
            "buying_power": usdt,
            "portfolio_value": usdt,
        }

    async def get_quote(self, symbol: str) -> QuoteResult:
        start_time = time.monotonic()
        try:
            ticker = await asyncio.wait_for(
                self.exchange.fetch_ticker(symbol), timeout=10.0
            )
            execution_time = time.monotonic() - start_time
            logger.info(
                "quote_fetched",
                symbol=symbol,
                bid=round(float(ticker["bid"]), 8),
                ask=round(float(ticker["ask"]), 8),
                execution_time_ms=round(execution_time * 1000, 2),
            )
            return QuoteResult(
                symbol=symbol,
                bid=float(ticker["bid"]),
                ask=float(ticker["ask"]),
                last=float(ticker["last"]),
                volume=float(ticker.get("baseVolume", 0)),
            )
        except asyncio.TimeoutError:
            execution_time = time.monotonic() - start_time
            logger.warning(
                "quote_timeout",
                symbol=symbol,
                execution_time_ms=round(execution_time * 1000, 2),
            )
            raise BrokerError(f"Binance quote timed out for {symbol}")
        except Exception as e:
            execution_time = time.monotonic() - start_time
            logger.error(
                "quote_fetch_failed",
                symbol=symbol,
                error=str(e),
                execution_time_ms=round(execution_time * 1000, 2),
            )
            raise

    async def get_historical(
        self, symbol: str, interval: str = "1d", limit: int = 500
    ) -> list[dict]:
        start_time = time.monotonic()
        tf = INTERVAL_MAP.get(interval, "1d")
        ohlcv = await self.exchange.fetch_ohlcv(symbol, tf, limit=limit)
        
        execution_time = time.monotonic() - start_time
        logger.info(
            "historical_fetched",
            symbol=symbol,
            interval=interval,
            bar_count=len(ohlcv),
            execution_time_ms=round(execution_time * 1000, 2),
        )
        
        return [
            {
                "ts": self.exchange.iso8601(bar[0]),
                "open": bar[1],
                "high": bar[2],
                "low": bar[3],
                "close": bar[4],
                "volume": bar[5],
            }
            for bar in ohlcv
        ]

    async def get_order_book(self, symbol: str, limit: int = 20) -> dict:
        start_time = time.monotonic()
        orderbook = await self.exchange.fetch_order_book(symbol, limit)
        
        execution_time = time.monotonic() - start_time
        logger.info(
            "orderbook_fetched",
            symbol=symbol,
            limit=limit,
            execution_time_ms=round(execution_time * 1000, 2),
        )
        
        return orderbook

    async def get_all_tickers(self, cache_ttl: int = 30) -> dict:
        """Fetch all tickers for triangular arb scanning with simple TTL caching."""
        start_time = time.monotonic()
        async with self._ticker_lock:
            now = time.monotonic()
            if (
                self._ticker_cache["data"] is not None
                and now - self._ticker_cache["timestamp"] < cache_ttl
            ):
                cache_age = now - self._ticker_cache["timestamp"]
                logger.info(
                    "tickers_cached",
                    ticker_count=len(self._ticker_cache["data"]),
                    cache_age_sec=round(cache_age, 2),
                    execution_time_ms=round((time.monotonic() - start_time) * 1000, 2),
                )
                return self._ticker_cache["data"]
            try:
                data = await self.exchange.fetch_tickers()
                self._ticker_cache.update({"data": data, "timestamp": now})
                execution_time = time.monotonic() - start_time
                logger.info(
                    "tickers_fetched",
                    ticker_count=len(data),
                    execution_time_ms=round(execution_time * 1000, 2),
                )
                return data
            except Exception as e:
                execution_time = time.monotonic() - start_time
                logger.error(
                    "ticker_fetch_failed",
                    error=str(e),
                    execution_time_ms=round(execution_time * 1000, 2),
                )
                raise BrokerError(f"Binance ticker fetch error: {e}")