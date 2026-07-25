"""
Binance broker integration via CCXT async.
Supports spot trading, real-time order book, and triangular arb scanning.
"""
import asyncio
import time
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

        # Monitoring counters
        self._order_counter = 0

    async def close(self):
        await self.exchange.close()

    async def place_order(self, request: OrderRequest) -> OrderResult:
        start_time = time.perf_counter()
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

            result = OrderResult(
                broker_order_id=str(order["id"]),
                status=order["status"],
                filled_qty=float(order.get("filled", 0)),
                avg_fill_price=float(order["average"])
                if order.get("average")
                else None,
                raw_payload=order,
            )
            # Monitoring
            self._order_counter += 1
            exec_time = time.perf_counter() - start_time
            logger.info(
                "Binance place_order executed",
                order_id=result.broker_order_id,
                symbol=request.symbol,
                side=request.side,
                quantity=request.quantity,
                order_type=request.order_type,
                filled_qty=result.filled_qty,
                avg_fill_price=result.avg_fill_price,
                execution_time_ms=round(exec_time * 1000, 2),
                total_orders=self._order_counter,
            )
            return result
        except Exception as e:
            logger.error(
                "Binance place_order failed",
                symbol=request.symbol,
                side=request.side,
                quantity=request.quantity,
                order_type=request.order_type,
                error=str(e),
            )
            raise BrokerError(f"Binance: {e}")

    async def cancel_order(self, broker_order_id: str, symbol: str = "") -> bool:
        try:
            await self.exchange.cancel_order(broker_order_id, symbol)
            logger.info(
                "Binance cancel_order succeeded",
                order_id=broker_order_id,
                symbol=symbol,
            )
            return True
        except Exception as e:
            logger.warning(
                "Binance cancel_order failed",
                order_id=broker_order_id,
                symbol=symbol,
                error=str(e),
            )
            return False

    async def get_order(self, broker_order_id: str, symbol: str = "") -> dict:
        start_time = time.perf_counter()
        order = await self.exchange.fetch_order(broker_order_id, symbol)
        exec_time = time.perf_counter() - start_time
        logger.info(
            "Binance fetch_order executed",
            order_id=broker_order_id,
            symbol=symbol,
            execution_time_ms=round(exec_time * 1000, 2),
        )
        return order

    async def get_positions(self) -> list[dict]:
        start_time = time.perf_counter()
        balance = await self.exchange.fetch_balance()
        exec_time = time.perf_counter() - start_time
        logger.info(
            "Binance fetch_balance executed",
            execution_time_ms=round(exec_time * 1000, 2),
        )
        positions = []
        for asset, info in balance["total"].items():
            if info > 0 and asset != "USDT":
                positions.append({"symbol": f"{asset}/USDT", "qty": info, "side": "long"})
        return positions

    async def get_account(self) -> dict:
        start_time = time.perf_counter()
        balance = await self.exchange.fetch_balance()
        exec_time = time.perf_counter() - start_time
        usdt = balance["total"].get("USDT", 0)
        logger.info(
            "Binance get_account executed",
            usdt=usdt,
            execution_time_ms=round(exec_time * 1000, 2),
        )
        return {
            "equity": usdt,
            "cash": usdt,
            "buying_power": usdt,
            "portfolio_value": usdt,
        }

    async def get_quote(self, symbol: str) -> QuoteResult:
        start_time = time.perf_counter()
        try:
            ticker = await asyncio.wait_for(
                self.exchange.fetch_ticker(symbol), timeout=10.0
            )
        except asyncio.TimeoutError:
            logger.warning("Binance fetch_ticker timed out", symbol=symbol)
            raise BrokerError(f"Binance quote timed out for {symbol}")
        exec_time = time.perf_counter() - start_time
        logger.info(
            "Binance get_quote executed",
            symbol=symbol,
            bid=ticker.get("bid"),
            ask=ticker.get("ask"),
            last=ticker.get("last"),
            execution_time_ms=round(exec_time * 1000, 2),
        )
        return QuoteResult(
            symbol=symbol,
            bid=float(ticker["bid"]),
            ask=float(ticker["ask"]),
            last=float(ticker["last"]),
            volume=float(ticker.get("baseVolume", 0)),
        )

    async def get_historical(
        self, symbol: str, interval: str = "1d", limit: int = 500
    ) -> list[dict]:
        start_time = time.perf_counter()
        tf = INTERVAL_MAP.get(interval, "1d")
        ohlcv = await self.exchange.fetch_ohlcv(symbol, tf, limit=limit)
        exec_time = time.perf_counter() - start_time
        logger.info(
            "Binance get_historical executed",
            symbol=symbol,
            interval=interval,
            limit=limit,
            execution_time_ms=round(exec_time * 1000, 2),
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
        start_time = time.perf_counter()
        order_book = await self.exchange.fetch_order_book(symbol, limit)
        exec_time = time.perf_counter() - start_time
        logger.info(
            "Binance get_order_book executed",
            symbol=symbol,
            limit=limit,
            execution_time_ms=round(exec_time * 1000, 2),
        )
        return order_book

    async def get_all_tickers(self, cache_ttl: int = 30) -> dict:
        """Fetch all tickers for triangular arb scanning with simple TTL caching."""
        async with self._ticker_lock:
            now = time.monotonic()
            if (
                self._ticker_cache["data"] is not None
                and now - self._ticker_cache["timestamp"] < cache_ttl
            ):
                logger.info(
                    "Binance get_all_tickers cache hit",
                    cache_ttl=cache_ttl,
                )
                return self._ticker_cache["data"]
            try:
                data = await self.exchange.fetch_tickers()
                self._ticker_cache.update({"data": data, "timestamp": now})
                logger.info(
                    "Binance get_all_tickers fetched",
                    ticker_count=len(data),
                    execution_time_ms=round((time.monotonic() - now) * 1000, 2),
                )
                return data
            except Exception as e:
                logger.error(
                    "Failed to fetch tickers from Binance",
                    error=str(e),
                )
                raise BrokerError(f"Binance ticker fetch error: {e}")