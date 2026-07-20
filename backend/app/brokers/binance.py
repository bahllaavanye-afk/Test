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
    from ccxt.base.errors import (
        BaseError,
        NetworkError,
        ExchangeError,
        AuthenticationError,
        InvalidOrder,
    )
    CCXT_AVAILABLE = True
except ImportError:  # pragma: no cover
    ccxt = None  # type: ignore
    BaseError = Exception  # fallback for type checking
    NetworkError = Exception
    ExchangeError = Exception
    AuthenticationError = Exception
    InvalidOrder = Exception
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

    async def close(self):
        await self.exchange.close()

    async def place_order(self, request: OrderRequest) -> OrderResult:
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

            return OrderResult(
                broker_order_id=str(order["id"]),
                status=order["status"],
                filled_qty=float(order.get("filled", 0)),
                avg_fill_price=float(order["average"])
                if order.get("average")
                else None,
                raw_payload=order,
            )
        except (InvalidOrder, NetworkError, ExchangeError, AuthenticationError) as e:
            logger.error(
                "Binance place_order failed",
                symbol=request.symbol,
                side=request.side,
                qty=request.quantity,
                order_type=request.order_type,
                error=str(e),
            )
            raise BrokerError(f"Binance place_order error: {e}") from e
        except BaseError as e:
            logger.error(
                "Binance unexpected error during place_order",
                symbol=request.symbol,
                error=str(e),
            )
            raise BrokerError(f"Binance unexpected error: {e}") from e

    async def cancel_order(self, broker_order_id: str, symbol: str = "") -> bool:
        try:
            await self.exchange.cancel_order(broker_order_id, symbol)
            return True
        except (NetworkError, ExchangeError) as e:
            logger.warning(
                "Binance cancel_order failed",
                order_id=broker_order_id,
                symbol=symbol,
                error=str(e),
            )
            return False
        except BaseError as e:
            logger.error(
                "Binance unexpected error during cancel_order",
                order_id=broker_order_id,
                error=str(e),
            )
            return False

    async def get_order(self, broker_order_id: str, symbol: str = "") -> dict:
        try:
            return await self.exchange.fetch_order(broker_order_id, symbol)
        except (NetworkError, ExchangeError) as e:
            logger.error(
                "Binance get_order failed",
                order_id=broker_order_id,
                symbol=symbol,
                error=str(e),
            )
            raise BrokerError(f"Binance get_order error: {e}") from e
        except BaseError as e:
            logger.error(
                "Binance unexpected error during get_order",
                order_id=broker_order_id,
                error=str(e),
            )
            raise BrokerError(f"Binance unexpected error: {e}") from e

    async def get_positions(self) -> list[dict]:
        try:
            balance = await self.exchange.fetch_balance()
        except (NetworkError, ExchangeError) as e:
            logger.error("Binance fetch_balance failed for positions", error=str(e))
            raise BrokerError(f"Binance fetch_balance error: {e}") from e
        positions = []
        for asset, info in balance["total"].items():
            if info > 0 and asset != "USDT":
                positions.append({"symbol": f"{asset}/USDT", "qty": info, "side": "long"})
        return positions

    async def get_account(self) -> dict:
        try:
            balance = await self.exchange.fetch_balance()
        except (NetworkError, ExchangeError) as e:
            logger.error("Binance fetch_balance failed for account", error=str(e))
            raise BrokerError(f"Binance fetch_balance error: {e}") from e
        usdt = balance["total"].get("USDT", 0)
        return {
            "equity": usdt,
            "cash": usdt,
            "buying_power": usdt,
            "portfolio_value": usdt,
        }

    async def get_quote(self, symbol: str) -> QuoteResult:
        try:
            ticker = await asyncio.wait_for(
                self.exchange.fetch_ticker(symbol), timeout=10.0
            )
        except asyncio.TimeoutError:
            logger.warning("Binance fetch_ticker timed out", symbol=symbol)
            raise BrokerError(f"Binance quote timed out for {symbol}")
        except (NetworkError, ExchangeError) as e:
            logger.error("Binance fetch_ticker failed", symbol=symbol, error=str(e))
            raise BrokerError(f"Binance quote error for {symbol}: {e}") from e
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
        tf = INTERVAL_MAP.get(interval, "1d")
        try:
            ohlcv = await self.exchange.fetch_ohlcv(symbol, tf, limit=limit)
        except (NetworkError, ExchangeError) as e:
            logger.error(
                "Binance fetch_ohlcv failed",
                symbol=symbol,
                interval=interval,
                limit=limit,
                error=str(e),
            )
            raise BrokerError(f"Binance historical data error: {e}") from e
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
        try:
            return await self.exchange.fetch_order_book(symbol, limit)
        except (NetworkError, ExchangeError) as e:
            logger.error(
                "Binance fetch_order_book failed",
                symbol=symbol,
                limit=limit,
                error=str(e),
            )
            raise BrokerError(f"Binance order book error: {e}") from e

    async def get_all_tickers(self, cache_ttl: int = 30) -> dict:
        """Fetch all tickers for triangular arb scanning with simple TTL caching."""
        async with self._ticker_lock:
            now = time.monotonic()
            if (
                self._ticker_cache["data"] is not None
                and now - self._ticker_cache["timestamp"] < cache_ttl
            ):
                return self._ticker_cache["data"]
            try:
                data = await self.exchange.fetch_tickers()
                self._ticker_cache.update({"data": data, "timestamp": now})
                return data
            except (NetworkError, ExchangeError) as e:
                logger.error("Failed to fetch tickers from Binance", error=str(e))
                raise BrokerError(f"Binance ticker fetch error: {e}") from e
            except BaseError as e:
                logger.error("Unexpected error fetching Binance tickers", error=str(e))
                raise BrokerError(f"Binance unexpected ticker error: {e}") from e