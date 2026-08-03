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
        AuthenticationError,
        ExchangeError,
        InvalidOrder,
        NetworkError,
        RateLimitExceeded,
        InsufficientFunds,
    )
    CCXT_AVAILABLE = True
except ImportError:  # pragma: no cover
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

    async def close(self) -> None:
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
        except (AuthenticationError, InsufficientFunds) as e:
            logger.error(
                "Authentication or funds error while placing order",
                symbol=request.symbol,
                side=request.side,
                qty=request.quantity,
                error=str(e),
            )
            raise BrokerError(f"Binance authentication/funds error: {e}") from e
        except InvalidOrder as e:
            logger.warning(
                "Invalid order parameters",
                symbol=request.symbol,
                side=request.side,
                qty=request.quantity,
                limit_price=request.limit_price,
                error=str(e),
            )
            raise BrokerError(f"Binance invalid order: {e}") from e
        except RateLimitExceeded as e:
            logger.warning(
                "Rate limit exceeded while placing order",
                symbol=request.symbol,
                error=str(e),
            )
            raise BrokerError(f"Binance rate limit exceeded: {e}") from e
        except (NetworkError, ExchangeError) as e:
            logger.error(
                "Network or exchange error while placing order",
                symbol=request.symbol,
                error=str(e),
            )
            raise BrokerError(f"Binance network/exchange error: {e}") from e
        except Exception as e:  # pragma: no cover
            logger.exception(
                "Unexpected error in place_order",
                symbol=request.symbol,
                side=request.side,
                qty=request.quantity,
                error=str(e),
            )
            raise BrokerError(f"Binance unexpected error: {e}") from e

    async def cancel_order(self, broker_order_id: str, symbol: str = "") -> bool:
        try:
            await self.exchange.cancel_order(broker_order_id, symbol)
            return True
        except (AuthenticationError, InvalidOrder) as e:
            logger.warning(
                "Authentication or invalid order error during cancel",
                order_id=broker_order_id,
                symbol=symbol,
                error=str(e),
            )
            return False
        except RateLimitExceeded as e:
            logger.warning(
                "Rate limit exceeded while canceling order",
                order_id=broker_order_id,
                symbol=symbol,
                error=str(e),
            )
            return False
        except (NetworkError, ExchangeError) as e:
            logger.error(
                "Network or exchange error during cancel",
                order_id=broker_order_id,
                symbol=symbol,
                error=str(e),
            )
            return False
        except Exception as e:  # pragma: no cover
            logger.exception(
                "Unexpected error in cancel_order",
                order_id=broker_order_id,
                symbol=symbol,
                error=str(e),
            )
            return False

    async def get_order(self, broker_order_id: str, symbol: str = "") -> dict:
        try:
            return await self.exchange.fetch_order(broker_order_id, symbol)
        except (AuthenticationError, InvalidOrder) as e:
            logger.warning(
                "Failed to fetch order due to auth/invalid parameters",
                order_id=broker_order_id,
                symbol=symbol,
                error=str(e),
            )
            raise BrokerError(f"Binance order fetch error: {e}") from e
        except (NetworkError, ExchangeError) as e:
            logger.error(
                "Network or exchange error while fetching order",
                order_id=broker_order_id,
                symbol=symbol,
                error=str(e),
            )
            raise BrokerError(f"Binance order fetch error: {e}") from e
        except Exception as e:  # pragma: no cover
            logger.exception(
                "Unexpected error in get_order",
                order_id=broker_order_id,
                symbol=symbol,
                error=str(e),
            )
            raise BrokerError(f"Binance unexpected error: {e}") from e

    async def get_positions(self) -> list[dict]:
        try:
            balance = await self.exchange.fetch_balance()
        except (AuthenticationError, NetworkError, ExchangeError) as e:
            logger.error("Failed to fetch positions", error=str(e))
            raise BrokerError(f"Binance fetch_balance error: {e}") from e
        positions = []
        for asset, info in balance["total"].items():
            if info > 0 and asset != "USDT":
                positions.append({"symbol": f"{asset}/USDT", "qty": info, "side": "long"})
        return positions

    async def get_account(self) -> dict:
        try:
            balance = await self.exchange.fetch_balance()
        except (AuthenticationError, NetworkError, ExchangeError) as e:
            logger.error("Failed to fetch account balance", error=str(e))
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
            logger.error("Error fetching ticker", symbol=symbol, error=str(e))
            raise BrokerError(f"Binance quote error for {symbol}: {e}") from e
        except Exception as e:  # pragma: no cover
            logger.exception("Unexpected error in get_quote", symbol=symbol, error=str(e))
            raise BrokerError(f"Binance unexpected quote error: {e}") from e

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
                "Failed to fetch historical data",
                symbol=symbol,
                interval=tf,
                error=str(e),
            )
            raise BrokerError(f"Binance historical fetch error: {e}") from e
        except Exception as e:  # pragma: no cover
            logger.exception(
                "Unexpected error in get_historical",
                symbol=symbol,
                interval=tf,
                error=str(e),
            )
            raise BrokerError(f"Binance unexpected historical error: {e}") from e

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
                "Failed to fetch order book",
                symbol=symbol,
                limit=limit,
                error=str(e),
            )
            raise BrokerError(f"Binance order book error: {e}") from e
        except Exception as e:  # pragma: no cover
            logger.exception(
                "Unexpected error in get_order_book",
                symbol=symbol,
                limit=limit,
                error=str(e),
            )
            raise BrokerError(f"Binance unexpected order book error: {e}") from e

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
            except Exception as e:  # pragma: no cover
                logger.exception("Unexpected error in get_all_tickers", error=str(e))
                raise BrokerError(f"Binance unexpected ticker error: {e}") from e