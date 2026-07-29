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

    async def close(self):
        await self.exchange.close()

    async def place_order(self, request: OrderRequest) -> OrderResult:
        if request is None:
            raise BrokerError("Binance: OrderRequest cannot be None")
        if not request.symbol or not request.side or request.quantity is None:
            raise BrokerError("Binance: Incomplete OrderRequest fields")
        try:
            if request.order_type == "market":
                order = await self.exchange.create_market_order(
                    request.symbol, request.side, request.quantity
                )
            elif request.order_type == "limit" and request.limit_price is not None:
                order = await self.exchange.create_limit_order(
                    request.symbol,
                    request.side,
                    request.quantity,
                    request.limit_price,
                )
            else:
                # Fallback to market order if limit_price missing
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
        except Exception as e:
            raise BrokerError(f"Binance: {e}")

    async def cancel_order(self, broker_order_id: str, symbol: str = "") -> bool:
        if not broker_order_id:
            logger.warning("Binance cancel_order called with empty order id")
            return False
        try:
            await self.exchange.cancel_order(broker_order_id, symbol)
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
        if not broker_order_id:
            raise BrokerError("Binance: broker_order_id cannot be empty")
        return await self.exchange.fetch_order(broker_order_id, symbol)

    async def get_positions(self) -> list[dict]:
        balance = await self.exchange.fetch_balance()
        positions = []
        for asset, info in balance.get("total", {}).items():
            if info > 0 and asset != "USDT":
                positions.append({"symbol": f"{asset}/USDT", "qty": info, "side": "long"})
        return positions

    async def get_account(self) -> dict:
        balance = await self.exchange.fetch_balance()
        usdt = balance.get("total", {}).get("USDT", 0)
        return {
            "equity": usdt,
            "cash": usdt,
            "buying_power": usdt,
            "portfolio_value": usdt,
        }

    async def get_quote(self, symbol: str) -> QuoteResult:
        if not symbol:
            raise BrokerError("Binance: symbol for quote cannot be empty")
        try:
            ticker = await asyncio.wait_for(
                self.exchange.fetch_ticker(symbol), timeout=10.0
            )
        except asyncio.TimeoutError:
            logger.warning("Binance fetch_ticker timed out", symbol=symbol)
            raise BrokerError(f"Binance quote timed out for {symbol}")
        except Exception as e:
            logger.error("Binance fetch_ticker failed", symbol=symbol, error=str(e))
            raise BrokerError(f"Binance failed to fetch ticker for {symbol}: {e}")

        # Guard against missing keys
        bid = ticker.get("bid")
        ask = ticker.get("ask")
        last = ticker.get("last")
        if bid is None or ask is None or last is None:
            raise BrokerError(f"Binance ticker missing price data for {symbol}")

        return QuoteResult(
            symbol=symbol,
            bid=float(bid),
            ask=float(ask),
            last=float(last),
            volume=float(ticker.get("baseVolume", 0)),
        )

    async def get_historical(
        self, symbol: str, interval: str = "1d", limit: int = 500
    ) -> list[dict]:
        if not symbol:
            raise BrokerError("Binance: symbol for historical data cannot be empty")
        if limit is None or limit <= 0:
            logger.warning("Binance get_historical called with non-positive limit; defaulting to 1")
            limit = 1
        tf = INTERVAL_MAP.get(interval, "1d")
        try:
            ohlcv = await self.exchange.fetch_ohlcv(symbol, tf, limit=limit)
        except Exception as e:
            logger.error("Binance fetch_ohlcv failed", symbol=symbol, error=str(e))
            raise BrokerError(f"Binance failed to fetch historical data for {symbol}: {e}")

        if not ohlcv:
            return []

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
        if not symbol:
            raise BrokerError("Binance: symbol for order book cannot be empty")
        if limit is None or limit <= 0:
            logger.warning("Binance get_order_book called with non-positive limit; defaulting to 1")
            limit = 1
        return await self.exchange.fetch_order_book(symbol, limit)

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
                # Ensure we have a dict; fallback to empty dict if None
                if not isinstance(data, dict):
                    logger.warning("Binance fetch_tickers returned non-dict, defaulting to empty")
                    data = {}
                self._ticker_cache.update({"data": data, "timestamp": now})
                return data
            except Exception as e:
                logger.error("Failed to fetch tickers from Binance", error=str(e))
                raise BrokerError(f"Binance ticker fetch error: {e}")