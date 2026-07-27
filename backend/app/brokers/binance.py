"""
Binance broker integration via CCXT async.
Supports spot trading, real-time order book, and triangular arb scanning.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional

from app.brokers.base import AbstractBroker, OrderRequest, OrderResult, QuoteResult
from app.utils.exceptions import BrokerError
from app.utils.logging import logger

try:
    import ccxt.async_support as ccxt
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
    """
    Binance broker implementation using the asynchronous CCXT client.

    Args:
        api_key: Binance API key.
        secret: Binance secret key.
        testnet: Whether to use Binance testnet sandbox mode. Defaults to ``True``.
    """

    def __init__(self, api_key: str, secret: str, testnet: bool = True) -> None:
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
        self._ticker_cache: Dict[str, Any] = {"data": None, "timestamp": 0.0}
        self._ticker_lock = asyncio.Lock()

    async def close(self) -> None:
        """Close the underlying CCXT exchange connection."""
        await self.exchange.close()

    async def place_order(self, request: OrderRequest) -> OrderResult:
        """
        Place a market or limit order on Binance.

        Args:
            request: An :class:`OrderRequest` containing order details.

        Returns:
            An :class:`OrderResult` with execution information.

        Raises:
            BrokerError: If the order cannot be placed.
        """
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
        except Exception as e:
            raise BrokerError(f"Binance: {e}")

    async def cancel_order(self, broker_order_id: str, symbol: str = "") -> bool:
        """
        Cancel an existing order.

        Args:
            broker_order_id: The Binance order identifier.
            symbol: Optional trading pair symbol; required by some exchanges.

        Returns:
            ``True`` if cancellation succeeded, ``False`` otherwise.
        """
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

    async def get_order(self, broker_order_id: str, symbol: str = "") -> Dict[str, Any]:
        """
        Retrieve details of a specific order.

        Args:
            broker_order_id: The Binance order identifier.
            symbol: Optional trading pair symbol.

        Returns:
            A dictionary with order information as returned by CCXT.
        """
        return await self.exchange.fetch_order(broker_order_id, symbol)

    async def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get current non‑USDT positions.

        Returns:
            A list of dictionaries, each containing ``symbol``, ``qty``, and ``side``.
        """
        balance = await self.exchange.fetch_balance()
        positions: List[Dict[str, Any]] = []
        for asset, info in balance["total"].items():
            if info > 0 and asset != "USDT":
                positions.append({"symbol": f"{asset}/USDT", "qty": info, "side": "long"})
        return positions

    async def get_account(self) -> Dict[str, float]:
        """
        Retrieve a simplified account snapshot.

        Returns:
            A dictionary with keys ``equity``, ``cash``, ``buying_power``, and ``portfolio_value``.
        """
        balance = await self.exchange.fetch_balance()
        usdt = balance["total"].get("USDT", 0)
        return {
            "equity": usdt,
            "cash": usdt,
            "buying_power": usdt,
            "portfolio_value": usdt,
        }

    async def get_quote(self, symbol: str) -> QuoteResult:
        """
        Fetch the latest market quote for a symbol.

        Args:
            symbol: Trading pair symbol, e.g., ``BTC/USDT``.

        Returns:
            A :class:`QuoteResult` containing bid, ask, last price, and volume.

        Raises:
            BrokerError: If the ticker request times out or fails.
        """
        try:
            ticker = await asyncio.wait_for(
                self.exchange.fetch_ticker(symbol), timeout=10.0
            )
        except asyncio.TimeoutError:
            logger.warning("Binance fetch_ticker timed out", symbol=symbol)
            raise BrokerError(f"Binance quote timed out for {symbol}")
        return QuoteResult(
            symbol=symbol,
            bid=float(ticker["bid"]),
            ask=float(ticker["ask"]),
            last=float(ticker["last"]),
            volume=float(ticker.get("baseVolume", 0)),
        )

    async def get_historical(
        self, symbol: str, interval: str = "1d", limit: int = 500
    ) -> List[Dict[str, Any]]:
        """
        Retrieve historical OHLCV data.

        Args:
            symbol: Trading pair symbol.
            interval: Timeframe identifier (e.g., ``1m``, ``1h``). Defaults to ``1d``.
            limit: Maximum number of bars to fetch. Defaults to ``500``.

        Returns:
            A list of dictionaries with timestamp and OHLCV fields.
        """
        tf = INTERVAL_MAP.get(interval, "1d")
        ohlcv = await self.exchange.fetch_ohlcv(symbol, tf, limit=limit)
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

    async def get_order_book(self, symbol: str, limit: int = 20) -> Dict[str, Any]:
        """
        Fetch the current order book for a symbol.

        Args:
            symbol: Trading pair symbol.
            limit: Number of price levels to retrieve. Defaults to ``20``.

        Returns:
            The raw order book dictionary as provided by CCXT.
        """
        return await self.exchange.fetch_order_book(symbol, limit)

    async def get_all_tickers(self, cache_ttl: int = 30) -> Dict[str, Any]:
        """
        Fetch all tickers for triangular arbitrage scanning with TTL caching.

        Args:
            cache_ttl: Cache time‑to‑live in seconds. Defaults to ``30``.

        Returns:
            A dictionary of ticker data.

        Raises:
            BrokerError: If fetching tickers fails.
        """
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
            except Exception as e:
                logger.error("Failed to fetch tickers from Binance", error=str(e))
                raise BrokerError(f"Binance ticker fetch error: {e}")