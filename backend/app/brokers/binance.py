"""
Binance broker integration via CCXT async.
Supports spot trading, real-time order book, and triangular arb scanning.
Enhanced with lightweight signal helpers for tighter entry/exit decisions.
"""
import asyncio
import time
from typing import Optional, Dict, Any, List

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
    Binance broker implementation with additional signal utilities.
    """

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
        self._ticker_cache: Dict[str, Any] = {"data": None, "timestamp": 0.0}
        self._ticker_lock = asyncio.Lock()

        # Simple in‑memory cache for recent OHLCV (used by signal helpers)
        self._ohlcv_cache: Dict[str, Dict[str, Any]] = {}
        self._ohlcv_lock = asyncio.Lock()

    async def close(self) -> None:
        await self.exchange.close()

    # ----------------------------------------------------------------------
    # Core broker methods (unchanged behaviour)
    # ----------------------------------------------------------------------
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
        except Exception as e:
            raise BrokerError(f"Binance: {e}")

    async def cancel_order(self, broker_order_id: str, symbol: str = "") -> bool:
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
        return await self.exchange.fetch_order(broker_order_id, symbol)

    async def get_positions(self) -> List[dict]:
        balance = await self.exchange.fetch_balance()
        positions = []
        for asset, info in balance["total"].items():
            if info > 0 and asset != "USDT":
                positions.append({"symbol": f"{asset}/USDT", "qty": info, "side": "long"})
        return positions

    async def get_account(self) -> dict:
        balance = await self.exchange.fetch_balance()
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
        return QuoteResult(
            symbol=symbol,
            bid=float(ticker["bid"]),
            ask=float(ticker["ask"]),
            last=float(ticker["last"]),
            volume=float(ticker.get("baseVolume", 0)),
        )

    async def get_historical(
        self, symbol: str, interval: str = "1d", limit: int = 500
    ) -> List[dict]:
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

    async def get_order_book(self, symbol: str, limit: int = 20) -> dict:
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
                self._ticker_cache.update({"data": data, "timestamp": now})
                return data
            except Exception as e:
                logger.error("Failed to fetch tickers from Binance", error=str(e))
                raise BrokerError(f"Binance ticker fetch error: {e}")

    # ----------------------------------------------------------------------
    # Signal helper methods – tightened entry/exit logic
    # ----------------------------------------------------------------------
    async def _cached_ohlcv(
        self, symbol: str, interval: str = "5m", limit: int = 100
    ) -> List[dict]:
        """Retrieve OHLCV with a short‑term cache to reduce API pressure."""
        key = f"{symbol}:{interval}:{limit}"
        async with self._ohlcv_lock:
            entry = self._ohlcv_cache.get(key)
            now = time.monotonic()
            if entry and now - entry["timestamp"] < 30:  # 30‑second TTL
                return entry["data"]
            data = await self.get_historical(symbol, interval, limit)
            self._ohlcv_cache[key] = {"data": data, "timestamp": now}
            return data

    def _simple_moving_average(self, data: List[dict], window: int = 20) -> float:
        """Calculate SMA of close prices; returns 0 if insufficient data."""
        closes = [bar["close"] for bar in data[-window:]]
        return sum(closes) / len(closes) if closes else 0.0

    async def should_enter(
        self,
        symbol: str,
        side: str = "buy",
        price_threshold: float = 0.001,
        volume_multiplier: float = 1.5,
    ) -> bool:
        """
        Determine if entry conditions are satisfied.

        Tightened criteria:
        1. Current price must be within `price_threshold` of the short‑term SMA.
        2. Recent volume must exceed `volume_multiplier` × median recent volume.
        3. Order‑book imbalance > 5 % (bid side for buys, ask side for sells).

        Returns True only when **all** conditions hold.
        """
        try:
            ticker = await self.get_quote(symbol)
            ohlcv = await self._cached_ohlcv(symbol, interval="5m", limit=100)

            sma = self._simple_moving_average(ohlcv, window=20)
            if sma == 0:
                return False

            price = ticker.last
            price_diff = abs(price - sma) / sma

            # Condition 1: price proximity
            if price_diff > price_threshold:
                return False

            # Condition 2: volume spike
            recent_volumes = [bar["volume"] for bar in ohlcv[-20:]]
            median_vol = sorted(recent_volumes)[len(recent_volumes) // 2] or 1
            if ticker.volume < median_vol * volume_multiplier:
                return False

            # Condition 3: order‑book imbalance
            order_book = await self.get_order_book(symbol, limit=20)
            bids = sum([b[1] for b in order_book["bids"]])
            asks = sum([a[1] for a in order_book["asks"]])
            if side == "buy":
                imbalance = (bids - asks) / (bids + asks + 1e-9)
            else:
                imbalance = (asks - bids) / (bids + asks + 1e-9)
            if imbalance < 0.05:  # require at least 5 % imbalance
                return False

            return True
        except Exception as e:
            logger.warning("Signal entry check failed", symbol=symbol, error=str(e))
            return False

    async def should_exit(
        self,
        symbol: str,
        entry_price: float,
        profit_target: float = 0.02,
        stop_loss: float = 0.01,
        trailing_percent: float = 0.015,
    ) -> bool:
        """
        Evaluate exit conditions.

        Exit triggers:
        * Profit target reached (price ≥ entry_price * (1 + profit_target))
        * Stop‑loss breached (price ≤ entry_price * (1 - stop_loss))
        * Trailing stop: price falls > `trailing_percent` from recent high.

        Returns True when any condition is met.
        """
        try:
            ticker = await self.get_quote(symbol)
            price = ticker.last

            # Profit target
            if price >= entry_price * (1 + profit_target):
                return True

            # Stop loss
            if price <= entry_price * (1 - stop_loss):
                return True

            # Trailing stop logic
            ohlcv = await self._cached_ohlcv(symbol, interval="5m", limit=50)
            recent_high = max(bar["high"] for bar in ohlcv)
            if price < recent_high * (1 - trailing_percent):
                return True

            return False
        except Exception as e:
            logger.warning("Signal exit check failed", symbol=symbol, error=str(e))
            return False

    async def generate_signal(
        self,
        symbol: str,
        side: str = "buy",
        entry_price: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Produce a signal dict containing entry/exit flags.

        If `entry_price` is None, only entry evaluation is performed.
        When an `entry_price` is provided, exit evaluation is performed.
        """
        signal: Dict[str, Any] = {"symbol": symbol, "side": side, "enter": False, "exit": False}
        if entry_price is None:
            signal["enter"] = await self.should_enter(symbol, side=side)
        else:
            signal["exit"] = await self.should_exit(symbol, entry_price=entry_price)
        return signal

    # ----------------------------------------------------------------------
    # End of BinanceBroker
    # ----------------------------------------------------------------------