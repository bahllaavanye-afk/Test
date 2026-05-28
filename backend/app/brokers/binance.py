"""
Binance broker integration via CCXT async.
Supports spot trading, real-time order book, and triangular arb scanning.
"""
import asyncio
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


INTERVAL_MAP = {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "1h", "4h": "4h", "1d": "1d"}


class BinanceBroker(AbstractBroker):
    def __init__(self, api_key: str, secret: str, testnet: bool = True):
        self.exchange = ccxt.binance({
            "apiKey": api_key,
            "secret": secret,
            "options": {"defaultType": "spot"},
            "enableRateLimit": True,
        })
        if testnet:
            self.exchange.set_sandbox_mode(True)

    async def close(self):
        await self.exchange.close()

    async def place_order(self, request: OrderRequest) -> OrderResult:
        try:
            if request.order_type == "market":
                order = await self.exchange.create_market_order(
                    request.symbol, request.side, request.quantity)
            elif request.order_type == "limit" and request.limit_price:
                order = await self.exchange.create_limit_order(
                    request.symbol, request.side, request.quantity, request.limit_price)
            else:
                order = await self.exchange.create_market_order(
                    request.symbol, request.side, request.quantity)

            return OrderResult(
                broker_order_id=str(order["id"]),
                status=order["status"],
                filled_qty=float(order.get("filled", 0)),
                avg_fill_price=float(order["average"]) if order.get("average") else None,
                raw_payload=order,
            )
        except Exception as e:
            raise BrokerError(f"Binance: {e}")

    async def cancel_order(self, broker_order_id: str, symbol: str = "") -> bool:
        try:
            await self.exchange.cancel_order(broker_order_id, symbol)
            return True
        except Exception:
            return False

    async def get_order(self, broker_order_id: str, symbol: str = "") -> dict:
        return await self.exchange.fetch_order(broker_order_id, symbol)

    async def get_positions(self) -> list[dict]:
        balance = await self.exchange.fetch_balance()
        positions = []
        for asset, info in balance["total"].items():
            if info > 0 and asset != "USDT":
                positions.append({"symbol": f"{asset}/USDT", "qty": info, "side": "long"})
        return positions

    async def get_account(self) -> dict:
        balance = await self.exchange.fetch_balance()
        usdt = balance["total"].get("USDT", 0)
        return {"equity": usdt, "cash": usdt, "buying_power": usdt, "portfolio_value": usdt}

    async def get_quote(self, symbol: str) -> QuoteResult:
        ticker = await self.exchange.fetch_ticker(symbol)
        return QuoteResult(symbol=symbol, bid=float(ticker["bid"]), ask=float(ticker["ask"]),
                           last=float(ticker["last"]), volume=float(ticker.get("baseVolume", 0)))

    async def get_historical(self, symbol: str, interval: str = "1d", limit: int = 500) -> list[dict]:
        tf = INTERVAL_MAP.get(interval, "1d")
        ohlcv = await self.exchange.fetch_ohlcv(symbol, tf, limit=limit)
        return [{"ts": self.exchange.iso8601(bar[0]), "open": bar[1], "high": bar[2],
                 "low": bar[3], "close": bar[4], "volume": bar[5]} for bar in ohlcv]

    async def get_order_book(self, symbol: str, limit: int = 20) -> dict:
        return await self.exchange.fetch_order_book(symbol, limit)

    async def get_all_tickers(self) -> dict:
        """Fetch all tickers for triangular arb scanning."""
        return await self.exchange.fetch_tickers()
