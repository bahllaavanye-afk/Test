"""
Alpaca broker — equities AND crypto on a single API key.

Routing:
  • Symbols containing '/' or ending with 'USD/USDT' → CryptoHistoricalDataClient
    for data, same TradingClient for orders (Alpaca unified account).
  • Everything else → StockHistoricalDataClient / StockLatestQuoteClient.

Alpaca crypto coverage (paper + live, commission-free):
  BTC/USD, ETH/USD, SOL/USD, AVAX/USD, DOGE/USD, SHIB/USD,
  LTC/USD, BCH/USD, LINK/USD, UNI/USD, AAVE/USD, BAT/USD,
  CRV/USD, GRT/USD, MKR/USD, SUSHI/USD, XTZ/USD, ALGO/USD,
  MATIC/USD, DOT/USD — and growing.

For perpetual futures, funding rates, liquidation data, or stablecoin
pairs (USDC/USDT, DAI/USDT) use BinanceBroker — those are not
available on Alpaca spot.
"""
import asyncio
from datetime import datetime, timezone
from app.brokers.base import AbstractBroker, OrderRequest, OrderResult, QuoteResult
from app.config import settings
from app.utils.exceptions import BrokerError
from app.utils.logging import logger

# Alpaca enforces 200 requests/minute. Cap concurrent calls at 10 to stay
# well within that limit even under heavy multi-symbol strategy runners.
_ALPACA_CONCURRENCY = 10

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import (
        MarketOrderRequest, LimitOrderRequest, StopOrderRequest,
        GetOrdersRequest,
    )
    from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
    from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
    from alpaca.data.requests import (
        StockBarsRequest, StockLatestQuoteRequest,
        CryptoBarsRequest, CryptoLatestQuoteRequest,
    )
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    logger.warning("alpaca-py not installed — Alpaca broker unavailable")


TF_MAP = {
    "1m":  TimeFrame(1,  TimeFrameUnit.Minute),
    "5m":  TimeFrame(5,  TimeFrameUnit.Minute),
    "15m": TimeFrame(15, TimeFrameUnit.Minute),
    "1h":  TimeFrame(1,  TimeFrameUnit.Hour),
    "4h":  TimeFrame(4,  TimeFrameUnit.Hour),
    "1d":  TimeFrame(1,  TimeFrameUnit.Day),
}

# Alpaca uses "BTC/USD" format for crypto
CRYPTO_SUFFIXES = ("/USD", "/USDT", "/BTC", "/ETH")


def _is_crypto(symbol: str) -> bool:
    return "/" in symbol or any(symbol.endswith(s) for s in ("BTC", "ETH", "SOL", "DOGE"))


def create_alpaca_broker(paper: bool = True) -> "AlpacaBroker | None":
    """Factory that returns an AlpacaBroker when keys are present, or None.

    In paper/dev mode without API keys the process must not crash — the strategy
    runner simply runs in signal-only mode (no orders submitted) when broker is None.
    """
    from app.config import settings

    api_key = settings.alpaca_api_key
    secret_key = settings.alpaca_secret_key

    if not api_key or not secret_key:
        logger.warning(
            "ALPACA_API_KEY / ALPACA_SECRET_KEY not set — Alpaca broker disabled. "
            "Strategies will run in signal-only mode (no orders submitted)."
        )
        return None

    if not ALPACA_AVAILABLE:
        logger.warning("alpaca-py not installed — Alpaca broker unavailable")
        return None

    try:
        return AlpacaBroker(api_key=api_key, secret_key=secret_key, paper=paper)
    except Exception as exc:
        logger.warning("Failed to initialise AlpacaBroker", error=str(exc))
        return None


class AlpacaBroker(AbstractBroker):
    """Unified Alpaca broker for both equities and crypto spot."""

    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        if not ALPACA_AVAILABLE:
            raise ImportError("alpaca-py required: pip install alpaca-py")
        if not api_key or not secret_key:
            raise ValueError("Alpaca API key and secret key are required")
        self.paper = paper
        self.trading     = TradingClient(api_key, secret_key, paper=paper)
        self.stock_data  = StockHistoricalDataClient(api_key, secret_key)
        self.crypto_data = CryptoHistoricalDataClient(api_key, secret_key)
        # Rate limiter: max _ALPACA_CONCURRENCY simultaneous API calls
        self._limiter = asyncio.Semaphore(_ALPACA_CONCURRENCY)

    async def _call(self, fn, *args, **kwargs):
        """Throttled wrapper around blocking SDK calls."""
        async with self._limiter:
            return await asyncio.to_thread(fn, *args, **kwargs)

    # ── Orders ────────────────────────────────────────────────────────────────

    async def place_order(self, request: OrderRequest) -> OrderResult:
        try:
            side = OrderSide.BUY if request.side.lower() == "buy" else OrderSide.SELL
            tif  = TimeInForce.GTC

            # Crypto requires IOC or GTC (no DAY orders on 24/7 markets)
            if _is_crypto(request.symbol):
                tif = TimeInForce.GTC

            if request.order_type in ("market", "moc"):
                req = MarketOrderRequest(
                    symbol=request.symbol, qty=request.quantity,
                    side=side, time_in_force=tif,
                )
            elif request.order_type == "limit" and request.limit_price:
                req = LimitOrderRequest(
                    symbol=request.symbol, qty=request.quantity,
                    side=side, time_in_force=tif,
                    limit_price=request.limit_price,
                )
            elif request.order_type == "stop" and request.stop_price:
                req = StopOrderRequest(
                    symbol=request.symbol, qty=request.quantity,
                    side=side, time_in_force=tif,
                    stop_price=request.stop_price,
                )
            else:
                req = MarketOrderRequest(
                    symbol=request.symbol, qty=request.quantity,
                    side=side, time_in_force=tif,
                )

            order = await self._call(self.trading.submit_order, order_data=req)
            return OrderResult(
                broker_order_id=str(order.id),
                status=str(order.status),
                filled_qty=float(order.filled_qty or 0),
                avg_fill_price=(float(order.filled_avg_price)
                                if order.filled_avg_price else None),
                raw_payload={"id": str(order.id), "symbol": request.symbol},
            )
        except Exception as e:
            logger.error("Alpaca order failed", symbol=request.symbol, error=str(e))
            raise BrokerError(f"Alpaca: {e}")

    async def cancel_order(self, broker_order_id: str) -> bool:
        try:
            await self._call(self.trading.cancel_order_by_id,
                             order_id=broker_order_id)
            return True
        except Exception:
            return False

    async def get_order(self, broker_order_id: str) -> dict:
        order = await self._call(self.trading.get_order_by_id, broker_order_id)
        return {
            "id": str(order.id),
            "status": str(order.status),
            "filled_qty": float(order.filled_qty or 0),
        }

    # ── Account / positions ───────────────────────────────────────────────────

    async def get_positions(self) -> list[dict]:
        positions = await self._call(self.trading.get_all_positions)
        return [
            {
                "symbol": p.symbol,
                "qty": float(p.qty),
                "avg_cost": float(p.avg_entry_price),
                "unrealized_pnl": float(p.unrealized_pl),
                "market_value": float(p.market_value),
                "side": "long" if float(p.qty) > 0 else "short",
            }
            for p in positions
        ]

    async def get_account(self) -> dict:
        acct = await self._call(self.trading.get_account)
        return {
            "equity": float(acct.equity),
            "cash": float(acct.cash),
            "buying_power": float(acct.buying_power),
            "portfolio_value": float(acct.portfolio_value),
            "status": str(acct.status) if hasattr(acct, "status") else "ACTIVE",
        }

    # ── Market data — auto-routes equity vs crypto ────────────────────────────

    async def get_quote(self, symbol: str) -> QuoteResult:
        try:
            if _is_crypto(symbol):
                req = CryptoLatestQuoteRequest(symbol_or_symbols=symbol)
                quotes = await self._call(self.crypto_data.get_crypto_latest_quote, req)
                q = quotes[symbol]
            else:
                req = StockLatestQuoteRequest(symbol_or_symbols=symbol)
                quotes = await self._call(self.stock_data.get_stock_latest_quote, req)
                q = quotes[symbol]
            return QuoteResult(
                symbol=symbol,
                bid=float(q.bid_price),
                ask=float(q.ask_price),
                last=float(q.ask_price),
                volume=None,
            )
        except Exception as e:
            raise BrokerError(f"Alpaca quote failed for {symbol}: {e}")

    async def get_historical(
        self,
        symbol: str,
        interval: str = "1d",
        limit: int = 500,
    ) -> list[dict]:
        tf = TF_MAP.get(interval, TimeFrame(1, TimeFrameUnit.Day))
        try:
            if _is_crypto(symbol):
                req = CryptoBarsRequest(symbol_or_symbols=symbol, timeframe=tf, limit=limit)
                bars_resp = await self._call(self.crypto_data.get_crypto_bars, req)
            else:
                req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=tf, limit=limit)
                bars_resp = await self._call(self.stock_data.get_stock_bars, req)

            return [
                {
                    "ts":     bar.timestamp.isoformat(),
                    "open":   float(bar.open),
                    "high":   float(bar.high),
                    "low":    float(bar.low),
                    "close":  float(bar.close),
                    "volume": float(bar.volume),
                }
                for bar in bars_resp[symbol]
            ]
        except Exception as e:
            logger.warning("Alpaca get_historical failed", symbol=symbol, error=str(e))
            return []


async def validate_alpaca_connection(broker: "AlpacaBroker") -> bool:
    """Returns True if Alpaca API responds with an ACTIVE account."""
    try:
        account = await broker.get_account()
        if account and account.get("status", "").upper() in ("ACTIVE",):
            logger.info("Alpaca connection OK", status=account.get("status"))
            return True
    except Exception as e:
        logger.warning("Alpaca connection check failed", error=str(e))
    return False
