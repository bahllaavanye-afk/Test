"""
Alpaca broker integration — primary equity broker.
Commission-free, excellent Python SDK, TradingView integration.
Paper trading: https://paper-api.alpaca.markets
Live trading:  https://api.alpaca.markets
"""
import asyncio
from datetime import datetime, timezone
from app.brokers.base import AbstractBroker, OrderRequest, OrderResult, QuoteResult
from app.config import settings
from app.utils.exceptions import BrokerError
from app.utils.logging import logger

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, StopOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    logger.warning("alpaca-py not installed — Alpaca broker unavailable")


TF_MAP = {
    "1m": TimeFrame(1, TimeFrameUnit.Minute),
    "5m": TimeFrame(5, TimeFrameUnit.Minute),
    "15m": TimeFrame(15, TimeFrameUnit.Minute),
    "1h": TimeFrame(1, TimeFrameUnit.Hour),
    "4h": TimeFrame(4, TimeFrameUnit.Hour),
    "1d": TimeFrame(1, TimeFrameUnit.Day),
}


class AlpacaBroker(AbstractBroker):
    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        if not ALPACA_AVAILABLE:
            raise ImportError("alpaca-py required")
        self.paper = paper
        base_url = "https://paper-api.alpaca.markets" if paper else "https://api.alpaca.markets"
        self.trading = TradingClient(api_key, secret_key, paper=paper, url_override=base_url)
        self.data = StockHistoricalDataClient(api_key, secret_key)

    async def place_order(self, request: OrderRequest) -> OrderResult:
        try:
            side = OrderSide.BUY if request.side.lower() == "buy" else OrderSide.SELL
            tif = TimeInForce.GTC

            if request.order_type == "market":
                req = MarketOrderRequest(symbol=request.symbol, qty=request.quantity, side=side, time_in_force=tif)
            elif request.order_type == "limit" and request.limit_price:
                req = LimitOrderRequest(symbol=request.symbol, qty=request.quantity, side=side,
                                        time_in_force=tif, limit_price=request.limit_price)
            elif request.order_type == "stop" and request.stop_price:
                req = StopOrderRequest(symbol=request.symbol, qty=request.quantity, side=side,
                                       time_in_force=tif, stop_price=request.stop_price)
            else:
                req = MarketOrderRequest(symbol=request.symbol, qty=request.quantity, side=side, time_in_force=tif)

            order = await asyncio.to_thread(self.trading.submit_order, order_data=req)
            return OrderResult(
                broker_order_id=str(order.id),
                status=str(order.status),
                filled_qty=float(order.filled_qty or 0),
                avg_fill_price=float(order.filled_avg_price) if order.filled_avg_price else None,
                raw_payload={"id": str(order.id), "symbol": request.symbol},
            )
        except Exception as e:
            logger.error("Alpaca order failed", symbol=request.symbol, error=str(e))
            raise BrokerError(f"Alpaca: {e}")

    async def cancel_order(self, broker_order_id: str) -> bool:
        try:
            await asyncio.to_thread(self.trading.cancel_order_by_id, order_id=broker_order_id)
            return True
        except Exception:
            return False

    async def get_order(self, broker_order_id: str) -> dict:
        order = await asyncio.to_thread(self.trading.get_order_by_id, broker_order_id)
        return {"id": str(order.id), "status": str(order.status), "filled_qty": float(order.filled_qty or 0)}

    async def get_positions(self) -> list[dict]:
        positions = await asyncio.to_thread(self.trading.get_all_positions)
        return [{"symbol": p.symbol, "qty": float(p.qty), "avg_cost": float(p.avg_entry_price),
                 "unrealized_pnl": float(p.unrealized_pl), "side": "long" if float(p.qty) > 0 else "short"}
                for p in positions]

    async def get_account(self) -> dict:
        acct = await asyncio.to_thread(self.trading.get_account)
        return {"equity": float(acct.equity), "cash": float(acct.cash),
                "buying_power": float(acct.buying_power), "portfolio_value": float(acct.portfolio_value)}

    async def get_quote(self, symbol: str) -> QuoteResult:
        req = StockLatestQuoteRequest(symbol_or_symbols=symbol)
        quotes = await asyncio.to_thread(self.data.get_stock_latest_quote, req)
        q = quotes[symbol]
        return QuoteResult(symbol=symbol, bid=float(q.bid_price), ask=float(q.ask_price),
                           last=float(q.ask_price), volume=None)

    async def get_historical(self, symbol: str, interval: str = "1d", limit: int = 500) -> list[dict]:
        from alpaca.data.requests import StockBarsRequest
        from datetime import timedelta
        tf = TF_MAP.get(interval, TimeFrame(1, TimeFrameUnit.Day))
        req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=tf, limit=limit)
        bars = await asyncio.to_thread(self.data.get_stock_bars, req)
        result = []
        for bar in bars[symbol]:
            result.append({"ts": bar.timestamp.isoformat(), "open": float(bar.open),
                           "high": float(bar.high), "low": float(bar.low),
                           "close": float(bar.close), "volume": float(bar.volume)})
        return result
