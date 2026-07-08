"""
Polymarket CLOB broker integration via py-clob-client.
Supports YES/NO binary market trading and arbitrage scanning.
"""
from app.brokers.base import AbstractBroker, OrderRequest, OrderResult, QuoteResult
from app.utils.exceptions import BrokerError
from app.utils.logging import logger
from app.config import settings

try:
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import OrderArgs, OrderType
    POLY_AVAILABLE = True
except ImportError:
    POLY_AVAILABLE = False

import asyncio


class PolymarketBroker(AbstractBroker):
    def __init__(self, private_key: str, chain_id: int = 137):
        if not POLY_AVAILABLE:
            raise ImportError("py-clob-client required")
        self.client = ClobClient(
            host="https://clob.polymarket.com",
            key=private_key,
            chain_id=chain_id,
        )

    async def get_markets(self, min_open_interest: float = 10000) -> list[dict]:
        """Auto-discover active markets with sufficient liquidity."""
        try:
            markets = await asyncio.to_thread(self.client.get_markets)
            return [
                m for m in markets
                if float(m.get("openInterest", 0)) >= min_open_interest
            ]
        except Exception as exc:
            logger.error(
                "Polymarket market fetch failed",
                error=str(exc),
                exc_info=True,
                extra={"min_open_interest": min_open_interest},
            )
            return []

    async def get_order_book(self, token_id: str) -> dict:
        try:
            return await asyncio.to_thread(self.client.get_order_book, token_id)
        except Exception as exc:
            logger.error(
                "Polymarket get_order_book failed",
                token_id=token_id,
                error=str(exc),
                exc_info=True,
            )
            raise BrokerError(f"Failed to retrieve order book for {token_id}") from exc

    async def place_order(self, request: OrderRequest) -> OrderResult:
        try:
            args = OrderArgs(
                token_id=request.symbol,
                price=request.limit_price or 0.5,
                size=request.quantity,
                side=request.side.upper(),
            )
            order = await asyncio.to_thread(self.client.create_and_post_order, args)
            return OrderResult(
                broker_order_id=str(order.get("orderID", "")),
                status=order.get("status", "pending"),
                raw_payload=order,
            )
        except (ValueError, TypeError) as exc:
            logger.error(
                "Polymarket place_order validation error",
                request=request,
                error=str(exc),
                exc_info=True,
            )
            raise BrokerError(f"Invalid order parameters: {exc}") from exc
        except Exception as exc:
            logger.error(
                "Polymarket place_order failed",
                request=request,
                error=str(exc),
                exc_info=True,
            )
            raise BrokerError(f"Polymarket: {exc}") from exc

    async def cancel_order(self, broker_order_id: str) -> bool:
        try:
            await asyncio.to_thread(self.client.cancel, broker_order_id)
            return True
        except Exception as exc:
            logger.warning(
                "Polymarket cancel_order failed",
                order_id=broker_order_id,
                error=str(exc),
                exc_info=True,
            )
            return False

    async def get_order(self, broker_order_id: str) -> dict:
        try:
            return await asyncio.to_thread(self.client.get_order, broker_order_id)
        except Exception as exc:
            logger.error(
                "Polymarket get_order failed",
                order_id=broker_order_id,
                error=str(exc),
                exc_info=True,
            )
            raise BrokerError(f"Failed to retrieve order {broker_order_id}") from exc

    async def get_positions(self) -> list[dict]:
        try:
            # Placeholder for future implementation
            return []
        except Exception as exc:
            logger.error(
                "Polymarket get_positions failed",
                error=str(exc),
                exc_info=True,
            )
            raise BrokerError("Failed to retrieve positions") from exc

    async def get_account(self) -> dict:
        try:
            # Placeholder for future implementation
            return {}
        except Exception as exc:
            logger.error(
                "Polymarket get_account failed",
                error=str(exc),
                exc_info=True,
            )
            raise BrokerError("Failed to retrieve account information") from exc

    async def get_quote(self, symbol: str) -> QuoteResult:
        try:
            ob = await self.get_order_book(symbol)
            bids = ob.get("bids", [])
            asks = ob.get("asks", [])
            best_bid = float(bids[0]["price"]) if bids else 0.0
            best_ask = float(asks[0]["price"]) if asks else 1.0
            return QuoteResult(
                symbol=symbol,
                bid=best_bid,
                ask=best_ask,
                last=(best_bid + best_ask) / 2,
            )
        except BrokerError:
            raise
        except Exception as exc:
            logger.error(
                "Polymarket get_quote failed",
                symbol=symbol,
                error=str(exc),
                exc_info=True,
            )
            raise BrokerError(f"Failed to compute quote for {symbol}") from exc

    async def get_historical(self, symbol: str, interval: str = "1d", limit: int = 500) -> list[dict]:
        try:
            # Polymarket doesn't have traditional OHLCV; return empty list
            return []
        except Exception as exc:
            logger.error(
                "Polymarket get_historical failed",
                symbol=symbol,
                interval=interval,
                limit=limit,
                error=str(exc),
                exc_info=True,
            )
            raise BrokerError("Failed to retrieve historical data") from exc