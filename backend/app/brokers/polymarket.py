"""
Polymarket CLOB broker integration via py-clob-client.
Supports YES/NO binary market trading and arbitrage scanning.
Provides tighter entry validation and basic exit confirmation.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List

from app.brokers.base import AbstractBroker, OrderRequest, OrderResult, QuoteResult
from app.utils.exceptions import BrokerError
from app.utils.logging import logger
from app.config import settings

try:
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import OrderArgs, OrderType
    POLY_AVAILABLE = True
except ImportError:  # pragma: no cover
    POLY_AVAILABLE = False


class PolymarketBroker(AbstractBroker):
    """Broker implementation for Polymarket CLOB."""

    def __init__(self, private_key: str, chain_id: int = 137):
        if not POLY_AVAILABLE:
            raise ImportError("py-clob-client required")
        self.client = ClobClient(
            host="https://clob.polymarket.com",
            key=private_key,
            chain_id=chain_id,
        )
        # Configurable thresholds for entry validation
        self.max_spread: float = getattr(settings, "POLY_MAX_SPREAD", 0.02)  # 2% default
        self.price_slippage_tolerance: float = getattr(
            settings, "POLY_SLIPPAGE_TOLERANCE", 0.01
        )  # 1% default

    async def get_markets(self, min_open_interest: float = 10000) -> List[Dict[str, Any]]:
        """Auto‑discover active markets with sufficient liquidity."""
        try:
            markets = await asyncio.to_thread(self.client.get_markets)
            return [
                m
                for m in markets
                if float(m.get("openInterest", 0)) >= min_open_interest
            ]
        except Exception as e:  # pragma: no cover
            logger.error("Polymarket market fetch failed", error=str(e))
            return []

    async def get_order_book(self, token_id: str) -> Dict[str, Any]:
        """Retrieve the current order book for a given market token."""
        return await asyncio.to_thread(self.client.get_order_book, token_id)

    async def _validate_entry(self, symbol: str, limit_price: float | None) -> bool:
        """
        Confirm that the market conditions satisfy tighter entry criteria.

        Checks:
        * Spread between best bid and ask is below ``self.max_spread``.
        * Provided limit price (if any) is within ``self.price_slippage_tolerance`` of the mid price.
        """
        ob = await self.get_order_book(symbol)
        bids = ob.get("bids", [])
        asks = ob.get("asks", [])
        if not bids or not asks:
            logger.debug("Order book empty for symbol", symbol=symbol)
            return False

        best_bid = float(bids[0].get("price", 0))
        best_ask = float(asks[0].get("price", 1))
        spread = best_ask - best_bid
        if spread < 0:
            logger.warning("Negative spread detected", symbol=symbol, spread=spread)
            return False
        if spread > self.max_spread:
            logger.debug(
                "Spread exceeds threshold",
                symbol=symbol,
                spread=spread,
                max_spread=self.max_spread,
            )
            return False

        mid_price = (best_bid + best_ask) / 2
        if limit_price is not None:
            deviation = abs(limit_price - mid_price) / mid_price
            if deviation > self.price_slippage_tolerance:
                logger.debug(
                    "Limit price deviates too much from mid price",
                    symbol=symbol,
                    limit_price=limit_price,
                    mid_price=mid_price,
                    deviation=deviation,
                    tolerance=self.price_slippage_tolerance,
                )
                return False
        return True

    async def _ensure_entry_valid(self, request: OrderRequest) -> None:
        """Raise ``BrokerError`` if entry validation fails."""
        is_valid = await self._validate_entry(
            symbol=request.symbol, limit_price=request.limit_price
        )
        if not is_valid:
            raise BrokerError(
                f"Polymarket: Entry validation failed for {request.symbol}"
            )

    def _build_order_args(self, request: OrderRequest) -> OrderArgs:
        """Construct ``OrderArgs`` from an ``OrderRequest``."""
        price = request.limit_price if request.limit_price is not None else 0.5
        return OrderArgs(
            token_id=request.symbol,
            price=price,
            size=request.quantity,
            side=request.side.upper(),
        )

    async def _submit_order(self, args: OrderArgs) -> Dict[str, Any]:
        """Submit the order to the Polymarket client."""
        return await asyncio.to_thread(self.client.create_and_post_order, args)

    async def _finalize_order_status(self, order: Dict[str, Any]) -> str:
        """
        Ensure the order status is final.

        If the initial status is not final, poll once after a short delay.
        """
        status = order.get("status", "pending")
        if status in {"filled", "canceled", "rejected"}:
            return status

        # Small pause before a second status check
        time.sleep(0.2)
        refreshed = await self.get_order(order.get("orderID", ""))
        return refreshed.get("status", status)

    async def place_order(self, request: OrderRequest) -> OrderResult:
        """
        Place an order after confirming entry conditions.

        Raises:
            BrokerError: If the order cannot be placed or validation fails.
        """
        try:
            await self._ensure_entry_valid(request)

            args = self._build_order_args(request)
            order = await self._submit_order(args)

            final_status = await self._finalize_order_status(order)

            return OrderResult(
                broker_order_id=str(order.get("orderID", "")),
                status=final_status,
                raw_payload=order,
            )
        except BrokerError:
            raise
        except Exception as e:  # pragma: no cover
            raise BrokerError(f"Polymarket: {e}")

    async def cancel_order(self, broker_order_id: str) -> bool:
        """Cancel an existing order."""
        try:
            await asyncio.to_thread(self.client.cancel, broker_order_id)
            return True
        except Exception as e:  # pragma: no cover
            logger.warning(
                "Polymarket cancel_order failed",
                order_id=broker_order_id,
                error=str(e),
            )
            return False

    async def get_order(self, broker_order_id: str) -> Dict[str, Any]:
        """Retrieve a specific order's details."""
        return await asyncio.to_thread(self.client.get_order, broker_order_id)

    async def get_positions(self) -> List[Dict[str, Any]]:
        """Polymarket does not expose positions via the CLOB client."""
        return []

    async def get_account(self) -> Dict[str, Any]:
        """Polymarket does not expose account balances via the CLOB client."""
        return {}

    async def get_quote(self, symbol: str) -> QuoteResult:
        """Return the best bid, ask, and mid price for a market."""
        ob = await self.get_order_book(symbol)
        bids = ob.get("bids", [])
        asks = ob.get("asks", [])
        best_bid = float(bids[0].get("price", 0)) if bids else 0.0
        best_ask = float(asks[0].get("price", 1)) if asks else 1.0
        mid = (best_bid + best_ask) / 2
        return QuoteResult(symbol=symbol, bid=best_bid, ask=best_ask, last=mid)

    async def get_historical(
        self, symbol: str, interval: str = "1d", limit: int = 500
    ) -> List[Dict[str, Any]]:
        """Polymarket doesn't have traditional OHLCV data."""
        return []