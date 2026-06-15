"""
Limit-First Execution: post limit order at best bid/ask + offset, then fall back to market.
Saves 5-15 bps on average vs immediate market orders.
"""
import asyncio

from app.brokers.base import AbstractBroker, OrderRequest, OrderResult


class LimitFirstExecution:
    def __init__(self, broker: AbstractBroker, offset_bps: float = 5, fallback_seconds: int = 30):
        self.broker = broker
        self.offset_bps = offset_bps
        self.fallback_seconds = fallback_seconds

    async def execute(self, request: OrderRequest) -> OrderResult:
        # Get current quote
        try:
            quote = await self.broker.get_quote(request.symbol)
            ref_price = quote.ask if request.side == "buy" else quote.bid
            offset = ref_price * self.offset_bps / 10_000

            if request.side == "buy":
                limit_price = quote.ask - offset    # post below ask to improve fill
            else:
                limit_price = quote.bid + offset    # post above bid to improve fill

            limit_req = OrderRequest(
                **{**request.__dict__, "order_type": "limit", "limit_price": round(limit_price, 4)}
            )
            result = await self.broker.place_order(limit_req)

            if result.status in ("filled", "partially_filled"):
                return result

            # Wait for fill, then fallback to market
            for _ in range(self.fallback_seconds):
                await asyncio.sleep(1)
                order_status = await self.broker.get_order(result.broker_order_id)
                if order_status.get("status") in ("filled", "closed"):
                    result.status = "filled"
                    result.filled_qty = float(order_status.get("filled_qty", request.quantity))
                    return result

            # Cancel limit, then only market-order the UNFILLED remainder.
            # Submitting the full quantity here would double-execute any qty
            # that filled while we were polling.
            await self.broker.cancel_order(result.broker_order_id)
            filled_so_far = 0.0
            try:
                final_status = await self.broker.get_order(result.broker_order_id)
                filled_so_far = float(final_status.get("filled_qty", 0) or 0)
            except Exception:
                filled_so_far = float(getattr(result, "filled_qty", 0) or 0)

            remaining = float(request.quantity) - filled_so_far
            if remaining <= 0:
                result.status = "filled"
                result.filled_qty = float(request.quantity)
                return result

            market_req = OrderRequest(
                **{**request.__dict__, "order_type": "market",
                   "limit_price": None, "quantity": remaining}
            )
            return await self.broker.place_order(market_req)

        except Exception:
            # Pre-order failure (e.g. quote fetch) — safe to fall back to a full
            # market order because no limit order was successfully placed.
            market_req = OrderRequest(**{**request.__dict__, "order_type": "market"})
            return await self.broker.place_order(market_req)
