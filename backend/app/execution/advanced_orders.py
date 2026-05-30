"""
Advanced order types:
  - BracketOrder: entry + take-profit + stop-loss together
  - OCOOrder: one-cancels-other (two opposing orders, fill one → cancel the other)
  - TrailingStop: stop that follows price by N% or $N
"""
from __future__ import annotations
import asyncio
from dataclasses import dataclass
from app.brokers.base import AbstractBroker, OrderRequest, OrderResult
from app.utils.logging import logger


@dataclass
class BracketOrderConfig:
    entry: OrderRequest
    take_profit_pct: float    # e.g. 0.05 = +5% TP
    stop_loss_pct: float      # e.g. 0.02 = -2% SL


class BracketOrder:
    """
    Submit entry, then watch for fill. Once filled, submit take-profit and stop-loss
    as OCO pair. Whichever fills cancels the other.
    """
    def __init__(self, broker: AbstractBroker):
        self.broker = broker

    async def execute(self, config: BracketOrderConfig) -> OrderResult:
        # 1. Submit entry
        entry_result = await self.broker.place_order(config.entry)
        if entry_result.status not in ("filled", "partially_filled"):
            logger.warning("Bracket entry didn't fill", status=entry_result.status)
            return entry_result

        fill_price = entry_result.avg_fill_price or 0
        is_buy = config.entry.side == "buy"

        # 2. Compute TP and SL prices
        if is_buy:
            tp_price = fill_price * (1 + config.take_profit_pct)
            sl_price = fill_price * (1 - config.stop_loss_pct)
            tp_side = "sell"
        else:
            tp_price = fill_price * (1 - config.take_profit_pct)
            sl_price = fill_price * (1 + config.stop_loss_pct)
            tp_side = "buy"

        sl_side = tp_side  # same side: both TP and SL close the position
        # 3. Submit TP limit + SL stop as OCO pair so only one fills
        tp_req = OrderRequest(
            account_id=config.entry.account_id,
            symbol=config.entry.symbol,
            side=tp_side,
            order_type="limit",
            quantity=entry_result.filled_qty,
            limit_price=round(tp_price, 4),
            stop_price=None,
            time_in_force="GTC",
            execution_algo="market",
            risk_bucket=config.entry.risk_bucket,
        )
        sl_req = OrderRequest(
            account_id=config.entry.account_id,
            symbol=config.entry.symbol,
            side=sl_side,
            order_type="stop",
            quantity=entry_result.filled_qty,
            limit_price=None,
            stop_price=round(sl_price, 4),
            time_in_force="GTC",
            execution_algo="market",
            risk_bucket=config.entry.risk_bucket,
        )

        oco = OCOOrder(self.broker)
        oco_result = await oco.execute(tp_req, sl_req)
        logger.info("Bracket OCO submitted",
                    symbol=config.entry.symbol, entry=fill_price,
                    tp=tp_price, sl=sl_price,
                    oco_order_id=oco_result.broker_order_id if oco_result else None)

        return oco_result or entry_result


class OCOOrder:
    """
    One-Cancels-Other: submit two opposing orders. Poll; whichever fills, cancel the other.
    """
    def __init__(self, broker: AbstractBroker, poll_seconds: int = 5, max_wait_seconds: int = 28800):
        self.broker = broker
        self.poll_seconds = poll_seconds
        self.max_wait_seconds = max_wait_seconds

    async def execute(self, order_a: OrderRequest, order_b: OrderRequest) -> OrderResult:
        ra = await self.broker.place_order(order_a)
        rb = await self.broker.place_order(order_b)
        elapsed = 0
        while elapsed < self.max_wait_seconds:
            try:
                sa = await self.broker.get_order(ra.broker_order_id)
                sb = await self.broker.get_order(rb.broker_order_id)
            except Exception:
                break
            if sa.get("status") in ("filled", "closed"):
                await self.broker.cancel_order(rb.broker_order_id)
                logger.info("OCO: order A filled, B cancelled")
                return ra
            if sb.get("status") in ("filled", "closed"):
                await self.broker.cancel_order(ra.broker_order_id)
                logger.info("OCO: order B filled, A cancelled")
                return rb
            await asyncio.sleep(self.poll_seconds)
            elapsed += self.poll_seconds
        return ra


class TrailingStop:
    """
    Trailing stop that follows price by trail_pct. Continually adjusts stop price upward
    (or downward for shorts) as price moves favorably.
    """
    def __init__(self, broker: AbstractBroker, poll_seconds: int = 5, max_hold_seconds: int = 28800):
        self.broker = broker
        self.poll_seconds = poll_seconds
        self.max_hold_seconds = max_hold_seconds

    async def execute(self, request: OrderRequest, trail_pct: float = 0.05) -> OrderResult:
        if request.side == "sell":
            # selling long position with trailing stop
            quote = await self.broker.get_quote(request.symbol)
            high_water = quote.last
            stop_price = high_water * (1 - trail_pct)
            logger.info("Trailing stop starting", symbol=request.symbol, hw=high_water, stop=stop_price)

            start_time = asyncio.get_running_loop().time()
            while True:
                if asyncio.get_running_loop().time() - start_time > self.max_hold_seconds:
                    logger.warning(f"TrailingStop for {request.symbol} timed out")
                    market_req = OrderRequest(**{**request.__dict__, "order_type": "market", "limit_price": None})
                    return await self.broker.place_order(market_req)
                await asyncio.sleep(self.poll_seconds)
                try:
                    quote = await self.broker.get_quote(request.symbol)
                except Exception:
                    continue
                if quote.last > high_water:
                    high_water = quote.last
                    stop_price = high_water * (1 - trail_pct)
                if quote.last <= stop_price:
                    # Fire market sell
                    market_req = OrderRequest(
                        **{**request.__dict__, "order_type": "market", "limit_price": None}
                    )
                    return await self.broker.place_order(market_req)
        else:
            # buying short / cover with trailing stop on the way down
            quote = await self.broker.get_quote(request.symbol)
            low_water = quote.last
            stop_price = low_water * (1 + trail_pct)
            start_time = asyncio.get_running_loop().time()
            while True:
                if asyncio.get_running_loop().time() - start_time > self.max_hold_seconds:
                    logger.warning(f"TrailingStop for {request.symbol} timed out")
                    market_req = OrderRequest(**{**request.__dict__, "order_type": "market", "limit_price": None})
                    return await self.broker.place_order(market_req)
                await asyncio.sleep(self.poll_seconds)
                try:
                    quote = await self.broker.get_quote(request.symbol)
                except Exception:
                    continue
                if quote.last < low_water:
                    low_water = quote.last
                    stop_price = low_water * (1 + trail_pct)
                if quote.last >= stop_price:
                    market_req = OrderRequest(
                        **{**request.__dict__, "order_type": "market", "limit_price": None}
                    )
                    return await self.broker.place_order(market_req)
