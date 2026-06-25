"""
Advanced order types for QuantEdge execution engine.

Provides:
- BracketOrder: entry order followed by a take‑profit and stop‑loss submitted as an OCO pair.
- OCOOrder: one‑cancels‑other logic for two opposing orders.
- TrailingStop: dynamic stop that trails the market price by a percentage.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from app.brokers.base import AbstractBroker, OrderRequest, OrderResult
from app.utils.logging import logger


@dataclass
class BracketOrderConfig:
    """Configuration for a bracket order.

    Attributes
    ----------
    entry: OrderRequest
        The initial entry order to be placed.
    take_profit_pct: float
        Desired take‑profit as a proportion of the entry price (e.g. ``0.05`` for 5 %).
    stop_loss_pct: float
        Desired stop‑loss as a proportion of the entry price (e.g. ``0.02`` for 2 %).
    """
    entry: OrderRequest
    take_profit_pct: float
    stop_loss_pct: float


class BracketOrder:
    """Handles a bracket order workflow.

    The entry order is placed first. Once it is filled (or partially filled) the
    take‑profit and stop‑loss orders are submitted as an OCO pair. Whichever of
    those fills first will cancel the other.
    """

    def __init__(self, broker: AbstractBroker) -> None:
        """
        Parameters
        ----------
        broker: AbstractBroker
            Broker implementation used to place and query orders.
        """
        self.broker = broker

    async def execute(self, config: BracketOrderConfig) -> OrderResult:
        """Execute the bracket order.

        Parameters
        ----------
        config: BracketOrderConfig
            Configuration containing the entry order and TP/SL percentages.

        Returns
        -------
        OrderResult
            Result of the OCO order if it was submitted, otherwise the entry
            order result.
        """
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
        logger.info(
            "Bracket OCO submitted",
            symbol=config.entry.symbol,
            entry=fill_price,
            tp=tp_price,
            sl=sl_price,
            oco_order_id=oco_result.broker_order_id if oco_result else None,
        )

        return oco_result or entry_result


class OCOOrder:
    """One‑Cancels‑Other order handling.

    Submits two opposing orders and continuously polls their status. When one
    order fills or closes, the counterpart is cancelled.
    """

    def __init__(self, broker: AbstractBroker, poll_seconds: int = 5, max_wait_seconds: int = 28800) -> None:
        """
        Parameters
        ----------
        broker: AbstractBroker
            Broker used for order placement and status queries.
        poll_seconds: int, optional
            Interval between status polls (default ``5`` seconds).
        max_wait_seconds: int, optional
            Maximum time to wait before giving up (default ``28800`` seconds).
        """
        self.broker = broker
        self.poll_seconds = poll_seconds
        self.max_wait_seconds = max_wait_seconds

    async def execute(self, order_a: OrderRequest, order_b: OrderRequest) -> OrderResult:
        """Execute an OCO pair.

        Parameters
        ----------
        order_a: OrderRequest
            First order in the pair.
        order_b: OrderRequest
            Second order in the pair.

        Returns
        -------
        OrderResult
            The result of the order that filled first; if neither fills within the
            wait window, returns the result of ``order_a``.
        """
        ra = await self.broker.place_order(order_a)
        rb = await self.broker.place_order(order_b)
        elapsed = 0
        while elapsed < self.max_wait_seconds:
            try:
                sa = await self.broker.get_order(ra.broker_order_id)
                sb = await self.broker.get_order(rb.broker_order_id)
            except Exception as exc:
                logger.warning("OCO poll failed — retrying", error=str(exc))
                await asyncio.sleep(self.poll_seconds)
                elapsed += self.poll_seconds
                continue

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
    """Trailing stop implementation.

    Adjusts the stop price as the market moves favorably. When the price
    retraces to the stop level, a market order is placed to exit the position.
    """

    def __init__(self, broker: AbstractBroker, poll_seconds: int = 5, max_hold_seconds: int = 28800) -> None:
        """
        Parameters
        ----------
        broker: AbstractBroker
            Broker used for quotes and order execution.
        poll_seconds: int, optional
            Frequency of price polling (default ``5`` seconds).
        max_hold_seconds: int, optional
            Maximum duration to keep the trailing stop active (default ``28800`` seconds).
        """
        self.broker = broker
        self.poll_seconds = poll_seconds
        self.max_hold_seconds = max_hold_seconds

    async def execute(self, request: OrderRequest, trail_pct: float = 0.05) -> OrderResult:
        """Run a trailing stop for the given order.

        Parameters
        ----------
        request: OrderRequest
            The original order to be protected by a trailing stop.
        trail_pct: float, optional
            Percentage distance from the high/low water mark to set the stop
            (default ``0.05`` for 5 %).

        Returns
        -------
        OrderResult
            Result of the final market order that exits the position.
        """
        if request.side == "sell":
            # Selling long position with trailing stop
            quote = await self.broker.get_quote(request.symbol)
            high_water = quote.last
            stop_price = high_water * (1 - trail_pct)
            logger.info(
                "Trailing stop starting",
                symbol=request.symbol,
                hw=high_water,
                stop=stop_price,
            )

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
                    market_req = OrderRequest(
                        **{**request.__dict__, "order_type": "market", "limit_price": None}
                    )
                    return await self.broker.place_order(market_req)
        else:
            # Buying short / covering with trailing stop on the way down
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