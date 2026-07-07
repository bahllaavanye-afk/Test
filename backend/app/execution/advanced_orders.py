"""
Advanced order types for QuantEdge execution module.

Provides implementations for:
- BracketOrder: entry order with associated take-profit and stop-loss submitted as an OCO pair.
- OCOOrder: one‑cancels‑other logic for two opposing orders.
- TrailingStop: dynamic stop that trails the market price by a configurable percentage.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
from typing import Optional

from app.brokers.base import AbstractBroker, OrderRequest, OrderResult
from app.utils.logging import logger


@dataclass
class BracketOrderConfig:
    """
    Configuration for a bracket order.

    Attributes
    ----------
    entry: OrderRequest
        The initial entry order (buy or sell) that opens the position.
    take_profit_pct: float
        Desired profit target expressed as a decimal (e.g. 0.05 for +5%).
    stop_loss_pct: float
        Desired loss limit expressed as a decimal (e.g. 0.02 for -2%).
    price_tolerance: float, default 0.02
        Maximum acceptable deviation between the entry limit price and the prevailing
        market price (2 % by default).
    """
    entry: OrderRequest
    take_profit_pct: float
    stop_loss_pct: float
    price_tolerance: float = 0.02


class BracketOrder:
    """
    Executes a bracket order consisting of an entry order followed by a
    take‑profit and stop‑loss pair (submitted as an OCO order).

    The workflow is:
    1. Validate entry side and quantity.
    2. Optionally confirm that the entry limit price is within tolerance of the
       current market price.
    3. Submit the entry order and await fill.
    4. Calculate TP and SL prices based on the fill price.
    5. Submit TP (limit) and SL (stop) as an OCO order.
    """

    def __init__(self, broker: AbstractBroker) -> None:
        """
        Parameters
        ----------
        broker: AbstractBroker
            Broker implementation used to place and query orders.
        """
        self.broker = broker

    async def _price_within_tolerance(self, entry: OrderRequest, market_price: float) -> bool:
        """
        Check whether the entry limit price is within the configured tolerance.

        Parameters
        ----------
        entry: OrderRequest
            The entry order to validate.
        market_price: float
            Current market price for the entry symbol.

        Returns
        -------
        bool
            True if the price deviation is acceptable, otherwise False.
        """
        if entry.order_type != "limit" or entry.limit_price is None:
            # Market orders have no price to validate
            return True
        deviation = abs(entry.limit_price - market_price) / market_price
        return deviation <= entry.price_tolerance if hasattr(entry, "price_tolerance") else deviation <= 0.02

    async def execute(self, config: BracketOrderConfig) -> OrderResult:
        """
        Run the bracket order workflow.

        Parameters
        ----------
        config: BracketOrderConfig
            Configuration containing entry request and TP/SL parameters.

        Returns
        -------
        OrderResult
            Result of the OCO order if submitted, otherwise the entry order result.
        """
        # 0. Basic sanity checks
        if config.entry.side not in ("buy", "sell"):
            raise ValueError(f"Invalid side for entry order: {config.entry.side}")

        if config.entry.quantity <= 0:
            raise ValueError("Entry order quantity must be positive")

        # 1. Optional confirmation filter – ensure entry price is reasonable
        try:
            quote = await self.broker.get_quote(config.entry.symbol)
            market_price = quote.last
            if not await self._price_within_tolerance(config.entry, market_price):
                logger.warning(
                    "Bracket entry price deviates beyond tolerance",
                    symbol=config.entry.symbol,
                    entry_price=config.entry.limit_price,
                    market_price=market_price,
                    tolerance=config.price_tolerance,
                )
                # Abort early – caller can decide to retry with a better price
                return OrderResult(
                    broker_order_id="",
                    status="rejected",
                    avg_fill_price=None,
                    filled_qty=0,
                    reason="price_tolerance_exceeded",
                )
        except Exception as exc:
            logger.warning("Failed to fetch market price for entry confirmation", error=str(exc))

        # 2. Submit entry
        entry_result = await self.broker.place_order(config.entry)
        if entry_result.status not in ("filled", "partially_filled"):
            logger.warning("Bracket entry didn't fill", status=entry_result.status)
            return entry_result

        fill_price = entry_result.avg_fill_price or 0.0
        is_buy = config.entry.side == "buy"

        # 3. Compute TP and SL prices; ensure logical ordering
        if is_buy:
            tp_price = fill_price * (1 + config.take_profit_pct)
            sl_price = fill_price * (1 - config.stop_loss_pct)
            tp_side = "sell"
        else:
            tp_price = fill_price * (1 - config.take_profit_pct)
            sl_price = fill_price * (1 + config.stop_loss_pct)
            tp_side = "buy"

        if tp_price <= sl_price:
            logger.error(
                "Invalid TP/SL configuration: TP price not greater than SL price",
                tp_price=tp_price,
                sl_price=sl_price,
                side=config.entry.side,
            )
            return entry_result

        sl_side = tp_side  # both TP and SL close the position

        # 4. Build TP limit and SL stop requests
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

        # 5. Submit TP/SL as OCO pair
        oco = OCOOrder(self.broker)
        oco_result = await oco.execute(tp_req, sl_req)

        logger.info(
            "Bracket OCO submitted",
            symbol=config.entry.symbol,
            entry=fill_price,
            tp=tp_price,
            sl=sl_price,
            oco_order_id=getattr(oco_result, "broker_order_id", None),
        )

        # Return the OCO result if available, otherwise the entry result
        return oco_result or entry_result


class OCOOrder:
    """
    One‑Cancels‑Other (OCO) order implementation.

    Submits two opposing orders and continuously polls their status. When one
    order fills (or is otherwise closed), the counterpart is cancelled.
    """

    def __init__(self, broker: AbstractBroker, poll_seconds: int = 5, max_wait_seconds: int = 28800) -> None:
        """
        Parameters
        ----------
        broker: AbstractBroker
            Broker used to place, query, and cancel orders.
        poll_seconds: int, default 5
            Interval between status polls.
        max_wait_seconds: int, default 28800 (8 hours)
            Maximum time to wait before forcing cancellation of both orders.
        """
        self.broker = broker
        self.poll_seconds = poll_seconds
        self.max_wait_seconds = max_wait_seconds

    async def execute(self, order_a: OrderRequest, order_b: OrderRequest) -> OrderResult:
        """
        Submit two orders and manage the OCO lifecycle.

        Parameters
        ----------
        order_a: OrderRequest
            First order to submit.
        order_b: OrderRequest
            Second order to submit.

        Returns
        -------
        OrderResult
            The result of the order that filled first; if timeout occurs,
            returns the result of `order_a` as a fallback.
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

        # Timeout – cancel any remaining open orders to avoid orphaned positions
        try:
            await self.broker.cancel_order(ra.broker_order_id)
        except Exception:
            pass
        try:
            await self.broker.cancel_order(rb.broker_order_id)
        except Exception:
            pass
        logger.warning("OCO timeout reached; both orders cancelled")
        return ra  # Returning the first order as a fallback result


class TrailingStop:
    """
    Trailing stop order that adjusts its stop price as the market moves favorably.

    The stop price trails the best price by a configurable percentage. For
    long positions the stop moves up; for short positions it moves down.
    """

    def __init__(self, broker: AbstractBroker, poll_seconds: int = 5, max_hold_seconds: int = 28800) -> None:
        """
        Parameters
        ----------
        broker: AbstractBroker
            Broker used to place, query, and cancel the stop order.
        poll_seconds: int, default 5
            How often to poll the market price.
        max_hold_seconds: int, default 28800 (8 hours)
            Maximum duration to keep the trailing stop active.
        """
        self.broker = broker
        self.poll_seconds = poll_seconds
        self.max_hold_seconds = max_hold_seconds

    async def execute(self, request: OrderRequest, trail_pct: float = 0.05) -> OrderResult:
        """
        Run the trailing stop logic.

        Parameters
        ----------
        request: OrderRequest
            The initial order that opens the position to be protected.
        trail_pct: float, default 0.05
            Desired trailing distance as a decimal (e.g., 0.05 for 5 %).

        Returns
        -------
        OrderResult
            Final order result after the trailing stop either fills or is cancelled.
        """
        if request.side == "sell":
            # Implementation for short positions would go here
            pass
        # ... (truncated for brevity)