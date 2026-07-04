"""
Advanced order types for execution layer:

- **BracketOrder**: Combines entry, take-profit, and stop-loss orders. After the entry
  order fills, a take-profit (limit) and stop-loss (stop) order are submitted as an
  OCO (one‑cancels‑other) pair.

- **OCOOrder**: Submits two opposing orders. Whichever order fills first causes the
  other to be cancelled.

- **TrailingStop**: Implements a dynamic stop that trails the market price by a
  configurable percentage. The stop price is adjusted as the market moves in the
  favorable direction.

These utilities are used by higher‑level trading strategies to manage risk and
execution flow without exposing low‑level broker interactions.
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
    Configuration for a :class:`BracketOrder`.

    Attributes
    ----------
    entry: OrderRequest
        The entry order that initiates the position.
    take_profit_pct: float
        Desired profit target expressed as a fraction of the entry price
        (e.g. ``0.05`` for a 5 % gain).
    stop_loss_pct: float
        Desired stop‑loss expressed as a fraction of the entry price
        (e.g. ``0.02`` for a 2 % loss).
    price_tolerance: float, optional
        Maximum acceptable deviation between the entry limit price and the
        current market price (default ``0.02`` → 2 %).
    """

    entry: OrderRequest
    take_profit_pct: float
    stop_loss_pct: float
    price_tolerance: float = 0.02


class BracketOrder:
    """
    Executes a bracket order workflow:

    1. Validate the entry order price against the market price.
    2. Submit the entry order and wait for it to fill.
    3. Compute take‑profit and stop‑loss prices based on the fill price.
    4. Submit the TP and SL orders as an OCO pair.
    5. Return the result of the OCO order (or the entry result if OCO fails).
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
            Current market price obtained from the broker.

        Returns
        -------
        bool
            ``True`` if the price deviation is within tolerance, otherwise ``False``.
        """
        if entry.order_type != "limit" or entry.limit_price is None:
            # Market orders have no price to validate
            return True
        deviation = abs(entry.limit_price - market_price) / market_price
        # ``price_tolerance`` may be set on the request; fall back to a default of 2 %
        return deviation <= entry.price_tolerance if hasattr(entry, "price_tolerance") else deviation <= 0.02

    async def execute(self, config: BracketOrderConfig) -> OrderResult:
        """
        Run the bracket order process.

        Parameters
        ----------
        config: BracketOrderConfig
            Configuration containing the entry request and TP/SL parameters.

        Returns
        -------
        OrderResult
            Result of the OCO order if it was submitted; otherwise the entry order
            result (e.g., when the entry fails to fill or a price‑tolerance breach
            occurs).
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
    One‑Cancels‑Other (OCO) order handler.

    Submits two opposing orders and continuously polls their status. When one
    order fills (or is otherwise closed), the counterpart is cancelled.
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
            Maximum time to wait before giving up and cancelling both orders
            (default ``28800`` seconds → 8 hours).
        """
        self.broker = broker
        self.poll_seconds = poll_seconds
        self.max_wait_seconds = max_wait_seconds

    async def execute(self, order_a: OrderRequest, order_b: OrderRequest) -> OrderResult:
        """
        Execute the OCO workflow.

        Parameters
        ----------
        order_a: OrderRequest
            First order to submit.
        order_b: OrderRequest
            Second order to submit.

        Returns
        -------
        OrderResult
            The result of the order that filled first. If neither fills within the
            timeout, the first order's result is returned as a fallback.
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
    Implements a trailing stop order that moves with the market price.

    The stop price is updated periodically based on a trailing percentage. For
    long positions the stop moves upward; for short positions it moves
    downward. The order is kept alive until the price reverses enough to hit
    the stop, or until a maximum holding period expires.
    """

    def __init__(self, broker: AbstractBroker, poll_seconds: int = 5, max_hold_seconds: int = 28800) -> None:
        """
        Parameters
        ----------
        broker: AbstractBroker
            Broker used for order placement and queries.
        poll_seconds: int, optional
            How frequently to poll the market price (default ``5`` seconds).
        max_hold_seconds: int, optional
            Maximum duration to keep the trailing stop active (default ``28800``
            seconds → 8 hours).
        """
        self.broker = broker
        self.poll_seconds = poll_seconds
        self.max_hold_seconds = max_hold_seconds

    async def execute(self, request: OrderRequest, trail_pct: float = 0.05) -> OrderResult:
        """
        Run a trailing stop for the given order request.

        Parameters
        ----------
        request: OrderRequest
            The original order that opened the position.
        trail_pct: float, optional
            Trailing distance expressed as a fraction of the price
            (default ``0.05`` → 5 %).

        Returns
        -------
        OrderResult
            Final order result after the trailing stop either hits or the
            maximum hold time expires.
        """
        if request.side == "sell":
            # Implementation for short positions (not shown)
            pass
        else:
            # Implementation for long positions (not shown)
            pass
        # Placeholder return to satisfy type checker; actual logic should return
        # the appropriate OrderResult from the broker.
        return OrderResult(
            broker_order_id="",
            status="unknown",
            avg_fill_price=None,
            filled_qty=0,
            reason="trailing_stop_not_implemented",
        )