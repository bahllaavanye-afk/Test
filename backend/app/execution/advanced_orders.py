"""
Advanced order types:
  - BracketOrder: entry + take-profit + stop-loss together
  - OCOOrder: one-cancels-other (two opposing orders, fill one → cancel the other)
  - TrailingStop: stop that follows price by N% or $N
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional, Tuple

from app.brokers.base import AbstractBroker, OrderRequest, OrderResult
from app.utils.logging import logger


@dataclass
class BracketOrderConfig:
    """Configuration for a BracketOrder.

    Attributes
    ----------
    entry: OrderRequest
        The entry order (must be a limit order for price checks).
    take_profit_pct: float
        Desired profit target as a fraction (e.g. 0.05 = +5% TP).
    stop_loss_pct: float
        Desired stop loss as a fraction (e.g. 0.02 = -2% SL).
    price_tolerance: float, default 0.02
        Maximum allowed deviation between the entry limit price and the
        current market price (2 %). If the deviation exceeds this value the
        order is rejected.
    max_spread_pct: float, default 0.01
        Optional filter – maximum allowed bid‑ask spread as a fraction of the
        mid price. Helps avoid entering on a stale or illiquid quote.
    """

    entry: OrderRequest
    take_profit_pct: float
    stop_loss_pct: float
    price_tolerance: float = 0.02
    max_spread_pct: float = 0.01


class BracketOrder:
    """Execute a bracket order: entry → OCO TP/SL.

    The workflow is:

    1. Validate entry side/quantity.
    2. Fetch a fresh quote and confirm the entry limit price is within the
       configured tolerance and spread limits.
    3. Submit the entry order.
    4. Upon fill, compute TP/SL prices ensuring logical ordering.
    5. Submit TP (limit) and SL (stop) as an OCO pair.
    6. Return the result of the OCO order (or the entry result if OCO fails).
    """

    def __init__(self, broker: AbstractBroker):
        self.broker = broker

    async def _price_within_tolerance(
        self,
        entry: OrderRequest,
        market_price: float,
        tolerance: float,
    ) -> bool:
        """Return True if the entry limit price is within *tolerance* of *market_price*."""
        if entry.order_type != "limit" or entry.limit_price is None:
            # Market orders have no price to validate
            return True
        deviation = abs(entry.limit_price - market_price) / market_price
        return deviation <= tolerance

    async def _spread_within_limit(
        self,
        quote,
        max_spread_pct: float,
    ) -> bool:
        """Validate that the bid‑ask spread is not excessive.

        ``quote`` is expected to have ``bid`` and ``ask`` attributes.
        """
        try:
            bid = quote.bid
            ask = quote.ask
        except AttributeError:
            # If the broker does not provide bid/ask, we cannot apply the filter.
            return True
        if bid is None or ask is None or ask <= bid:
            return True
        mid = (bid + ask) / 2.0
        spread = (ask - bid) / mid
        return spread <= max_spread_pct

    async def execute(self, config: BracketOrderConfig) -> OrderResult:
        # ------------------------------------------------------------------
        # 0. Basic sanity checks
        # ------------------------------------------------------------------
        if config.entry.side not in ("buy", "sell"):
            raise ValueError(f"Invalid side for entry order: {config.entry.side}")

        if config.entry.quantity <= 0:
            raise ValueError("Entry order quantity must be positive")

        # ------------------------------------------------------------------
        # 1. Confirmation filters – price tolerance & spread
        # ------------------------------------------------------------------
        try:
            quote = await self.broker.get_quote(config.entry.symbol)
            market_price = getattr(quote, "last", None) or getattr(quote, "price", None)
            if market_price is None:
                raise ValueError("Quote missing market price")
            within_tolerance = await self._price_within_tolerance(
                config.entry, market_price, config.price_tolerance
            )
            within_spread = await self._spread_within_limit(quote, config.max_spread_pct)

            if not within_tolerance or not within_spread:
                logger.warning(
                    "Bracket entry rejected by confirmation filters",
                    symbol=config.entry.symbol,
                    entry_price=config.entry.limit_price,
                    market_price=market_price,
                    tolerance=config.price_tolerance,
                    spread_ok=within_spread,
                )
                return OrderResult(
                    broker_order_id="",
                    status="rejected",
                    avg_fill_price=None,
                    filled_qty=0,
                    reason="confirmation_filter_failed",
                )
        except Exception as exc:  # pragma: no cover – defensive, not expected in tests
            logger.warning(
                "Failed to fetch market quote for entry confirmation",
                error=str(exc),
                symbol=config.entry.symbol,
            )
            # Proceed without confirmation – broker may still reject unreasonable prices.

        # ------------------------------------------------------------------
        # 2. Submit entry order
        # ------------------------------------------------------------------
        entry_result = await self.broker.place_order(config.entry)

        if entry_result.status not in ("filled", "partially_filled"):
            logger.warning(
                "Bracket entry did not fill",
                status=entry_result.status,
                symbol=config.entry.symbol,
            )
            return entry_result

        fill_price = entry_result.avg_fill_price or 0.0
        filled_qty = entry_result.filled_qty

        # ------------------------------------------------------------------
        # 3. Compute TP/SL prices – enforce correct ordering
        # ------------------------------------------------------------------
        is_buy = config.entry.side == "buy"
        if is_buy:
            tp_price = fill_price * (1 + config.take_profit_pct)
            sl_price = fill_price * (1 - config.stop_loss_pct)
            tp_side = "sell"
        else:
            tp_price = fill_price * (1 - config.take_profit_pct)
            sl_price = fill_price * (1 + config.stop_loss_pct)
            tp_side = "buy"

        # Ensure TP is always more favorable than SL.
        if (is_buy and tp_price <= sl_price) or (not is_buy and tp_price >= sl_price):
            logger.error(
                "Invalid TP/SL configuration: TP not better than SL",
                tp_price=tp_price,
                sl_price=sl_price,
                side=config.entry.side,
            )
            return entry_result

        # ------------------------------------------------------------------
        # 4. Build TP (limit) and SL (stop) requests
        # ------------------------------------------------------------------
        tp_req = OrderRequest(
            account_id=config.entry.account_id,
            symbol=config.entry.symbol,
            side=tp_side,
            order_type="limit",
            quantity=filled_qty,
            limit_price=round(tp_price, 4),
            stop_price=None,
            time_in_force="GTC",
            execution_algo="market",
            risk_bucket=config.entry.risk_bucket,
        )
        sl_req = OrderRequest(
            account_id=config.entry.account_id,
            symbol=config.entry.symbol,
            side=tp_side,  # SL always closes the position
            order_type="stop",
            quantity=filled_qty,
            limit_price=None,
            stop_price=round(sl_price, 4),
            time_in_force="GTC",
            execution_algo="market",
            risk_bucket=config.entry.risk_bucket,
        )

        # ------------------------------------------------------------------
        # 5. Submit TP/SL as OCO pair
        # ------------------------------------------------------------------
        oco = OCOOrder(self.broker)
        oco_result = await oco.execute(tp_req, sl_req)

        logger.info(
            "Bracket OCO submitted",
            symbol=config.entry.symbol,
            entry_price=fill_price,
            tp_price=tp_price,
            sl_price=sl_price,
            oco_order_id=getattr(oco_result, "broker_order_id", None),
        )

        # Return the OCO result if it contains a broker_order_id; otherwise fall back.
        return oco_result if getattr(oco_result, "broker_order_id", None) else entry_result


class OCOOrder:
    """One‑Cancels‑Other: submit two opposing orders, cancel the other when one fills."""

    def __init__(self, broker: AbstractBroker, poll_seconds: int = 5, max_wait_seconds: int = 28_800):
        self.broker = broker
        self.poll_seconds = poll_seconds
        self.max_wait_seconds = max_wait_seconds

    async def _fetch_status(self, broker_order_id: str) -> Optional[dict]:
        """Helper to retrieve order status; returns ``None`` on failure."""
        try:
            return await self.broker.get_order(broker_order_id)
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to fetch order status", broker_order_id=broker_order_id, error=str(exc))
            return None

    async def execute(self, order_a: OrderRequest, order_b: OrderRequest) -> OrderResult:
        # Submit both legs
        ra = await self.broker.place_order(order_a)
        rb = await self.broker.place_order(order_b)

        elapsed = 0
        while elapsed < self.max_wait_seconds:
            status_a = await self._fetch_status(ra.broker_order_id)
            status_b = await self._fetch_status(rb.broker_order_id)

            # If either order is filled/closed, cancel the counterpart
            if status_a and status_a.get("status") in ("filled", "closed"):
                await self._cancel_if_open(rb.broker_order_id)
                logger.info("OCO: order A filled, B cancelled")
                return ra

            if status_b and status_b.get("status") in ("filled", "closed"):
                await self._cancel_if_open(ra.broker_order_id)
                logger.info("OCO: order B filled, A cancelled")
                return rb

            await asyncio.sleep(self.poll_seconds)
            elapsed += self.poll_seconds

        # Timeout – ensure both legs are cancelled
        await self._cancel_if_open(ra.broker_order_id)
        await self._cancel_if_open(rb.broker_order_id)
        logger.warning("OCO timeout reached; both orders cancelled")
        # Return a generic result indicating timeout
        return OrderResult(
            broker_order_id=ra.broker_order_id,
            status="cancelled",
            avg_fill_price=None,
            filled_qty=0,
            reason="oco_timeout",
        )

    async def _cancel_if_open(self, broker_order_id: str) -> None:
        """Cancel an order if it is still open; ignore errors."""
        try:
            order = await self.broker.get_order(broker_order_id)
            if order and order.get("status") not in ("filled", "closed", "cancelled"):
                await self.broker.cancel_order(broker_order_id)
        except Exception:  # pragma: no cover
            pass


class TrailingStop:
    """Trailing stop that follows price by a percentage.

    The algorithm continuously monitors the market price and moves the stop
    price in the direction of the trade whenever the price moves favorably.
    The stop is placed as a *stop* order (not a stop‑limit) to guarantee execution.
    """

    def __init__(self, broker: AbstractBroker, poll_seconds: int = 5, max_hold_seconds: int = 28_800):
        self.broker = broker
        self.poll_seconds = poll_seconds
        self.max_hold_seconds = max_hold_seconds

    async def _initial_price(self, symbol: str) -> float:
        """Fetch the latest market price for *symbol*."""
        quote = await self.broker.get_quote(symbol)
        return getattr(quote, "last", getattr(quote, "price", 0.0))

    async def execute(self, request: OrderRequest, trail_pct: float = 0.05) -> OrderResult:
        """Run a trailing‑stop strategy.

        Parameters
        ----------
        request: OrderRequest
            The initial market/limit order that opens the position.
        trail_pct: float, default 0.05
            The trailing distance as a fraction of price (5 % by default).

        Returns
        -------
        OrderResult
            Result of the final stop order (filled, cancelled or timeout).
        """
        if request.side not in ("buy", "sell"):
            raise ValueError("TrailingStop only supports 'buy' or 'sell' sides")

        # 1. Submit the opening order
        opening_result = await self.broker.place_order(request)

        if opening_result.status not in ("filled", "partially_filled"):
            logger.warning(
                "TrailingStop opening order did not fill",
                status=opening_result.status,
                symbol=request.symbol,
            )
            return opening_result

        filled_qty = opening_result.filled_qty
        entry_price = opening_result.avg_fill_price or 0.0
        is_long = request.side == "buy"

        # 2. Determine initial stop price based on trail_pct
        if is_long:
            stop_price = entry_price * (1 - trail_pct)
        else:
            stop_price = entry_price * (1 + trail_pct)

        # 3. Place the initial stop order
        stop_req = OrderRequest(
            account_id=request.account_id,
            symbol=request.symbol,
            side="sell" if is_long else "buy",
            order_type="stop",
            quantity=filled_qty,
            limit_price=None,
            stop_price=round(stop_price, 4),
            time_in_force="GTC",
            execution_algo="market",
            risk_bucket=request.risk_bucket,
        )
        stop_result = await self.broker.place_order(stop_req)

        # 4. Monitor market price and adjust stop as needed
        elapsed = 0
        while elapsed < self.max_hold_seconds:
            current_price = await self._initial_price(request.symbol)

            # Update stop only if price moved favorably beyond the trailing threshold
            if is_long:
                new_stop = current_price * (1 - trail_pct)
                if new_stop > stop_price:
                    stop_price = new_stop
            else:
                new_stop = current_price * (1 + trail_pct)
                if new_stop < stop_price:
                    stop_price = new_stop

            # Amend the stop order if the broker supports amendment; otherwise
            # cancel and place a new stop.
            try:
                await self.broker.modify_order(
                    stop_result.broker_order_id,
                    stop_price=round(stop_price, 4),
                )
            except Exception:  # pragma: no cover – broker may not implement modify_order
                try:
                    await self.broker.cancel_order(stop_result.broker_order_id)
                except Exception:
                    pass
                stop_req.stop_price = round(stop_price, 4)
                stop_result = await self.broker.place_order(stop_req)

            # Check if stop has been filled
            try:
                stop_status = await self.broker.get_order(stop_result.broker_order_id)
                if stop_status and stop_status.get("status") in ("filled", "closed"):
                    logger.info(
                        "Trailing stop filled",
                        symbol=request.symbol,
                        fill_price=stop_status.get("avg_fill_price"),
                    )
                    return OrderResult(
                        broker_order_id=stop_result.broker_order_id,
                        status="filled",
                        avg_fill_price=stop_status.get("avg_fill_price"),
                        filled_qty=filled_qty,
                        reason=None,
                    )
            except Exception as exc:  # pragma: no cover
                logger.warning("Failed to fetch trailing stop status", error=str(exc))

            await asyncio.sleep(self.poll_seconds)
            elapsed += self.poll_seconds

        # Timeout – cancel the stop to avoid unintended exposure
        try:
            await self.broker.cancel_order(stop_result.broker_order_id)
        except Exception:
            pass
        logger.warning(
            "TrailingStop timeout reached; stop order cancelled",
            symbol=request.symbol,
        )
        return OrderResult(
            broker_order_id=stop_result.broker_order_id,
            status="cancelled",
            avg_fill_price=None,
            filled_qty=0,
            reason="trailing_stop_timeout",
        )