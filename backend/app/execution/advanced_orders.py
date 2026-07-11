"""
Advanced order types:
  - BracketOrder: entry + take-profit + stop-loss together
  - OCOOrder: one-cancels-other (two opposing orders, fill one → cancel the other)
  - TrailingStop: stop that follows price by N% or $N
"""
from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
from typing import Optional

from app.brokers.base import AbstractBroker, OrderRequest, OrderResult
from app.utils.logging import logger


@dataclass
class BracketOrderConfig:
    entry: OrderRequest
    take_profit_pct: float    # e.g. 0.05 = +5% TP
    stop_loss_pct: float      # e.g. 0.02 = -2% SL
    price_tolerance: float = 0.02  # allowable deviation between entry request price and market price (2%)


class BracketOrder:
    """
    Submit entry, then watch for fill. Once filled, submit take-profit and stop-loss
    as OCO pair. Whichever fills cancels the other.
    """
    def __init__(self, broker: AbstractBroker):
        self.broker = broker

    async def _price_within_tolerance(
        self,
        entry: OrderRequest,
        market_price: float,
        tolerance: float,
    ) -> bool:
        """Validate that the entry price is within the configured tolerance."""
        if entry.order_type != "limit" or entry.limit_price is None:
            # Market orders have no price to validate
            return True
        deviation = abs(entry.limit_price - market_price) / market_price
        return deviation <= tolerance

    async def _validate_entry(self, config: BracketOrderConfig) -> None:
        """Perform basic sanity checks on the entry order."""
        if config.entry.side not in ("buy", "sell"):
            raise ValueError(f"Invalid side for entry order: {config.entry.side}")
        if config.entry.quantity <= 0:
            raise ValueError("Entry order quantity must be positive")

    async def _fetch_market_price(self, symbol: str) -> Optional[float]:
        """Retrieve the latest market price; return None on failure."""
        try:
            quote = await self.broker.get_quote(symbol)
            return quote.last
        except Exception as exc:
            logger.warning("Failed to fetch market price for entry confirmation", error=str(exc))
            return None

    async def _abort_due_to_tolerance(self, config: BracketOrderConfig, market_price: float) -> OrderResult:
        """Return a rejected OrderResult when price tolerance is exceeded."""
        logger.warning(
            "Bracket entry price deviates beyond tolerance",
            symbol=config.entry.symbol,
            entry_price=config.entry.limit_price,
            market_price=market_price,
            tolerance=config.price_tolerance,
        )
        return OrderResult(
            broker_order_id="",
            status="rejected",
            avg_fill_price=None,
            filled_qty=0,
            reason="price_tolerance_exceeded",
        )

    def _compute_tp_sl(
        self,
        fill_price: float,
        is_buy: bool,
        take_profit_pct: float,
        stop_loss_pct: float,
    ) -> tuple[float, float, str]:
        """Calculate TP and SL prices and the side for the TP order."""
        if is_buy:
            tp_price = fill_price * (1 + take_profit_pct)
            sl_price = fill_price * (1 - stop_loss_pct)
            tp_side = "sell"
        else:
            tp_price = fill_price * (1 - take_profit_pct)
            sl_price = fill_price * (1 + stop_loss_pct)
            tp_side = "buy"
        return tp_price, sl_price, tp_side

    def _build_tp_sl_requests(
        self,
        config: BracketOrderConfig,
        filled_qty: float,
        tp_price: float,
        sl_price: float,
        tp_side: str,
    ) -> tuple[OrderRequest, OrderRequest]:
        """Create the OrderRequest objects for TP (limit) and SL (stop)."""
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
            side=tp_side,  # both TP and SL close the position
            order_type="stop",
            quantity=filled_qty,
            limit_price=None,
            stop_price=round(sl_price, 4),
            time_in_force="GTC",
            execution_algo="market",
            risk_bucket=config.entry.risk_bucket,
        )
        return tp_req, sl_req

    async def execute(self, config: BracketOrderConfig) -> OrderResult:
        # 0. Basic sanity checks
        await self._validate_entry(config)

        # 1. Optional confirmation filter – ensure entry price is reasonable
        market_price = await self._fetch_market_price(config.entry.symbol)
        if market_price is not None:
            within_tol = await self._price_within_tolerance(
                config.entry,
                market_price,
                config.price_tolerance,
            )
            if not within_tol:
                return await self._abort_due_to_tolerance(config, market_price)

        # 2. Submit entry
        entry_result = await self.broker.place_order(config.entry)
        if entry_result.status not in ("filled", "partially_filled"):
            logger.warning("Bracket entry didn't fill", status=entry_result.status)
            return entry_result

        fill_price = entry_result.avg_fill_price or 0.0
        is_buy = config.entry.side == "buy"

        # 3. Compute TP and SL prices; ensure logical ordering
        tp_price, sl_price, tp_side = self._compute_tp_sl(
            fill_price,
            is_buy,
            config.take_profit_pct,
            config.stop_loss_pct,
        )
        if tp_price <= sl_price:
            logger.error(
                "Invalid TP/SL configuration: TP price not greater than SL price",
                tp_price=tp_price,
                sl_price=sl_price,
                side=config.entry.side,
            )
            return entry_result

        # 4. Build TP limit and SL stop requests
        tp_req, sl_req = self._build_tp_sl_requests(
            config,
            entry_result.filled_qty,
            tp_price,
            sl_price,
            tp_side,
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
    Trailing stop that follows price by trail_pct. Continually adjusts stop price upward
    (or downward for shorts) as price moves favorably.
    """
    def __init__(self, broker: AbstractBroker, poll_seconds: int = 5, max_hold_seconds: int = 28800):
        self.broker = broker
        self.poll_seconds = poll_seconds
        self.max_hold_seconds = max_hold_seconds

    async def execute(self, request: OrderRequest, trail_pct: float = 0.05) -> OrderResult:
        if request.side == "sell":
            # Implementation omitted
            raise NotImplementedError("TrailingStop execution for sell side not implemented")
        else:
            # Implementation omitted
            raise NotImplementedError("TrailingStop execution for buy side not implemented")