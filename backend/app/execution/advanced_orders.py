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

    async def _price_within_tolerance(self, entry: OrderRequest, market_price: float) -> bool:
        """Validate that the entry price is within the configured tolerance."""
        if entry.order_type != "limit" or entry.limit_price is None:
            # Market orders have no price to validate
            return True
        deviation = abs(entry.limit_price - market_price) / market_price
        return deviation <= entry.price_tolerance if hasattr(entry, "price_tolerance") else deviation <= 0.02

    async def execute(self, config: BracketOrderConfig) -> OrderResult:
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
            ...

# --------------------------------------------------------------------
# Unit tests for edge cases
# --------------------------------------------------------------------
import pytest
from unittest.mock import AsyncMock, MagicMock

@pytest.fixture
def mock_broker():
    broker = MagicMock(spec=AbstractBroker)
    broker.get_quote = AsyncMock(return_value=MagicMock(last=100.0))
    broker.place_order = AsyncMock()
    broker.get_order = AsyncMock()
    broker.cancel_order = AsyncMock()
    return broker

@pytest.mark.asyncio
async def test_bracket_order_price_tolerance_boundary(mock_broker):
    """Entry price exactly at tolerance limit should be accepted."""
    entry_req = OrderRequest(
        account_id="acc1",
        symbol="TEST",
        side="buy",
        order_type="limit",
        quantity=10,
        limit_price=102.0,  # 2% above market price (100)
        stop_price=None,
        time_in_force="GTC",
        execution_algo="market",
        risk_bucket="default",
    )
    # Attach explicit tolerance attribute to entry
    entry_req.price_tolerance = 0.02

    # Mock filled entry result
    mock_broker.place_order.return_value = OrderResult(
        broker_order_id="ord123",
        status="filled",
        avg_fill_price=102.0,
        filled_qty=10,
        reason=None,
    )
    # Patch OCOOrder.execute to avoid long polling
    original_oco_execute = OCOOrder.execute
    OCOOrder.execute = AsyncMock(return_value=OrderResult(
        broker_order_id="oco123",
        status="filled",
        avg_fill_price=105.0,
        filled_qty=10,
        reason=None,
    ))

    bracket = BracketOrder(mock_broker)
    config = BracketOrderConfig(
        entry=entry_req,
        take_profit_pct=0.03,
        stop_loss_pct=0.01,
    )
    result = await bracket.execute(config)

    # Ensure we get the mocked OCO result, meaning price tolerance passed
    assert result.broker_order_id == "oco123"

    # Restore original method
    OCOOrder.execute = original_oco_execute

@pytest.mark.asyncio
async def test_bracket_order_invalid_tp_sl_configuration(mock_broker):
    """When TP price is not greater than SL price, execution should abort after entry."""
    entry_req = OrderRequest(
        account_id="acc2",
        symbol="TEST2",
        side="sell",
        order_type="limit",
        quantity=5,
        limit_price=100.0,
        stop_price=None,
        time_in_force="GTC",
        execution_algo="market",
        risk_bucket="default",
    )
    mock_broker.place_order.return_value = OrderResult(
        broker_order_id="ord456",
        status="filled",
        avg_fill_price=100.0,
        filled_qty=5,
        reason=None,
    )
    bracket = BracketOrder(mock_broker)
    # Set percentages that will make TP <= SL for a sell side
    config = BracketOrderConfig(
        entry=entry_req,
        take_profit_pct=0.10,   # TP = 90
        stop_loss_pct=0.05,     # SL = 105
    )
    result = await bracket.execute(config)

    # Should return the entry result because TP/SL check fails
    assert result.broker_order_id == "ord456"
    assert result.status == "filled"

@pytest.mark.asyncio
async def test_oco_order_timeout_fallback(mock_broker):
    """OCOOrder should timeout and return the first order when neither fills."""
    # Configure broker to always return open orders
    async def get_order_stub(order_id):
        return {"order_id": order_id, "status": "open"}
    mock_broker.place_order.side_effect = [
        OrderResult(broker_order_id="a1", status="open", avg_fill_price=None, filled_qty=0, reason=None),
        OrderResult(broker_order_id="b1", status="open", avg_fill_price=None, filled_qty=0, reason=None),
    ]
    mock_broker.get_order.side_effect = get_order_stub

    # Use a very short max_wait_seconds to trigger timeout quickly
    oco = OCOOrder(mock_broker, poll_seconds=1, max_wait_seconds=3)
    order_a = OrderRequest(
        account_id="acc3",
        symbol="T3",
        side="buy",
        order_type="limit",
        quantity=1,
        limit_price=10,
        stop_price=None,
        time_in_force="GTC",
        execution_algo="market",
        risk_bucket="default",
    )
    order_b = OrderRequest(
        account_id="acc3",
        symbol="T3",
        side="sell",
        order_type="stop",
        quantity=1,
        limit_price=None,
        stop_price=9,
        time_in_force="GTC",
        execution_algo="market",
        risk_bucket="default",
    )
    result = await oco.execute(order_a, order_b)

    # Expect fallback to first order result
    assert result.broker_order_id == "a1"
    assert result.status == "open"