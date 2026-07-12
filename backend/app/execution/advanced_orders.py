"""
Advanced order types:
  - BracketOrder: entry + take-profit + stop-loss together
  - OCOOrder: one-cancels-other (two opposing orders, fill one → cancel the other)
  - TrailingStop: stop that follows price by N% or $N
"""
from __future__ import annotations

import asyncio
import unittest
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

        # Timeout – cancel any remaining open orders to avoid orphaned positions.
        for leg, res in (("A", ra), ("B", rb)):
            try:
                await self.broker.cancel_order(res.broker_order_id)
            except Exception as exc:  # noqa: BLE001 — log and continue to next leg
                logger.error(
                    "OCO timeout: cancel of leg %s FAILED — order %s may still be live",
                    leg, res.broker_order_id, exc_info=exc,
                )
        logger.warning("OCO timeout reached; cancellation attempted on both legs")
        return ra  # Returning the first order as a fallback result


class TrailingStop:
    """
    Trailing stop that follows price by trail_pct. Continually adjusts stop price upward
    (or downward for shorts) as price moves favorably.
    """
    ...


# ----------------------------------------------------------------------
# Unit tests for edge cases
# ----------------------------------------------------------------------
class MockBroker(AbstractBroker):
    """A lightweight in‑memory broker mock for unit testing."""
    def __init__(self):
        self.orders = {}
        self.next_id = 1
        self.canceled = set()
        self.quote_price = 100.0

    async def get_quote(self, symbol: str):
        class Quote:  # simple container
            def __init__(self, last):
                self.last = last
        return Quote(self.quote_price)

    async def place_order(self, request: OrderRequest) -> OrderResult:
        oid = f"order-{self.next_id}"
        self.next_id += 1
        # Simulate immediate fill for entry orders; OCO legs stay open unless forced
        if request.order_type == "limit" and request.side in ("buy", "sell"):
            status = "filled"
            avg_price = request.limit_price or self.quote_price
        else:
            status = "open"
            avg_price = None
        result = OrderResult(
            broker_order_id=oid,
            status=status,
            avg_fill_price=avg_price,
            filled_qty=request.quantity if status == "filled" else 0,
            reason=None,
        )
        self.orders[oid] = {"status": status, "request": request}
        return result

    async def get_order(self, broker_order_id: str):
        # Return stored status; allow external manipulation in tests
        return self.orders.get(broker_order_id, {"status": "unknown"})

    async def cancel_order(self, broker_order_id: str):
        self.canceled.add(broker_order_id)
        # Update internal state to reflect cancellation
        if broker_order_id in self.orders:
            self.orders[broker_order_id]["status"] = "canceled"


class TestAdvancedOrders(unittest.IsolatedAsyncioTestCase):
    async def test_price_tolerance_boundary(self):
        """Deviation exactly equal to tolerance should be accepted."""
        broker = MockBroker()
        broker.quote_price = 100.0
        entry = OrderRequest(
            account_id="A1",
            symbol="TEST",
            side="buy",
            order_type="limit",
            quantity=10,
            limit_price=102.0,  # 2% above market
            stop_price=None,
            time_in_force="GTC",
            execution_algo="market",
            risk_bucket="default",
        )
        # Attach explicit tolerance attribute to entry request
        entry.price_tolerance = 0.02
        config = BracketOrderConfig(entry=entry, take_profit_pct=0.05, stop_loss_pct=0.02)
        bracket = BracketOrder(broker)
        result = await bracket.execute(config)
        self.assertEqual(result.status, "filled")
        # Ensure TP/SL OCO was submitted (order ids > 1)
        self.assertTrue(any(oid.startswith("order-") for oid in broker.orders if oid != "order-1"))

    async def test_invalid_tp_sl_configuration(self):
        """When TP price is not greater than SL price, the order should abort early."""
        broker = MockBroker()
        entry = OrderRequest(
            account_id="A1",
            symbol="TEST",
            side="buy",
            order_type="limit",
            quantity=5,
            limit_price=100.0,
            stop_price=None,
            time_in_force="GTC",
            execution_algo="market",
            risk_bucket="default",
        )
        # Set TP and SL such that TP <= SL
        config = BracketOrderConfig(entry=entry, take_profit_pct=0.01, stop_loss_pct=0.02)
        bracket = BracketOrder(broker)
        result = await bracket.execute(config)
        # The entry should have filled, but TP/SL OCO should not be created
        self.assertEqual(result.status, "filled")
        self.assertEqual(len(broker.orders), 1)  # only the entry order

    async def test_oco_timeout_cancels_both_legs(self):
        """When neither leg fills before timeout, both should be cancelled."""
        class FastTimeoutOCO(OCOOrder):
            def __init__(self, broker):
                super().__init__(broker, poll_seconds=0, max_wait_seconds=0)  # immediate timeout

        broker = MockBroker()
        order_a = OrderRequest(
            account_id="A1",
            symbol="TEST",
            side="sell",
            order_type="limit",
            quantity=10,
            limit_price=105.0,
            stop_price=None,
            time_in_force="GTC",
            execution_algo="market",
            risk_bucket="default",
        )
        order_b = OrderRequest(
            account_id="A1",
            symbol="TEST",
            side="sell",
            order_type="stop",
            quantity=10,
            limit_price=None,
            stop_price=95.0,
            time_in_force="GTC",
            execution_algo="market",
            risk_bucket="default",
        )
        oco = FastTimeoutOCO(broker)
        result = await oco.execute(order_a, order_b)
        # Both legs should be marked as cancelled in the mock broker
        self.assertIn(result.broker_order_id, broker.canceled)
        other_id = [oid for oid in broker.orders if oid != result.broker_order_id][0]
        self.assertIn(other_id, broker.canceled)


if __name__ == "__main__":
    unittest.main()