"""
Iceberg execution: show only small visible quantity, refill as each slice fills.
Prevents large orders from moving the market by hiding true size.
"""
from __future__ import annotations
import asyncio
from dataclasses import asdict
from app.brokers.base import AbstractBroker, OrderRequest, OrderResult
from app.utils.logging import logger

import unittest


class IcebergExecution:
    def __init__(self, broker: AbstractBroker, visible_pct: float = 0.10, refill_delay_seconds: int = 5):
        self.broker = broker
        self.visible_pct = visible_pct
        self.refill_delay_seconds = refill_delay_seconds

    async def execute(self, request: OrderRequest) -> OrderResult:
        visible_qty = max(1.0, request.quantity * self.visible_pct)
        remaining = request.quantity
        total_filled = 0.0
        total_cost = 0.0
        last_result: OrderResult | None = None

        while remaining > 0.01:
            slice_qty = min(visible_qty, remaining)
            slice_req = OrderRequest(
                **{**asdict(request), "quantity": slice_qty, "order_type": "market"}
            )
            try:
                result = await self.broker.place_order(slice_req)
                total_filled += result.filled_qty
                remaining -= result.filled_qty
                if result.avg_fill_price:
                    total_cost += result.avg_fill_price * result.filled_qty
                last_result = result
                logger.debug("Iceberg slice", filled=result.filled_qty, remaining=remaining)

                if remaining > 0.01:
                    await asyncio.sleep(self.refill_delay_seconds)
            except Exception as e:
                logger.warning("Iceberg slice failed", error=str(e))
                break

        avg_price = total_cost / total_filled if total_filled > 0 else None
        return OrderResult(
            broker_order_id=last_result.broker_order_id if last_result else "iceberg",
            status="filled" if total_filled >= request.quantity * 0.95 else "partial",
            filled_qty=total_filled,
            avg_fill_price=avg_price,
        )


# ----------------------------------------------------------------------
# Unit tests for edge cases
# ----------------------------------------------------------------------
class _FakeBroker(AbstractBroker):
    """A minimal broker stub that instantly fills the requested quantity."""
    async def place_order(self, request: OrderRequest) -> OrderResult:
        # Simulate an immediate full fill at a constant price
        return OrderResult(
            broker_order_id="fake",
            status="filled",
            filled_qty=request.quantity,
            avg_fill_price=100.0,
        )


class TestIcebergExecutionEdgeCases(unittest.IsolatedAsyncioTestCase):
    async def test_zero_quantity(self):
        """Zero quantity should result in a filled status with zero filled_qty."""
        broker = _FakeBroker()
        exec_engine = IcebergExecution(broker)
        request = OrderRequest(
            symbol="TEST",
            side="buy",
            quantity=0.0,
            order_type="limit",
            price=50.0,
        )
        result = await exec_engine.execute(request)
        self.assertEqual(result.filled_qty, 0.0)
        self.assertIsNone(result.avg_fill_price)
        self.assertEqual(result.status, "filled")
        self.assertEqual(result.broker_order_id, "iceberg")

    async def test_small_quantity_boundary(self):
        """Quantities just above the loop threshold should be processed in a single slice."""
        broker = _FakeBroker()
        exec_engine = IcebergExecution(broker, visible_pct=0.5)
        request = OrderRequest(
            symbol="TEST",
            side="sell",
            quantity=0.015,  # Slightly above the 0.01 loop termination threshold
            order_type="limit",
            price=75.0,
        )
        result = await exec_engine.execute(request)
        self.assertAlmostEqual(result.filled_qty, 0.015, places=6)
        self.assertAlmostEqual(result.avg_fill_price, 100.0, places=6)
        self.assertEqual(result.status, "filled")
        self.assertEqual(result.broker_order_id, "fake")

    async def test_no_refill_delay(self):
        """Refill delay set to zero should not introduce unnecessary sleep."""
        broker = _FakeBroker()
        exec_engine = IcebergExecution(broker, visible_pct=0.2, refill_delay_seconds=0)
        request = OrderRequest(
            symbol="TEST",
            side="buy",
            quantity=2.5,
            order_type="limit",
            price=60.0,
        )
        # Patch asyncio.sleep to ensure it's not called with a positive delay
        original_sleep = asyncio.sleep

        async def fake_sleep(delay, *args, **kwargs):
            self.assertEqual(delay, 0, "Sleep called with non-zero delay")
            return await original_sleep(0)

        asyncio.sleep = fake_sleep
        try:
            result = await exec_engine.execute(request)
        finally:
            asyncio.sleep = original_sleep

        self.assertAlmostEqual(result.filled_qty, 2.5, places=6)
        self.assertAlmostEqual(result.avg_fill_price, 100.0, places=6)
        self.assertEqual(result.status, "filled")
        self.assertEqual(result.broker_order_id, "fake")


if __name__ == "__main__":
    unittest.main()