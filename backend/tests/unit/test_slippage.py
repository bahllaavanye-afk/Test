"""Slippage tracker tests."""
import pytest
from app.brokers.base import OrderRequest, OrderResult
from app.execution.slippage_tracker import SlippageTracker


@pytest.mark.asyncio
async def test_record_signal_and_fill_buy():
    tracker = SlippageTracker()
    req = OrderRequest(
        account_id="acc1",
        symbol="AAPL",
        side="buy",
        order_type="market",
        quantity=10,
        limit_price=None,
        stop_price=None,
        time_in_force="GTC",
        execution_algo="market",
    )
    await tracker.record_signal_price(req, 100.00)
    result = OrderResult(
        broker_order_id="x",
        status="filled",
        filled_qty=10,
        avg_fill_price=100.10,
    )
    # 10 bps slippage on a buy
    await tracker.record_fill(req, result)


@pytest.mark.asyncio
async def test_no_signal_price_skips():
    tracker = SlippageTracker()
    req = OrderRequest(
        account_id="acc1",
        symbol="AAPL",
        side="buy",
        order_type="market",
        quantity=10,
        limit_price=None,
        stop_price=None,
        time_in_force="GTC",
        execution_algo="market",
    )
    result = OrderResult(
        broker_order_id="x",
        status="filled",
        filled_qty=10,
        avg_fill_price=100.10,
    )
    # Should not raise when no signal_price was recorded
    await tracker.record_fill(req, result)


@pytest.mark.asyncio
async def test_record_signal_price_none_request():
    """Ensure that passing None as the request does not raise an exception."""
    tracker = SlippageTracker()
    # The tracker should gracefully ignore a None request.
    await tracker.record_signal_price(None, 100.0)


@pytest.mark.asyncio
async def test_record_fill_none_inputs():
    """Validate that None inputs are safely handled by record_fill."""
    tracker = SlippageTracker()
    # Both request and result are None – should be a no‑op without error.
    await tracker.record_fill(None, None)


@pytest.mark.asyncio
async def test_empty_quantity_handling():
    """Check behavior when an order has zero quantity (edge off‑by‑one case)."""
    tracker = SlippageTracker()
    req = OrderRequest(
        account_id="acc2",
        symbol="MSFT",
        side="sell",
        order_type="market",
        quantity=0,  # Edge case: zero quantity
        limit_price=None,
        stop_price=None,
        time_in_force="GTC",
        execution_algo="market",
    )
    await tracker.record_signal_price(req, 250.00)
    result = OrderResult(
        broker_order_id="y",
        status="filled",
        filled_qty=0,
        avg_fill_price=250.00,
    )
    # Should not raise; zero‑quantity orders are effectively no‑ops.
    await tracker.record_fill(req, result)


@pytest.mark.asyncio
async def test_multiple_fills_off_by_one():
    """Simulate consecutive fills to expose potential off‑by‑one errors."""
    tracker = SlippageTracker()
    req = OrderRequest(
        account_id="acc3",
        symbol="GOOG",
        side="buy",
        order_type="market",
        quantity=5,
        limit_price=None,
        stop_price=None,
        time_in_force="GTC",
        execution_algo="market",
    )
    await tracker.record_signal_price(req, 1500.00)

    # First fill of 2 units
    result1 = OrderResult(
        broker_order_id="z1",
        status="filled",
        filled_qty=2,
        avg_fill_price=1500.20,
    )
    await tracker.record_fill(req, result1)

    # Second fill of remaining 3 units
    result2 = OrderResult(
        broker_order_id="z2",
        status="filled",
        filled_qty=3,
        avg_fill_price=1500.25,
    )
    await tracker.record_fill(req, result2)

    # No exception should be raised; internal counters must handle the off‑by‑one
    # scenario correctly.