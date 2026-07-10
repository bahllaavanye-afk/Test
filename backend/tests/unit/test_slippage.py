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


# -------------------------------------------------------------------------
# Edge case tests
# -------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_record_signal_price_with_none_request():
    """Passing None as the OrderRequest should raise a TypeError."""
    tracker = SlippageTracker()
    with pytest.raises(TypeError):
        await tracker.record_signal_price(None, 100.00)  # type: ignore


@pytest.mark.asyncio
async def test_record_signal_price_with_none_price():
    """Passing None as the price should raise a TypeError."""
    tracker = SlippageTracker()
    req = OrderRequest(
        account_id="acc2",
        symbol="MSFT",
        side="sell",
        order_type="market",
        quantity=5,
        limit_price=None,
        stop_price=None,
        time_in_force="GTC",
        execution_algo="market",
    )
    with pytest.raises(TypeError):
        await tracker.record_signal_price(req, None)  # type: ignore


@pytest.mark.asyncio
async def test_record_fill_with_none_result():
    """Calling record_fill with a None result should raise a TypeError."""
    tracker = SlippageTracker()
    req = OrderRequest(
        account_id="acc3",
        symbol="GOOG",
        side="buy",
        order_type="market",
        quantity=1,
        limit_price=None,
        stop_price=None,
        time_in_force="GTC",
        execution_algo="market",
    )
    with pytest.raises(TypeError):
        await tracker.record_fill(req, None)  # type: ignore


@pytest.mark.asyncio
async def test_record_fill_empty_quantity():
    """Zero quantity should be handled without division errors (off‑by‑one)."""
    tracker = SlippageTracker()
    req = OrderRequest(
        account_id="acc4",
        symbol="TSLA",
        side="sell",
        order_type="market",
        quantity=0,
        limit_price=None,
        stop_price=None,
        time_in_force="GTC",
        execution_algo="market",
    )
    await tracker.record_signal_price(req, 200.00)
    result = OrderResult(
        broker_order_id="y",
        status="filled",
        filled_qty=0,
        avg_fill_price=200.00,
    )
    # Ensure no exception is raised even though quantity is zero
    await tracker.record_fill(req, result)


@pytest.mark.asyncio
async def test_record_fill_off_by_one_quantity():
    """Test off‑by‑one scenario where filled_qty is one less than requested."""
    tracker = SlippageTracker()
    req = OrderRequest(
        account_id="acc5",
        symbol="NFLX",
        side="buy",
        order_type="market",
        quantity=11,
        limit_price=None,
        stop_price=None,
        time_in_force="GTC",
        execution_algo="market",
    )
    await tracker.record_signal_price(req, 500.00)
    result = OrderResult(
        broker_order_id="z",
        status="filled",
        filled_qty=10,  # one less than requested
        avg_fill_price=500.25,
    )
    # Should process without error; slippage calculation may be partial
    await tracker.record_fill(req, result)