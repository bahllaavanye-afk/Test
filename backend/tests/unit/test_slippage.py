"""Slippage tracker tests."""
import pytest
from app.brokers.base import OrderRequest, OrderResult
from app.execution.slippage_tracker import SlippageTracker


def _create_order_request(
    account_id: str = "acc1",
    symbol: str = "AAPL",
    side: str = "buy",
    order_type: str = "market",
    quantity: int = 10,
    limit_price: float | None = None,
    stop_price: float | None = None,
    time_in_force: str = "GTC",
    execution_algo: str = "market",
) -> OrderRequest:
    """Factory for a standard OrderRequest used in tests."""
    return OrderRequest(
        account_id=account_id,
        symbol=symbol,
        side=side,
        order_type=order_type,
        quantity=quantity,
        limit_price=limit_price,
        stop_price=stop_price,
        time_in_force=time_in_force,
        execution_algo=execution_algo,
    )


def _create_order_result(
    broker_order_id: str = "x",
    status: str = "filled",
    filled_qty: int = 10,
    avg_fill_price: float = 100.10,
) -> OrderResult:
    """Factory for a standard OrderResult used in tests."""
    return OrderResult(
        broker_order_id=broker_order_id,
        status=status,
        filled_qty=filled_qty,
        avg_fill_price=avg_fill_price,
    )


@pytest.mark.asyncio
async def test_record_signal_and_fill_buy():
    tracker = SlippageTracker()
    req = _create_order_request()
    await tracker.record_signal_price(req, 100.00)
    result = _create_order_result()
    # 10 bps slippage on a buy
    await tracker.record_fill(req, result)


@pytest.mark.asyncio
async def test_no_signal_price_skips():
    tracker = SlippageTracker()
    req = _create_order_request()
    result = _create_order_result()
    # Should not raise when no signal_price was recorded
    await tracker.record_fill(req, result)