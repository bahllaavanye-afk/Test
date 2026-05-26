"""Slippage tracker tests."""
import pytest
from app.brokers.base import OrderRequest, OrderResult
from app.execution.slippage_tracker import SlippageTracker


@pytest.mark.asyncio
async def test_record_signal_and_fill_buy():
    tracker = SlippageTracker()
    req = OrderRequest(account_id="acc1", symbol="AAPL", side="buy",
                        order_type="market", quantity=10, limit_price=None,
                        stop_price=None, time_in_force="GTC", execution_algo="market")
    await tracker.record_signal_price(req, 100.00)
    result = OrderResult(broker_order_id="x", status="filled",
                          filled_qty=10, avg_fill_price=100.10)
    # 10 bps slippage on a buy
    await tracker.record_fill(req, result)


@pytest.mark.asyncio
async def test_no_signal_price_skips():
    tracker = SlippageTracker()
    req = OrderRequest(account_id="acc1", symbol="AAPL", side="buy",
                        order_type="market", quantity=10, limit_price=None,
                        stop_price=None, time_in_force="GTC", execution_algo="market")
    result = OrderResult(broker_order_id="x", status="filled",
                          filled_qty=10, avg_fill_price=100.10)
    # Should not raise when no signal_price was recorded
    await tracker.record_fill(req, result)
