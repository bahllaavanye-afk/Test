"""Slippage tracker tests."""
import pytest
from app.brokers.base import OrderRequest, OrderResult
from app.execution.slippage_tracker import SlippageTracker

# Constants
ACCOUNT_ID = "acc1"
SYMBOL = "AAPL"
SIDE_BUY = "buy"
ORDER_TYPE_MARKET = "market"
QUANTITY = 10
TIME_IN_FORCE = "GTC"
EXECUTION_ALGO = "market"
BROKER_ORDER_ID = "x"
STATUS_FILLED = "filled"
SIGNAL_PRICE = 100.00
FILL_PRICE = 100.10


@pytest.mark.asyncio
async def test_record_signal_and_fill_buy():
    tracker = SlippageTracker()
    req = OrderRequest(
        account_id=ACCOUNT_ID,
        symbol=SYMBOL,
        side=SIDE_BUY,
        order_type=ORDER_TYPE_MARKET,
        quantity=QUANTITY,
        limit_price=None,
        stop_price=None,
        time_in_force=TIME_IN_FORCE,
        execution_algo=EXECUTION_ALGO,
    )
    await tracker.record_signal_price(req, SIGNAL_PRICE)
    result = OrderResult(
        broker_order_id=BROKER_ORDER_ID,
        status=STATUS_FILLED,
        filled_qty=QUANTITY,
        avg_fill_price=FILL_PRICE,
    )
    # 10 bps slippage on a buy
    await tracker.record_fill(req, result)


@pytest.mark.asyncio
async def test_no_signal_price_skips():
    tracker = SlippageTracker()
    req = OrderRequest(
        account_id=ACCOUNT_ID,
        symbol=SYMBOL,
        side=SIDE_BUY,
        order_type=ORDER_TYPE_MARKET,
        quantity=QUANTITY,
        limit_price=None,
        stop_price=None,
        time_in_force=TIME_IN_FORCE,
        execution_algo=EXECUTION_ALGO,
    )
    result = OrderResult(
        broker_order_id=BROKER_ORDER_ID,
        status=STATUS_FILLED,
        filled_qty=QUANTITY,
        avg_fill_price=FILL_PRICE,
    )
    # Should not raise when no signal_price was recorded
    await tracker.record_fill(req, result)