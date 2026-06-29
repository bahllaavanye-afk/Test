"""Execution algorithm tests using mock broker."""
import asyncio
from datetime import datetime, timedelta

import pytest

from app.brokers.base import OrderRequest, OrderResult, QuoteResult
from app.execution.twap import TWAPExecution
from app.execution.limit_first import LimitFirstExecution


class MockBroker:
    """In-memory broker that always fills orders at the last quote."""
    def __init__(self, last: float = 100.0, bid: float = 99.95, ask: float = 100.05):
        self.last = last
        self.bid = bid
        self.ask = ask
        self.placed: list[OrderRequest] = []
        self.cancelled: list[str] = []

    async def get_quote(self, symbol: str) -> QuoteResult:
        return QuoteResult(
            symbol=symbol,
            bid=self.bid,
            ask=self.ask,
            last=self.last,
            volume=1_000_000,
        )

    async def place_order(self, request: OrderRequest) -> OrderResult:
        self.placed.append(request)
        return OrderResult(
            broker_order_id=f"mock-{len(self.placed)}",
            status="filled",
            filled_qty=request.quantity,
            avg_fill_price=self.last,
        )

    async def cancel_order(self, order_id: str) -> bool:
        self.cancelled.append(order_id)
        return True

    async def get_order(self, order_id: str) -> dict:
        return {"status": "filled", "filled_qty": 1.0}


class MockBrokerNoFill(MockBroker):
    """Broker that never fills limit orders, forcing a fallback to market."""
    async def place_order(self, request: OrderRequest) -> OrderResult:
        self.placed.append(request)
        # Simulate a limit order that is not filled; market orders are filled.
        if request.order_type == "limit":
            return OrderResult(
                broker_order_id=f"mock-{len(self.placed)}",
                status="open",
                filled_qty=0.0,
                avg_fill_price=0.0,
            )
        return OrderResult(
            broker_order_id=f"mock-{len(self.placed)}",
            status="filled",
            filled_qty=request.quantity,
            avg_fill_price=self.last,
        )


@pytest.fixture
def request_obj() -> OrderRequest:
    """Standard order request used across tests."""
    return OrderRequest(
        account_id="test",
        symbol="AAPL",
        side="buy",
        order_type="market",
        quantity=100,
        limit_price=None,
        stop_price=None,
        time_in_force="GTC",
        execution_algo="auto",
    )


@pytest.mark.asyncio
async def test_twap_slices_evenly(request_obj: OrderRequest) -> None:
    """TWAP should split the total quantity evenly across slices."""
    broker = MockBroker()
    # Use 2 slices over a very short duration to keep the test fast.
    twap = TWAPExecution(broker, slices=2, duration_minutes=0.001)
    result = await twap.execute(request_obj)

    assert len(broker.placed) == 2, "TWAP must place exactly two slice orders"
    assert abs(broker.placed[0].quantity - 50) < 0.01, "First slice quantity should be half of total"
    assert result.filled_qty > 0, "Result should report filled quantity"


@pytest.mark.asyncio
async def test_limit_first_fills_immediately(request_obj: OrderRequest) -> None:
    """When the limit order can be filled instantly, LimitFirst should use it."""
    broker = MockBroker()
    lf = LimitFirstExecution(broker, offset_bps=5, fallback_seconds=1)
    result = await lf.execute(request_obj)

    assert broker.placed[0].order_type == "limit", "First order must be a limit order"
    assert result.status in ("filled", "partially_filled"), "Result should indicate a fill"


@pytest.mark.asyncio
async def test_limit_first_fallback_to_market(request_obj: OrderRequest) -> None:
    """If the limit order is not filled within the fallback window, a market order is placed."""
    broker = MockBrokerNoFill()
    lf = LimitFirstExecution(broker, offset_bps=5, fallback_seconds=0.1)

    # Run the execution; it should place a limit order first, wait, then place a market order.
    result = await lf.execute(request_obj)

    # Two orders should be placed: the initial limit and the fallback market order.
    assert len(broker.placed) == 2, "Both limit and fallback market orders should be placed"
    assert broker.placed[0].order_type == "limit", "First order must be a limit order"
    assert broker.placed[1].order_type == "market", "Fallback order must be a market order"
    # The final result should reflect the market order fill.
    assert result.status == "filled", "Result should be filled after fallback"
    assert result.filled_qty == request_obj.quantity, "Full quantity should be filled after fallback"


@pytest.mark.asyncio
async def test_twap_exit_logic(request_obj: OrderRequest) -> None:
    """TWAP should respect the total duration and not exceed it."""
    broker = MockBroker()
    slices = 5
    duration_minutes = 0.01  # short but measurable
    twap = TWAPExecution(broker, slices=slices, duration_minutes=duration_minutes)

    start_time = datetime.utcnow()
    result = await twap.execute(request_obj)
    end_time = datetime.utcnow()

    elapsed = (end_time - start_time).total_seconds() / 60.0
    # Allow a small tolerance due to execution overhead.
    assert elapsed <= duration_minutes * 1.1, "TWAP execution should not significantly exceed the duration"
    assert len(broker.placed) == slices, "TWAP must place the configured number of slices"
    assert result.filled_qty > 0, "Result should contain filled quantity"

"""End of test_execution.py"""