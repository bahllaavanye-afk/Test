"""Execution algorithm tests using mock broker."""
import pytest
from app.brokers.base import OrderRequest, OrderResult, QuoteResult
from app.execution.twap import TWAPExecution
from app.execution.limit_first import LimitFirstExecution


class MockBroker:
    """In-memory broker that always fills orders at last quote."""
    def __init__(self, last=100.0, bid=99.95, ask=100.05):
        self.last = last
        self.bid = bid
        self.ask = ask
        self.placed: list[OrderRequest] = []
        self.cancelled: list[str] = []

    async def get_quote(self, symbol):
        return QuoteResult(symbol=symbol, bid=self.bid, ask=self.ask, last=self.last, volume=1_000_000)

    async def place_order(self, request):
        self.placed.append(request)
        return OrderResult(
            broker_order_id=f"mock-{len(self.placed)}",
            status="filled",
            filled_qty=request.quantity,
            avg_fill_price=self.last,
        )

    async def cancel_order(self, order_id):
        self.cancelled.append(order_id)
        return True

    async def get_order(self, order_id):
        return {"status": "filled", "filled_qty": 1.0}


@pytest.fixture
def request_obj():
    return OrderRequest(
        account_id="test",
        symbol="AAPL", side="buy", order_type="market",
        quantity=100, limit_price=None, stop_price=None,
        time_in_force="GTC", execution_algo="auto",
    )


@pytest.mark.asyncio
async def test_twap_slices_evenly(request_obj):
    broker = MockBroker()
    # Use 2 slices over 0.01 min to keep test fast
    twap = TWAPExecution(broker, slices=2, duration_minutes=0.001)
    result = await twap.execute(request_obj)
    assert len(broker.placed) == 2
    assert abs(broker.placed[0].quantity - 50) < 0.01
    assert result.filled_qty > 0


@pytest.mark.asyncio
async def test_limit_first_fills_immediately(request_obj):
    broker = MockBroker()
    lf = LimitFirstExecution(broker, offset_bps=5, fallback_seconds=1)
    result = await lf.execute(request_obj)
    assert broker.placed[0].order_type == "limit"
    assert result.status in ("filled", "partially_filled")
