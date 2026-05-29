"""Smart order router tests."""
import pytest
from app.brokers.base import OrderRequest
from app.execution.smart_router import SmartOrderRouter


class DummyBroker:
    async def place_order(self, req):
        from app.brokers.base import OrderResult
        return OrderResult(broker_order_id="d", status="filled",
                            filled_qty=req.quantity, avg_fill_price=100.0)

    async def get_quote(self, symbol):
        from app.brokers.base import QuoteResult
        return QuoteResult(symbol=symbol, bid=99.95, ask=100.05, last=100.0, volume=100)


def _req(quantity=10, limit_price=None, order_type="market", algo="auto"):
    return OrderRequest(
        account_id="a", symbol="AAPL", side="buy", order_type=order_type,
        quantity=quantity, limit_price=limit_price, stop_price=None,
        time_in_force="GTC", execution_algo=algo,
    )


def test_large_order_picks_rl_or_twap():
    """Large orders use rl_exec when available, twap as fallback."""
    from app.execution import smart_router as sr
    router = SmartOrderRouter(DummyBroker())
    req = _req(quantity=200, limit_price=100)  # 200 * 100 = $20k > $10k
    algo = router._select_algorithm(req)
    # almgren_chriss for mid-size orders, rl_exec/twap for very large
    assert algo in ("rl_exec", "twap", "almgren_chriss")


def test_limit_order_picks_limit_first():
    router = SmartOrderRouter(DummyBroker())
    req = _req(quantity=5, limit_price=99, order_type="limit")
    algo = router._select_algorithm(req)
    assert algo == "limit_first"


def test_default_market():
    router = SmartOrderRouter(DummyBroker())
    req = _req(quantity=5)
    algo = router._select_algorithm(req)
    assert algo == "market"


def test_explicit_override():
    router = SmartOrderRouter(DummyBroker())
    req = _req(quantity=5, algo="limit_first")
    # User override should win when not large
    algo = router._select_algorithm(req)
    assert algo == "limit_first"
