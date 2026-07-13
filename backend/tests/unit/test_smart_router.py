"""Smart order router tests."""
import pytest
from app.brokers.base import OrderRequest
from app.execution.smart_router import SmartOrderRouter


class DummyBroker:
    async def place_order(self, req):
        from app.brokers.base import OrderResult
        return OrderResult(
            broker_order_id="d",
            status="filled",
            filled_qty=req.quantity,
            avg_fill_price=100.0,
        )

    async def get_quote(self, symbol):
        from app.brokers.base import QuoteResult
        return QuoteResult(
            symbol=symbol,
            bid=99.95,
            ask=100.05,
            last=100.0,
            volume=100,
        )


class LowVolumeBroker(DummyBroker):
    async def get_quote(self, symbol):
        from app.brokers.base import QuoteResult
        # Simulate low market depth to trigger confirmation filter
        return QuoteResult(
            symbol=symbol,
            bid=99.95,
            ask=100.05,
            last=100.0,
            volume=10,
        )


def _req(quantity=10, limit_price=None, order_type="market", algo="auto"):
    return OrderRequest(
        account_id="a",
        symbol="AAPL",
        side="buy",
        order_type=order_type,
        quantity=quantity,
        limit_price=limit_price,
        stop_price=None,
        time_in_force="GTC",
        execution_algo=algo,
    )


def test_large_order_picks_rl_or_twap():
    """Large orders use rl_exec when available, twap as fallback."""
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


def test_limit_price_out_of_spread_falls_back():
    """Limit orders outside the current bid‑ask spread should not use limit_first."""
    router = SmartOrderRouter(DummyBroker())
    # Set limit far away from market (e.g., 90 when spread is ~99.95‑100.05)
    req = _req(quantity=5, limit_price=90, order_type="limit")
    algo = router._select_algorithm(req)
    # Expect fallback to market or another algorithm, not limit_first
    assert algo != "limit_first"


def test_confirmation_filter_requires_sufficient_volume():
    """When market depth is low, the router should avoid aggressive limit algorithms."""
    router = SmartOrderRouter(LowVolumeBroker())
    req = _req(quantity=5, limit_price=99, order_type="limit")
    algo = router._select_algorithm(req)
    # With low volume, algorithm should fallback to a more conservative choice
    assert algo in ("market", "twap", "almgren_chriss")
    assert algo != "limit_first"


def test_mid_size_order_uses_almgren_chriss():
    """Orders around the $10k threshold should select the Almgren‑Chriss algorithm."""
    router = SmartOrderRouter(DummyBroker())
    # $10k / $100 = 100 shares; use slightly below threshold to stay in mid‑size range
    req = _req(quantity=95, limit_price=100)
    algo = router._select_algorithm(req)
    assert algo == "almgren_chriss"


def test_exit_logic_on_partial_fill():
    """Router should correctly handle partial fills and trigger appropriate exit logic."""
    class PartialFillBroker(DummyBroker):
        async def place_order(self, req):
            from app.brokers.base import OrderResult
            # Simulate a partial fill (50% filled)
            return OrderResult(
                broker_order_id="p",
                status="partial",
                filled_qty=req.quantity // 2,
                avg_fill_price=100.0,
            )

    router = SmartOrderRouter(PartialFillBroker())
    req = _req(quantity=10)
    # Execute the order; the router should process the partial result
    result = pytest.run(asyncio.run(router.execute_order(req)))  # type: ignore
    # Verify that the router marks the order as needing further action (e.g., continue execution)
    assert result.status == "partial"
    assert result.filled_qty == 5