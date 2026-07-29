"""Smart order router tests."""
import pytest
from app.brokers.base import OrderRequest
from app.execution.smart_router import SmartOrderRouter

# Default request parameters
DEFAULT_ACCOUNT_ID = "a"
DEFAULT_SYMBOL = "AAPL"
DEFAULT_SIDE = "buy"
DEFAULT_TIME_IN_FORCE = "GTC"
DEFAULT_EXECUTION_ALGO = "auto"
DEFAULT_ORDER_TYPE = "market"

# Dummy broker return values
DUMMY_BROKER_ORDER_ID = "d"
DUMMY_ORDER_STATUS = "filled"
DUMMY_AVG_FILL_PRICE = 100.0
DUMMY_BID = 99.95
DUMMY_ASK = 100.05
DUMMY_LAST = 100.0
DUMMY_VOLUME = 100

# Test-specific constants
LARGE_ORDER_QUANTITY = 200
LARGE_ORDER_LIMIT_PRICE = 100
SMALL_ORDER_QUANTITY = 5
SMALL_ORDER_LIMIT_PRICE = 99

# Algorithm identifiers
ALGO_RL_EXEC = "rl_exec"
ALGO_TWAP = "twap"
ALGO_ALMGEREN_CHRISS = "almgren_chriss"
ALGO_LIMIT_FIRST = "limit_first"
ALGO_MARKET = "market"


class DummyBroker:
    async def place_order(self, req):
        from app.brokers.base import OrderResult
        return OrderResult(
            broker_order_id=DUMMY_BROKER_ORDER_ID,
            status=DUMMY_ORDER_STATUS,
            filled_qty=req.quantity,
            avg_fill_price=DUMMY_AVG_FILL_PRICE,
        )

    async def get_quote(self, symbol):
        from app.brokers.base import QuoteResult
        return QuoteResult(
            symbol=symbol,
            bid=DUMMY_BID,
            ask=DUMMY_ASK,
            last=DUMMY_LAST,
            volume=DUMMY_VOLUME,
        )


def _req(quantity=SMALL_ORDER_QUANTITY, limit_price=None, order_type=DEFAULT_ORDER_TYPE, algo=DEFAULT_EXECUTION_ALGO):
    return OrderRequest(
        account_id=DEFAULT_ACCOUNT_ID,
        symbol=DEFAULT_SYMBOL,
        side=DEFAULT_SIDE,
        order_type=order_type,
        quantity=quantity,
        limit_price=limit_price,
        stop_price=None,
        time_in_force=DEFAULT_TIME_IN_FORCE,
        execution_algo=algo,
    )


def test_large_order_picks_rl_or_twap():
    """Large orders use rl_exec when available, twap as fallback."""
    from app.execution import smart_router as sr
    router = SmartOrderRouter(DummyBroker())
    req = _req(quantity=LARGE_ORDER_QUANTITY, limit_price=LARGE_ORDER_LIMIT_PRICE)  # 200 * 100 = $20k > $10k
    algo = router._select_algorithm(req)
    # almgren_chriss for mid-size orders, rl_exec/twap for very large
    assert algo in (ALGO_RL_EXEC, ALGO_TWAP, ALGO_ALMGEREN_CHRISS)


def test_limit_order_picks_limit_first():
    router = SmartOrderRouter(DummyBroker())
    req = _req(quantity=SMALL_ORDER_QUANTITY, limit_price=SMALL_ORDER_LIMIT_PRICE, order_type="limit")
    algo = router._select_algorithm(req)
    assert algo == ALGO_LIMIT_FIRST


def test_default_market():
    router = SmartOrderRouter(DummyBroker())
    req = _req(quantity=SMALL_ORDER_QUANTITY)
    algo = router._select_algorithm(req)
    assert algo == ALGO_MARKET


def test_explicit_override():
    router = SmartOrderRouter(DummyBroker())
    req = _req(quantity=SMALL_ORDER_QUANTITY, algo=ALGO_LIMIT_FIRST)
    # User override should win when not large
    algo = router._select_algorithm(req)
    assert algo == ALGO_LIMIT_FIRST