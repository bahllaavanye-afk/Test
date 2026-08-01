"""A failed bracket order must not look like a clean fill.

When Alpaca rejects a bracket, place_order() falls through and fills a PLAIN
order — the take-profit and stop-loss legs the caller asked for are silently
dropped, but the entry still executes. The strategy sized that trade assuming a
stop exists, so the result has to say the position is unprotected rather than
returning an ordinary-looking success.

NOTE: these stub the alpaca-py symbols rather than importing them. alpaca-py is
NOT in the CI dependency list, so a test gated on `ALPACA_AVAILABLE` would skip
in CI and never actually run — and the logic under test here is ours, not the
SDK's.
"""

from __future__ import annotations

import pytest

from app.brokers import alpaca as mod
from app.brokers.base import OrderRequest

# -------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------

DEFAULT_ORDER_ID = "ord-1"
DEFAULT_ORDER_STATUS = "accepted"
CALL_TYPE_BRACKET = "bracket"
CALL_TYPE_PLAIN = "plain"

BRACKET_DEGRADED_KEY = "bracket_degraded"
UNAPPLIED_STOP_LOSS_KEY = "unapplied_stop_loss"
UNAPPLIED_TAKE_PROFIT_KEY = "unapplied_take_profit"


class _Req:
    """Stands in for the SDK request objects; only bracket calls get order_class."""

    def __init__(self, **kw):
        self.__dict__.update(kw)
        self.order_class = kw.get("order_class")


class _Order:
    id = DEFAULT_ORDER_ID
    status = DEFAULT_ORDER_STATUS
    filled_qty = 0
    filled_avg_price = None


@pytest.fixture
def broker(monkeypatch):
    """AlpacaBroker with the SDK symbols and network calls stubbed out."""
    monkeypatch.setattr(mod, "ALPACA_AVAILABLE", True, raising=False)
    monkeypatch.setattr(mod, "ALPACA_BRACKET_AVAILABLE", True, raising=False)
    for name in ("MarketOrderRequest", "LimitOrderRequest", "StopOrderRequest",
                 "TakeProfitRequest", "StopLossRequest"):
        monkeypatch.setattr(mod, name, _Req, raising=False)
    monkeypatch.setattr(mod, "OrderSide", type("S", (), {"BUY": "buy", "SELL": "sell"}),
                        raising=False)
    monkeypatch.setattr(mod, "TimeInForce", type("T", (), {"GTC": "gtc", "DAY": "day"}),
                        raising=False)
    monkeypatch.setattr(mod, "OrderClass", type("C", (), {"BRACKET": "bracket"}),
                        raising=False)

    b = mod.AlpacaBroker.__new__(mod.AlpacaBroker)  # bypass __init__/credentials
    # place_order dereferences self.trading.submit_order before calling _call
    b.trading = type("T", (), {"submit_order": staticmethod(lambda **kw: None)})()
    b._calls = []

    def make_call(bracket_raises: bool):
        async def fake_call(fn, *a, **kw):
            data = kw.get("order_data")
            is_bracket = getattr(data, "order_class", None) is not None
            b._calls.append(CALL_TYPE_BRACKET if is_bracket else CALL_TYPE_PLAIN)
            if is_bracket and bracket_raises:
                raise RuntimeError("bracket rejected by Alpaca")
            return _Order()
        return fake_call

    b._make_call = make_call
    return b


@pytest.mark.asyncio
async def test_failed_bracket_is_flagged_as_unprotected(broker, monkeypatch):
    monkeypatch.setattr(broker, "_call", broker._make_call(True), raising=False)
    res = await broker.place_order(OrderRequest(
        symbol="AAPL", side="buy", quantity=1, order_type="market",
        stop_loss=90.0, take_profit=110.0,
    ))

    assert broker._calls == [CALL_TYPE_BRACKET, CALL_TYPE_PLAIN], "the entry must still fill"
    payload = res.raw_payload
    assert payload.get(BRACKET_DEGRADED_KEY) is True, (
        "a dropped stop-loss must be visible to callers, not look like a clean fill"
    )
    assert payload.get(UNAPPLIED_STOP_LOSS_KEY) == 90.0
    assert payload.get(UNAPPLIED_TAKE_PROFIT_KEY) == 110.0


@pytest.mark.asyncio
async def test_successful_bracket_is_not_flagged(broker, monkeypatch):
    monkeypatch.setattr(broker, "_call", broker._make_call(False), raising=False)
    res = await broker.place_order(OrderRequest(
        symbol="AAPL", side="buy", quantity=1, order_type="market",
        stop_loss=90.0, take_profit=110.0,
    ))
    assert broker._calls == [CALL_TYPE_BRACKET]
    assert BRACKET_DEGRADED_KEY not in res.raw_payload


@pytest.mark.asyncio
async def test_plain_order_without_bracket_is_not_flagged(broker, monkeypatch):
    """A genuinely unbracketed order must not be mislabelled as degraded."""
    monkeypatch.setattr(broker, "_call", broker._make_call(False), raising=False)
    res = await broker.place_order(OrderRequest(
        symbol="AAPL", side="buy", quantity=1, order_type="market",
    ))
    assert broker._calls == [CALL_TYPE_PLAIN]
    assert BRACKET_DEGRADED_KEY not in res.raw_payload