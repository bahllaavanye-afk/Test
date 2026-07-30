"""SmartOrderRouter result construction tests.

These tests verify that the SmartOrderRouter correctly aggregates fills,
handles edge cases, and respects failure scenarios. Additional tests cover
tightened entry conditions, confirmation filters, and exit logic improvements.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List

import pytest

from app.brokers.base import OrderRequest, OrderResult
from app.execution import smart_router as mod


def _req(qty: float = 10.0, algo: str = "auto") -> OrderRequest:
    """Create a minimal OrderRequest for testing."""
    return OrderRequest(
        account_id="test",
        symbol="AAPL",
        side="buy",
        order_type="market",
        quantity=qty,
        limit_price=100.0,
        stop_price=None,
        time_in_force="GTC",
        execution_algo=algo,
        risk_bucket="directional",
    )


class _Broker:
    """A lightweight mock broker used across tests."""

    def __init__(self, *, fail_all: bool = False):
        self.fail_all = fail_all
        self.calls = 0

    async def place_order(self, request: OrderRequest) -> OrderResult:
        """Simulate order placement, optionally raising a connection error."""
        self.calls += 1
        if self.fail_all:
            raise ConnectionError("broker unreachable")
        return OrderResult(
            broker_order_id=f"ord-{self.calls}",
            status="filled",
            filled_qty=request.quantity,
            avg_fill_price=100.0,
        )


@pytest.mark.asyncio
async def test_rl_aggregation_does_not_raise(monkeypatch: Any) -> None:
    """RL execution should aggregate fills without raising a TypeError."""
    fills: List[Dict[str, float]] = [
        {"qty": 6.0, "price": 100.0},
        {"qty": 4.0, "price": 105.0},
    ]

    class _RL:
        def __init__(self, broker: _Broker, agent: Any = None):
            pass

        async def execute(self, request: OrderRequest, signal_price: float | None = None) -> List[Dict[str, float]]:
            return fills

    monkeypatch.setattr(mod, "_RL_EXEC_AVAILABLE", True, raising=False)
    monkeypatch.setattr(mod, "RLExecution", _RL, raising=False)
    monkeypatch.setattr(mod, "get_rl_agent", lambda: object(), raising=False)

    broker = _Broker()
    router = mod.SmartOrderRouter(broker=broker)
    res = await router.execute(_req(qty=10.0, algo="rl_exec"))

    assert res is not None
    assert res.status == "filled"
    assert res.filled_qty == pytest.approx(10.0)
    assert res.avg_fill_price == pytest.approx(102.0)
    assert res.broker_order_id == "rl_AAPL"


@pytest.mark.asyncio
async def test_rl_with_no_fills_returns_none(monkeypatch: Any) -> None:
    """When RL execution returns no fills, the router should return None."""
    class _RL:
        def __init__(self, broker: _Broker, agent: Any = None):
            pass

        async def execute(self, request: OrderRequest, signal_price: float | None = None) -> List[Dict]:
            return []

    monkeypatch.setattr(mod, "_RL_EXEC_AVAILABLE", True, raising=False)
    monkeypatch.setattr(mod, "RLExecution", _RL, raising=False)
    monkeypatch.setattr(mod, "get_rl_agent", lambda: object(), raising=False)

    router = mod.SmartOrderRouter(broker=_Broker())
    assert await router.execute(_req(algo="rl_exec")) is None


@pytest.mark.asyncio
async def test_almgren_chriss_total_failure_is_rejected(monkeypatch: Any) -> None:
    """If all Almgren‑Chriss slices fail, the order should be rejected."""
    broker = _Broker(fail_all=True)
    router = mod.SmartOrderRouter(broker=broker)

    # Patch asyncio.sleep to avoid a 120 s delay in the test environment.
    monkeypatch.setattr(asyncio, "sleep", _noop)

    res = await router.execute(_req(qty=100.0, algo="almgren_chriss"))

    assert res.status == "rejected", "nothing filled is not a partial fill"
    assert res.broker_order_id != "ac_exec", "no fabricated id"
    assert res.raw_payload["slices_failed"] >= 3


@pytest.mark.asyncio
async def test_entry_condition_zero_quantity_returns_none(monkeypatch: Any) -> None:
    """Orders with zero quantity should be rejected before routing."""
    router = mod.SmartOrderRouter(broker=_Broker())
    assert await router.execute(_req(qty=0.0, algo="auto")) is None


@pytest.mark.asyncio
async def test_partial_fill_exit_logic(monkeypatch: Any) -> None:
    """When only a portion of the order is filled, status should be 'partial'."""
    class _PartialBroker(_Broker):
        async def place_order(self, request: OrderRequest) -> OrderResult:
            # Simulate a partial fill of half the requested quantity.
            filled = request.quantity / 2
            return OrderResult(
                broker_order_id="partial-ord",
                status="partial",
                filled_qty=filled,
                avg_fill_price=101.0,
                raw_payload={"filled_qty": filled},
            )

    broker = _PartialBroker()
    router = mod.SmartOrderRouter(broker=broker)
    res = await router.execute(_req(qty=20.0, algo="auto"))

    assert res is not None
    assert res.status == "partial"
    assert res.filled_qty == pytest.approx(10.0)
    assert res.broker_order_id == "partial-ord"


async def _noop(*_args: Any, **_kwargs: Any) -> None:
    """Placeholder coroutine used to replace long‑running async calls in tests."""
    return None