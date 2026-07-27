"""SmartOrderRouter result construction.

The RL branch aggregated its fills into
`OrderResult(order_id=..., symbol=..., ...)` — neither is a field on that
dataclass, so every successful RL execution raised TypeError *after* the fills
had already happened at the broker. The caller saw a crash for an order that
had in fact executed.
"""
from __future__ import annotations

import asyncio

import pytest

from app.brokers.base import OrderRequest, OrderResult
from app.execution import smart_router as mod


def _req(qty=10.0, algo="auto"):
    return OrderRequest(
        account_id="test", symbol="AAPL", side="buy", order_type="market",
        quantity=qty, limit_price=100.0, stop_price=None, time_in_force="GTC",
        execution_algo=algo, risk_bucket="directional",
    )


class _Broker:
    def __init__(self, *, fail_all=False):
        self.fail_all = fail_all
        self.calls = 0

    async def place_order(self, request):
        self.calls += 1
        if self.fail_all:
            raise ConnectionError("broker unreachable")
        return OrderResult(broker_order_id=f"ord-{self.calls}", status="filled",
                           filled_qty=request.quantity, avg_fill_price=100.0)


@pytest.mark.asyncio
async def test_rl_aggregation_does_not_raise(monkeypatch):
    """order_id=/symbol= are not OrderResult fields — this used to TypeError."""
    fills = [{"qty": 6.0, "price": 100.0}, {"qty": 4.0, "price": 105.0}]

    class _RL:
        def __init__(self, broker, agent=None):
            pass

        async def execute(self, request, signal_price=None):
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
async def test_rl_with_no_fills_returns_none(monkeypatch):
    class _RL:
        def __init__(self, broker, agent=None):
            pass

        async def execute(self, request, signal_price=None):
            return []

    monkeypatch.setattr(mod, "_RL_EXEC_AVAILABLE", True, raising=False)
    monkeypatch.setattr(mod, "RLExecution", _RL, raising=False)
    monkeypatch.setattr(mod, "get_rl_agent", lambda: object(), raising=False)

    router = mod.SmartOrderRouter(broker=_Broker())
    assert await router.execute(_req(algo="rl_exec")) is None


@pytest.mark.asyncio
async def test_almgren_chriss_total_failure_is_rejected(monkeypatch):
    broker = _Broker(fail_all=True)
    router = mod.SmartOrderRouter(broker=broker)
    # _execute_almgren_chriss imports asyncio inside the method, so patch the
    # module itself — the inter-slice sleep is 120s of real time otherwise.
    monkeypatch.setattr(asyncio, "sleep", _noop)

    res = await router.execute(_req(qty=100.0, algo="almgren_chriss"))

    assert res.status == "rejected", "nothing filled is not a partial fill"
    assert res.broker_order_id != "ac_exec", "no fabricated id"
    assert res.raw_payload["slices_failed"] >= 3


async def _noop(*_args, **_kwargs):
    return None
