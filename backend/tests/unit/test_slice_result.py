"""A sliced execution that filled nothing must not report a partial fill.

TWAP, VWAP, iceberg and Almgren-Chriss all ended with the same construction:

    broker_order_id=last_result.broker_order_id if last_result else "vwap",
    status="filled" if total_filled >= request.quantity * 0.95 else "partial",

so a run where every slice failed came back as `status="partial"`,
`filled_qty=0`, and a broker_order_id of `"vwap"` — an id no broker ever
issued. `strategy_runner` then took that as a submitted order (`if result:` is
always true for a dataclass) and wrote a Redis exit config for a position that
does not exist.

NOTE: log assertions use `capsys`, not `caplog` — structlog does not propagate
to stdlib logging, so caplog would pass for the wrong reason.
"""
from __future__ import annotations

import pytest
from functools import lru_cache

from app.brokers.base import OrderRequest, OrderResult
from app.execution.slice_result import build_slice_result


@lru_cache(maxsize=None)
def _req(qty: float = 100.0) -> OrderRequest:
    """Create a reusable OrderRequest; cached to avoid repeated object construction."""
    return OrderRequest(
        account_id="test",
        symbol="AAPL",
        side="buy",
        order_type="market",
        quantity=qty,
        limit_price=None,
        stop_price=None,
        time_in_force="GTC",
        execution_algo="market",
        risk_bucket="directional",
    )


class _Broker:
    """Fails the first `fail_first` slices, then fills. `fail_all` fails every one."""

    def __init__(self, *, fail_all: bool = False, fail_first: int = 0,
                 fill_price: float = 100.0, fill_ratio: float = 1.0):
        self.fail_all = fail_all
        self.fail_first = fail_first
        self.fill_price = fill_price
        self.fill_ratio = fill_ratio
        self.calls = 0

    async def place_order(self, request):
        self.calls += 1
        if self.fail_all or self.calls <= self.fail_first:
            raise ConnectionError("broker unreachable")
        return OrderResult(
            broker_order_id=f"ord-{self.calls}",
            status="filled",
            filled_qty=request.quantity * self.fill_ratio,
            avg_fill_price=self.fill_price,
        )

    async def get_bars(self, symbol, timeframe: str = "30Min", limit: int = 13):
        raise RuntimeError("no bars")      # forces the empirical VWAP profile


# ── the helper itself ────────────────────────────────────────────────────────

def test_zero_fill_is_rejected_not_partial(capsys):
    res = build_slice_result(
        "TWAP",
        _req(),
        total_filled=0.0,
        total_cost=0.0,
        last_result=None,
        slices_attempted=10,
        slices_failed=10,
        last_error="broker unreachable",
    )

    assert res.status == "rejected", "nothing filled is not a partial fill"
    assert res.filled_qty == 0
    assert res.broker_order_id == "", "must not invent a broker order id"
    assert res.raw_payload["slices_failed"] == 10
    assert res.raw_payload["last_error"] == "broker unreachable"
    assert "filled NOTHING" in capsys.readouterr().out


def test_complete_fill_is_filled():
    last = OrderResult(broker_order_id="ord-9", status="filled", filled_qty=10)
    res = build_slice_result(
        "TWAP",
        _req(100),
        total_filled=100.0,
        total_cost=10_000.0,
        last_result=last,
        slices_attempted=10,
    )
    assert res.status == "filled"
    assert res.broker_order_id == "ord-9"
    assert res.avg_fill_price == pytest.approx(100.0)


def test_genuine_partial_still_reports_partial():
    last = OrderResult(broker_order_id="ord-3", status="filled", filled_qty=10)
    res = build_slice_result(
        "TWAP",
        _req(100),
        total_filled=30.0,
        total_cost=3_000.0,
        last_result=last,
        slices_attempted=10,
        slices_failed=7,
    )
    assert res.status == "partial"
    assert res.filled_qty == 30.0
    assert res.raw_payload["slices_failed"] == 7


# ── each algorithm's ending ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_twap_total_failure_is_rejected():
    # Local import to avoid loading heavy modules when not needed
    from app.execution.twap import TWAPExecution

    broker = _Broker(fail_all=True)
    res = await TWAPExecution(broker, slices=10, duration_minutes=0).execute(_req())

    assert res.status == "rejected"
    assert res.broker_order_id != "twap", "no fabricated id"
    assert broker.calls == 3, "aborts after 3 consecutive failures"


@pytest.mark.asyncio
async def test_vwap_total_failure_is_rejected():
    from app.execution.vwap import VWAPExecution

    broker = _Broker(fail_all=True)
    vwap = VWAPExecution(broker)
    vwap.sleep_seconds = 0
    res = await vwap.execute(_req())

    assert res.status == "rejected"
    assert res.broker_order_id != "vwap"


@pytest.mark.asyncio
async def test_vwap_stops_hammering_a_dead_broker(capsys):
    """VWAP had no consecutive-failure abort — it worked the whole schedule."""
    from app.execution.vwap import VWAPExecution

    broker = _Broker(fail_all=True)
    vwap = VWAPExecution(broker)
    vwap.sleep_seconds = 0
    await vwap.execute(_req())

    assert broker.calls == 3, f"expected abort after 3 failures, got {broker.calls}"
    assert "aborting after consecutive failures" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_iceberg_total_failure_is_rejected():
    from app.execution.iceberg import IcebergExecution

    broker = _Broker(fail_all=True)
    ice = IcebergExecution(broker, refill_delay_seconds=0)
    res = await ice.execute(_req())

    assert res.status == "rejected"
    assert res.broker_order_id != "iceberg"


@pytest.mark.asyncio
async def test_iceberg_does_not_spin_on_zero_fill_slices(capsys):
    """filled_qty=0 leaves `remaining` untouched — the loop used to never exit."""
    from app.execution.iceberg import IcebergExecution

    broker = _Broker(fill_ratio=0.0)
    ice = IcebergExecution(broker, refill_delay_seconds=0)

    res = await ice.execute(_req())

    assert broker.calls == 1, "must stop once a slice makes no progress"
    assert res.status == "rejected"
    assert "unbounded refill loop" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_partial_run_keeps_the_real_order_id():
    """Some slices fail, some fill — the id must be a real one, status partial."""
    from app.execution.twap import TWAPExecution

    broker = _Broker(fail_first=2)
    res = await TWAPExecution(broker, slices=10, duration_minutes=0).execute(_req(100))

    assert res.status == "partial"
    assert res.broker_order_id.startswith("ord-")
    assert res.filled_qty == pytest.approx(80.0)   # 8 of 10 slices at 10 each
    assert res.raw_payload["slices_failed"] == 2