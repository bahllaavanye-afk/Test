"""Risk gate tests — proves RiskManager.check_order() actually BLOCKS bad orders.

The safety-critical gate (every order passes through it before a broker) had no
direct test (57% coverage was just imports). These exercise each block/allow
branch: circuit-breaker halts, zero-equity halt, position-size cap, and the
normal allow path. Pure/offline.
"""
from __future__ import annotations

import pytest

from app.brokers.base import OrderRequest
from app.risk.manager import RiskManager

# ----------------------------------------------------------------------
# Constants – extracted from magic numbers / hard‑coded strings
# ----------------------------------------------------------------------
DEFAULT_INITIAL_EQUITY: int = 100_000
DEFAULT_MAX_DRAWDOWN_PCT: float = 0.10
DEFAULT_ARB_DRAWDOWN_PCT: float = 0.05
DEFAULT_MAX_POSITION_PCT: float = 0.05
DEFAULT_MAX_CLUSTER_PCT: float = 0.30
DEFAULT_MAX_POSITION_PCT_CLUSTER: float = 0.50

ORDER_SIDE: str = "buy"
ORDER_TYPE: str = "limit"

BUCKET_DIRECTIONAL: str = "directional"
BUCKET_ARBITRAGE: str = "arbitrage"

REASON_OK: str = "ok"
REASON_SIZE_CAPPED: str = "size capped"
REASON_CIRCUIT_BREAKER: str = "circuit breaker halted"

# ----------------------------------------------------------------------


def _order(qty: float = 10.0, price: float = 100.0, bucket: str = BUCKET_DIRECTIONAL, symbol: str = "AAPL") -> OrderRequest:
    return OrderRequest(
        symbol=symbol,
        side=ORDER_SIDE,
        order_type=ORDER_TYPE,
        quantity=qty,
        limit_price=price,
        risk_bucket=bucket,
    )


@pytest.mark.asyncio
async def test_normal_order_allowed():
    rm = RiskManager(initial_equity=DEFAULT_INITIAL_EQUITY)
    rm.update_equity(DEFAULT_INITIAL_EQUITY)
    d = await rm.check_order(_order(qty=10, price=100))  # $1k << 5% of 100k
    assert d.allowed and d.reason == REASON_OK


@pytest.mark.asyncio
async def test_global_breaker_halt_blocks_all_orders():
    rm = RiskManager(max_drawdown_pct=DEFAULT_MAX_DRAWDOWN_PCT, initial_equity=DEFAULT_INITIAL_EQUITY)
    rm.update_equity(DEFAULT_INITIAL_EQUITY)   # set peak
    rm.update_equity(85_000)    # -15% drawdown > 10% → halt
    assert rm.global_breaker.is_halted
    d = await rm.check_order(_order())
    assert not d.allowed
    assert REASON_CIRCUIT_BREAKER in d.reason.lower()


@pytest.mark.asyncio
async def test_arb_breaker_halts_only_arbitrage_bucket():
    rm = RiskManager(arb_drawdown_pct=DEFAULT_ARB_DRAWDOWN_PCT, initial_equity=DEFAULT_INITIAL_EQUITY)
    rm.update_equity(DEFAULT_INITIAL_EQUITY)
    rm.arb_breaker.update(DEFAULT_INITIAL_EQUITY)
    rm.arb_breaker.update(90_000)   # -10% > 5% → arb halt
    assert rm.arb_breaker.is_halted
    # arbitrage order blocked
    assert not (await rm.check_order(_order(bucket=BUCKET_ARBITRAGE))).allowed
    # directional order still allowed (global breaker not halted)
    assert (await rm.check_order(_order(bucket=BUCKET_DIRECTIONAL))).allowed


@pytest.mark.asyncio
async def test_zero_or_negative_equity_halts():
    rm = RiskManager(initial_equity=DEFAULT_INITIAL_EQUITY)
    rm.update_equity(0.0)
    d = await rm.check_order(_order())
    assert not d.allowed and "equity" in d.reason.lower()


@pytest.mark.asyncio
async def test_position_size_cap_adjusts_quantity():
    rm = RiskManager(max_position_pct=DEFAULT_MAX_POSITION_PCT, initial_equity=DEFAULT_INITIAL_EQUITY)
    rm.update_equity(DEFAULT_INITIAL_EQUITY)
    # 100 shares @ $100 = $10k > 5% ($5k) → must be capped to ~50 shares
    d = await rm.check_order(_order(qty=100, price=100))
    assert d.allowed and d.reason == REASON_SIZE_CAPPED
    assert d.adjusted_quantity == pytest.approx(50.0, rel=0.01)


@pytest.mark.asyncio
async def test_unconfirmed_equity_still_gates_on_estimate():
    # Before a broker snapshot, it uses the conservative seed but still enforces caps.
    rm = RiskManager(max_position_pct=DEFAULT_MAX_POSITION_PCT, initial_equity=DEFAULT_INITIAL_EQUITY)
    assert not rm._equity_confirmed
    d = await rm.check_order(_order(qty=10, price=100))
    assert d.allowed  # within the seed equity's cap


@pytest.mark.asyncio
async def test_correlation_cluster_limit_blocks_overconcentration():
    rm = RiskManager(
        max_position_pct=DEFAULT_MAX_POSITION_PCT,
        max_cluster_pct=DEFAULT_MAX_CLUSTER_PCT,
        initial_equity=DEFAULT_INITIAL_EQUITY,
    )
    rm.update_equity(DEFAULT_INITIAL_EQUITY)
    # Tech cluster already holds $28k; a new $5k AAPL order pushes the cluster
    # past 30% of NAV ($30k) → must be blocked.
    rm._clusters = {"tech": ["AAPL", "MSFT", "NVDA"]}
    rm.update_positions([{"symbol": "MSFT", "market_value": 28_000}])
    d = await rm.check_order(_order(qty=50, price=100, symbol="AAPL"))  # $5k
    assert not d.allowed


def test_update_positions_and_kelly_size():
    rm = RiskManager(max_position_pct=DEFAULT_MAX_POSITION_PCT, initial_equity=DEFAULT_INITIAL_EQUITY)
    rm.update_positions([{"symbol": "AAPL", "market_value": 5000}, {"symbol": "MSFT", "market_value": 3000}])
    assert rm._positions == {"AAPL": 5000.0, "MSFT": 3000.0}
    # Favourable edge → positive, capped size
    size = rm.kelly_size("AAPL", price=100.0, win_rate=0.6, avg_win_pct=0.05, avg_loss_pct=0.03)
    assert isinstance(size, int) and size >= 0


def test_update_returns_builds_correlation_clusters():
    # Feed ≥20 rows so update_returns recomputes clusters; AAPL/MSFT move together
    # (corr ~1.0) while TLT is anti-correlated → AAPL+MSFT form one cluster.
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    base = rng.normal(0, 0.01, 40)
    df = pd.DataFrame({
        "AAPL": base,
        "MSFT": base + rng.normal(0, 0.0001, 40),  # nearly identical → high corr
        "TLT": -base,                               # inverse
    })
    rm = RiskManager(initial_equity=DEFAULT_INITIAL_EQUITY)
    rm.update_returns(df)
    # AAPL and MSFT must land in the same cluster.
    members = [set(v) for v in rm._clusters.values()]
    assert any({"AAPL", "MSFT"} <= m for m in members)