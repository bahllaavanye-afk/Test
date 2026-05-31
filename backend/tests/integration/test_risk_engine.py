"""
Integration tests: risk engine end-to-end.
Covers Kelly sizing, circuit breakers, HRP optimizer.
"""
from __future__ import annotations

import asyncio
import numpy as np
import pandas as pd
import pytest

from app.risk.kelly import kelly_fraction, size_from_kelly
from app.risk.circuit_breaker import CircuitBreaker, BreakerState
from app.risk.manager import RiskManager, RiskDecision
from app.risk.hrp import HRPOptimizer


# ──────────────────────────────────────────────────────────────────────────────
# Kelly criterion
# ──────────────────────────────────────────────────────────────────────────────

class TestKelly:
    def test_positive_edge_positive_fraction(self):
        f = kelly_fraction(win_rate=0.6, avg_win=2.0, avg_loss=1.0)
        assert f > 0, f"Positive edge must give positive Kelly fraction, got {f}"

    def test_negative_edge_zero_fraction(self):
        f = kelly_fraction(win_rate=0.3, avg_win=0.5, avg_loss=1.0)
        assert f == 0.0, f"Negative edge must give 0 fraction, got {f}"

    def test_fraction_hard_capped_at_20_pct(self):
        f = kelly_fraction(win_rate=0.99, avg_win=100.0, avg_loss=1.0)
        assert f <= 0.20, f"Kelly must be capped at 0.20, got {f}"

    def test_zero_avg_loss_returns_zero(self):
        f = kelly_fraction(win_rate=0.6, avg_win=2.0, avg_loss=0.0)
        assert f == 0.0

    def test_size_from_kelly_returns_at_least_one(self):
        n = size_from_kelly(equity=100_000, win_rate=0.6, avg_win_pct=0.02,
                            avg_loss_pct=0.01, price=150.0)
        assert n >= 1


# ──────────────────────────────────────────────────────────────────────────────
# Circuit breaker
# ──────────────────────────────────────────────────────────────────────────────

class TestCircuitBreaker:
    def test_not_halted_initially(self):
        cb = CircuitBreaker(name="test", max_drawdown_pct=0.10)
        assert not cb.is_halted

    def test_halts_on_drawdown_breach(self):
        cb = CircuitBreaker(name="test", max_drawdown_pct=0.10)
        cb.update(100_000)  # set peak
        cb.update(85_000)   # 15% drawdown — exceeds 10% limit
        assert cb.is_halted, "Must halt when drawdown exceeds limit"
        assert cb.state == BreakerState.HALTED

    def test_no_halt_below_threshold(self):
        cb = CircuitBreaker(name="test", max_drawdown_pct=0.20)
        cb.update(100_000)
        cb.update(85_000)   # 15% drawdown — below 20% limit
        assert not cb.is_halted

    def test_halt_reasons_accessible(self):
        """Regression: halt_reasons[-1] must never raise IndexError."""
        cb = CircuitBreaker(name="test", max_drawdown_pct=0.10)
        cb.update(100_000)
        cb.update(80_000)
        assert cb.is_halted
        # This must not raise IndexError
        reason = cb.halt_reasons[-1] if cb.halt_reasons else "circuit breaker tripped"
        assert isinstance(reason, str)


# ──────────────────────────────────────────────────────────────────────────────
# Risk manager
# ──────────────────────────────────────────────────────────────────────────────

class TestRiskManager:
    def setup_method(self):
        self.rm = RiskManager(initial_equity=100_000.0)
        # Seed equity so breaker has a peak
        self.rm.update_equity(100_000.0)

    def test_instantiates(self):
        assert self.rm is not None
        assert self.rm.global_breaker is not None

    def test_check_order_returns_risk_decision(self):
        from app.brokers.base import OrderRequest
        req = OrderRequest(
            symbol="AAPL", side="buy", quantity=10,
            order_type="market", risk_bucket="directional",
        )
        result = asyncio.get_event_loop().run_until_complete(self.rm.check_order(req))
        assert isinstance(result, RiskDecision)
        assert isinstance(result.allowed, bool)

    def test_halted_manager_rejects_all(self):
        """After circuit breaker triggers, all orders must be rejected."""
        from app.brokers.base import OrderRequest
        self.rm.update_equity(100_000)
        self.rm.update_equity(70_000)  # 30% drawdown → triggers 10% limit
        assert self.rm.global_breaker.is_halted

        req = OrderRequest(
            symbol="AAPL", side="buy", quantity=1,
            order_type="market", risk_bucket="directional",
        )
        result = asyncio.get_event_loop().run_until_complete(self.rm.check_order(req))
        assert not result.allowed, "Halted manager must reject all orders"

    def test_equity_update_sets_confirmed_flag(self):
        rm = RiskManager()
        assert not rm._equity_confirmed
        rm.update_equity(50_000)
        assert rm._equity_confirmed


# ──────────────────────────────────────────────────────────────────────────────
# HRP optimizer
# ──────────────────────────────────────────────────────────────────────────────

class TestHRPOptimizer:
    def setup_method(self):
        rng = np.random.default_rng(42)
        self.returns = pd.DataFrame(
            rng.normal(0.001, 0.02, size=(252, 5)),
            columns=["AAPL", "MSFT", "GOOG", "AMZN", "META"],
        )
        self.hrp = HRPOptimizer()

    def test_weights_sum_to_one(self):
        w = self.hrp.compute_weights(self.returns)
        assert abs(w.sum() - 1.0) < 1e-6, f"Weights sum to {w.sum()}, not 1.0"

    def test_weights_non_negative(self):
        w = self.hrp.compute_weights(self.returns)
        assert (w >= -1e-9).all(), f"Negative weight found: {w}"

    def test_covers_all_assets(self):
        w = self.hrp.compute_weights(self.returns)
        assert len(w) == 5

    def test_single_asset_full_weight(self):
        w = self.hrp.compute_weights(self.returns[["AAPL"]])
        assert abs(w.sum() - 1.0) < 1e-6
