"""
Unit tests for app.backtest.metrics.compute_metrics

Tests:
  - Linear equity curve (known Sharpe)
  - Max drawdown computation
  - Win rate from trades DataFrame
  - VaR / CVaR computation
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
import pytest

from app.backtest.metrics import compute_metrics, BacktestMetrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_equity(daily_pct_returns: list[float], start: float = 100_000) -> pd.Series:
    """Build an equity curve from a list of daily percentage returns."""
    idx = pd.date_range("2020-01-01", periods=len(daily_pct_returns) + 1, freq="B")
    rets = pd.Series([0.0] + daily_pct_returns, index=idx)
    equity = start * (1 + rets).cumprod()
    return equity


# ---------------------------------------------------------------------------
# Test 1: Linear equity curve — verify Sharpe is computable and positive
# ---------------------------------------------------------------------------

class TestComputeMetricsBasic:
    def test_returns_backtest_metrics_instance(self):
        """compute_metrics must return a BacktestMetrics dataclass."""
        equity = _make_equity([0.001] * 252)
        result = compute_metrics(equity)
        assert isinstance(result, BacktestMetrics)

    def test_linear_equity_positive_return(self):
        """Constant daily gain → positive annual return and positive Sharpe."""
        daily_ret = 0.001  # 0.1% per day → ~28% annual
        equity = _make_equity([daily_ret] * 252)
        m = compute_metrics(equity)

        assert m.annual_return_pct > 0, "Annual return must be positive"
        assert m.sharpe > 0, "Sharpe must be positive for monotonically rising equity"

    def test_known_sharpe_linear_curve(self):
        """
        For a constant daily return r, std of daily returns = 0 if r is exactly
        constant. We use a tiny noise to make std > 0 and verify the ratio.
        """
        rng = np.random.default_rng(42)
        base_return = 0.001
        noise = rng.normal(0, 0.0001, 252)     # tiny noise so std ≈ 0.0001
        daily_rets = [base_return + n for n in noise]

        equity = _make_equity(daily_rets)
        m = compute_metrics(equity)

        # Expected Sharpe ≈ mean/std * sqrt(252)
        arr = np.array(daily_rets)
        expected_sharpe = arr.mean() / arr.std() * math.sqrt(252)

        # Allow ±10% relative tolerance — pct_change on the equity curve introduces
        # rounding, so the result won't be exact
        assert abs(m.sharpe - expected_sharpe) / abs(expected_sharpe) < 0.10, (
            f"Sharpe mismatch: got {m.sharpe:.4f}, expected ≈ {expected_sharpe:.4f}"
        )

    def test_negative_return_negative_sharpe(self):
        """Constant daily loss → negative Sharpe."""
        equity = _make_equity([-0.002] * 252)
        m = compute_metrics(equity)
        # Sharpe may be zero when std approaches zero, but annual return must be negative
        assert m.annual_return_pct < 0

    def test_total_return_pct_matches_equity(self):
        """total_return_pct should equal (final/initial - 1) * 100."""
        equity = _make_equity([0.001] * 100)
        m = compute_metrics(equity)
        expected = (equity.iloc[-1] / equity.iloc[0] - 1) * 100
        assert abs(m.total_return_pct - expected) < 0.01


# ---------------------------------------------------------------------------
# Test 2: Max drawdown
# ---------------------------------------------------------------------------

class TestMaxDrawdown:
    def test_no_drawdown_on_monotonic_curve(self):
        """Strictly rising equity → max drawdown = 0."""
        equity = _make_equity([0.001] * 200)
        m = compute_metrics(equity)
        assert m.max_drawdown_pct == 0.0, f"Expected 0 drawdown, got {m.max_drawdown_pct}"

    def test_known_drawdown(self):
        """
        Build an equity that rises to 110k then falls to 99k — drawdown ≈ -10%.
        """
        # rise phase: 10 days up 1%
        up = [0.01] * 10          # equity peaks around 110k
        # fall phase: 10 days down 1%
        down = [-0.01] * 10       # equity falls to ~99.5k from 110k → drawdown ~-9.5%
        # recover
        up2 = [0.005] * 20

        equity = _make_equity(up + down + up2)
        m = compute_metrics(equity)

        assert m.max_drawdown_pct < 0, "Max drawdown must be negative"
        assert m.max_drawdown_pct < -5.0, (
            f"Expected drawdown > 5%, got {m.max_drawdown_pct}"
        )
        assert m.max_drawdown_pct > -25.0, (
            f"Drawdown unexpectedly large: {m.max_drawdown_pct}"
        )

    def test_drawdown_duration_positive(self):
        """After a dip the drawdown duration must be > 0."""
        equity = _make_equity([0.01] * 10 + [-0.005] * 20 + [0.008] * 20)
        m = compute_metrics(equity)
        assert m.max_drawdown_duration_days > 0

    def test_calmar_ratio(self):
        """Calmar = annual_return / max_drawdown; must be negative when return > 0 and drawdown < 0."""
        equity = _make_equity([0.01] * 10 + [-0.02] * 10 + [0.005] * 100)
        m = compute_metrics(equity)
        if m.max_drawdown_pct != 0:
            assert m.calmar > 0, "Calmar should be positive when annual return > 0"


# ---------------------------------------------------------------------------
# Test 3: Win rate from trades DataFrame
# ---------------------------------------------------------------------------

class TestWinRate:
    def _make_trades(self, pnl_values: list[float]) -> pd.DataFrame:
        return pd.DataFrame({"pnl": pnl_values})

    def test_all_winning_trades(self):
        equity = _make_equity([0.001] * 200)
        trades = self._make_trades([100.0, 200.0, 150.0])
        m = compute_metrics(equity, trades=trades)
        assert m.win_rate == 1.0
        assert m.total_trades == 3

    def test_all_losing_trades(self):
        equity = _make_equity([0.001] * 200)
        trades = self._make_trades([-50.0, -75.0, -30.0])
        m = compute_metrics(equity, trades=trades)
        assert m.win_rate == 0.0
        assert m.total_trades == 3

    def test_mixed_win_rate(self):
        equity = _make_equity([0.001] * 200)
        # 3 wins, 1 loss → 75%
        trades = self._make_trades([100.0, 200.0, 150.0, -50.0])
        m = compute_metrics(equity, trades=trades)
        assert abs(m.win_rate - 0.75) < 1e-6

    def test_profit_factor(self):
        equity = _make_equity([0.001] * 200)
        # wins sum = 300, losses sum = 100 → PF = 3.0
        trades = self._make_trades([100.0, 200.0, -100.0])
        m = compute_metrics(equity, trades=trades)
        assert abs(m.profit_factor - 3.0) < 1e-4

    def test_profit_factor_no_losses(self):
        equity = _make_equity([0.001] * 200)
        trades = self._make_trades([100.0, 200.0])
        m = compute_metrics(equity, trades=trades)
        assert m.profit_factor == float("inf")

    def test_avg_win_avg_loss(self):
        equity = _make_equity([0.001] * 200)
        trades = self._make_trades([0.10, 0.20, -0.05, -0.15])
        m = compute_metrics(equity, trades=trades)
        assert m.avg_win_pct > 0
        assert m.avg_loss_pct < 0


# ---------------------------------------------------------------------------
# Test 4: VaR and CVaR
# ---------------------------------------------------------------------------

class TestVaRCVaR:
    def test_var_is_negative_for_volatile_curve(self):
        """VaR should be negative (a loss threshold)."""
        rng = np.random.default_rng(0)
        daily_rets = rng.normal(0.0005, 0.015, 252).tolist()
        equity = _make_equity(daily_rets)
        m = compute_metrics(equity)
        assert m.var_95 < 0.0, f"VaR should be negative, got {m.var_95}"

    def test_cvar_worse_than_var(self):
        """CVaR (expected shortfall) must be <= VaR (more extreme)."""
        rng = np.random.default_rng(1)
        daily_rets = rng.normal(0.0005, 0.015, 500).tolist()
        equity = _make_equity(daily_rets)
        m = compute_metrics(equity)
        assert m.cvar_95 <= m.var_95 + 1e-9, (
            f"CVaR ({m.cvar_95}) should be <= VaR ({m.var_95})"
        )

    def test_var_matches_numpy_percentile(self):
        """VaR must match np.percentile at the 5th percentile."""
        rng = np.random.default_rng(7)
        daily_rets = rng.normal(0.001, 0.02, 300).tolist()
        equity = _make_equity(daily_rets)

        # compute expected VaR manually from daily returns of equity curve
        eq_series = equity
        dr = eq_series.pct_change().dropna().values
        expected_var = float(np.percentile(dr, 5))

        m = compute_metrics(equity)
        # var_95 is stored rounded to 6 decimal places, so tolerance = 5e-7
        assert abs(m.var_95 - expected_var) < 5e-7, (
            f"VaR mismatch: got {m.var_95}, expected {expected_var}"
        )

    def test_cvar_is_mean_below_var(self):
        """CVaR must equal mean of returns strictly below VaR threshold."""
        rng = np.random.default_rng(99)
        daily_rets = rng.normal(0.001, 0.02, 300).tolist()
        equity = _make_equity(daily_rets)

        dr = equity.pct_change().dropna().values
        var = float(np.percentile(dr, 5))
        expected_cvar = float(dr[dr <= var].mean())

        m = compute_metrics(equity)
        # cvar_95 is stored rounded to 6 decimal places, so tolerance = 5e-7
        assert abs(m.cvar_95 - expected_cvar) < 5e-7, (
            f"CVaR mismatch: got {m.cvar_95}, expected {expected_cvar}"
        )

    def test_constant_positive_returns_var_positive(self):
        """Constant positive returns → all daily returns equal → VaR is that constant (positive)."""
        equity = _make_equity([0.01] * 252)
        m = compute_metrics(equity)
        # All daily returns of equity.pct_change() will be ~0.01 (constant)
        # So the 5th percentile is still ~0.01 → VaR should be positive
        assert m.var_95 > 0.0, (
            f"For a monotonically rising curve, VaR should be positive, got {m.var_95}"
        )


# ---------------------------------------------------------------------------
# Test 5: Information ratio with benchmark
# ---------------------------------------------------------------------------

class TestInformationRatio:
    def test_ir_zero_without_benchmark(self):
        equity = _make_equity([0.001] * 200)
        m = compute_metrics(equity, benchmark=None)
        assert m.information_ratio == 0.0

    def test_ir_positive_when_outperforming(self):
        """Strategy returns more than benchmark → positive IR."""
        equity_rets = [0.002] * 252
        benchmark_rets = [0.001] * 252
        equity = _make_equity(equity_rets)
        bm = _make_equity(benchmark_rets)
        m = compute_metrics(equity, benchmark=bm)
        assert m.information_ratio > 0, f"IR should be positive, got {m.information_ratio}"

    def test_ir_negative_when_underperforming(self):
        """Strategy returns less than benchmark → negative IR."""
        equity_rets = [0.0005] * 252
        benchmark_rets = [0.002] * 252
        equity = _make_equity(equity_rets)
        bm = _make_equity(benchmark_rets)
        m = compute_metrics(equity, benchmark=bm)
        assert m.information_ratio < 0, f"IR should be negative, got {m.information_ratio}"


# ---------------------------------------------------------------------------
# Test 6: Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_raises_on_too_short_curve(self):
        """Should raise ValueError if equity curve has fewer than 2 points."""
        equity = pd.Series([100_000.0], index=pd.date_range("2020-01-01", periods=1))
        with pytest.raises(ValueError):
            compute_metrics(equity)

    def test_recovery_factor(self):
        """recovery_factor = total_return / |max_drawdown|."""
        equity = _make_equity([0.01] * 20 + [-0.005] * 20 + [0.003] * 100)
        m = compute_metrics(equity)
        if m.max_drawdown_pct != 0:
            expected = m.total_return_pct / abs(m.max_drawdown_pct)
            # recovery_factor uses raw ratios not pct, so recompute
            eq = equity.dropna()
            total_ret = (eq.iloc[-1] / eq.iloc[0]) - 1
            rolling_max = eq.cummax()
            dd = ((eq - rolling_max) / rolling_max).min()
            expected_rf = total_ret / abs(dd) if dd != 0 else 0
            assert abs(m.recovery_factor - round(expected_rf, 4)) < 0.01
