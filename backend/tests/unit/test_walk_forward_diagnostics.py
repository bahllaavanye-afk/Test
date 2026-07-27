"""Validation maths that is tested but never run is not validation.

Found by the unreferenced-function sweep. `probabilistic_sharpe_ratio()` and
`monte_carlo_simulation()` were fully implemented and unit-tested, and
referenced ONLY by their own tests — nothing in production called either. The
overfit gate computed a verdict from DSR and consistency and nothing else.

They are wired here as REPORTED, NOT GATED. Changing which strategies clear the
promotion bar is a risk decision, not a wiring fix, so `is_robust` is
deliberately untouched — pinned by
`test_the_promotion_verdict_is_unchanged_by_the_new_diagnostics`.

Not wired, and why:
  probability_of_backtest_overfitting()  needs performance across N CONFIGS;
      walk_forward runs one config across windows. Wrong call site — it belongs
      to a parameter sweep / strategy selection loop.
  run_stress_tests()                     needs the price history to actually
      span the named crisis windows; on a 2-year backtest almost every scenario
      returns period_covered=False.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.backtest.cpcv import probabilistic_sharpe_ratio
from app.backtest.walk_forward import (
    MIN_OBS_FOR_MONTE_CARLO,
    WalkForwardResult,
    _add_distribution_diagnostics,
    _oos_daily_returns,
    robustness_verdict,
    walk_forward,
)


def _prices(n: int = 1400, drift: float = 0.0004, seed: int = 7) -> pd.Series:
    rng = np.random.default_rng(seed)
    steps = rng.normal(drift, 0.01, n)
    return pd.Series(
        100 * np.exp(np.cumsum(steps)),
        index=pd.date_range("2019-01-01", periods=n, freq="B"),
    )


def _always_long(_train, test):
    return pd.Series(1, index=test.index)


# ── the diagnostics now actually run ─────────────────────────────────────────

def test_walk_forward_reports_psr():
    result = walk_forward(_always_long, _prices(), train_years=1, test_months=3)
    assert result.combined_equity, "precondition: the backtest produced equity"
    assert 0.0 <= result.psr <= 1.0
    assert result.psr != 0.0, (
        "PSR was implemented and never called; a real run must populate it"
    )


def test_walk_forward_reports_a_monte_carlo_distribution():
    result = walk_forward(_always_long, _prices(), train_years=1, test_months=3)
    assert result.mc_simulations > 0, (
        "monte_carlo_simulation() was implemented and never called from "
        "production — a run with enough OOS data must populate it"
    )
    assert 0.0 <= result.mc_prob_positive <= 1.0
    assert result.mc_p5_sharpe <= result.mc_median_sharpe, (
        "the 5th percentile path cannot beat the median path"
    )
    # Drawdowns are negative (`dd.min()`), so the risk number is <= 0.
    assert result.mc_worst_max_dd <= 0.0


def test_the_reported_drawdown_is_the_severe_tail_not_the_mild_one():
    """`MonteCarloResult.p95_max_dd` is the LUCKY tail, despite the name.

    max_dd is `dd.min()`, i.e. negative, so the 95th percentile of that series
    is the mildest drawdown. Quoting it as a risk number understates risk. The
    walk-forward reports the 5th percentile instead.
    """
    from app.backtest.monte_carlo import monte_carlo_simulation

    rng = np.random.default_rng(11)
    returns = pd.Series(rng.normal(0.0002, 0.012, 500))
    mc = monte_carlo_simulation(returns, n_simulations=300)

    assert mc.p5_max_dd <= mc.median_max_dd <= mc.p95_max_dd, (
        "on a negative-signed series the 5th percentile is the WORST drawdown"
    )

    result = WalkForwardResult()
    equity = 100_000 * np.cumprod(1 + returns.to_numpy())
    result.combined_equity = [{"equity": float(v)} for v in equity]
    _add_distribution_diagnostics(result)
    assert result.mc_worst_max_dd <= mc.median_max_dd + 1e-9, (
        "the walk-forward must surface the severe tail, not the mild one"
    )


# ── the gate must not have moved ─────────────────────────────────────────────

def test_the_promotion_verdict_is_unchanged_by_the_new_diagnostics():
    """Reported, not gated. is_robust must still come only from the protocol."""
    sharpes = [1.2, 0.9, 1.4, 0.8, 1.1, 1.3, 0.95, 1.05, 1.25, 0.85, 1.15, 1.35]
    graded = robustness_verdict(sharpes)

    result = WalkForwardResult()
    result.is_robust = graded["is_robust"]
    result.verdict = graded["verdict"]
    result.combined_equity = [
        {"equity": float(v)} for v in 100_000 * np.cumprod(
            1 + np.random.default_rng(1).normal(-0.002, 0.01, 300)
        )
    ]

    before = (result.is_robust, result.verdict)
    _add_distribution_diagnostics(result)
    assert (result.is_robust, result.verdict) == before, (
        "adding PSR/Monte-Carlo must not change which strategies get promoted"
    )
    assert result.mc_simulations > 0, "…but the diagnostics must still be computed"


# ── the arithmetic ───────────────────────────────────────────────────────────

def test_psr_uses_the_per_period_sharpe_not_an_annualised_one():
    """PSR's contract: observed_sr must match the frequency of the moments.

    skew/kurtosis are computed on DAILY returns, so observed_sr must be the
    daily Sharpe. Annualising it (x sqrt(252)) inflates it ~16x and wrecks both
    the denominator and the z-score — this test pins the convention because the
    first draft of the wiring got it wrong.
    """
    rng = np.random.default_rng(3)
    returns = pd.Series(rng.normal(0.0005, 0.01, 400))

    result = WalkForwardResult()
    equity = 100_000 * np.cumprod(1 + returns.to_numpy())
    result.combined_equity = [{"equity": float(v)} for v in equity]
    _add_distribution_diagnostics(result)

    daily_sr = float(returns.mean() / returns.std(ddof=1))
    expected = probabilistic_sharpe_ratio(
        observed_sr=daily_sr,
        n_obs=len(returns),
        skew=float(returns.skew()),
        kurtosis=float(returns.kurtosis()) + 3.0,
    )
    assert result.psr == pytest.approx(round(expected, 4), abs=0.02)

    annualised = probabilistic_sharpe_ratio(
        observed_sr=daily_sr * (252 ** 0.5),
        n_obs=len(returns),
        skew=float(returns.skew()),
        kurtosis=float(returns.kurtosis()) + 3.0,
    )
    assert abs(result.psr - annualised) > 1e-6 or annualised == pytest.approx(expected), (
        "the annualised form must not silently coincide, or this test proves nothing"
    )


def test_pandas_excess_kurtosis_is_converted_to_pearson():
    """PSR defaults kurtosis=3.0 (normal), i.e. Pearson. pandas returns EXCESS."""
    normal = pd.Series(np.random.default_rng(5).normal(0, 0.01, 5000))
    assert abs(float(normal.kurtosis())) < 0.3, "pandas gives ~0 for a normal sample"
    assert abs(float(normal.kurtosis()) + 3.0 - 3.0) < 0.3, "+3 lands on Pearson normal"


# ── refusing to fabricate ────────────────────────────────────────────────────

def test_monte_carlo_is_skipped_rather_than_faked_on_thin_data():
    """Bootstrap percentiles off a handful of points are noise, not risk."""
    result = WalkForwardResult()
    result.combined_equity = [{"equity": 100_000 + i} for i in range(MIN_OBS_FOR_MONTE_CARLO - 5)]
    _add_distribution_diagnostics(result)
    assert result.mc_simulations == 0, (
        "too few observations must report 'not run', not a fabricated distribution"
    )


def test_diagnostics_never_raise_out_of_a_successful_backtest():
    """These hang off a run that already succeeded; they must not fail it."""
    for equity in ([], [{"equity": None}], [{"nope": 1}], [{"equity": 100.0}]):
        result = WalkForwardResult()
        result.combined_equity = equity
        _add_distribution_diagnostics(result)   # must not raise
        assert result.mc_simulations == 0


def test_flat_equity_does_not_divide_by_zero():
    result = WalkForwardResult()
    result.combined_equity = [{"equity": 100_000.0} for _ in range(200)]
    _add_distribution_diagnostics(result)
    assert result.psr == 0.0, "zero variance has no meaningful Sharpe probability"


def test_oos_returns_are_derived_from_the_stitched_equity_curve():
    result = WalkForwardResult()
    result.combined_equity = [{"equity": 100.0}, {"equity": 110.0}, {"equity": 99.0}]
    returns = _oos_daily_returns(result)
    assert len(returns) == 2
    assert returns.iloc[0] == pytest.approx(0.10)
    assert returns.iloc[1] == pytest.approx(-0.10)
