"""Overfit killers: Deflated Sharpe, Probabilistic Sharpe, PBO + the walk-forward
robustness gate.

These implement the "what top firms do that we don't" line item — multiple-testing
and backtest-overfitting haircuts (Bailey & López de Prado) that gate a strategy
before it is promoted, on top of plain walk-forward. Pure math, synthetic data,
deterministic seeds — no network, no DB.
"""
import numpy as np
import pandas as pd
import pytest

from app.backtest.cpcv import (
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
    probability_of_backtest_overfitting,
)
from app.backtest.walk_forward import robustness_verdict, walk_forward


# ── Deflated Sharpe Ratio (probability) ──────────────────────────────────────

def test_dsr_is_a_probability():
    assert 0.0 <= deflated_sharpe_ratio([0.8, 0.9, 0.7, 0.85, 0.75], 5) <= 1.0


def test_dsr_high_for_consistent_positive_sharpes():
    # 12 tightly-clustered good Sharpes → very unlikely to be luck.
    sr = [0.80, 0.85, 0.75, 0.82, 0.78, 0.80, 0.83, 0.77, 0.81, 0.79, 0.84, 0.76]
    assert deflated_sharpe_ratio(sr, len(sr)) > 0.95


def test_dsr_low_for_noisy_sharpes_around_zero():
    # High-variance Sharpes centred near zero → within the luck benchmark.
    sr = [1.5, -1.3, 0.9, -1.1, 0.7, -0.8, 1.2, -1.0, 0.6, -0.9, 1.1, -1.2]
    assert deflated_sharpe_ratio(sr, len(sr)) < 0.5


def test_dsr_empty_and_singletons():
    assert deflated_sharpe_ratio([], 1) == 0.0
    assert deflated_sharpe_ratio([1.5], 1) == 1.0     # single positive
    assert deflated_sharpe_ratio([-0.5], 1) == 0.0    # single negative


def test_dsr_more_trials_lowers_confidence():
    sr = [0.9, 1.0, 0.8, 0.95, 0.85, 0.92]
    # Same Sharpes, but claiming they came from a 200-config sweep → harsher.
    assert deflated_sharpe_ratio(sr, n_trials=len(sr)) >= deflated_sharpe_ratio(sr, n_trials=200)


def test_dsr_zero_variance_negative_sharpes():
    # All Sharpe values are identical and negative → probability should be 0.
    sr = [-0.3, -0.3, -0.3]
    assert deflated_sharpe_ratio(sr, len(sr)) == 0.0


# ── Probabilistic Sharpe Ratio ───────────────────────────────────────────────

def test_psr_bounds_and_zero_point():
    assert probabilistic_sharpe_ratio(0.0, 250) == pytest.approx(0.5, abs=1e-6)
    assert 0.0 <= probabilistic_sharpe_ratio(0.1, 250) <= 1.0


def test_psr_monotonic_in_sharpe():
    lo = probabilistic_sharpe_ratio(0.02, 250)
    mid = probabilistic_sharpe_ratio(0.05, 250)
    hi = probabilistic_sharpe_ratio(0.10, 250)
    assert lo < mid < hi
    assert probabilistic_sharpe_ratio(-0.05, 250) < 0.5


def test_psr_more_observations_more_confident():
    assert probabilistic_sharpe_ratio(0.05, 1000) > probabilistic_sharpe_ratio(0.05, 100)


def test_psr_negative_skew_hurts():
    # Negative skewness (crash risk) lowers confidence in the same Sharpe.
    base = probabilistic_sharpe_ratio(0.08, 500, skew=0.0, kurtosis=3.0)
    skewed = probabilistic_sharpe_ratio(0.08, 500, skew=-1.5, kurtosis=6.0)
    assert skewed < base


def test_psr_too_few_obs():
    assert probabilistic_sharpe_ratio(0.5, 1) == 0.0


def test_psr_extreme_sharpe_bounds():
    # Very high Sharpe values should be capped at 1.0.
    assert probabilistic_sharpe_ratio(5.0, 100) == pytest.approx(1.0, abs=1e-6)


# ── Probability of Backtest Overfitting (CSCV) ───────────────────────────────

def test_pbo_high_for_pure_noise():
    # 20 skill-less configs → selecting the in-sample best is ~coin-flip OOS,
    # so PBO sits near 0.5 (theory). N=20 keeps the CSCV estimate stable.
    rng = np.random.default_rng(7)
    M = rng.normal(0.0, 0.01, size=(1000, 20))
    out = probability_of_backtest_overfitting(M, n_splits=10)
    assert out["n_configs"] == 20 and out["n_combinations"] == 252
    assert out["pbo"] > 0.3   # selection is ~no better than chance


def test_pbo_low_for_one_genuine_edge():
    rng = np.random.default_rng(11)
    M = rng.normal(0.0, 0.01, size=(1000, 20))
    # Column 0 has a real, persistent positive drift → best in- AND out-of-sample.
    M[:, 0] += 0.004
    out = probability_of_backtest_overfitting(M, n_splits=10)
    assert out["pbo"] < 0.15


def test_pbo_failsoft_on_tiny_matrix():
    out = probability_of_backtest_overfitting(np.zeros((5, 3)), n_splits=16)
    assert out["pbo"] == 1.0 and out["n_combinations"] == 0


def test_pbo_failsoft_on_single_config():
    out = probability_of_backtest_overfitting(np.zeros((500, 1)), n_splits=10)
    assert out["pbo"] == 1.0


def test_pbo_zero_splits_returns_soft_fail():
    # Zero splits should trigger a soft‑fail and return pbo=1.0.
    rng = np.random.default_rng(42)
    M = rng.normal(0, 0.01, size=(100, 10))
    out = probability_of_backtest_overfitting(M, n_splits=0)
    assert out["pbo"] == 1.0


# ── Walk-forward robustness verdict ──────────────────────────────────────────

def test_verdict_robust_when_consistent_and_plentiful():
    sr = [0.9, 1.0, 0.8, 0.95, 0.85, 0.92, 0.88, 0.9, 0.83, 0.97, 0.86, 0.91]
    v = robustness_verdict(sr)
    assert v["is_robust"] is True and v["verdict"] == "robust"
    assert v["n_windows"] == 12 and v["consistency"] == 1.0


def test_verdict_rejects_too_few_windows():
    v = robustness_verdict([1.2, 1.1, 1.3, 1.0])   # only 4 windows
    assert v["is_robust"] is False and "windows" in v["verdict"]


def test_verdict_rejects_weak_average():
    v = robustness_verdict([0.1, 0.2, 0.0, 0.15, 0.05, 0.1, 0.2, 0.1, 0.0, 0.1, 0.15, 0.05])
    assert v["is_robust"] is False and "avg Sharpe" in v["verdict"]


def test_verdict_rejects_high_variance_lucky_windows():
    # Average scraped up by a few huge windows; most are losers → not robust.
    sr = [4.0, 4.0, 4.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
    v = robustness_verdict(sr)
    assert v["is_robust"] is False


def test_verdict_empty():
    v = robustness_verdict([])
    assert v["is_robust"] is False and v["verdict"] == "insufficient_data"


def test_verdict_boundary_min_windows():
    # Minimal acceptable number of windows with decent Sharpe values should pass.
    sr = [0.6, 0.7, 0.65, 0.68, 0.66, 0.7]
    v = robustness_verdict(sr)
    assert v["is_robust"] is True


# ── walk_forward() integration: verdict populated + initial_equity accepted ──

def test_walk_forward_populates_verdict_and_accepts_initial_equity():
    rng = np.random.default_rng(3)
    n = 252 * 4
    prices = pd.Series(
        100 * np.cumprod(1 + rng.normal(0.0004, 0.012, n)),
        index=pd.date_range("2019-01-01", periods=n, freq="D"),
    )

    def signals_fn(train, test):
        sma = test.rolling(20).mean()
        return (test > sma).astype(int).shift(1).fillna(0) * 2 - 1

    # initial_equity previously raised TypeError (walk_forward lacked the param).
    result = walk_forward(signals_fn, prices, train_years=1, test_months=3,
                          initial_equity=50_000)
    assert result.windows
    assert isinstance(result.is_robust, bool)
    assert isinstance(result.deflated_sharpe, float)
    assert result.n_windows == len([w for w in result.windows if "sharpe" in w])
    assert result.verdict  # non-empty string


def test_walk_forward_empty_prices_returns_insufficient():
    empty_prices = pd.Series([], dtype=float)
    def dummy_signals(train, test):
        return pd.Series([], index=test.index)
    result = walk_forward(dummy_signals, empty_prices, train_years=1, test_months=3, initial_equity=10_000)
    assert result.is_robust is False
    assert result.verdict == "insufficient_data"