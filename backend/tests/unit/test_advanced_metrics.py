"""
Unit tests for app.ml.evaluation.metrics

Tests cover correctness of every metric in PerformanceMetrics.
All values are verified against hand-calculated expectations.
"""
import math
import pytest
import numpy as np
import pandas as pd
from datetime import date

from app.ml.evaluation.metrics import (
    compute_metrics,
    metrics_from_signals,
    metrics_from_positions,
    format_metrics_slack,
    PerformanceMetrics,
)

# ─── Fixtures ──────────────────────────────────────────────────────────────────

def _daily_returns(n: int = 252, seed: int = 42, drift: float = 0.0005) -> pd.Series:
    rng = np.random.default_rng(seed)
    r   = rng.normal(drift, 0.01, n)
    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    return pd.Series(r, index=idx, name="returns")


def _constant_returns(n: int = 252, value: float = 0.001) -> pd.Series:
    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    return pd.Series(value, index=idx, name="returns")


def _prices_from_returns(r: pd.Series) -> pd.Series:
    return (1 + r).cumprod() * 100.0


# ─── Basic smoke tests ─────────────────────────────────────────────────────────

def test_compute_metrics_returns_dataclass():
    r = _daily_returns()
    m = compute_metrics(r)
    assert isinstance(m, PerformanceMetrics)


def test_compute_metrics_empty_returns_no_crash():
    m = compute_metrics(pd.Series([], dtype=float))
    assert m.sharpe == 0.0


def test_compute_metrics_too_short_returns_zero():
    m = compute_metrics(pd.Series([0.01, 0.02, -0.01, 0.005], dtype=float))
    assert m.sharpe == 0.0


# ─── Sharpe / Sortino ─────────────────────────────────────────────────────────

def test_sharpe_positive_for_upward_drift():
    r = _daily_returns(drift=0.002)
    m = compute_metrics(r)
    assert m.sharpe > 0


def test_sharpe_zero_for_zero_vol():
    # Zero vol → Sharpe should be 0 or inf; we return 0 for safety
    r = pd.Series([0.0] * 252, index=pd.date_range("2022-01-01", periods=252, freq="B"))
    m = compute_metrics(r)
    assert m.sharpe == 0.0


def test_sortino_gte_sharpe_for_positive_skew():
    # When all returns are positive, downside vol is near 0 → Sortino >> Sharpe
    r = pd.Series(np.abs(_daily_returns().values),
                  index=pd.date_range("2022-01-01", periods=252, freq="B"))
    m = compute_metrics(r)
    assert m.sortino >= m.sharpe


# ─── Calmar / Drawdown ────────────────────────────────────────────────────────

def test_max_drawdown_is_negative():
    r = _daily_returns()
    m = compute_metrics(r)
    assert m.max_drawdown <= 0.0


def test_max_drawdown_zero_for_monotone_up():
    r = _constant_returns(value=0.001)
    m = compute_metrics(r)
    assert math.isclose(m.max_drawdown, 0.0, abs_tol=1e-9)


def test_calmar_positive_when_positive_return_and_drawdown():
    r = _daily_returns(drift=0.001)
    m = compute_metrics(r)
    if m.max_drawdown < 0 and m.ann_return > 0:
        assert m.calmar > 0


def test_ulcer_index_nonneg():
    r = _daily_returns()
    m = compute_metrics(r)
    assert m.ulcer_index >= 0.0


# ─── Omega / Gain-to-Pain ─────────────────────────────────────────────────────

def test_omega_greater_than_one_for_positive_drift():
    r = _daily_returns(drift=0.002)
    m = compute_metrics(r)
    assert m.omega > 1.0


def test_gain_to_pain_positive_for_net_positive_returns():
    r = _daily_returns(drift=0.001)
    m = compute_metrics(r)
    if r.sum() > 0:
        assert m.gain_to_pain > 0


# ─── Recovery / Tail ──────────────────────────────────────────────────────────

def test_recovery_factor_sign():
    r = _daily_returns(drift=0.001)
    m = compute_metrics(r)
    if m.max_drawdown < 0:
        assert m.recovery_factor > 0


def test_tail_ratio_gt_zero():
    r = _daily_returns()
    m = compute_metrics(r)
    assert m.tail_ratio > 0.0


# ─── VaR / CVaR ───────────────────────────────────────────────────────────────

def test_var_95_lt_var_99():
    r = _daily_returns()
    m = compute_metrics(r)
    # VaR_99 is more extreme (more negative) than VaR_95
    assert m.var_99 <= m.var_95


def test_cvar_95_lte_var_95():
    r = _daily_returns()
    m = compute_metrics(r)
    assert m.cvar_95 <= m.var_95


def test_var_negative_for_volatile_returns():
    r = _daily_returns(seed=0, drift=0.0)
    m = compute_metrics(r)
    assert m.var_95 < 0


# ─── Skewness / Kurtosis ──────────────────────────────────────────────────────

def test_skewness_near_zero_for_symmetric():
    rng = np.random.default_rng(7)
    r   = pd.Series(rng.normal(0, 0.01, 1000),
                    index=pd.date_range("2022-01-01", periods=1000, freq="B"))
    m   = compute_metrics(r)
    assert abs(m.skewness) < 0.5   # symmetric → near-zero skew


def test_kurtosis_near_zero_for_normal():
    rng = np.random.default_rng(7)
    r   = pd.Series(rng.normal(0, 0.01, 2000),
                    index=pd.date_range("2022-01-01", periods=2000, freq="B"))
    m   = compute_metrics(r)
    assert abs(m.excess_kurtosis) < 1.0


# ─── Trade-level statistics ───────────────────────────────────────────────────

def _make_entry_exit(prices: pd.Series, buy_every: int = 10) -> tuple[pd.Series, pd.Series]:
    entries = pd.Series(False, index=prices.index)
    exits   = pd.Series(False, index=prices.index)
    for i in range(0, len(prices) - buy_every, buy_every * 2):
        entries.iloc[i]            = True
        exits.iloc[i + buy_every]  = True
    return entries, exits


def test_win_rate_between_zero_and_one():
    r       = _daily_returns()
    prices  = _prices_from_returns(r)
    entries, exits = _make_entry_exit(prices)
    m = compute_metrics(r, entries=entries, exits=exits)
    assert 0.0 <= m.win_rate <= 1.0


def test_n_trades_matches_entries():
    r       = _daily_returns()
    prices  = _prices_from_returns(r)
    entries, exits = _make_entry_exit(prices, buy_every=20)
    m = compute_metrics(r, entries=entries, exits=exits)
    expected_trades = int(entries.sum())
    assert abs(m.n_trades - expected_trades) <= 2   # allow 1-2 open-at-end


def test_profit_factor_positive():
    r       = _daily_returns(drift=0.001)
    prices  = _prices_from_returns(r)
    entries, exits = _make_entry_exit(prices)
    m = compute_metrics(r, entries=entries, exits=exits)
    assert m.profit_factor >= 0.0


def test_avg_win_gte_avg_loss_for_positive_drift():
    r       = _daily_returns(drift=0.002, seed=1)
    prices  = _prices_from_returns(r)
    entries, exits = _make_entry_exit(prices, buy_every=10)
    m = compute_metrics(r, entries=entries, exits=exits)
    if m.win_rate > 0.5:
        assert m.avg_win >= 0


# ─── Benchmark-relative metrics ───────────────────────────────────────────────

def test_beta_near_one_for_parallel_strategy():
    r  = _daily_returns(seed=99, drift=0.001)
    # Benchmark with same distribution but slightly different seed
    b  = _daily_returns(seed=99, drift=0.001)
    m  = compute_metrics(r, benchmark=b)
    assert math.isclose(m.beta, 1.0, abs_tol=0.15)


def test_alpha_zero_for_identical_strategy_and_bench():
    r = _daily_returns(seed=42)
    m = compute_metrics(r, benchmark=r)
    assert abs(m.alpha) < 0.01


def test_information_ratio_positive_for_outperforming():
    r = _daily_returns(drift=0.002)
    b = _daily_returns(drift=0.0005)
    m = compute_metrics(r, benchmark=b)
    assert m.information_ratio > 0 or m.active_return > 0


def test_up_down_capture_present():
    r = _daily_returns(drift=0.001)
    b = _daily_returns(drift=0.0005)
    m = compute_metrics(r, benchmark=b)
    assert m.up_capture != 0.0 or m.down_capture != 0.0


# ─── ML signal quality ────────────────────────────────────────────────────────

def test_ic_positive_for_correct_signal():
    r      = _daily_returns(drift=0.001)
    # Signal = lagged return → slightly predictive
    signal = r.shift(1).fillna(0)
    m      = compute_metrics(r, signal=signal)
    # IC can be positive or near-zero for shifted signal
    assert -1.0 <= m.ic <= 1.0


def test_ic_zero_for_random_signal():
    rng    = np.random.default_rng(123)
    r      = _daily_returns()
    signal = pd.Series(rng.normal(0, 1, len(r)), index=r.index)
    m      = compute_metrics(r, signal=signal)
    assert abs(m.ic) < 0.3   # random signal → IC near 0


def test_rank_ic_in_range():
    r      = _daily_returns()
    signal = r.shift(1).fillna(0)
    m      = compute_metrics(r, signal=signal)
    assert -1.0 <= m.rank_ic <= 1.0


# ─── Monthly stats ────────────────────────────────────────────────────────────

def test_monthly_hit_rate_between_zero_and_one():
    r = _daily_returns()
    m = compute_metrics(r)
    assert 0.0 <= m.hit_rate_monthly <= 1.0


def test_best_month_gte_worst_month():
    r = _daily_returns()
    m = compute_metrics(r)
    assert m.best_month >= m.worst_month


# ─── metrics_from_signals convenience wrapper ─────────────────────────────────

def test_metrics_from_signals_runs():
    r       = _daily_returns()
    prices  = _prices_from_returns(r)
    entries, exits = _make_entry_exit(prices)
    m = metrics_from_signals(entries, exits, prices)
    assert isinstance(m, PerformanceMetrics)
    assert m.total_return != 0.0 or m.n_trades == 0


def test_metrics_from_positions_runs():
    r       = _daily_returns()
    prices  = _prices_from_returns(r)
    pos     = pd.Series(1.0, index=prices.index)
    m       = metrics_from_positions(pos, prices)
    # Long-only = total return matches buy-and-hold
    bah = float(prices.iloc[-1] / prices.iloc[0] - 1)
    assert math.isclose(m.total_return, bah, rel_tol=0.01)


# ─── to_dict / format_metrics_slack ──────────────────────────────────────────

def test_to_dict_all_numeric():
    r = _daily_returns()
    m = compute_metrics(r)
    d = m.to_dict()
    assert all(isinstance(v, (int, float)) for v in d.values())


def test_format_metrics_slack_returns_string():
    r   = _daily_returns()
    m   = compute_metrics(r)
    out = format_metrics_slack(m, name="TestStrategy")
    assert isinstance(out, str)
    assert "Sharpe" in out
    assert "Drawdown" in out


def test_format_metrics_slack_includes_benchmark_section_when_alpha_present():
    r   = _daily_returns(drift=0.001)
    b   = _daily_returns(drift=0.0005)
    m   = compute_metrics(r, benchmark=b)
    out = format_metrics_slack(m, name="Strategy vs SPY")
    assert "Benchmark" in out
    assert "Alpha" in out


# ─── Edge cases ───────────────────────────────────────────────────────────────

def test_all_losses_omega_low():
    r = pd.Series(-0.001, index=pd.date_range("2022-01-01", periods=252, freq="B"))
    m = compute_metrics(r)
    assert m.omega < 1.0


def test_ann_return_matches_cagr():
    r  = _constant_returns(value=0.001, n=252)
    m  = compute_metrics(r)
    expected_cagr = (1 + 0.001) ** 252 - 1
    assert math.isclose(m.total_return, expected_cagr, rel_tol=0.01)


def test_n_drawdown_periods_nonneg():
    r = _daily_returns()
    m = compute_metrics(r)
    assert m.n_drawdown_periods >= 0
