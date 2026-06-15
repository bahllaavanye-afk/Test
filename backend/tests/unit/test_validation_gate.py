"""
Tests for backend/app/backtest/validation_gate.py
Uses synthetic price/signal data — no network calls.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from app.backtest.validation_gate import (
    ValidationReport,
    validate_experiment,
    summarize_for_results,
)
from app.backtest.walk_forward import WalkForwardResult


# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_prices(n: int = 1500, seed: int = 42) -> pd.Series:
    """Generate a synthetic price series long enough for 2+ walk-forward windows."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0003, 0.01, n)
    prices = 100.0 * np.cumprod(1 + returns)
    idx = pd.date_range("2018-01-01", periods=n, freq="B")
    return pd.Series(prices, index=idx, name="close")


def _good_signal_fn(train: pd.Series, test: pd.Series) -> pd.Series:
    """Always long — will produce modest positive returns on upward trending data."""
    return pd.Series(1.0, index=test.index)


def _flat_signal_fn(train: pd.Series, test: pd.Series) -> pd.Series:
    """Always flat — zero returns, Sharpe ≈ 0."""
    return pd.Series(0.0, index=test.index)


# ── 1. Pass: enough windows + good sharpe ─────────────────────────────────────


def test_validation_gate_pass():
    """With a trending price series and enough data, validation should pass."""
    # Build a strongly trending price series so buy-and-hold gets Sharpe > 0.3
    n = 1500
    returns = np.full(n, 0.002)  # deterministic 0.2%/day → very high Sharpe
    prices = 100.0 * np.cumprod(1 + returns)
    idx = pd.date_range("2018-01-01", periods=n, freq="B")
    prices = pd.Series(prices, index=idx)

    report = validate_experiment(_good_signal_fn, prices)

    assert report.n_windows >= 2
    assert report.passed is True
    assert report.oos_sharpe >= 0.3
    assert len(report.failures) == 0


# ── 2. Fail: low Sharpe ────────────────────────────────────────────────────────


def test_validation_gate_fail_low_sharpe():
    """A flat (zero-signal) strategy should fail the Sharpe threshold."""
    # We mock walk_forward so we control the output precisely
    mock_result = WalkForwardResult(
        windows=[
            {"start": "2020-01-01", "end": "2020-06-30", "sharpe": 0.1, "max_drawdown": -0.05, "total_return": 0.01, "num_trades": 5},
            {"start": "2020-07-01", "end": "2020-12-31", "sharpe": 0.05, "max_drawdown": -0.03, "total_return": 0.005, "num_trades": 3},
        ],
        avg_sharpe=0.075,
        avg_drawdown=-0.04,
    )

    with patch("app.backtest.walk_forward.walk_forward", return_value=mock_result):
        prices = _make_prices()
        report = validate_experiment(_flat_signal_fn, prices)

    assert report.passed is False
    assert any("Sharpe" in f for f in report.failures)
    assert report.oos_sharpe == 0.075


# ── 3. Fail: fewer than min_windows ───────────────────────────────────────────


def test_validation_gate_fail_few_windows():
    """Only 1 valid window should fail the minimum-windows check."""
    mock_result = WalkForwardResult(
        windows=[
            {"start": "2020-01-01", "end": "2020-06-30", "sharpe": 1.5, "max_drawdown": -0.05, "total_return": 0.10, "num_trades": 10},
        ],
        avg_sharpe=1.5,
        avg_drawdown=-0.05,
    )

    with patch("app.backtest.walk_forward.walk_forward", return_value=mock_result):
        prices = _make_prices()
        report = validate_experiment(_good_signal_fn, prices)

    assert report.passed is False
    assert any("windows" in f.lower() for f in report.failures)
    assert report.n_windows == 1


# ── 4. summarize_for_results dict structure ────────────────────────────────────


def test_summarize_for_results():
    """summarize_for_results should return the correct nested dict structure."""
    report = ValidationReport(
        passed=True,
        oos_sharpe=1.2,
        oos_drawdown=-0.08,
        n_windows=4,
        window_results=[{"sharpe": 1.2}],
        failures=[],
        warnings=["OOS Sharpe 0.42 is borderline"],
    )
    summary = summarize_for_results(report)

    assert "validation" in summary
    v = summary["validation"]
    assert v["passed"] is True
    assert v["oos_sharpe"] == 1.2
    assert v["oos_drawdown"] == -0.08
    assert v["n_windows"] == 4
    assert isinstance(v["failures"], list)
    assert isinstance(v["warnings"], list)
    assert len(v["warnings"]) == 1


# ── 5. ValidationReport field completeness ────────────────────────────────────


def test_validation_report_fields():
    """All fields should be populated and have correct types."""
    report = ValidationReport(
        passed=False,
        oos_sharpe=-0.2,
        oos_drawdown=-0.55,
        n_windows=3,
        window_results=[
            {"start": "2021-01-01", "end": "2021-06-30", "sharpe": -0.2, "max_drawdown": -0.55, "total_return": -0.10},
        ],
        failures=["OOS Sharpe -0.200 below minimum 0.300"],
        warnings=[],
    )

    assert isinstance(report.passed, bool)
    assert isinstance(report.oos_sharpe, float)
    assert isinstance(report.oos_drawdown, float)
    assert isinstance(report.n_windows, int)
    assert isinstance(report.window_results, list)
    assert isinstance(report.failures, list)
    assert isinstance(report.warnings, list)
    assert report.n_windows == 3
    assert report.passed is False
    assert len(report.failures) == 1

    # Class-level threshold constants should still be accessible
    assert ValidationReport.MIN_SHARPE == 0.3
    assert ValidationReport.MIN_WINDOWS == 2
    assert ValidationReport.MAX_DRAWDOWN == -0.40
