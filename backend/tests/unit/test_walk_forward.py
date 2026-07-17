"""Walk-forward validation tests."""
import pandas as pd
import numpy as np
from app.backtest.walk_forward import walk_forward


def test_walk_forward_basic():
    rng = np.random.default_rng(42)
    n = 252 * 3   # 3 years
    returns = rng.normal(0.0005, 0.015, n)
    prices = pd.Series(100 * np.cumprod(1 + returns),
                        index=pd.date_range("2020-01-01", periods=n, freq="D"))

    def signals_fn(train, test):
        # Simple: buy on positive 20-day SMA momentum, computed from train only
        sma = test.rolling(20).mean()
        signals = (test > sma).astype(int).shift(1).fillna(0) * 2 - 1
        return signals

    result = walk_forward(signals_fn, prices, train_years=1, test_months=3)
    assert result.windows  # at least one window
    assert isinstance(result.avg_sharpe, float)


def test_walk_forward_too_short():
    prices = pd.Series([100] * 50, index=pd.date_range("2024-01-01", periods=50, freq="D"))
    result = walk_forward(lambda t, e: pd.Series(0, index=e.index), prices,
                           train_years=2, test_months=6)
    assert result.windows == []


def test_walk_forward_exact_boundary():
    """
    Verify that a series exactly long enough for one train/test window
    produces a single window.
    """
    # 1 year of daily data (~365 days) + 1 month of daily data (~30 days)
    total_days = 365 + 30
    dates = pd.date_range("2022-01-01", periods=total_days, freq="D")
    prices = pd.Series(np.linspace(100, 200, total_days), index=dates)

    def signals_fn(train, test):
        # Return a flat signal (hold) for the test period
        return pd.Series(0, index=test.index)

    result = walk_forward(signals_fn, prices, train_years=1, test_months=1)
    assert len(result.windows) == 1
    # The window should contain both train and test slices; verify lengths
    train_slice, test_slice = result.windows[0]
    assert len(train_slice) >= 365  # at least one year of training data
    assert len(test_slice) >= 30    # at least one month of testing data


def test_walk_forward_zero_test_months():
    """
    Edge case where test_months is set to zero; expect no windows.
    """
    dates = pd.date_range("2021-01-01", periods=500, freq="D")
    prices = pd.Series(100 + np.arange(500), index=dates)

    result = walk_forward(lambda t, e: pd.Series(0, index=e.index),
                          prices,
                          train_years=1,
                          test_months=0)
    assert result.windows == []


def test_walk_forward_zero_train_years():
    """
    Edge case where train_years is zero; expect no windows because no training period.
    """
    dates = pd.date_range("2021-01-01", periods=200, freq="D")
    prices = pd.Series(100 + np.arange(200), index=dates)

    result = walk_forward(lambda t, e: pd.Series(0, index=e.index),
                          prices,
                          train_years=0,
                          test_months=1)
    assert result.windows == []