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


def test_walk_forward_minimal_length():
    # Exactly the minimal number of days for 1 year training + 1 month testing
    train_days = 365
    test_days = 30
    total_days = train_days + test_days
    rng = np.random.default_rng(0)
    returns = rng.normal(0, 0.01, total_days)
    prices = pd.Series(100 * np.cumprod(1 + returns),
                       index=pd.date_range("2021-01-01", periods=total_days, freq="D"))

    def signals_fn(train, test):
        # Use a short rolling window to avoid NaNs on the minimal series
        sma = test.rolling(5).mean()
        signals = (test > sma).astype(int).shift(1).fillna(0) * 2 - 1
        return signals

    result = walk_forward(signals_fn, prices, train_years=1, test_months=1)
    # Expect at least one window because the series length matches the required window size
    assert len(result.windows) >= 1
    assert isinstance(result.avg_sharpe, float)


def test_walk_forward_flat_prices():
    # Flat price series should still produce windows; signals are all zero
    n = 500
    prices = pd.Series([100.0] * n, index=pd.date_range("2022-01-01", periods=n, freq="D"))

    def signals_fn(train, test):
        # Return zero signals regardless of input
        return pd.Series(0, index=test.index)

    result = walk_forward(signals_fn, prices, train_years=1, test_months=2)
    assert result.windows  # windows should be generated
    # Sharpe may be NaN or zero; ensure it is a float (including nan)
    assert isinstance(result.avg_sharpe, float)


def test_walk_forward_business_days():
    # Use business day frequency to verify handling of non‑daily calendars
    rng = np.random.default_rng(123)
    n = 300  # enough for training and testing
    returns = rng.normal(0.0003, 0.012, n)
    prices = pd.Series(100 * np.cumprod(1 + returns),
                       index=pd.date_range("2020-01-01", periods=n, freq="B"))

    def signals_fn(train, test):
        sma = test.rolling(10).mean()
        signals = (test > sma).astype(int).shift(1).fillna(0) * 2 - 1
        return signals

    result = walk_forward(signals_fn, prices, train_years=1, test_months=1)
    assert result.windows  # at least one window should be produced
    assert isinstance(result.avg_sharpe, float)