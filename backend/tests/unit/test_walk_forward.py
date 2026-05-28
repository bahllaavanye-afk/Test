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
