"""Unit tests for the BreakoutStrategy.

These tests validate that the BreakoutStrategy conforms to the expected
interface, produces correctly typed back‑test signals, avoids look‑ahead bias,
and handles edge cases in its asynchronous ``analyze`` method.
"""

import pytest
import pandas as pd
import numpy as np
from app.strategies.manual.breakout import BreakoutStrategy


@pytest.fixture
def ohlcv():
    """Generate a synthetic OHLCV DataFrame for testing."""
    n = 300
    rng = np.random.default_rng(99)
    close = 100 * np.cumprod(1 + rng.normal(0.0003, 0.012, n))
    high = close * (1 + rng.uniform(0, 0.01, n))
    low = close * (1 - rng.uniform(0, 0.01, n))
    volume = rng.integers(500_000, 5_000_000, n).astype(float)
    idx = pd.date_range("2023-01-01", periods=n, freq="1D")
    return pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


@pytest.fixture
def strategy():
    """Instantiate a fresh BreakoutStrategy for each test."""
    return BreakoutStrategy()


def test_has_required_attrs(strategy):
    """Check that the strategy exposes the required attribute values."""
    assert strategy.name == "breakout"
    assert strategy.market_type == "equity"
    assert strategy.strategy_type == "manual"
    assert strategy.risk_bucket == "directional"


def test_backtest_signals_returns_backtestsignals(strategy, ohlcv):
    """Validate the type of the object returned by ``backtest_signals``."""
    from app.strategies.base import BacktestSignals

    result = strategy.backtest_signals(ohlcv)
    assert isinstance(result, BacktestSignals)
    assert isinstance(result.entries, pd.Series)
    assert isinstance(result.exits, pd.Series)


def test_backtest_signals_boolean_dtype(strategy, ohlcv):
    """Ensure the signal series use a boolean dtype."""
    result = strategy.backtest_signals(ohlcv)
    assert result.entries.dtype == bool
    assert result.exits.dtype == bool


def test_backtest_signals_no_lookahead(strategy, ohlcv):
    """Confirm that ``backtest_signals`` does not contain a shift(0) look‑ahead."""
    import inspect

    src = inspect.getsource(strategy.backtest_signals)
    assert "shift(0)" not in src, "lookahead bias: shift(0) found in backtest_signals"
    assert "shift(2)" in src or "shift(1)" in src


def test_backtest_signals_same_length(strategy, ohlcv):
    """The length of the signal series must match the input OHLCV length."""
    result = strategy.backtest_signals(ohlcv)
    assert len(result.entries) == len(ohlcv)
    assert len(result.exits) == len(ohlcv)


@pytest.mark.asyncio
async def test_analyze_returns_none_on_insufficient_data(strategy):
    """When data is insufficient, ``analyze`` should return ``None``."""
    tiny = pd.DataFrame(
        {"open": [1.0], "high": [1.1], "low": [0.9], "close": [1.0], "volume": [1000.0]}
    )
    result = await strategy.analyze(tiny, "AAPL")
    assert result is None


@pytest.mark.asyncio
async def test_analyze_returns_none_when_no_breakout(strategy, ohlcv):
    """If no breakout occurs, ``analyze`` must return ``None``."""
    flat_close = pd.Series(50.0, index=ohlcv.index)
    flat_ohlcv = ohlcv.copy()
    flat_ohlcv["close"] = flat_close
    flat_ohlcv["high"] = flat_close * 1.001
    flat_ohlcv["low"] = flat_close * 0.999
    result = await strategy.analyze(flat_ohlcv, "AAPL")
    assert result is None


def test_custom_params():
    """Custom initialization parameters should be reflected on the instance."""
    s = BreakoutStrategy(params={"lookback": 20, "vol_mult": 2.0, "atr_mult": 0.3})
    assert s.lookback == 20
    assert s.vol_mult == 2.0
    assert s.atr_mult == 0.3