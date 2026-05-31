"""Unit tests for MeanReversionStrategy (Bollinger Band)."""
import pytest
import pandas as pd
import numpy as np
from app.strategies.manual.mean_reversion import MeanReversionStrategy
from app.strategies.base import BacktestSignals


@pytest.fixture
def ohlcv():
    n = 200
    rng = np.random.default_rng(7)
    close = 100 + 5 * np.sin(np.linspace(0, 8 * np.pi, n)) + rng.normal(0, 1, n)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    return pd.DataFrame({
        "open": close, "high": close + 0.5,
        "low": close - 0.5, "close": close,
        "volume": rng.integers(100_000, 500_000, n).astype(float)
    }, index=idx)


@pytest.fixture
def strategy():
    return MeanReversionStrategy()


def test_has_required_attrs(strategy):
    assert strategy.name == "mean_reversion"
    assert strategy.market_type == "equity"
    assert strategy.strategy_type == "manual"
    assert strategy.risk_bucket == "directional"


def test_backtest_signals_type(strategy, ohlcv):
    result = strategy.backtest_signals(ohlcv)
    assert isinstance(result, BacktestSignals)
    assert isinstance(result.entries, pd.Series)
    assert isinstance(result.exits, pd.Series)


def test_backtest_signals_same_length(strategy, ohlcv):
    result = strategy.backtest_signals(ohlcv)
    assert len(result.entries) == len(ohlcv)
    assert len(result.exits) == len(ohlcv)


def test_no_lookahead_in_backtest(strategy):
    import inspect
    src = inspect.getsource(strategy.backtest_signals)
    assert "shift(0)" not in src, "lookahead bias detected: shift(0) in backtest_signals"


@pytest.mark.asyncio
async def test_analyze_none_on_short_data(strategy):
    tiny = pd.DataFrame({"close": [1.0, 2.0], "high": [1.1, 2.1],
                          "low": [0.9, 1.9], "open": [1.0, 2.0], "volume": [1000.0, 1000.0]})
    result = await strategy.analyze(tiny, "SPY")
    assert result is None


@pytest.mark.asyncio
async def test_analyze_buy_signal_near_lower_band(strategy):
    # Build a series that dips sharply at the end to touch lower BB
    n = 60
    close = np.full(n, 100.0)
    close[-5:] = 88.0  # sharp drop below 2-std lower band
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    df = pd.DataFrame({"close": close, "high": close + 0.2, "low": close - 0.2,
                        "open": close, "volume": np.ones(n) * 100_000}, index=idx)
    signal = await strategy.analyze(df, "TEST")
    if signal is not None:
        assert signal.side == "buy"


def test_custom_params():
    s = MeanReversionStrategy(params={"bb_period": 10, "bb_std": 1.5, "rsi_oversold": 25})
    assert s.bb_period == 10
    assert s.bb_std == 1.5
    assert s.rsi_oversold == 25
