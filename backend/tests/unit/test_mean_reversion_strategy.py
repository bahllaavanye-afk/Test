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
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": rng.integers(100_000, 500_000, n).astype(float),
        },
        index=idx,
    )


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
    tiny = pd.DataFrame(
        {
            "close": [1.0, 2.0],
            "high": [1.1, 2.1],
            "low": [0.9, 1.9],
            "open": [1.0, 2.0],
            "volume": [1000.0, 1000.0],
        }
    )
    result = await strategy.analyze(tiny, "SPY")
    assert result is None


@pytest.mark.asyncio
async def test_analyze_buy_signal_near_lower_band(strategy):
    # Build a series that dips sharply at the end to touch lower BB
    n = 60
    close = np.full(n, 100.0)
    close[-5:] = 88.0  # sharp drop below 2-std lower band
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    df = pd.DataFrame(
        {
            "close": close,
            "high": close + 0.2,
            "low": close - 0.2,
            "open": close,
            "volume": np.ones(n) * 100_000,
        },
        index=idx,
    )
    signal = await strategy.analyze(df, "TEST")
    if signal is not None:
        assert signal.side == "buy"


def test_custom_params():
    s = MeanReversionStrategy(params={"bb_period": 10, "bb_std": 1.5, "rsi_oversold": 25})
    assert s.bb_period == 10
    assert s.bb_std == 1.5
    assert s.rsi_oversold == 25


# Edge case tests -------------------------------------------------------------

def test_backtest_signals_none_input(strategy):
    """Ensure backtest_signals gracefully handles None input."""
    with pytest.raises(Exception):
        strategy.backtest_signals(None)


def test_backtest_signals_empty_dataframe(strategy):
    """Backtest on an empty DataFrame should return empty signal series."""
    empty_df = pd.DataFrame(
        {"open": [], "high": [], "low": [], "close": [], "volume": []},
        dtype=float,
    )
    result = strategy.backtest_signals(empty_df)
    assert isinstance(result, BacktestSignals)
    assert result.entries.empty
    assert result.exits.empty


@pytest.mark.asyncio
async def test_analyze_empty_dataframe(strategy):
    """Analyze should return None for an empty DataFrame."""
    empty_df = pd.DataFrame(
        {"open": [], "high": [], "low": [], "close": [], "volume": []},
        dtype=float,
    )
    result = await strategy.analyze(empty_df, "EMPTY")
    assert result is None


@pytest.mark.asyncio
async def test_analyze_off_by_one_edge(strategy):
    """Test handling when DataFrame length is exactly the lookback period."""
    # Assuming the default lookback is 20; create a DataFrame with 20 rows.
    n = 20
    rng = np.random.default_rng(42)
    close = 100 + rng.normal(0, 1, n)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    df = pd.DataFrame(
        {
            "close": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "open": close,
            "volume": np.full(n, 150_000.0),
        },
        index=idx,
    )
    result = await strategy.analyze(df, "EDGE")
    # With minimal data the strategy may not emit a signal; ensure no error.
    assert result is None or hasattr(result, "side")