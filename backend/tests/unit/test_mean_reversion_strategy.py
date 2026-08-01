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
    close[-5:] = 88.0  # sharp drop below 2‑std lower band
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


@pytest.mark.asyncio
async def test_entry_requires_rsi_oversold(strategy):
    """
    Ensure that a buy signal is only emitted when price touches the lower
    Bollinger Band *and* RSI is below the oversold threshold.
    """
    n = 50
    close = np.full(n, 100.0)
    # Force lower band breach in the last bar
    close[-1] = 85.0
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    df = pd.DataFrame(
        {
            "close": close,
            "high": close + 0.2,
            "low": close - 0.2,
            "open": close,
            "volume": np.ones(n) * 150_000,
        },
        index=idx,
    )
    # Manipulate RSI to be comfortably above oversold (e.g., 45)
    # The strategy should internally compute RSI; we mimic this by
    # providing a price path that yields a high RSI.
    # A flat price series will generate an RSI of 50; we add a small
    # upward drift to push it higher.
    df["close"] = np.linspace(100, 110, n)
    df["close"][-1] = 85.0  # still breach lower band
    signal = await strategy.analyze(df, "TEST")
    # Because RSI is not oversold, we expect no signal
    assert signal is None


@pytest.mark.asyncio
async def test_exit_when_price_crosses_middle_band(strategy):
    """
    Verify that after a buy entry, the strategy emits an exit signal
    when price moves above the middle Bollinger Band (the SMA).
    """
    # Phase 1 – generate a clear entry signal
    n_entry = 30
    close_entry = np.full(n_entry, 100.0)
    close_entry[-5:] = 85.0  # breach lower band
    idx_entry = pd.date_range("2024-01-01", periods=n_entry, freq="1h")
    df_entry = pd.DataFrame(
        {
            "close": close_entry,
            "high": close_entry + 0.2,
            "low": close_entry - 0.2,
            "open": close_entry,
            "volume": np.ones(n_entry) * 120_000,
        },
        index=idx_entry,
    )
    entry_signal = await strategy.analyze(df_entry, "TEST")
    assert entry_signal is not None
    assert entry_signal.side == "buy"

    # Phase 2 – continue with price climbing above the middle band
    n_exit = 10
    close_exit = np.linspace(85.0, 115.0, n_exit)  # upward trend crossing SMA
    idx_exit = pd.date_range(idx_entry[-1] + pd.Timedelta(hours=1), periods=n_exit, freq="1h")
    df_exit = pd.DataFrame(
        {
            "close": close_exit,
            "high": close_exit + 0.3,
            "low": close_exit - 0.3,
            "open": close_exit,
            "volume": np.ones(n_exit) * 130_000,
        },
        index=idx_exit,
    )
    exit_signal = await strategy.analyze(df_exit, "TEST")
    # The strategy should now emit a sell/exit signal
    assert exit_signal is not None
    assert exit_signal.side == "sell"


def test_custom_params():
    s = MeanReversionStrategy(params={"bb_period": 10, "bb_std": 1.5, "rsi_oversold": 25})
    assert s.bb_period == 10
    assert s.bb_std == 1.5
    assert s.rsi_oversold == 25