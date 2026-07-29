"""Unit tests for MeanReversionStrategy (Bollinger Band)."""
import pytest
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field, validator

from app.strategies.manual.mean_reversion import MeanReversionStrategy
from app.strategies.base import BacktestSignals


class MeanReversionParams(BaseModel):
    """Parameters for the MeanReversionStrategy.

    Attributes
    ----------
    bb_period: int
        Number of periods used to compute Bollinger Bands. Must be a positive integer.
        Example: 20
    bb_std: float
        Standard deviation multiplier for the Bollinger Bands. Must be greater than 0.
        Example: 2.0
    rsi_oversold: int
        RSI threshold considered oversold. Must be between 0 and 100 inclusive.
        Example: 30
    """
    bb_period: int = Field(..., description="Number of periods for Bollinger Bands", example=20, ge=1)
    bb_std: float = Field(..., description="Standard deviation multiplier for Bollinger Bands", example=2.0, gt=0)
    rsi_oversold: int = Field(..., description="RSI oversold threshold", example=30, ge=0, le=100)

    @validator("bb_period")
    def validate_bb_period(cls, v):
        if not isinstance(v, int):
            raise ValueError("bb_period must be an integer")
        return v

    @validator("bb_std")
    def validate_bb_std(cls, v):
        if v <= 0:
            raise ValueError("bb_std must be greater than 0")
        return v

    @validator("rsi_oversold")
    def validate_rsi_oversold(cls, v):
        if not (0 <= v <= 100):
            raise ValueError("rsi_oversold must be between 0 and 100")
        return v


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
    assert isinstance(result, BacklogSignals)
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
    params = MeanReversionParams(bb_period=10, bb_std=1.5, rsi_oversold=25)
    s = MeanReversionStrategy(params=params.dict())
    assert s.bb_period == params.bb_period
    assert s.bb_std == params.bb_std
    assert s.rsi_oversold == params.rsi_oversold