"""`regime` bot condition — OA-style "trade only in these market conditions".

The user directive: different strategies for different market conditions. A bot
condition of type `regime` fires only when the detected (trend, vol) regime is
in the bot's allowed list. Fail‑soft: no filter → always allowed; degenerate
data → sideways/calm, never an exception.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pandas")

from app.bots.engine import detect_regime, evaluate_condition  # noqa: E402
from app.schemas.bot import ConditionConfig  # noqa: E402


def _series(drift: float, vol: float, n: int = 120) -> pd.DataFrame:
    """Generate a synthetic price series using geometric Brownian motion."""
    rng = np.random.default_rng(0)
    rets = rng.normal(drift, vol, n)
    close = 100 * np.exp(np.cumsum(rets))
    return pd.DataFrame({"close": close})


def _cond(regimes: list[str] | None) -> ConditionConfig:
    """Convenient constructor for a regime condition."""
    return ConditionConfig(type="regime", regimes=regimes)


@pytest.fixture
def bull_calm_series() -> pd.DataFrame:
    """A deterministic series that should be classified as bull/calm."""
    return _series(0.003, 0.004)


@pytest.fixture
def bear_stressed_series() -> pd.DataFrame:
    """Series that mimics a recent sell‑off with elevated volatility."""
    early = [100.0 - i * 0.01 for i in range(100)]  # flat‑ish, low vol
    px, late = early[-1], []
    for k in range(24):
        px *= 0.95 if k % 2 == 0 else 0.99
        late.append(px)
    return pd.DataFrame({"close": early + late})


def test_detect_regime_bull_calm(bull_calm_series: pd.DataFrame) -> None:
    trend, vol = detect_regime(bull_calm_series)
    assert trend == "bull" and vol == "calm"


def test_detect_regime_bear_stressed(bear_stressed_series: pd.DataFrame) -> None:
    trend, vol = detect_regime(bear_stressed_series)
    assert trend == "bear" and vol == "stressed"


@pytest.mark.parametrize(
    "allowed_regimes,expected",
    [
        (["bull"], True),
        (["calm"], True),
        (["bull", "stressed"], True),  # OR semantics across trend/vol
        (["bear"], False),
        (["stressed"], False),
    ],
)
def test_condition_regime_matching(
    bull_calm_series: pd.DataFrame, allowed_regimes: list[str], expected: bool
) -> None:
    """Validate that the condition passes only when the detected regime is allowed."""
    assert evaluate_condition(_cond(allowed_regimes), bull_calm_series, 100.0) is expected


def test_empty_regimes_always_allowed() -> None:
    """When no regimes are supplied the condition should never block."""
    series = _series(0.003, 0.004)
    assert evaluate_condition(_cond([]), series, 100.0)
    assert evaluate_condition(_cond(None), series, 100.0)


def test_degenerate_data_is_sideways_calm_not_error() -> None:
    """Degenerate constant price data should be classified as sideways/calm."""
    trend, vol = detect_regime(pd.DataFrame({"close": [100.0, 100.0]}))
    assert trend == "sideways" and vol == "calm"


def test_schema_accepts_regime_type() -> None:
    """ConditionConfig should correctly store type and regimes."""
    c = ConditionConfig(type="regime", regimes=["bull", "sideways"])
    assert c.type == "regime" and c.regimes == ["bull", "sideways"]


def test_regime_stability_over_short_term_noise() -> None:
    """The regime detector should be robust to short‑term noise spikes."""
    # Create a bullish series then inject a brief volatile dip.
    base = _series(0.004, 0.003)
    noisy = base.copy()
    # Introduce a 5‑bar high‑volatility dip around the middle of the series.
    mid = len(noisy) // 2
    rng = np.random.default_rng(1)
    dip = rng.normal(-0.02, 0.08, 5)
    noisy.loc[mid : mid + 4, "close"] *= np.exp(np.cumsum(dip))
    trend, vol = detect_regime(noisy)
    # Expect the overall regime to remain bull/calm despite the local noise.
    assert trend == "bull" and vol == "calm"


def test_transition_detection() -> None:
    """Ensure that a clear regime transition is captured by the detector."""
    # First half bullish, second half bearish with higher volatility.
    bullish = _series(0.005, 0.002, n=60)
    bearish = _series(-0.005, 0.006, n=60)
    combined = pd.concat([bullish, bearish], ignore_index=True)
    trend, vol = detect_regime(combined)
    # The detector should prioritize the most recent regime characteristics.
    assert trend == "bear" and vol == "stressed"