"""`regime` bot condition — OA-style "trade only in these market conditions".

The user directive: different strategies for different market conditions. A bot
condition of type `regime` fires only when the detected (trend, vol) regime is
in the bot's allowed list. Fail-soft: no filter → always allowed; degenerate
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
    """Generate a synthetic price series.

    Parameters
    ----------
    drift : float
        Expected return per period. Must be a numeric type.
    vol : float
        Standard deviation of returns. Must be a numeric type.
    n : int, optional
        Number of observations. Must be a positive integer. Default is 120.

    Returns
    -------
    pd.DataFrame
        DataFrame with a single column ``close`` containing the generated series.

    Raises
    ------
    ValueError
        If any argument is of an invalid type or out of expected range.
    """
    if not isinstance(drift, (int, float)):
        raise ValueError("drift must be a numeric type (int or float).")
    if not isinstance(vol, (int, float)):
        raise ValueError("vol must be a numeric type (int or float).")
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer.")

    rng = np.random.default_rng(0)
    rets = rng.normal(drift, vol, n)
    close = 100 * np.exp(np.cumsum(rets))
    return pd.DataFrame({"close": close})


def _cond(regimes):
    """Create a ConditionConfig for regime checking.

    Parameters
    ----------
    regimes : list[str] | tuple[str, ...] | None
        Allowed regimes. Must be a list/tuple of strings or None. Empty list or
        None disables filtering (always allowed).

    Returns
    -------
    ConditionConfig
        Config object with type set to ``regime`` and the provided regimes.

    Raises
    ------
    ValueError
        If ``regimes`` is not None, list, or tuple, or contains non‑string items.
    """
    if regimes is not None and not isinstance(regimes, (list, tuple)):
        raise ValueError("regimes must be a list, tuple, or None.")
    if isinstance(regimes, (list, tuple)):
        for r in regimes:
            if not isinstance(r, str):
                raise ValueError("each regime must be a string.")
    return ConditionConfig(type="regime", regimes=regimes)


def test_detect_regime_bull_calm():
    trend, vol = detect_regime(_series(0.003, 0.004))
    assert trend == "bull" and vol == "calm"


def test_detect_regime_bear_stressed():
    # Deterministic recent selloff with elevated late vol: calm early, sharp
    # decline in the last 20 bars → recent_ret < −0.002 AND vol_ratio > 1.3.
    early = [100.0 - i * 0.01 for i in range(100)]         # flat-ish, low vol
    # Volatile selloff: alternating −5%/−1% steps → net negative drift AND high
    # realized vol (a constant-rate decline has ZERO vol and reads as sideways).
    px, late = early[-1], []
    for k in range(24):
        px *= 0.95 if k % 2 == 0 else 0.99
        late.append(px)
    df = pd.DataFrame({"close": early + late})
    trend, vol = detect_regime(df)
    assert trend == "bear" and vol == "stressed"


def test_condition_passes_when_regime_matches():
    df = _series(0.003, 0.004)  # bull/calm
    assert evaluate_condition(_cond(["bull"]), df, 100.0)
    assert evaluate_condition(_cond(["calm"]), df, 100.0)
    assert evaluate_condition(_cond(["bull", "stressed"]), df, 100.0)  # OR semantics


def test_condition_blocks_when_regime_absent():
    df = _series(0.003, 0.004)  # bull/calm
    assert not evaluate_condition(_cond(["bear"]), df, 100.0)
    assert not evaluate_condition(_cond(["stressed"]), df, 100.0)


def test_empty_regimes_always_allowed():
    assert evaluate_condition(_cond([]), _series(0.003, 0.004), 100.0)
    assert evaluate_condition(_cond(None), _series(0.003, 0.004), 100.0)


def test_degenerate_data_is_sideways_calm_not_error():
    trend, vol = detect_regime(pd.DataFrame({"close": [100.0, 100.0]}))
    assert trend == "sideways" and vol == "calm"


def test_schema_accepts_regime_type():
    c = ConditionConfig(type="regime", regimes=["bull", "sideways"])
    assert c.type == "regime" and c.regimes == ["bull", "sideways"]