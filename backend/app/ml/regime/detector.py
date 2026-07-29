"""
HMM-based market regime detector.

Provides utilities to classify the current market regime into one of three states:
TRENDING, MEAN_REVERTING, or HIGH_VOL. The resulting regime is used to scale
position sizing according to a Kelly framework.

Classes
-------
RegimeState
    Dataclass representing the detected regime and associated metrics.
RegimeMonitor
    In‑memory cache that stores the latest ``RegimeState`` per symbol.

Functions
---------
detect_regime(prices, high_vol_threshold)
    Detects the market regime from a list of close prices.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional, dict, list


class Regime(str, Enum):
    """Enumeration of possible market regimes."""

    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"
    HIGH_VOL = "high_vol"
    UNKNOWN = "unknown"


REGIME_SIZING_MULTIPLIER: dict[Regime, float] = {
    Regime.TRENDING: 1.0,
    Regime.MEAN_REVERTING: 0.85,
    Regime.HIGH_VOL: 0.50,
    Regime.UNKNOWN: 0.75,
}


@dataclass
class RegimeState:
    """
    Container for the detected market regime and supporting statistics.

    Attributes
    ----------
    regime : Regime
        Detected regime.
    confidence : float
        Confidence level in the detection (0‑1).
    vol_20d : float
        Realised 20‑day volatility, annualised (as a decimal).
    hurst : float
        Hurst exponent; >0.5 indicates trending, <0.5 mean‑reverting.
    sizing_multiplier : float
        Multiplier to apply to the Kelly position size.
    updated_at : datetime
        Timestamp of the detection.
    """

    regime: Regime
    confidence: float       # 0-1
    vol_20d: float          # realized vol (annualized %)
    hurst: float            # Hurst exponent: >0.5 trending, <0.5 mean-reverting
    sizing_multiplier: float
    updated_at: datetime

    def to_dict(self) -> dict[str, Any]:
        """
        Serialise the ``RegimeState`` to a JSON‑compatible dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary containing human‑readable fields and a description.
        """
        return {
            "regime": self.regime.value,
            "confidence": round(self.confidence, 3),
            "vol_20d_pct": round(self.vol_20d * 100, 2),
            "hurst_exponent": round(self.hurst, 3),
            "sizing_multiplier": self.sizing_multiplier,
            "updated_at": self.updated_at.isoformat(),
            "description": self._describe(),
        }

    def _describe(self) -> str:
        """
        Generate a short textual description of the current regime.

        Returns
        -------
        str
            Human‑readable description.
        """
        if self.regime == Regime.TRENDING:
            return f"Trending market (Hurst={self.hurst:.2f}). Full position sizing."
        elif self.regime == Regime.MEAN_REVERTING:
            return f"Mean-reverting market (Hurst={self.hurst:.2f}). Reduce size 15%."
        elif self.regime == Regime.HIGH_VOL:
            return f"High-volatility regime (vol={self.vol_20d*100:.1f}%). Halve position size."
        return "Unknown regime. Using conservative sizing."


def _hurst_exponent(prices: np.ndarray, max_lag: int = 20) -> float:
    """
    Compute the Hurst exponent using the rescaled range (R/S) method.

    Parameters
    ----------
    prices : np.ndarray
        Array of price values.
    max_lag : int, optional
        Maximum lag to consider when constructing the R/S statistics. Default is 20.

    Returns
    -------
    float
        Estimated Hurst exponent clipped to the range [0.1, 0.9].
    """
    if len(prices) < 30:
        return 0.5
    returns = np.diff(np.log(prices + 1e-10))
    lags = range(2, min(max_lag, len(returns) // 2))
    rs_values = []
    for lag in lags:
        chunks = [returns[i:i+lag] for i in range(0, len(returns) - lag, lag)]
        if not chunks:
            continue
        rs_per_chunk = []
        for chunk in chunks:
            if len(chunk) < 2:
                continue
            mean = np.mean(chunk)
            deviations = np.cumsum(chunk - mean)
            r = np.max(deviations) - np.min(deviations)
            s = np.std(chunk, ddof=1)
            if s > 0:
                rs_per_chunk.append(r / s)
        if rs_per_chunk:
            rs_values.append((lag, np.mean(rs_per_chunk)))
    if len(rs_values) < 3:
        return 0.5
    lags_log = np.log([x[0] for x in rs_values])
    rs_log = np.log([x[1] for x in rs_values])
    hurst = np.polyfit(lags_log, rs_log, 1)[0]
    return float(np.clip(hurst, 0.1, 0.9))


def detect_regime(prices: list[float], high_vol_threshold: float = 0.25) -> RegimeState:
    """
    Detect the market regime from a sequence of close prices.

    Parameters
    ----------
    prices : list[float]
        List of close prices, ordered from oldest to newest (most recent last).
    high_vol_threshold : float, optional
        Annualised volatility threshold above which the regime is classified as
        ``HIGH_VOL``. Default is 0.25 (25 %).

    Returns
    -------
    RegimeState
        The detected regime together with confidence, volatility, Hurst exponent,
        and sizing multiplier.
    """
    arr = np.array(prices, dtype=float)
    if len(arr) < 30:
        return RegimeState(
            regime=Regime.UNKNOWN,
            confidence=0.0,
            vol_20d=0.0,
            hurst=0.5,
            sizing_multiplier=REGIME_SIZING_MULTIPLIER[Regime.UNKNOWN],
            updated_at=datetime.now(timezone.utc),
        )

    # 20‑day realised volatility (annualised)
    rets = np.diff(np.log(arr[-21:] + 1e-10))
    vol_20d = float(np.std(rets) * np.sqrt(252))

    # Hurst exponent on the most recent data (up to 100 bars)
    hurst = _hurst_exponent(arr[-min(100, len(arr)) :])

    # Classification logic
    if vol_20d > high_vol_threshold:
        regime = Regime.HIGH_VOL
        confidence = min(1.0, (vol_20d - high_vol_threshold) / high_vol_threshold + 0.6)
    elif hurst > 0.55:
        regime = Regime.TRENDING
        confidence = min(1.0, (hurst - 0.5) * 4)
    elif hurst < 0.45:
        regime = Regime.MEAN_REVERTING
        confidence = min(1.0, (0.5 - hurst) * 4)
    else:
        # Borderline case – use volatility to break the tie
        regime = Regime.TRENDING if vol_20d < 0.15 else Regime.MEAN_REVERTING
        confidence = 0.5

    return RegimeState(
        regime=regime,
        confidence=float(confidence),
        vol_20d=vol_20d,
        hurst=hurst,
        sizing_multiplier=REGIME_SIZING_MULTIPLIER[regime],
        updated_at=datetime.now(timezone.utc),
    )


class RegimeMonitor:
    """
    In‑memory cache that stores the latest ``RegimeState`` for each symbol.

    The monitor is updated by the price‑feed task and queried by risk management
    components to obtain the appropriate sizing multiplier.
    """

    def __init__(self) -> None:
        """Initialize an empty cache."""
        self._states: dict[str, RegimeState] = {}

    def update(self, symbol: str, prices: list[float]) -> RegimeState:
        """
        Detect and store the regime for a given symbol.

        Parameters
        ----------
        symbol : str
            Ticker symbol for which the regime is being updated.
        prices : list[float]
            List of recent close prices for the symbol.

        Returns
        -------
        RegimeState
            The newly detected regime state.
        """
        state = detect_regime(prices)
        self._states[symbol] = state
        return state

    def get(self, symbol: str) -> Optional[RegimeState]:
        """
        Retrieve the cached regime state for a symbol.

        Parameters
        ----------
        symbol : str
            Ticker symbol to query.

        Returns
        -------
        Optional[RegimeState]
            The cached ``RegimeState`` if present, otherwise ``None``.
        """
        return self._states.get(symbol)

    def get_multiplier(self, symbol: str) -> float:
        """
        Obtain the sizing multiplier for a symbol.

        If the symbol is not cached, a default multiplier of 0.75 is returned.

        Parameters
        ----------
        symbol : str
            Ticker symbol to query.

        Returns
        -------
        float
            Sizing multiplier associated with the cached regime or the default.
        """
        state = self._states.get(symbol)
        return state.sizing_multiplier if state else 0.75

    def all_states(self) -> dict[str, dict]:
        """
        Return a snapshot of all cached regime states as serialisable dictionaries.

        Returns
        -------
        dict[str, dict]
            Mapping from symbol to its ``RegimeState`` representation.
        """
        return {sym: state.to_dict() for sym, state in self._states.items()}


regime_monitor = RegimeMonitor()