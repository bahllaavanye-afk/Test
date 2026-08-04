"""
Market regime detection utilities.

This module provides a Hidden Markov Model‑style detector that classifies
the current market into one of three regimes: trending, mean‑reverting,
or high volatility. The resulting regime is used to scale Kelly position
sizing throughout the trading platform.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class Regime(str, Enum):
    """Enum representing possible market regimes."""
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
    Container for the current market regime and associated metrics.

    Attributes
    ----------
    regime: Regime
        The identified market regime.
    confidence: float
        Confidence in the classification (0‑1).
    vol_20d: float
        Realized 20‑day volatility, annualized (as a decimal).
    hurst: float
        Hurst exponent; >0.5 indicates trending, <0.5 mean‑reverting.
    sizing_multiplier: float
        Multiplier applied to Kelly sizing based on the regime.
    updated_at: datetime
        Timestamp of the most recent calculation.
    """
    regime: Regime
    confidence: float       # 0-1
    vol_20d: float          # realized vol (annualized %)
    hurst: float            # Hurst exponent: >0.5 trending, <0.5 mean-reverting
    sizing_multiplier: float
    updated_at: datetime

    def to_dict(self) -> dict:
        """
        Convert the state to a JSON‑serialisable dictionary.

        Returns
        -------
        dict
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
        Generate a short textual description of the regime.

        Returns
        -------
        str
            Human‑readable description summarising the regime and sizing.
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
    Estimate the Hurst exponent using rescaled range (R/S) analysis.

    Parameters
    ----------
    prices : np.ndarray
        Array of price values.
    max_lag : int, optional
        Maximum lag to consider when computing the rescaled range, by default 20.

    Returns
    -------
    float
        Estimated Hurst exponent, clipped to the interval [0.1, 0.9].
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
    Detect the market regime from a series of closing prices.

    The function requires at least 30 price points; otherwise it returns an
    ``UNKNOWN`` regime with zero confidence.

    Parameters
    ----------
    prices : list[float]
        List of close prices, ordered from oldest to newest (most recent last).
    high_vol_threshold : float, optional
        Annualized volatility threshold above which the regime is considered
        ``HIGH_VOL``, by default 0.25 (25 %).

    Returns
    -------
    RegimeState
        Structured result containing the identified regime and supporting metrics.
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

    # 20‑day realized volatility (annualized)
    rets = np.diff(np.log(arr[-21:] + 1e-10))
    vol_20d = float(np.std(rets) * np.sqrt(252))

    # Hurst exponent on last 60+ bars
    hurst = _hurst_exponent(arr[-min(100, len(arr)):])

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
    In‑memory cache of regime states per ticker symbol.

    The monitor is updated by the price‑feed task and queried by the risk
    manager to retrieve sizing multipliers.
    """
    def __init__(self) -> None:
        self._states: dict[str, RegimeState] = {}

    def update(self, symbol: str, prices: list[float]) -> RegimeState:
        """
        Compute and store the regime for a given symbol.

        Parameters
        ----------
        symbol : str
            Ticker symbol.
        prices : list[float]
            Recent price series for the symbol.

        Returns
        -------
        RegimeState
            The newly calculated regime state.
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
            Ticker symbol.

        Returns
        -------
        Optional[RegimeState]
            Cached state if present, otherwise ``None``.
        """
        return self._states.get(symbol)

    def get_multiplier(self, symbol: str) -> float:
        """
        Obtain the sizing multiplier for a symbol.

        Returns a default multiplier of 0.75 when the regime is unknown.

        Parameters
        ----------
        symbol : str
            Ticker symbol.

        Returns
        -------
        float
            Sizing multiplier associated with the current regime.
        """
        state = self._states.get(symbol)
        return state.sizing_multiplier if state else 0.75

    def all_states(self) -> dict[str, dict]:
        """
        Export all cached regime states as dictionaries.

        Returns
        -------
        dict[str, dict]
            Mapping from symbol to its regime state dictionary.
        """
        return {sym: state.to_dict() for sym, state in self._states.items()}


regime_monitor = RegimeMonitor()