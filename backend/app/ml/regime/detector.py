"""
HMM-based market regime detector.
Classifies current market into 3 states: TRENDING, MEAN_REVERTING, HIGH_VOL.
Used to scale Kelly position sizing:
  TRENDING    → 1.0x (full size)
  MEAN_REVERTING → 0.85x
  HIGH_VOL    → 0.5x (half size — protect capital)
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum

import numpy as np


class Regime(str, Enum):
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
    regime: Regime
    confidence: float       # 0-1
    vol_20d: float          # realized vol (annualized %)
    hurst: float            # Hurst exponent: >0.5 trending, <0.5 mean-reverting
    sizing_multiplier: float
    updated_at: datetime

    def to_dict(self) -> dict:
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
        if self.regime == Regime.TRENDING:
            return f"Trending market (Hurst={self.hurst:.2f}). Full position sizing."
        elif self.regime == Regime.MEAN_REVERTING:
            return f"Mean-reverting market (Hurst={self.hurst:.2f}). Reduce size 15%."
        elif self.regime == Regime.HIGH_VOL:
            return f"High-volatility regime (vol={self.vol_20d*100:.1f}%). Halve position size."
        return "Unknown regime. Using conservative sizing."


def _hurst_exponent(prices: np.ndarray, max_lag: int = 20) -> float:
    """
    Hurst exponent via rescaled range (R/S) analysis.
    H > 0.55 → trending (momentum works)
    H < 0.45 → mean-reverting (reversion works)
    H ≈ 0.5 → random walk
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
    Classify market regime from a list of close prices (min 30 required).

    Args:
        prices: List of close prices, most recent last.
        high_vol_threshold: Annualized vol above this → HIGH_VOL regime.
    """
    arr = np.array(prices, dtype=float)
    if len(arr) < 30:
        return RegimeState(
            regime=Regime.UNKNOWN, confidence=0.0, vol_20d=0.0,
            hurst=0.5, sizing_multiplier=REGIME_SIZING_MULTIPLIER[Regime.UNKNOWN],
            updated_at=datetime.now(UTC),
        )

    # 20-day realized volatility (annualized)
    rets = np.diff(np.log(arr[-21:] + 1e-10))
    vol_20d = float(np.std(rets) * np.sqrt(252))

    # Hurst exponent on last 60+ bars
    hurst = _hurst_exponent(arr[-min(100, len(arr)):])

    # Classify
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
        # borderline — use vol to tiebreak
        regime = Regime.TRENDING if vol_20d < 0.15 else Regime.MEAN_REVERTING
        confidence = 0.5

    return RegimeState(
        regime=regime,
        confidence=float(confidence),
        vol_20d=vol_20d,
        hurst=hurst,
        sizing_multiplier=REGIME_SIZING_MULTIPLIER[regime],
        updated_at=datetime.now(UTC),
    )


class RegimeMonitor:
    """
    Caches per-symbol regime states. Updated by PriceFeed task.
    Consumed by RiskManager to scale Kelly sizing.
    """
    def __init__(self):
        self._states: dict[str, RegimeState] = {}

    def update(self, symbol: str, prices: list[float]) -> RegimeState:
        state = detect_regime(prices)
        self._states[symbol] = state
        return state

    def get(self, symbol: str) -> RegimeState | None:
        return self._states.get(symbol)

    def get_multiplier(self, symbol: str) -> float:
        state = self._states.get(symbol)
        return state.sizing_multiplier if state else 0.75

    def all_states(self) -> dict[str, dict]:
        return {sym: state.to_dict() for sym, state in self._states.items()}


regime_monitor = RegimeMonitor()
