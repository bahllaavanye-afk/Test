"""
Measures IC (Information Coefficient) decay for each strategy signal.
IC = Spearman correlation between signal and subsequent forward return.
Fits exponential decay: IC(t) = IC_0 * exp(-lambda * t)

Usage:
    tracker = AlphaDecayTracker()
    profile = tracker.compute_ic_profile(signals, prices, "momentum")
    scaled_conf = tracker.scale_confidence(0.7, profile, staleness_hours=2)
"""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, validator
from scipy.optimize import curve_fit
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


class DecayProfile(BaseModel):
    """
    Pydantic model representing the decay characteristics of a strategy's predictive power.
    """

    strategy_name: str = Field(
        ...,
        description="Name of the strategy for which the decay profile is computed.",
        example="momentum",
    )
    ic_0: float = Field(
        ...,
        description="IC at time zero (t=0). Correlation between signal and immediate forward return.",
        ge=-1.0,
        le=1.0,
        example=0.12,
    )
    half_life_hours: float = Field(
        ...,
        description=(
            "Half‑life of IC decay expressed in hours. "
            "Use `float('inf')` if decay is not observed."
        ),
        example=48.0,
    )
    horizons: dict[int, float] = Field(
        default_factory=dict,
        description=(
            "Mapping from horizon (in hours) to the IC value observed at that horizon."
        ),
        example={1: 0.12, 4: 0.08, 24: 0.04},
    )

    @validator("half_life_hours")
    def validate_half_life(cls, v: float) -> float:
        if v != float("inf") and v <= 0:
            raise ValueError("half_life_hours must be positive or infinite")
        return v

    @validator("horizons")
    def validate_horizons(cls, v: dict[int, float]) -> dict[int, float]:
        for horizon, ic_val in v.items():
            if not isinstance(horizon, int) or horizon <= 0:
                raise ValueError("horizon keys must be positive integers")
            if not isinstance(ic_val, (float, int)):
                raise ValueError("IC values must be numeric")
            if ic_val < -1 or ic_val > 1:
                raise ValueError("IC values must be between -1 and 1")
        return v


class AlphaDecayTracker:
    """
    Measures how quickly a strategy's predictive power decays over time.

    Fits an exponential decay model to Spearman IC across multiple horizons.
    Used to scale down signal confidence when signals are stale.
    """

    # Horizons to measure IC at: 1h, 4h, 1d, 5d, 20d
    HORIZONS: list[int] = [1, 4, 24, 120, 480]

    def compute_ic_profile(
        self,
        signals: pd.Series,
        prices: pd.DataFrame,
        strategy_name: str,
    ) -> DecayProfile:
        """
        Compute IC at each horizon and fit exponential decay.

        Args:
            signals: pd.Series of -1/0/+1 indexed by datetime
            prices: pd.DataFrame with 'close' column at same frequency as signals
            strategy_name: name for labelling the profile

        Returns:
            DecayProfile with IC at each horizon and fitted half-life in hours.
            Raises ValueError if prices has no 'close' column.
        """
        start_time = time.time()
        signal_count = int(signals.shape[0])

        if "close" not in prices.columns:
            raise ValueError("prices DataFrame must contain a 'close' column")

        ics: dict[int, float] = {}

        for h in self.HORIZONS:
            fwd_ret = prices["close"].pct_change(h).shift(-h)
            common = signals.index.intersection(fwd_ret.index)
            if len(common) < 30:
                continue

            s = signals.loc[common].dropna()
            r = fwd_ret.loc[s.index].dropna()
            s = s.loc[r.index]

            if len(s) < 20:
                continue

            ic_val, _ = spearmanr(s, r)
            if not np.isnan(ic_val):
                ics[h] = float(ic_val)

        if len(ics) < 2:
            profile = DecayProfile(
                strategy_name=strategy_name,
                ic_0=0.0,
                half_life_hours=float("inf"),
                horizons=ics,
            )
        else:
            horizons_arr = np.array(list(ics.keys()), dtype=float)
            ic_arr = np.array(list(ics.values()), dtype=float)

            try:

                def exp_decay(t: np.ndarray, ic0: float, lam: float) -> np.ndarray:
                    return ic0 * np.exp(-lam * t)

                popt, _ = curve_fit(
                    exp_decay,
                    horizons_arr,
                    ic_arr,
                    p0=[float(ic_arr[0]), 0.01],
                    maxfev=1000,
                )
                ic_0, lam = float(popt[0]), float(popt[1])
                half_life = np.log(2) / lam if lam > 0 else float("inf")
            except Exception:
                ic_0 = float(ic_arr[0]) if len(ic_arr) > 0 else 0.0
                half_life = float("inf")

            profile = DecayProfile(
                strategy_name=strategy_name,
                ic_0=ic_0,
                half_life_hours=float(half_life),
                horizons=ics,
            )

        # Approximate P&L as sum of one‑step forward returns aligned with signals
        fwd_one = prices["close"].pct_change().shift(-1)
        common_one = signals.index.intersection(fwd_one.index)
        pnl = float((signals.loc[common_one] * fwd_one.loc[common_one]).sum())

        elapsed = time.time() - start_time
        logger.info(
            {
                "event": "compute_ic_profile",
                "strategy": strategy_name,
                "signal_count": signal_count,
                "execution_time_sec": elapsed,
                "pnl": pnl,
                "half_life_hours": profile.half_life_hours,
                "ic_0": profile.ic_0,
            }
        )
        return profile

    def scale_confidence(
        self,
        base_confidence: float,
        profile: DecayProfile,
        staleness_hours: float,
    ) -> float:
        """
        Scale a signal's confidence downward based on how stale it is.

        Args:
            base_confidence: raw confidence score [0, 1]
            profile: fitted DecayProfile for the strategy
            staleness_hours: hours since the signal was generated

        Returns:
            Adjusted confidence in [0, 1].  Returns base_confidence unchanged
            when half-life is infinite (signal does not decay).
        """
        start_time = time.time()
        if profile.half_life_hours == float("inf") or profile.half_life_hours <= 0:
            adjusted = float(base_confidence)
        else:
            decay = np.exp(
                -staleness_hours * np.log(2) / profile.half_life_hours
            )
            adjusted = float(base_confidence * max(float(decay), 0.0))

        elapsed = time.time() - start_time
        logger.info(
            {
                "event": "scale_confidence",
                "strategy": profile.strategy_name,
                "base_confidence": base_confidence,
                "staleness_hours": staleness_hours,
                "adjusted_confidence": adjusted,
                "execution_time_sec": elapsed,
            }
        )
        return adjusted