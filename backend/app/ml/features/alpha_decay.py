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
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


@dataclass
class DecayProfile:
    strategy_name: str
    ic_0: float  # IC at t=0
    half_life_hours: float  # hours until IC halves
    horizons: dict = field(default_factory=dict)  # {horizon_hours: ic_value}


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
        start_time = time.perf_counter()
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
            exec_time = time.perf_counter() - start_time
            logger.info(
                "AlphaDecay compute_ic_profile completed (insufficient data)",
                extra={
                    "strategy_name": strategy_name,
                    "signal_count": signal_count,
                    "execution_time_sec": exec_time,
                    "ic_0": profile.ic_0,
                    "half_life_hours": profile.half_life_hours,
                },
            )
            return profile

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
        exec_time = time.perf_counter() - start_time
        logger.info(
            "AlphaDecay compute_ic_profile completed",
            extra={
                "strategy_name": strategy_name,
                "signal_count": signal_count,
                "execution_time_sec": exec_time,
                "ic_0": profile.ic_0,
                "half_life_hours": profile.half_life_hours,
                "horizons": profile.horizons,
            },
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
        if profile.half_life_hours == float("inf") or profile.half_life_hours <= 0:
            adjusted = float(base_confidence)
            logger.info(
                "AlphaDecay scale_confidence skipped (no decay)",
                extra={
                    "strategy_name": profile.strategy_name,
                    "base_confidence": base_confidence,
                    "staleness_hours": staleness_hours,
                    "adjusted_confidence": adjusted,
                },
            )
            return adjusted

        decay = np.exp(-staleness_hours * np.log(2) / profile.half_life_hours)
        adjusted = float(base_confidence * max(float(decay), 0.0))
        logger.info(
            "AlphaDecay scale_confidence applied",
            extra={
                "strategy_name": profile.strategy_name,
                "base_confidence": base_confidence,
                "staleness_hours": staleness_hours,
                "half_life_hours": profile.half_life_hours,
                "decay_factor": float(decay),
                "adjusted_confidence": adjusted,
            },
        )
        return adjusted