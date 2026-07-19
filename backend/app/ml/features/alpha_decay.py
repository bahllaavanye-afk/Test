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

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from scipy.stats import spearmanr
from scipy.optimize import curve_fit


@dataclass
class DecayProfile:
    strategy_name: str
    ic_0: float           # IC at t=0
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
        signals: pd.Series | None,
        prices: pd.DataFrame | None,
        strategy_name: str,
    ) -> DecayProfile:
        """
        Compute IC at each horizon and fit exponential decay.

        Args:
            signals: pd.Series of -1/0/+1 indexed by datetime, or None
            prices: pd.DataFrame with 'close' column at same frequency as signals, or None
            strategy_name: name for labelling the profile

        Returns:
            DecayProfile with IC at each horizon and fitted half-life in hours.
            Returns a zeroed profile when inputs are missing or insufficient.
            Raises ValueError if a non‑empty ``prices`` DataFrame lacks a 'close' column.
        """
        # Guard against None or empty inputs
        if signals is None or signals.empty:
            return DecayProfile(
                strategy_name=strategy_name,
                ic_0=0.0,
                half_life_hours=float("inf"),
                horizons={},
            )

        if prices is None or prices.empty:
            raise ValueError("prices DataFrame must contain a non‑empty 'close' column")

        if "close" not in prices.columns:
            raise ValueError("prices DataFrame must contain a 'close' column")

        ics: dict[int, float] = {}

        for h in self.HORIZONS:
            # Ensure horizon is positive to avoid off‑by‑one or zero‑division issues
            if h <= 0:
                continue

            # pct_change with a large horizon on a tiny DataFrame can return all NaNs;
            # shift(-h) aligns forward returns correctly.
            fwd_ret = prices["close"].pct_change(h).shift(-h)
            common = signals.index.intersection(fwd_ret.index)

            # Require a minimal overlap to produce a meaningful correlation
            if len(common) < 30:
                continue

            s = signals.loc[common].dropna()
            r = fwd_ret.loc[s.index].dropna()
            # Align again after dropping NaNs
            s = s.loc[r.index]

            if len(s) < 20:
                continue

            ic_val, _ = spearmanr(s, r)
            if not np.isnan(ic_val):
                ics[h] = float(ic_val)

        # If we have fewer than two points, we cannot fit a decay curve
        if len(ics) < 2:
            return DecayProfile(
                strategy_name=strategy_name,
                ic_0=0.0,
                half_life_hours=float("inf"),
                horizons=ics,
            )

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
            ic_0 = float(ic_arr[0]) if ic_arr.size > 0 else 0.0
            half_life = float("inf")

        return DecayProfile(
            strategy_name=strategy_name,
            ic_0=ic_0,
            half_life_hours=float(half_life),
            horizons=ics,
        )

    def scale_confidence(
        self,
        base_confidence: float | None,
        profile: DecayProfile | None,
        staleness_hours: float | None,
    ) -> float:
        """
        Scale a signal's confidence downward based on how stale it is.

        Args:
            base_confidence: raw confidence score [0, 1]; None treated as 0.0
            profile: fitted DecayProfile for the strategy; None returns 0.0
            staleness_hours: hours since the signal was generated; None treated as 0

        Returns:
            Adjusted confidence in [0, 1]. Returns base_confidence unchanged
            when half-life is infinite (signal does not decay) or when inputs are invalid.
        """
        # Defensive defaults
        if base_confidence is None:
            base_confidence = 0.0
        if staleness_hours is None:
            staleness_hours = 0.0
        if profile is None:
            return float(base_confidence)

        if profile.half_life_hours == float("inf") or profile.half_life_hours <= 0:
            return float(base_confidence)

        decay = np.exp(
            -staleness_hours * np.log(2) / profile.half_life_hours
        )
        return float(base_confidence * max(float(decay), 0.0))