"""
Measures IC (Information Coefficient) decay for each strategy signal.
IC = Spearman correlation between signal and subsequent forward return.
Fits exponential decay: IC(t) = IC_0 * exp(-lambda * t)

Usage:
    tracker = AlphaDecayTracker()
    profile = tracker.compute_ic_profile(signals, prices, "momentum")
    scaled_conf = tracker.scale_confidence(0.7, profile, staleness_hours=2)
    valid = tracker.filter_signals(signals, profile, min_ic=0.1)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from scipy.stats import spearmanr
from scipy.optimize import curve_fit
from typing import Dict, Tuple


@dataclass
class DecayProfile:
    strategy_name: str
    ic_0: float  # IC at t=0
    half_life_hours: float  # hours until IC halves
    horizons: Dict[int, float] = field(default_factory=dict)  # {horizon_hours: ic_value}


class AlphaDecayTracker:
    """
    Measures how quickly a strategy's predictive power decays over time.

    Fits an exponential decay model to Spearman IC across multiple horizons.
    Used to scale down signal confidence when signals are stale and to
    provide confirmation filters for entry decisions.
    """

    # Horizons to measure IC at: 1h, 4h, 1d, 5d, 20d
    HORIZONS: Tuple[int, ...] = (1, 4, 24, 120, 480)

    def _validate_inputs(self, signals: pd.Series, prices: pd.DataFrame) -> None:
        if not isinstance(signals, pd.Series):
            raise TypeError("signals must be a pandas Series")
        if not isinstance(prices, pd.DataFrame):
            raise TypeError("prices must be a pandas DataFrame")
        if "close" not in prices.columns:
            raise ValueError("prices DataFrame must contain a 'close' column")
        if signals.empty:
            raise ValueError("signals Series is empty")
        if prices.empty:
            raise ValueError("prices DataFrame is empty")

    def compute_ic_profile(
        self,
        signals: pd.Series,
        prices: pd.DataFrame,
        strategy_name: str,
    ) -> DecayProfile:
        """
        Compute IC at each horizon and fit exponential decay.

        Args:
            signals: pd.Series of -1/0/+1 indexed by datetime.
            prices: pd.DataFrame with a 'close' column at the same frequency as signals.
            strategy_name: Name used for labeling the profile.

        Returns:
            DecayProfile containing IC values per horizon and a fitted half‑life.
            If insufficient data, returns a profile with ic_0 = 0 and infinite half‑life.
        """
        self._validate_inputs(signals, prices)

        ics: Dict[int, float] = {}

        # Minimum number of overlapping observations required per horizon
        min_common = 30
        min_valid = 20

        for h in self.HORIZONS:
            # forward return over horizon h
            fwd_ret = prices["close"].pct_change(h).shift(-h)
            common_idx = signals.index.intersection(fwd_ret.index)

            if len(common_idx) < min_common:
                continue

            s = signals.loc[common_idx].dropna()
            r = fwd_ret.loc[s.index].dropna()
            s = s.loc[r.index]  # align both series

            if len(s) < min_valid:
                continue

            ic_val, _ = spearmanr(s, r)
            if np.isnan(ic_val):
                continue

            # Apply a simple outlier filter: discard extreme IC values > 0.99
            if abs(ic_val) > 0.99:
                continue

            ics[h] = float(ic_val)

        # Fallback when not enough horizons have valid ICs
        if len(ics) < 2:
            return DecayProfile(
                strategy_name=strategy_name,
                ic_0=0.0,
                half_life_hours=float("inf"),
                horizons=ics,
            )

        horizons_arr = np.array(list(ics.keys()), dtype=float)
        ic_arr = np.array(list(ics.values()), dtype=float)

        # Fit exponential decay with bounds to enforce positive decay rate
        def exp_decay(t: np.ndarray, ic0: float, lam: float) -> np.ndarray:
            return ic0 * np.exp(-lam * t)

        try:
            popt, _ = curve_fit(
                exp_decay,
                horizons_arr,
                ic_arr,
                p0=[ic_arr[0], 0.01],
                bounds=([0, 0], [1, np.inf]),
                maxfev=2000,
            )
            ic_0, lam = float(popt[0]), float(popt[1])
            half_life = np.log(2) / lam if lam > 0 else float("inf")
        except Exception:
            ic_0 = float(ic_arr[0])
            half_life = float("inf")

        return DecayProfile(
            strategy_name=strategy_name,
            ic_0=ic_0,
            half_life_hours=half_life,
            horizons=ics,
        )

    def scale_confidence(
        self,
        base_confidence: float,
        profile: DecayProfile,
        staleness_hours: float,
    ) -> float:
        """
        Scale a signal's confidence downward based on how stale it is.

        Args:
            base_confidence: Raw confidence score in [0, 1].
            profile: Fitted DecayProfile for the strategy.
            staleness_hours: Hours since the signal was generated.

        Returns:
            Adjusted confidence in [0, 1]. Returns base_confidence unchanged
            when half‑life is infinite (signal does not decay) or when the
            provided confidence is out of bounds.
        """
        if not (0.0 <= base_confidence <= 1.0):
            raise ValueError("base_confidence must be within [0, 1]")

        if profile.half_life_hours == float("inf") or profile.half_life_hours <= 0:
            return float(base_confidence)

        decay_factor = np.exp(-staleness_hours * np.log(2) / profile.half_life_hours)
        adjusted = base_confidence * max(decay_factor, 0.0)
        return float(min(max(adjusted, 0.0), 1.0))

    def filter_signals(
        self,
        signals: pd.Series,
        profile: DecayProfile,
        min_ic: float = 0.05,
    ) -> pd.Series:
        """
        Apply a confirmation filter based on the decay profile.

        Signals are retained only if the absolute IC at the shortest horizon
        exceeds ``min_ic`` and the signal's absolute value is at least ``min_ic``.

        Args:
            signals: Original signal series.
            profile: DecayProfile containing IC values.
            min_ic: Minimum absolute IC required for a signal to be considered valid.

        Returns:
            A boolean Series (indexed like ``signals``) where True indicates the
            signal passes the confirmation filter.
        """
        if not profile.horizons:
            # No IC information – be conservative and reject all signals
            return pd.Series(False, index=signals.index)

        # Use the smallest horizon available as a proxy for immediate predictive power
        shortest_h = min(profile.horizons.keys())
        ic_at_shortest = abs(profile.horizons[shortest_h])

        if ic_at_shortest < min_ic:
            return pd.Series(False, index=signals.index)

        # Apply strength filter on the signal itself
        strength_mask = signals.abs() >= min_ic
        return strength_mask.reindex(signals.index, fill_value=False)