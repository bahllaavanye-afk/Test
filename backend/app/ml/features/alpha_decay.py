"""
Measures IC (Information Coefficient) decay for each strategy signal.
IC = Spearman correlation between signal and subsequent forward return.
Fits exponential decay: IC(t) = IC_0 * exp(-lambda * t)

Usage:
    tracker = AlphaDecayTracker()
    profile = tracker.compute_ic_profile(signals, prices, "momentum")
    scaled_conf = tracker.scale_confidence(0.7, profile, staleness_hours=2)

    # Entry decision
    if tracker.is_entry_allowed(base_confidence=0.8, profile=profile,
                               staleness_hours=0.5, min_ic=0.05):
        # place order ...

    # Exit decision
    if tracker.should_exit(base_confidence=0.8, profile=profile,
                           staleness_hours=5, exit_conf_threshold=0.2):
        # close position ...
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
    ic_0: float                       # IC at t=0
    half_life_hours: float            # hours until IC halves
    horizons: dict = field(default_factory=dict)  # {horizon_hours: ic_value}


class AlphaDecayTracker:
    """
    Measures how quickly a strategy's predictive power decays over time.

    Fits an exponential decay model to Spearman IC across multiple horizons.
    Used to scale down signal confidence when signals are stale and to
    provide entry/exit decision helpers.
    """

    # Horizons to measure IC at: 1h, 4h, 1d, 5d, 20d
    HORIZONS: list[int] = [1, 4, 24, 120, 480]

    # Default thresholds – can be overridden per call
    DEFAULT_MIN_IC: float = 0.05
    DEFAULT_EXIT_CONF_THRESHOLD: float = 0.2

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
        if "close" not in prices.columns:
            raise ValueError("prices DataFrame must contain a 'close' column")

        ics: dict[int, float] = {}

        for h in self.HORIZONS:
            # forward return over horizon h
            fwd_ret = prices["close"].pct_change(h).shift(-h)
            common = signals.index.intersection(fwd_ret.index)

            # Require a minimal sample size for statistical relevance
            if len(common) < 30:
                continue

            s = signals.loc[common].dropna()
            r = fwd_ret.loc[s.index].dropna()
            s = s.loc[r.index]

            if len(s) < 20:
                continue

            ic_val, _ = spearmanr(s, r)
            if np.isnan(ic_val):
                continue

            # Filter out noisy or insignificant IC values
            if abs(ic_val) < self.DEFAULT_MIN_IC:
                continue

            ics[h] = float(ic_val)

        # If we have too few points, fall back to a trivial profile
        if len(ics) < 2:
            return DecayProfile(
                strategy_name=strategy_name,
                ic_0=0.0,
                half_life_hours=float("inf"),
                horizons=ics,
            )

        # Ensure horizons are sorted for curve fitting
        horizons_arr = np.array(sorted(ics.keys()), dtype=float)
        ic_arr = np.array([ics[int(h)] for h in horizons_arr], dtype=float)

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
        base_confidence: float,
        profile: DecayProfile,
        staleness_hours: float,
        *,
        min_confidence: float = 0.0,
    ) -> float:
        """
        Scale a signal's confidence downward based on how stale it is.

        Args:
            base_confidence: raw confidence score [0, 1]
            profile: fitted DecayProfile for the strategy
            staleness_hours: hours since the signal was generated
            min_confidence: floor to apply after scaling (default 0.0)

        Returns:
            Adjusted confidence in [0, 1]. Returns base_confidence unchanged
            when half-life is infinite (signal does not decay).
        """
        if profile.half_life_hours == float("inf") or profile.half_life_hours <= 0:
            return float(base_confidence)

        decay = np.exp(
            -staleness_hours * np.log(2) / profile.half_life_hours
        )
        scaled = float(base_confidence * max(decay, 0.0))
        return max(scaled, min_confidence)

    def is_entry_allowed(
        self,
        base_confidence: float,
        profile: DecayProfile,
        staleness_hours: float,
        *,
        min_ic: float = DEFAULT_MIN_IC,
        confidence_threshold: float = 0.5,
    ) -> bool:
        """
        Determine whether a new position should be opened.

        Entry is allowed only if:
        * The fitted IC at horizon 0 exceeds ``min_ic``.
        * The scaled confidence (accounting for staleness) is above
          ``confidence_threshold``.
        * The short‑term horizon IC (e.g., 1h) is not weaker than the
          longer‑term horizon IC (e.g., 24h), providing a confirmation filter.

        Args:
            base_confidence: raw confidence score [0, 1] for the signal.
            profile: DecayProfile for the strategy.
            staleness_hours: hours elapsed since signal generation.
            min_ic: minimum absolute IC_0 required for entry.
            confidence_threshold: scaled confidence minimum to permit entry.

        Returns:
            True if entry conditions are satisfied, False otherwise.
        """
        # Check IC magnitude
        if abs(profile.ic_0) < min_ic:
            return False

        # Apply staleness scaling
        scaled_conf = self.scale_confidence(
            base_confidence, profile, staleness_hours
        )
        if scaled_conf < confidence_threshold:
            return False

        # Confirmation filter: short‑term IC should be at least as strong
        # as the longer‑term IC (if both are available)
        short_h = min(self.HORIZONS)
        long_h = max(self.HORIZONS)
        short_ic = profile.horizons.get(short_h)
        long_ic = profile.horizons.get(long_h)
        if short_ic is not None and long_ic is not None:
            if abs(short_ic) < abs(long_ic):
                return False

        return True

    def should_exit(
        self,
        base_confidence: float,
        profile: DecayProfile,
        staleness_hours: float,
        *,
        exit_conf_threshold: float = DEFAULT_EXIT_CONF_THRESHOLD,
    ) -> bool:
        """
        Determine whether an existing position should be closed.

        Exit is triggered when the scaled confidence falls below
        ``exit_conf_threshold``. This captures decay of predictive power and
        prevents holding positions with diminishing edge.

        Args:
            base_confidence: original confidence score when the position was opened.
            profile: DecayProfile for the strategy.
            staleness_hours: hours elapsed since the position entry.
            exit_conf_threshold: confidence level below which to exit.

        Returns:
            True if the position should be exited, False otherwise.
        """
        scaled_conf = self.scale_confidence(
            base_confidence, profile, staleness_hours
        )
        return scaled_conf < exit_conf_threshold