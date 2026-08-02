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
from dataclasses import dataclass, field
from typing import Dict, List

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
    horizons: Dict[int, float] = field(default_factory=dict)  # {horizon_hours: ic_value}


class AlphaDecayTracker:
    """
    Measures how quickly a strategy's predictive power decays over time.

    Fits an exponential decay model to Spearman IC across multiple horizons.
    Used to scale down signal confidence when signals are stale.
    """

    # Horizons to measure IC at: 1h, 4h, 1d, 5d, 20d (in hours)
    HORIZONS: List[int] = [1, 4, 24, 120, 480]

    def _validate_inputs(self, signals: pd.Series, prices: pd.DataFrame) -> None:
        """Validate basic pre‑conditions for the computation."""
        if "close" not in prices.columns:
            raise ValueError("prices DataFrame must contain a 'close' column")

        if signals.empty:
            raise ValueError("signals Series is empty")

        if signals.nunique() <= 1:
            raise ValueError("signals contain no variation; IC cannot be computed")

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
            strategy_name: Name for labeling the profile.

        Returns:
            DecayProfile with IC at each horizon and fitted half‑life in hours.
            If insufficient data, returns a profile with ic_0 = 0 and infinite half‑life.
        """
        self._validate_inputs(signals, prices)

        ics: Dict[int, float] = {}

        for h in self.HORIZONS:
            # Forward return over horizon h
            fwd_ret = prices["close"].pct_change(h).shift(-h)

            # Align timestamps
            common_idx = signals.index.intersection(fwd_ret.index)

            # Require a minimum amount of overlapping data for statistical relevance
            if len(common_idx) < 30:
                logger.debug("Horizon %s: insufficient overlap (%d points)", h, len(common_idx))
                continue

            s = signals.loc[common_idx].dropna()
            r = fwd_ret.loc[s.index].dropna()
            s = s.loc[r.index]

            if len(s) < 20:
                logger.debug("Horizon %s: post‑filter sample size too small (%d)", h, len(s))
                continue

            # Volatility filter: ignore periods where forward return volatility is too low
            if np.std(r) < 1e-4:
                logger.debug("Horizon %s: forward return volatility below threshold", h)
                continue

            ic_val, p_val = spearmanr(s, r)
            if np.isnan(ic_val) or np.isnan(p_val):
                continue

            # Confirmation filter: keep only statistically significant correlations
            if p_val > 0.05:
                logger.debug(
                    "Horizon %s: IC not significant (p=%.3f); skipping", h, p_val
                )
                continue

            ics[h] = float(ic_val)
            logger.debug("Horizon %s: IC=%.4f (p=%.3f)", h, ic_val, p_val)

        # Fallback when not enough points to fit a decay curve
        if len(ics) < 2:
            logger.info("Insufficient IC points (%d); returning default profile", len(ics))
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
                p0=[ic_arr[0], 0.01],
                maxfev=1000,
            )
            ic_0, lam = float(popt[0]), float(popt[1])
            half_life = np.log(2) / lam if lam > 0 else float("inf")
        except Exception as exc:  # pragma: no cover
            logger.warning("Curve fitting failed: %s", exc)
            ic_0 = float(ic_arr[0])
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
            resulting confidence falls below a minimal threshold (set to 0).
        """
        if profile.half_life_hours in (float("inf"), 0):
            return float(base_confidence)

        decay_factor = np.exp(-staleness_hours * np.log(2) / profile.half_life_hours)
        adjusted = base_confidence * max(decay_factor, 0.0)

        # Exit filter: if confidence drops below 5 % treat as zero to avoid noisy trades
        if adjusted < 0.05:
            logger.debug(
                "Confidence %.4f below threshold after decay; returning 0.0",
                adjusted,
            )
            return 0.0

        return float(min(adjusted, 1.0))