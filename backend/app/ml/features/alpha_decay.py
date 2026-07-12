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
from typing import Dict

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import spearmanr

# Configure module‑level logger
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

    # Horizons to measure IC at: 1h, 4h, 1d, 5d, 20d
    HORIZONS: list[int] = [1, 4, 24, 120, 480]

    def _validate_inputs(
        self,
        signals: pd.Series,
        prices: pd.DataFrame,
        strategy_name: str,
    ) -> None:
        """Validate input types and required columns."""
        if not isinstance(signals, pd.Series):
            logger.error("Invalid type for signals: %s", type(signals))
            raise TypeError("signals must be a pandas Series")
        if not isinstance(prices, pd.DataFrame):
            logger.error("Invalid type for prices: %s", type(prices))
            raise TypeError("prices must be a pandas DataFrame")
        if not isinstance(strategy_name, str):
            logger.error("Invalid type for strategy_name: %s", type(strategy_name))
            raise TypeError("strategy_name must be a string")
        if "close" not in prices.columns:
            logger.error("prices DataFrame missing required 'close' column")
            raise ValueError("prices DataFrame must contain a 'close' column")

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

        Raises:
            TypeError: if input types are incorrect.
            ValueError: if required columns are missing.
        """
        try:
            self._validate_inputs(signals, prices, strategy_name)
        except Exception as exc:
            logger.exception("Input validation failed")
            raise

        ics: dict[int, float] = {}

        for h in self.HORIZONS:
            try:
                fwd_ret = prices["close"].pct_change(h).shift(-h)
            except Exception as exc:
                logger.exception("Failed to compute forward returns for horizon %s", h)
                continue

            common = signals.index.intersection(fwd_ret.index)
            if len(common) < 30:
                continue

            s = signals.loc[common].dropna()
            r = fwd_ret.loc[s.index].dropna()
            s = s.loc[r.index]

            if len(s) < 20:
                continue

            try:
                ic_val, _ = spearmanr(s, r)
            except Exception as exc:
                logger.exception(
                    "Spearman correlation failed for horizon %s (strategy %s)",
                    h,
                    strategy_name,
                )
                continue

            if not np.isnan(ic_val):
                ics[h] = float(ic_val)

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
        except (RuntimeError, ValueError) as exc:
            logger.exception(
                "Curve fitting failed for strategy %s; using fallback values", strategy_name
            )
            ic_0 = float(ic_arr[0]) if ic_arr.size > 0 else 0.0
            half_life = float("inf")
        except Exception as exc:
            logger.exception(
                "Unexpected error during curve fitting for strategy %s", strategy_name
            )
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
    ) -> float:
        """
        Scale a signal's confidence downward based on how stale it is.

        Args:
            base_confidence: raw confidence score [0, 1]
            profile: fitted DecayProfile for the strategy
            staleness_hours: hours since the signal was generated

        Returns:
            Adjusted confidence in [0, 1]. Returns base_confidence unchanged
            when half-life is infinite (signal does not decay).

        Raises:
            ValueError: if base_confidence is outside [0, 1] or staleness_hours is negative.
        """
        if not (0.0 <= base_confidence <= 1.0):
            logger.error(
                "Invalid base_confidence %s; must be within [0, 1]", base_confidence
            )
            raise ValueError("base_confidence must be between 0 and 1")
        if staleness_hours < 0:
            logger.error("Negative staleness_hours %s supplied", staleness_hours)
            raise ValueError("staleness_hours cannot be negative")

        if profile.half_life_hours == float("inf") or profile.half_life_hours <= 0:
            return float(base_confidence)

        decay = np.exp(-staleness_hours * np.log(2) / profile.half_life_hours)
        return float(base_confidence * max(decay, 0.0))