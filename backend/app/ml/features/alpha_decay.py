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

    # Horizons to measure IC at: 1h, 4h, 1d, 5d, 20d
    HORIZONS: List[int] = [1, 4, 24, 120, 480]

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
            ValueError: If ``prices`` does not contain a ``close`` column.
            TypeError: If inputs are not of expected pandas types.
        """
        if not isinstance(signals, pd.Series):
            logger.error("Invalid type for signals: expected pd.Series, got %s", type(signals))
            raise TypeError("signals must be a pandas Series")
        if not isinstance(prices, pd.DataFrame):
            logger.error("Invalid type for prices: expected pd.DataFrame, got %s", type(prices))
            raise TypeError("prices must be a pandas DataFrame")

        if "close" not in prices.columns:
            logger.error("prices DataFrame missing required 'close' column")
            raise ValueError("prices DataFrame must contain a 'close' column")

        ics: Dict[int, float] = {}

        for h in self.HORIZONS:
            try:
                fwd_ret = prices["close"].pct_change(h).shift(-h)
            except Exception as exc:
                logger.error(
                    "Failed to compute forward returns for horizon %d: %s",
                    h,
                    exc,
                    exc_info=True,
                )
                continue

            common = signals.index.intersection(fwd_ret.index)
            if len(common) < 30:
                logger.debug(
                    "Insufficient overlapping data for horizon %d (found %d points)",
                    h,
                    len(common),
                )
                continue

            s = signals.loc[common].dropna()
            r = fwd_ret.loc[s.index].dropna()
            s = s.loc[r.index]

            if len(s) < 20:
                logger.debug(
                    "After alignment, insufficient points for horizon %d (found %d points)",
                    h,
                    len(s),
                )
                continue

            try:
                ic_val, _ = spearmanr(s, r)
            except Exception as exc:
                logger.error(
                    "Spearman correlation failed for horizon %d: %s",
                    h,
                    exc,
                    exc_info=True,
                )
                continue

            if not np.isnan(ic_val):
                ics[h] = float(ic_val)

        if len(ics) < 2:
            logger.info(
                "Not enough IC points to fit decay model for strategy %s; returning default profile",
                strategy_name,
            )
            return DecayProfile(
                strategy_name=strategy_name,
                ic_0=0.0,
                half_life_hours=float("inf"),
                horizons=ics,
            )

        horizons_arr = np.array(list(ics.keys()), dtype=float)
        ic_arr = np.array(list(ics.values()), dtype=float)

        def exp_decay(t: np.ndarray, ic0: float, lam: float) -> np.ndarray:
            return ic0 * np.exp(-lam * t)

        try:
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
            logger.error(
                "Curve fitting failed for strategy %s: %s",
                strategy_name,
                exc,
                exc_info=True,
            )
            ic_0 = float(ic_arr[0]) if ic_arr.size > 0 else 0.0
            half_life = float("inf")
        except Exception as exc:
            logger.error(
                "Unexpected error during curve fitting for strategy %s: %s",
                strategy_name,
                exc,
                exc_info=True,
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
            Adjusted confidence in [0, 1]. Returns ``base_confidence`` unchanged
            when half-life is infinite (signal does not decay).

        Raises:
            ValueError: If ``base_confidence`` is outside the [0, 1] range.
            TypeError: If ``profile`` is not a DecayProfile instance.
        """
        if not isinstance(profile, DecayProfile):
            logger.error(
                "Invalid profile type: expected DecayProfile, got %s",
                type(profile),
            )
            raise TypeError("profile must be a DecayProfile instance")
        if not (0.0 <= base_confidence <= 1.0):
            logger.error(
                "Base confidence out of bounds: %s (must be within [0, 1])",
                base_confidence,
            )
            raise ValueError("base_confidence must be between 0 and 1")

        if profile.half_life_hours == float("inf") or profile.half_life_hours <= 0:
            return float(base_confidence)

        try:
            decay = np.exp(-staleness_hours * np.log(2) / profile.half_life_hours)
        except Exception as exc:
            logger.error(
                "Failed to compute decay factor for staleness %s hours: %s",
                staleness_hours,
                exc,
                exc_info=True,
            )
            return float(base_confidence)

        return float(base_confidence * max(float(decay), 0.0))