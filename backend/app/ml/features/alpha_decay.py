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
from typing import Dict, List


@dataclass
class DecayProfile:
    """
    Container for the decay characteristics of a strategy's predictive power.

    Attributes
    ----------
    strategy_name: str
        Human‑readable name of the strategy.
    ic_0: float
        Information coefficient at horizon zero.
    half_life_hours: float
        Estimated half‑life of the IC decay expressed in hours.
    horizons: Dict[int, float]
        Mapping from horizon (in hours) to measured IC value.
    """
    strategy_name: str
    ic_0: float                     # IC at t=0
    half_life_hours: float          # hours until IC halves
    horizons: Dict[int, float] = field(default_factory=dict)  # {horizon_hours: ic_value}


class AlphaDecayTracker:
    """
    Measures how quickly a strategy's predictive power decays over time.

    The class fits an exponential decay model to Spearman IC values across
    multiple horizons and provides a utility to down‑scale confidence for
    stale signals.
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
        Compute the information coefficient (IC) at each predefined horizon
        and fit an exponential decay curve.

        Parameters
        ----------
        signals : pd.Series
            Series of signal values (-1, 0, +1) indexed by datetime.
        prices : pd.DataFrame
            DataFrame containing a ``close`` column with the same frequency as
            ``signals``.
        strategy_name : str
            Identifier for the strategy; stored in the returned profile.

        Returns
        -------
        DecayProfile
            Profile containing the measured IC values, the fitted IC at t=0,
            and the estimated half‑life in hours.

        Raises
        ------
        ValueError
            If ``prices`` does not contain a ``close`` column.
        """
        if "close" not in prices.columns:
            raise ValueError("prices DataFrame must contain a 'close' column")

        ics: Dict[int, float] = {}

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
                """
                Exponential decay model used for curve fitting.

                Parameters
                ----------
                t : np.ndarray
                    Horizon values (in hours).
                ic0 : float
                    IC at horizon zero.
                lam : float
                    Decay rate parameter.

                Returns
                -------
                np.ndarray
                    Predicted IC values for each horizon.
                """
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

        Parameters
        ----------
        base_confidence : float
            Raw confidence score in the range [0, 1].
        profile : DecayProfile
            Fitted decay profile for the strategy.
        staleness_hours : float
            Number of hours elapsed since the signal was generated.

        Returns
        -------
        float
            Adjusted confidence in the range [0, 1]. If the half‑life is
            infinite (i.e., no decay), the original ``base_confidence`` is
            returned unchanged.
        """
        if profile.half_life_hours == float("inf") or profile.half_life_hours <= 0:
            return float(base_confidence)

        decay = np.exp(
            -staleness_hours * np.log(2) / profile.half_life_hours
        )
        return float(base_confidence * max(float(decay), 0.0))