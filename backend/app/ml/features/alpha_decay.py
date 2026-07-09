"""
Alpha decay utilities.

This module provides tools to measure the decay of a strategy's predictive
power (Information Coefficient) over time and to scale signal confidence
according to signal staleness.

Typical usage::

    tracker = AlphaDecayTracker()
    profile = tracker.compute_ic_profile(signals, prices, "momentum")
    scaled_conf = tracker.scale_confidence(0.7, profile, staleness_hours=2)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import spearmanr


@dataclass
class DecayProfile:
    """
    Container for the exponential decay fit of a strategy's IC.

    Attributes
    ----------
    strategy_name: str
        Human‑readable identifier for the strategy.
    ic_0: float
        Estimated IC at horizon zero (t = 0).
    half_life_hours: float
        Half‑life of the IC decay in hours. ``float('inf')`` indicates no decay.
    horizons: Dict[float, float]
        Mapping from horizon (in hours) to the measured IC value for that horizon.
    """

    strategy_name: str
    ic_0: float
    half_life_hours: float
    horizons: Dict[float, float] = field(default_factory=dict)


class AlphaDecayTracker:
    """
    Measures how quickly a strategy's predictive power decays over time.

    The tracker computes the Spearman rank correlation (IC) between a signal
    and forward returns at several horizons, fits an exponential decay model,
    and provides a method to scale confidence based on signal staleness.
    """

    # Horizons to measure IC at: 1 h, 4 h, 1 d, 5 d, 20 d (expressed in hours)
    HORIZONS: List[int] = [1, 4, 24, 120, 480]

    def compute_ic_profile(
        self,
        signals: pd.Series,
        prices: pd.DataFrame,
        strategy_name: str,
    ) -> DecayProfile:
        """
        Compute the IC profile for a set of signals.

        Parameters
        ----------
        signals : pd.Series
            Series of signal values (typically -1, 0, +1) indexed by timestamps.
        prices : pd.DataFrame
            DataFrame containing at least a ``'close'`` column with the same
            frequency as ``signals``.
        strategy_name : str
            Identifier used when constructing the returned :class:`DecayProfile`.

        Returns
        -------
        DecayProfile
            Profile containing IC values for each horizon and the fitted
            exponential decay parameters.  If insufficient data are available,
            ``ic_0`` is set to ``0.0`` and ``half_life_hours`` to ``float('inf')``.

        Raises
        ------
        ValueError
            If ``prices`` does not contain a ``'close'`` column.
        """
        if "close" not in prices.columns:
            raise ValueError("prices DataFrame must contain a 'close' column")

        ics: Dict[int, float] = {}

        for h in self.HORIZONS:
            # Forward return over horizon ``h`` expressed as a percentage change.
            fwd_ret = prices["close"].pct_change(h).shift(-h)
            common = signals.index.intersection(fwd_ret.index)

            # Require a minimum number of overlapping observations.
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
                """Exponential decay model: IC(t) = ic0 * exp(-lam * t)."""
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
    ) -> float:
        """
        Scale a signal's confidence downward based on its staleness.

        Parameters
        ----------
        base_confidence : float
            Raw confidence score in the range ``[0, 1]``.
        profile : DecayProfile
            The decay profile fitted for the strategy.
        staleness_hours : float
            Number of hours elapsed since the signal was generated.

        Returns
        -------
        float
            Adjusted confidence in ``[0, 1]``.  If the profile indicates an
            infinite half‑life (i.e., no decay) the original ``base_confidence`` is
            returned unchanged.
        """
        if profile.half_life_hours == float("inf") or profile.half_life_hours <= 0:
            return float(base_confidence)

        decay = np.exp(-staleness_hours * np.log(2) / profile.half_life_hours)
        return float(base_confidence * max(float(decay), 0.0))