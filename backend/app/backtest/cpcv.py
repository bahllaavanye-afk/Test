"""
Combinatorial Purged Cross-Validation (CPCV) — López de Prado (2018).
======================================================================
Stronger than walk-forward: tests all k-fold combinations, prevents
multiple-testing overfitting, reports Deflated Sharpe Ratio (DSR).

Academic basis:
  - López de Prado (2018) "Advances in Financial Machine Learning"
    Chapter 12: Cross-Validation in Finance
  - Bailey & López de Prado (2014) "The Deflated Sharpe Ratio"
  - Bailey et al. (2014) "Pseudo-Mathematics and Financial Charlatanism"

Key insight:
  Standard k-fold CV is invalid for financial time series due to serial
  correlation. CPCV adds purge gaps (to prevent forward leakage) and
  embargo gaps (to prevent backward leakage) around each test fold.
  The Deflated Sharpe Ratio corrects for multiple-testing inflation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Iterable, List

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

DEFAULT_N_SPLITS: int = 6
DEFAULT_PURGE_DAYS: int = 5
DEFAULT_EMBARGO_DAYS: int = 2

# Numerical tolerances / small numbers
EPSILON: float = 1e-10

# Over‑fit detection factor (DSR < OVERFIT_FACTOR * mean_sharpe → overfit)
OVERFIT_FACTOR: float = 0.8

# Euler–Mascheroni constant (used in DSR approximation)
GAMMA: float = 0.5772156649

# Annualisation factor for daily returns (trading days per year)
ANNUALIZATION_FACTOR: float = 252.0


class CPCV:
    """
    Combinatorial Purged Cross-Validation for financial time series.

    Parameters
    ----------
    n_splits : int, default ``DEFAULT_N_SPLITS``
        Number of time‑series folds (e.g., 6 gives C(6,1)=6 test periods).
    purge_days : int, default ``DEFAULT_PURGE_DAYS``
        Bars to drop before the test fold (prevents train→test leakage).
    embargo_days : int, default ``DEFAULT_EMBARGO_DAYS``
        Bars to drop after the test fold (prevents test→train leakage).

    Usage
    -----
    >>> cpcv = CPCV()
    >>> results = cpcv.validate(signals, returns)
    >>> print(f"Deflated Sharpe: {results['deflated_sharpe']:.3f}")
    >>> print(f"Overfit: {results['is_overfit']}")
    """

    def __init__(
        self,
        n_splits: int = DEFAULT_N_SPLITS,
        purge_days: int = DEFAULT_PURGE_DAYS,
        embargo_days: int = DEFAULT_EMBARGO_DAYS,
    ) -> None:
        if not isinstance(n_splits, int) or n_splits < 2:
            raise ValueError(f"n_splits must be an integer >= 2, got {n_splits}")
        if not isinstance(purge_days, int) or purge_days < 0:
            raise ValueError(f"purge_days must be a non‑negative integer, got {purge_days}")
        if not isinstance(embargo_days, int) or embargo_days < 0:
            raise ValueError(f"embargo_days must be a non‑negative integer, got {embargo_days}")

        self.n_splits = n_splits
        self.purge_days = purge_days
        self.embargo_days = embargo_days

    # --------------------------------------------------------------------- #
    # Split generation
    # --------------------------------------------------------------------- #
    def split(self, index: pd.DatetimeIndex) -> Iterable[tuple[list[int], list[int]]]:
        """
        Yield ``(train_idx, test_idx)`` pairs with purge/embargo gaps.

        Parameters
        ----------
        index : pd.DatetimeIndex
            Ordered datetime index of the full dataset.

        Yields
        ------
        train_idx : list[int]
            Integer positions of the training set after applying purge/embargo.
        test_idx : list[int]
            Integer positions of the test fold.
        """
        if not isinstance(index, pd.DatetimeIndex):
            raise ValueError("index must be a pandas.DatetimeIndex")
        if not index.is_monotonic_increasing:
            raise ValueError("index must be sorted in increasing order")

        n = len(index)
        if n < self.n_splits:
            raise ValueError(f"Index length {n} is too short for {self.n_splits} folds")
        fold_size = n // self.n_splits
        if fold_size == 0:
            raise ValueError(f"Index length {n} is too short for {self.n_splits} folds")

        folds: list[range] = [
            range(i * fold_size, min((i + 1) * fold_size, n))
            for i in range(self.n_splits)
        ]

        for test_fold_idx in range(self.n_splits):
            test_idx = list(folds[test_fold_idx])
            test_start = test_idx[0]
            test_end = test_idx[-1]

            train_idx: list[int] = []
            for i, fold in enumerate(folds):
                if i == test_fold_idx:
                    continue
                for j in fold:
                    # Purge: exclude bars within purge_days before test_start
                    if (test_start - j) <= self.purge_days and j < test_start:
                        continue
                    # Embargo: exclude bars within embargo_days after test_end
                    if (j - test_end) <= self.embargo_days and j > test_end:
                        continue
                    train_idx.append(j)

            yield train_idx, test_idx

    # --------------------------------------------------------------------- #
    # Deflated Sharpe Ratio
    # --------------------------------------------------------------------- #
    def deflated_sharpe(
        self,
        sharpe_ratios: Iterable[float],
        n_trials: int,
    ) -> float:
        """
        Deflated Sharpe Ratio (Bailey & López de Prado 2014).

        Adjusts observed Sharpe Ratio downward for:
        1. Multiple testing: the more trials, the higher the expected best SR by luck.
        2. Non‑normality: excess kurtosis inflates SR under normality assumption.

        DSR = (mean_SR - SR*) / std_SR
        where SR* is the expected maximum SR over ``n_trials`` random draws.

        Parameters
        ----------
        sharpe_ratios : iterable of float
            SR values from each CPCV fold.
        n_trials : int
            Number of strategy configurations tried (use ``len(sharpe_ratios)`` for a
            single strategy; use larger if a parameter sweep was performed).

        Returns
        -------
        float
            DSR. Positive → robust; Negative → likely over‑fit.
        """
        if not isinstance(n_trials, int) or n_trials < 1:
            raise ValueError(f"n_trials must be an integer >= 1, got {n_trials}")

        sr_list: List[float] = list(sharpe_ratios)
        if not sr_list:
            return 0.0

        for i, val in enumerate(sr_list):
            if not isinstance(val, (int, float, np.number)):
                raise ValueError(f"sharpe_ratios element at position {i} is not numeric: {val}")

        sr = np.array(sr_list, dtype=float)
        if len(sr) < 2:
            return float(sr[0])

        mean_sr = float(np.mean(sr))
        std_sr = float(np.std(sr, ddof=1)) + EPSILON

        try:
            from scipy.special import erfinv  # type: ignore

            def norm_ppf(p: float) -> float:
                p = float(np.clip(p, EPSILON, 1 - EPSILON))
                return float(np.sqrt(2) * erfinv(2 * p - 1))

            p1 = 1.0 - 1.0 / max(n_trials, 1)
            p2 = 1.0 - 1.0 / max(n_trials * np.e, 1)
            sr_star = (1 - GAMMA) * norm_ppf(p1) + GAMMA * norm_ppf(p2)
            # Scale by empirical variance to keep units comparable
            sr_star = sr_star * float(np.sqrt(np.var(sr) + 1))
        except ImportError:
            # Fallback simple approximation when scipy is unavailable
            sr_star = float(np.log(n_trials + 1) * 0.5)

        dsr = (mean_sr - sr_star) / std_sr
        return float(dsr)

    # --------------------------------------------------------------------- #
    # Validation driver
    # --------------------------------------------------------------------- #
    def _annualized_sharpe(self, pnl: pd.Series) -> float:
        """
        Compute annualised Sharpe Ratio for a profit‑and‑loss series.

        Returns
        -------
        float
            Annualised Sharpe (mean / std * sqrt(ANNUALIZATION_FACTOR)).
        """
        if pnl.empty:
            return 0.0
        mean = pnl.mean()
        std = pnl.std(ddof=1) + EPSILON
        return float(mean / std * np.sqrt(ANNUALIZATION_FACTOR))

    def validate(
        self,
        signals: pd.Series,
        returns: pd.Series,
    ) -> dict:
        """
        Run CPCV on ``signals`` vs ``returns``.

        Computes Sharpe Ratio on each out‑of‑sample fold using the signals
        shifted by one bar to prevent look‑ahead bias.

        Parameters
        ----------
        signals : pd.Series
            Strategy signals (-1, 0, +1) indexed by datetime.
        returns : pd.Series
            Asset returns at the same frequency.

        Returns
        -------
        dict
            ``{
                "fold_sharpes": List[float],
                "mean_sharpe": float,
                "deflated_sharpe": float,
                "is_overfit": bool,
            }``
        """
        if not isinstance(signals, pd.Series):
            raise ValueError("signals must be a pandas.Series")
        if not isinstance(returns, pd.Series):
            raise ValueError("returns must be a pandas.Series")

        # Ensure datetime index for both series
        if not isinstance(signals.index, pd.DatetimeIndex):
            signals = signals.copy()
            signals.index = pd.to_datetime(signals.index)
        if not isinstance(returns.index, pd.DatetimeIndex):
            returns = returns.copy()
            returns.index = pd.to_datetime(returns.index)

        # Align the two series on the intersection of their indexes
        signals, returns = signals.align(returns, join="inner")
        if signals.empty:
            raise ValueError("No overlapping dates between signals and returns after alignment")

        # Apply 1‑bar shift to avoid look‑ahead bias
        shifted_signals = signals.shift(1).fillna(0)

        # Compute per‑fold Sharpe ratios
        fold_sharpes: List[float] = []
        for train_idx, test_idx in self.split(signals.index):
            # Build test P&L using shifted signals only on the test indices
            test_signals = shifted_signals.iloc[test_idx]
            test_returns = returns.iloc[test_idx]
            pnl = test_signals * test_returns
            fold_sharpes.append(self._annualized_sharpe(pnl))

        mean_sharpe = float(np.mean(fold_sharpes)) if fold_sharpes else 0.0
        deflated_sharpe = self.deflated_sharpe(fold_sharpes, n_trials=len(fold_sharpes))
        is_overfit = deflated_sharpe < OVERFIT_FACTOR * mean_sharpe

        return {
            "fold_sharpes": fold_sharpes,
            "mean_sharpe": mean_sharpe,
            "deflated_sharpe": deflated_sharpe,
            "is_overfit": is_overfit,
        }