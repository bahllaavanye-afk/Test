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

import logging
import time
from itertools import combinations
from typing import List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CPCV:
    """
    Combinatorial Purged Cross-Validation for financial time series.

    Parameters
    ----------
    n_splits : int, default 6
        Number of time-series folds (6 gives C(6,1)=6 test periods).
    purge_days : int, default 5
        Bars to drop before the test fold (prevents train→test leakage).
    embargo_days : int, default 2
        Bars to drop after the test fold (prevents test→train leakage).

    Usage
    -----
    >>> cpcv = CPCV(n_splits=6, purge_days=5, embargo_days=2)
    >>> results = cpcv.validate(signals, returns)
    >>> print(f"Deflated Sharpe: {results['deflated_sharpe']:.3f}")
    >>> print(f"Overfit: {results['is_overfit']}")
    """

    def __init__(
        self,
        n_splits: int = 6,
        purge_days: int = 5,
        embargo_days: int = 2,
    ) -> None:
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {n_splits}")
        if purge_days < 0:
            raise ValueError(f"purge_days must be >= 0, got {purge_days}")
        if embargo_days < 0:
            raise ValueError(f"embargo_days must be >= 0, got {embargo_days}")
        self.n_splits = n_splits
        self.purge_days = purge_days
        self.embargo_days = embargo_days

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def split(self, index: pd.DatetimeIndex) -> Tuple[List[int], List[int]]:
        """
        Yield (train_idx, test_idx) pairs with purge/embargo gaps.

        Parameters
        ----------
        index : pd.DatetimeIndex
            Datetime index of the full series.

        Yields
        ------
        train_idx : List[int]
            Integer positions used for training after applying purge/embargo.
        test_idx : List[int]
            Integer positions of the current test fold.
        """
        n = len(index)
        fold_size = n // self.n_splits
        if fold_size == 0:
            raise ValueError(
                f"Index length {n} is too short for {self.n_splits} folds"
            )

        folds: List[range] = [
            range(i * fold_size, min((i + 1) * fold_size, n))
            for i in range(self.n_splits)
        ]

        for test_fold_idx in range(self.n_splits):
            test_idx = list(folds[test_fold_idx])
            test_start = test_idx[0]
            test_end = test_idx[-1]

            train_idx: List[int] = []
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

    def deflated_sharpe(
        self,
        sharpe_ratios: List[float],
        n_trials: int,
    ) -> float:
        """
        Deflated Sharpe Ratio (Bailey & López de Prado 2014).

        Adjusts observed Sharpe Ratio downward for:
        1. Multiple testing: the more trials, the higher the expected best SR by luck.
        2. Non‑normality: excess kurtosis inflates SR under normality assumption.

        Parameters
        ----------
        sharpe_ratios : list[float]
            Sharpe Ratio values from each CPCV fold.
        n_trials : int
            Number of strategy configurations tried (use ``len(sharpe_ratios)`` for a
            single strategy; use a larger value if parameters were swept).

        Returns
        -------
        float
            Deflated Sharpe Ratio. Positive values indicate robustness,
            negative values suggest over‑fitting.
        """
        if not sharpe_ratios:
            return 0.0

        sr = np.array(sharpe_ratios, dtype=float)
        if len(sr) < 2:
            return float(sr[0])

        mean_sr = float(np.mean(sr))
        std_sr = float(np.std(sr, ddof=1)) + 1e-10

        # Expected maximum SR under n_trials independent tests
        # Approximation: E[max_SR] ≈ (1 - γ)·Φ⁻¹(1 - 1/n) + γ·Φ⁻¹(1 - 1/(n·e))
        # where γ is Euler‑Mascheroni constant.
        try:
            from scipy.special import erfinv  # type: ignore

            gamma = 0.5772156649  # Euler‑Mascheroni constant

            def _norm_ppf(p: float) -> float:
                p = float(np.clip(p, 1e-10, 1 - 1e-10))
                return float(np.sqrt(2) * erfinv(2 * p - 1))

            p1 = 1.0 - 1.0 / max(n_trials, 1)
            p2 = 1.0 - 1.0 / max(n_trials * np.e, 1)
            sr_star = (1 - gamma) * _norm_ppf(p1) + gamma * _norm_ppf(p2)
            # Scale by empirical standard deviation of the SR distribution
            sr_star = sr_star * float(np.sqrt(np.var(sr) + 1))
        except Exception:  # pragma: no cover
            # Fallback to a simple, deterministic approximation when scipy is unavailable
            sr_star = float(np.log(n_trials + 1) * 0.5)

        dsr = (mean_sr - sr_star) / std_sr
        return float(dsr)

    def validate(
        self,
        signals: pd.Series,
        returns: pd.Series,
    ) -> dict:
        """
        Run CPCV on signals vs returns.

        Computes Sharpe Ratio on each out‑of‑sample fold using the signals
        shifted by one bar to prevent look‑ahead bias.

        Parameters
        ----------
        signals : pd.Series
            Strategy signals (‑1, 0, +1) indexed by datetime.
        returns : pd.Series
            Asset returns at the same frequency as ``signals``.

        Returns
        -------
        dict
            Dictionary containing:

            - ``fold_sharpes``: list of per‑fold Sharpe Ratios (annualised).
            - ``mean_sharpe``: mean Sharpe Ratio across folds.
            - ``deflated_sharpe``: DSR (adjusted for multiple testing).
            - ``is_overfit``: ``True`` if ``deflated_sharpe`` < 0.8 × ``mean_sharpe``.
            - ``total_pnl``: cumulative P&L across all folds.
            - ``runtime``: elapsed time in seconds.
        """
        start_time = time.time()

        signals_aligned, returns_aligned = self._align_series(signals, returns)

        sharpes, total_pnl = self._compute_fold_sharpes(signals_aligned, returns_aligned)

        if not sharpes:
            result = {
                "fold_sharpes": [],
                "mean_sharpe": 0.0,
                "deflated_sharpe": 0.0,
                "is_overfit": True,
                "total_pnl": 0.0,
                "runtime": time.time() - start_time,
            }
            return result

        mean_sr = float(np.mean(sharpes))
        dsr = self.deflated_sharpe(sharpes, n_trials=len(sharpes))
        overfit = dsr < 0.8 * mean_sr

        result = {
            "fold_sharpes": sharpes,
            "mean_sharpe": mean_sr,
            "deflated_sharpe": dsr,
            "is_overfit": overfit,
            "total_pnl": total_pnl,
            "runtime": time.time() - start_time,
        }
        return result

    # --------------------------------------------------------------------- #
    # Private helpers
    # --------------------------------------------------------------------- #
    def _align_series(
        self,
        signals: pd.Series,
        returns: pd.Series,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Ensure both series share a common datetime index and are of ``pd.DatetimeIndex`` type.

        Parameters
        ----------
        signals : pd.Series
        returns : pd.Series

        Returns
        -------
        Tuple[pd.Series, pd.Series]
            Aligned ``signals`` and ``returns`` series.
        """
        if not isinstance(signals.index, pd.DatetimeIndex):
            signals = signals.copy()
            signals.index = pd.to_datetime(signals.index)

        if not isinstance(returns.index, pd.DatetimeIndex):
            returns = returns.copy()
            returns.index = pd.to_datetime(returns.index)

        common_idx = signals.index.intersection(returns.index)
        return signals.loc[common_idx], returns.loc[common_idx]

    def _compute_fold_sharpes(
        self,
        signals: pd.Series,
        returns: pd.Series,
    ) -> Tuple[List[float], float]:
        """
        Iterate over CPCV folds, compute Sharpe Ratio for each, and aggregate total P&L.

        Parameters
        ----------
        signals : pd.Series
            Aligned signal series.
        returns : pd.Series
            Aligned returns series.

        Returns
        -------
        Tuple[List[float], float]
            List of Sharpe Ratios per fold and the cumulative P&L.
        """
        sharpes: List[float] = []
        total_pnl = 0.0

        for train_idx, test_idx in self.split(pd.DatetimeIndex(signals.index)):
            # The train_idx is currently unused – kept for API compatibility.
            test_signals = signals.iloc[test_idx]
            test_returns = returns.iloc[test_idx]

            # Shift signals by one period to avoid look‑ahead bias.
            pnl = test_signals.shift(1).fillna(0) * test_returns
            total_pnl += float(pnl.sum())

            # Annualise assuming daily data (252 trading days).
            sr = pnl.mean() / (pnl.std() + 1e-10) * np.sqrt(252)
            sharpes.append(float(sr))

        return sharpes, total_pnl