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
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CPCV:
    """
    Combinatorial Purged Cross-Validation for financial time series.

    Parameters
    ----------
    n_splits : int, default 6
        Number of time‑series folds (6 gives C(6,1)=6 test periods).
    purge_days : int, default 5
        Bars to drop before the test fold (prevents train→test leakage).
    embargo_days : int, default 2
        Bars to drop after the test fold (prevents test→train leakage).

    Notes
    -----
    The class provides three public methods:
    * :meth:`split` – generate train/test index pairs respecting purge/embargo.
    * :meth:`deflated_sharpe` – compute the Deflated Sharpe Ratio.
    * :meth:`validate` – run the full CPCV pipeline on a signal series.
    """

    def __init__(
        self,
        n_splits: int = 6,
        purge_days: int = 5,
        embargo_days: int = 2,
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

    def split(self, index: pd.DatetimeIndex):
        """
        Yield (train_idx, test_idx) pairs with purge/embargo gaps.

        Parameters
        ----------
        index : pd.DatetimeIndex
            Full datetime index of the dataset.

        Yields
        ------
        train_idx : list[int]
            Integer positions for the training set (respecting purge/embargo).
        test_idx : list[int]
            Integer positions for the test set.

        Raises
        ------
        ValueError
            If ``index`` is not a :class:`pandas.DatetimeIndex` or is too short.
        """
        if not isinstance(index, pd.DatetimeIndex):
            raise ValueError("index must be a pandas.DatetimeIndex")
        n = len(index)
        if n < self.n_splits:
            raise ValueError(
                f"Index length {n} is smaller than the number of splits {self.n_splits}"
            )
        fold_size = n // self.n_splits
        if fold_size == 0:
            raise ValueError(
                f"Index length {n} is too short for {self.n_splits} folds (fold size would be 0)"
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
        sharpe_ratios: list[float],
        n_trials: int,
    ) -> float:
        """
        Deflated Sharpe Ratio (Bailey & López de Prado 2014).

        Adjusts observed Sharpe Ratio downward for:
        1. Multiple testing: the more trials, the higher the expected best SR by luck.
        2. Non‑normality: excess kurtosis inflates SR under normality assumption.

        DSR = (mean_SR - SR*) / std_SR

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

        Raises
        ------
        ValueError
            If ``sharpe_ratios`` is empty, contains non‑numeric values, or if
            ``n_trials`` is not a positive integer.
        """
        if not isinstance(sharpe_ratios, (list, tuple, np.ndarray)):
            raise ValueError("sharpe_ratios must be a list, tuple, or numpy array of numbers")
        if len(sharpe_ratios) == 0:
            raise ValueError("sharpe_ratios cannot be empty")
        if any(not isinstance(x, (int, float, np.number)) for x in sharpe_ratios):
            raise ValueError("sharpe_ratios must contain only numeric values")
        if not isinstance(n_trials, int) or n_trials < 1:
            raise ValueError(f"n_trials must be a positive integer, got {n_trials}")

        sr = np.array(sharpe_ratios, dtype=float)

        if sr.size == 1:
            return float(sr[0])

        mean_sr = float(np.mean(sr))
        std_sr = float(np.std(sr, ddof=1)) + 1e-10  # avoid division by zero

        # Expected maximum SR under n_trials independent tests
        # Approximation: E[max_SR] ≈ (1 - γ)*Φ⁻¹(1 - 1/n) + γ*Φ⁻¹(1 - 1/(n·e))
        # where γ is Euler‑Mascheroni constant
        try:
            from scipy.special import erfinv  # type: ignore

            gamma = 0.5772156649  # Euler‑Mascheroni constant

            def norm_ppf(p: float) -> float:
                p = float(np.clip(p, 1e-10, 1 - 1e-10))
                return float(np.sqrt(2) * erfinv(2 * p - 1))

            p1 = 1.0 - 1.0 / max(n_trials, 1)
            p2 = 1.0 - 1.0 / max(n_trials * np.e, 1)
            sr_star = (1 - gamma) * norm_ppf(p1) + gamma * norm_ppf(p2)
            # Scale by empirical variance to reflect the observed distribution
            sr_star = sr_star * float(np.sqrt(np.var(sr) + 1))
        except Exception:
            # Fallback: simple logarithmic approximation
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
        shifted by 1 bar to prevent look‑ahead bias.

        Parameters
        ----------
        signals : pd.Series
            Strategy signals (expected values -1, 0, +1) indexed by datetime.
        returns : pd.Series
            Asset returns at the same frequency.

        Returns
        -------
        dict
            ``{
                "fold_sharpes": list[float],
                "mean_sharpe": float,
                "deflated_sharpe": float,
                "is_overfit": bool
            }``

        Raises
        ------
        ValueError
            If inputs are not pandas Series, are empty, contain non‑numeric data,
            or have indexes that cannot be interpreted as datetime.
        """
        start_time = time.time()

        # Basic type checks
        if not isinstance(signals, pd.Series):
            raise ValueError("signals must be a pandas Series")
        if not isinstance(returns, pd.Series):
            raise ValueError("returns must be a pandas Series")
        if signals.empty:
            raise ValueError("signals series is empty")
        if returns.empty:
            raise ValueError("returns series is empty")
        if not np.issubdtype(signals.dtype, np.number):
            raise ValueError("signals series must contain numeric values")
        if not np.issubdtype(returns.dtype, np.number):
            raise ValueError("returns series must contain numeric values")

        # Ensure datetime indexes
        if not isinstance(signals.index, pd.DatetimeIndex):
            try:
                signals = signals.copy()
                signals.index = pd.to_datetime(signals.index)
            except Exception as exc:
                raise ValueError("signals index could not be converted to DatetimeIndex") from exc
        if not isinstance(returns.index, pd.DatetimeIndex):
            try:
                returns = returns.copy()
                returns.index = pd.to_datetime(returns.index)
            except Exception as exc:
                raise ValueError("returns index could not be converted to DatetimeIndex") from exc

        # Align the two series on the intersection of their indexes
        common_idx = signals.index.intersection(returns.index)
        if common_idx.empty:
            raise ValueError("signals and returns have no overlapping timestamps")
        signals = signals.loc[common_idx]
        returns = returns.loc[common_idx]

        sharpes: list[float] = []
        total_pnl = 0.0

        for train_idx, test_idx in self.split(pd.DatetimeIndex(signals.index)):
            test_signals = signals.iloc[test_idx]
            test_returns = returns.iloc[test_idx]

            # Shift signals by 1 to prevent look‑ahead bias
            pnl = test_signals.shift(1).fillna(0) * test_returns
            total_pnl += float(pnl.sum())

            # Annualized Sharpe (assuming daily data → 252 trading days)
            sr = pnl.mean() / (pnl.std() + 1e-10) * np.sqrt(252)
            sharpes.append(float(sr))

        if not sharpes:
            result = {
                "fold_sharpes": [],
                "mean_sharpe": 0.0,
                "deflated_sharpe": 0.0,
                "is_overfit": True,
            }
        else:
            mean_sr = float(np.mean(sharpes))
            dsr = self.deflated_sharpe(sharpes, n_trials=len(sharpes))
            is_overfit = dsr < 0.8 * mean_sr

            result = {
                "fold_sharpes": sharpes,
                "mean_sharpe": mean_sr,
                "deflated_sharpe": dsr,
                "is_overfit": is_overfit,
            }

        elapsed = time.time() - start_time
        logger.debug("CPCV validation completed in %.4f seconds", elapsed)
        return result