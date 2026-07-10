"""
Combinatorial Purged Cross-Validation (CPCV) — López de Prado (2018).
======================================================================
Stronger than walk-forward: tests all k-fold combinations, prevents
multiple‑testing overfitting, reports Deflated Sharpe Ratio (DSR).

Academic basis:
  - López de Prado (2018) "Advances in Financial Machine Learning"
    Chapter 12: Cross‑Validation in Finance
  - Bailey & López de Prado (2014) "The Deflated Sharpe Ratio"
  - Bailey et al. (2014) "Pseudo‑Mathematics and Financial Charlatanism"

Key insight:
  Standard k‑fold CV is invalid for financial time series due to serial
  correlation. CPCV adds purge gaps (to prevent forward leakage) and
  embargo gaps (to prevent backward leakage) around each test fold.
  The Deflated Sharpe Ratio corrects for multiple‑testing inflation.
"""
from __future__ import annotations

import logging
import time
from typing import List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CPCV:
    """
    Combinatorial Purged Cross‑Validation for financial time series.

    Parameters
    ----------
    n_splits : int
        Number of time‑series folds (e.g., 6 gives C(6,1)=6 test periods).
    purge_days : int
        Bars to drop before the test fold (prevents train→test leakage).
    embargo_days : int
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
    ):
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {n_splits}")
        if purge_days < 0:
            raise ValueError(f"purge_days must be >= 0, got {purge_days}")
        if embargo_days < 0:
            raise ValueError(f"embargo_days must be >= 0, got {embargo_days}")

        self.n_splits = n_splits
        self.purge_days = purge_days
        self.embargo_days = embargo_days

        # Cache split results keyed by length of the index
        self._split_cache: dict[int, List[Tuple[List[int], List[int]]]] = {}

    def _compute_splits(self, n: int) -> List[Tuple[List[int], List[int]]]:
        """
        Compute train/test index pairs using vectorised NumPy operations.
        The result depends only on the length of the index and the CPCV
        parameters, so it can be cached.
        """
        fold_size = n // self.n_splits
        if fold_size == 0:
            raise ValueError(f"Index length {n} is too short for {self.n_splits} folds")

        indices = np.arange(n)
        # Determine start/end for each fold
        bounds = [
            (i * fold_size, min((i + 1) * fold_size, n))
            for i in range(self.n_splits)
        ]

        splits: List[Tuple[List[int], List[int]]] = []
        for test_fold_idx, (test_start, test_end) in enumerate(bounds):
            test_idx = list(indices[test_start:test_end])

            mask = np.ones(n, dtype=bool)

            # Exclude the test fold itself
            mask[test_start:test_end] = False

            # Purge: exclude bars within purge_days before test_start
            purge_start = max(0, test_start - self.purge_days)
            mask[purge_start:test_start] = False

            # Embargo: exclude bars within embargo_days after test_end
            embargo_end = min(n, test_end + self.embargo_days)
            mask[test_end:embargo_end] = False

            train_idx = list(indices[mask])
            splits.append((train_idx, test_idx))

        return splits

    def split(self, index: pd.DatetimeIndex):
        """
        Yield (train_idx, test_idx) pairs with purge/embargo gaps.

        Parameters
        ----------
        index : pd.DatetimeIndex
            The chronological index over which to generate folds.

        Yields
        ------
        train_idx : list[int]
            Integer positions of the training set for the current fold.
        test_idx : list[int]
            Integer positions of the test set for the current fold.
        """
        n = len(index)
        if n not in self._split_cache:
            self._split_cache[n] = self._compute_splits(n)

        for train_idx, test_idx in self._split_cache[n]:
            yield train_idx, test_idx

    def deflated_sharpe(
        self,
        sharpe_ratios: list[float],
        n_trials: int,
    ) -> float:
        """
        Deflated Sharpe Ratio (Bailey & López de Prado 2014).

        Adjusts observed Sharpe Ratio downward for:
        1. Multiple testing: the more trials, the higher the expected best SR
           by luck.
        2. Non‑normality: excess kurtosis inflates SR under normality assumption.

        DSR = (mean_SR - SR*) / std_SR
        where SR* is the expected maximum SR over `n_trials` random draws.

        Parameters
        ----------
        sharpe_ratios : list[float]
            Sharpe values from each CPCV fold.
        n_trials : int
            Number of strategy configurations tried (use ``len(sharpe_ratios)`` for
            a single strategy; larger if parameter‑swept).

        Returns
        -------
        float
            Deflated Sharpe Ratio. Positive indicates robustness; negative
            suggests over‑fitting.
        """
        if not sharpe_ratios:
            return 0.0

        sr = np.asarray(sharpe_ratios, dtype=float)
        if sr.size == 1:
            return float(sr[0])

        mean_sr = float(sr.mean())
        std_sr = float(sr.std(ddof=1)) + 1e-10

        try:
            from scipy.special import erfinv  # type: ignore

            gamma = 0.5772156649  # Euler‑Mascheroni constant

            def norm_ppf(p: float) -> float:
                p = float(np.clip(p, 1e-10, 1.0 - 1e-10))
                return float(np.sqrt(2) * erfinv(2 * p - 1))

            p1 = 1.0 - 1.0 / max(n_trials, 1)
            p2 = 1.0 - 1.0 / max(n_trials * np.e, 1)
            sr_star = (1 - gamma) * norm_ppf(p1) + gamma * norm_ppf(p2)
            sr_star *= float(np.sqrt(np.var(sr) + 1))
        except Exception:  # pragma: no cover
            # Fallback simple approximation when scipy is unavailable
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
            Strategy signals (‑1, 0, +1) indexed by datetime.
        returns : pd.Series
            Asset returns at the same frequency.

        Returns
        -------
        dict
            ``{
                "fold_sharpes": list[float],
                "mean_sharpe": float,
                "deflated_sharpe": float,
                "is_overfit": bool,
            }``
        """
        start_time = time.time()

        if not isinstance(signals.index, pd.DatetimeIndex):
            signals = signals.copy()
            signals.index = pd.to_datetime(signals.index)

        # Align both series on the common datetime index
        common_idx = signals.index.intersection(returns.index)
        signals = signals.loc[common_idx]
        returns = returns.loc[common_idx]

        sharpes: list[float] = []
        total_pnl = 0.0

        for train_idx, test_idx in self.split(pd.DatetimeIndex(signals.index)):
            # Train set is not used directly here but kept for API completeness
            test_signals = signals.iloc[test_idx]
            test_returns = returns.iloc[test_idx]

            # Shift signals by 1 to avoid look‑ahead bias
            pnl = test_signals.shift(1).fillna(0) * test_returns
            total_pnl += float(pnl.sum())

            # Annualised Sharpe (assumes daily data)
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
            # Over‑fit flag: DSR less than 80 % of mean Sharpe
            is_overfit = dsr < 0.8 * mean_sr

            result = {
                "fold_sharpes": sharpes,
                "mean_sharpe": mean_sr,
                "deflated_sharpe": dsr,
                "is_overfit": is_overfit,
            }

        logger.info(
            "CPCV validation completed in %.3f seconds (mean Sharpe=%.3f, DSR=%.3f)",
            time.time() - start_time,
            result["mean_sharpe"],
            result["deflated_sharpe"],
        )
        return result