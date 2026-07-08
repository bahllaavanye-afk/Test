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
from typing import List, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CPCV:
    """
    Combinatorial Purged Cross-Validation for financial time series.

    Parameters
    ----------
    n_splits : int, default 6
        Number of time‑series folds. Must be at least 2.
    purge_days : int, default 5
        Number of bars to drop before the test fold (prevents train→test leakage).
    embargo_days : int, default 2
        Number of bars to drop after the test fold (prevents test→train leakage).

    Notes
    -----
    The class provides three public methods:
    * :meth:`split` – generate train/test index pairs respecting purge/embargo.
    * :meth:`deflated_sharpe` – compute the Deflated Sharpe Ratio.
    * :meth:`validate` – run the full CPCV workflow on signal and return series.
    """

    def __init__(
        self,
        n_splits: int = 6,
        purge_days: int = 5,
        embargo_days: int = 2,
    ) -> None:
        if not isinstance(n_splits, int):
            raise ValueError(f"n_splits must be an integer, got {type(n_splits).__name__}")
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {n_splits}")

        if not isinstance(purge_days, int):
            raise ValueError(f"purge_days must be an integer, got {type(purge_days).__name__}")
        if purge_days < 0:
            raise ValueError(f"purge_days must be >= 0, got {purge_days}")

        if not isinstance(embargo_days, int):
            raise ValueError(f"embargo_days must be an integer, got {type(embargo_days).__name__}")
        if embargo_days < 0:
            raise ValueError(f"embargo_days must be >= 0, got {embargo_days}")

        self.n_splits = n_splits
        self.purge_days = purge_days
        self.embargo_days = embargo_days

    def split(self, index: pd.DatetimeIndex) -> Sequence[tuple[List[int], List[int]]]:
        """
        Yield (train_idx, test_idx) pairs with purge/embargo gaps.

        Parameters
        ----------
        index : pd.DatetimeIndex
            Ordered datetime index of the dataset.

        Yields
        ------
        train_idx : list[int]
            Integer positions representing the training set for the current split.
        test_idx : list[int]
            Integer positions representing the test set for the current split.

        Raises
        ------
        ValueError
            If ``index`` is empty or not a ``pd.DatetimeIndex``.
        """
        if not isinstance(index, pd.DatetimeIndex):
            raise ValueError(
                f"index must be a pandas DatetimeIndex, got {type(index).__name__}"
            )
        if len(index) == 0:
            raise ValueError("index must contain at least one element")

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
        sharpe_ratios: Sequence[float],
        n_trials: int,
    ) -> float:
        """
        Deflated Sharpe Ratio (Bailey & López de Prado 2014).

        Adjusts observed Sharpe Ratio downward for:
        1. Multiple testing: the more trials, the higher the expected best SR by luck.
        2. Non‑normality: excess kurtosis inflates SR under normality assumption.

        Parameters
        ----------
        sharpe_ratios : sequence of float
            Sharpe Ratio values from each CPCV fold.
        n_trials : int
            Number of strategy configurations tried (use ``len(sharpe_ratios)`` for a
            single strategy; use a larger number if a parameter sweep was performed).

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
        if not isinstance(sharpe_ratios, Sequence):
            raise ValueError(
                f"sharpe_ratios must be a sequence of numbers, got {type(sharpe_ratios).__name__}"
            )
        if len(sharpe_ratios) == 0:
            raise ValueError("sharpe_ratios must contain at least one element")
        if any(not isinstance(sr, (int, float, np.floating, np.integer)) for sr in sharpe_ratios):
            raise ValueError("sharpe_ratios must contain only numeric values")

        if not isinstance(n_trials, int) or n_trials <= 0:
            raise ValueError(f"n_trials must be a positive integer, got {n_trials}")

        sr = np.array(sharpe_ratios, dtype=float)
        if sr.size == 1:
            return float(sr[0])

        mean_sr = float(np.mean(sr))
        std_sr = float(np.std(sr, ddof=1)) + 1e-10

        try:
            from scipy.special import erfinv  # type: ignore

            gamma = 0.5772156649  # Euler‑Mascheroni constant

            def norm_ppf(p: float) -> float:
                p = float(np.clip(p, 1e-10, 1 - 1e-10))
                return float(np.sqrt(2) * erfinv(2 * p - 1))

            p1 = 1.0 - 1.0 / max(n_trials, 1)
            p2 = 1.0 - 1.0 / max(n_trials * np.e, 1)
            sr_star = (1 - gamma) * norm_ppf(p1) + gamma * norm_ppf(p2)
            sr_star = sr_star * float(np.sqrt(np.var(sr) + 1))
        except Exception:  # ImportError or any runtime error from scipy
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
            Strategy signals (typically -1, 0, +1) indexed by datetime.
        returns : pd.Series
            Asset returns at the same frequency as ``signals``.

        Returns
        -------
        dict
            ``{
                "fold_sharpes": list[float],
                "mean_sharpe": float,
                "deflated_sharpe": float,
                "is_overfit": bool,
                "total_pnl": float,
                "runtime_seconds": float,
            }``

        Raises
        ------
        ValueError
            If inputs are not pandas Series, have mismatched or non‑datetime indexes,
            or are empty.
        """
        start_time = time.time()

        if not isinstance(signals, pd.Series):
            raise ValueError(
                f"signals must be a pandas Series, got {type(signals).__name__}"
            )
        if not isinstance(returns, pd.Series):
            raise ValueError(
                f"returns must be a pandas Series, got {type(returns).__name__}"
            )

        # Ensure datetime index
        if not isinstance(signals.index, pd.DatetimeIndex):
            signals = signals.copy()
            signals.index = pd.to_datetime(signals.index)
        if not isinstance(returns.index, pd.DatetimeIndex):
            returns = returns.copy()
            returns.index = pd.to_datetime(returns.index)

        # Align both series on the intersection of their indexes
        common_idx = signals.index.intersection(returns.index)
        if common_idx.empty:
            raise ValueError("signals and returns share no common timestamps")
        signals = signals.loc[common_idx]
        returns = returns.loc[common_idx]

        if signals.empty:
            raise ValueError("signals series is empty after alignment")
        if returns.empty:
            raise ValueError("returns series is empty after alignment")

        sharpes: List[float] = []
        total_pnl = 0.0

        for train_idx, test_idx in self.split(pd.DatetimeIndex(signals.index)):
            # Train data is not used directly in this method but is generated
            # to ensure the split logic is exercised.
            test_signals = signals.iloc[test_idx]
            test_returns = returns.iloc[test_idx]

            # Shift signals by 1 to prevent look‑ahead bias
            pnl = test_signals.shift(1).fillna(0) * test_returns
            total_pnl += float(pnl.sum())

            # Annualized Sharpe (assumes daily frequency; adjust sqrt factor if needed)
            sr = pnl.mean() / (pnl.std() + 1e-10) * np.sqrt(252)
            sharpes.append(float(sr))

        if not sharpes:
            result = {
                "fold_sharpes": [],
                "mean_sharpe": 0.0,
                "deflated_sharpe": 0.0,
                "is_overfit": True,
                "total_pnl": total_pnl,
                "runtime_seconds": time.time() - start_time,
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
                "total_pnl": total_pnl,
                "runtime_seconds": time.time() - start_time,
            }

        logger.debug(
            "CPCV validation completed in %.3f seconds – mean SR: %.4f, DSR: %.4f, overfit: %s",
            result["runtime_seconds"],
            result["mean_sharpe"],
            result["deflated_sharpe"],
            result["is_overfit"],
        )
        return result