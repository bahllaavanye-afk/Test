"""Monte Carlo simulation: bootstrap equity curve for robustness confidence intervals."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd


DAYS_PER_YEAR = 252
INITIAL_CAPITAL = 100_000


@dataclass
class MonteCarloResult:
    median_sharpe: float
    p5_sharpe: float
    p95_sharpe: float
    median_max_dd: float
    p95_max_dd: float
    prob_positive_return: float
    num_simulations: int


def _sample_returns(rng: np.random.Generator, returns_array: np.ndarray, n_days: int) -> np.ndarray:
    """Draw a bootstrap sample of daily returns."""
    return rng.choice(returns_array, size=n_days, replace=True)


def _compute_equity(sampled_returns: np.ndarray) -> np.ndarray:
    """Calculate the equity curve given sampled returns."""
    return np.cumprod(1 + sampled_returns) * INITIAL_CAPITAL


def _compute_max_drawdown(equity: np.ndarray) -> float:
    """Maximum drawdown expressed as a fraction of peak equity."""
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    return drawdown.min()


def _compute_sharpe(sampled_returns: np.ndarray, risk_free_daily: float) -> float:
    """Annualized Sharpe ratio of the sampled path."""
    excess = sampled_returns - risk_free_daily
    std = excess.std()
    if std > 0:
        return excess.mean() / std * np.sqrt(DAYS_PER_YEAR)
    return 0.0


def _is_positive_return(equity: np.ndarray) -> bool:
    """Check if the final equity exceeds the initial capital."""
    return equity[-1] > INITIAL_CAPITAL


def monte_carlo_simulation(
    daily_returns: pd.Series,
    n_simulations: int = 1000,
    n_years: int = 3,
    risk_free_daily: float = 0.05 / DAYS_PER_YEAR,
) -> MonteCarloResult:
    """Bootstrap daily returns to simulate N years of paths."""
    n_days = n_years * DAYS_PER_YEAR
    returns_array = daily_returns.dropna().values

    rng = np.random.default_rng(42)

    sharpes: List[float] = []
    max_dds: List[float] = []
    positive_count = 0

    for _ in range(n_simulations):
        sampled = _sample_returns(rng, returns_array, n_days)

        equity = _compute_equity(sampled)
        max_dd = _compute_max_drawdown(equity)
        sharpe = _compute_sharpe(sampled, risk_free_daily)

        sharpes.append(sharpe)
        max_dds.append(max_dd)

        if _is_positive_return(equity):
            positive_count += 1

    return MonteCarloResult(
        median_sharpe=round(float(np.median(sharpes)), 4),
        p5_sharpe=round(float(np.percentile(sharpes, 5)), 4),
        p95_sharpe=round(float(np.percentile(sharpes, 95)), 4),
        median_max_dd=round(float(np.median(max_dds)), 4),
        p95_max_dd=round(float(np.percentile(max_dds, 95)), 4),
        prob_positive_return=round(positive_count / n_simulations, 4),
        num_simulations=n_simulations,
    )