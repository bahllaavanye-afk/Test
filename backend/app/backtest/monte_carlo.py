"""Monte Carlo simulation: bootstrap equity curve for robustness confidence intervals."""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class MonteCarloResult:
    median_sharpe: float
    p5_sharpe: float
    p95_sharpe: float
    median_max_dd: float
    p95_max_dd: float
    prob_positive_return: float
    num_simulations: int


def _sample_returns(
    rng: np.random.Generator,
    returns_array: np.ndarray,
    n_days: int,
) -> np.ndarray:
    """Draw a bootstrap sample of daily returns."""
    return rng.choice(returns_array, size=n_days, replace=True)


def _compute_equity_path(
    sampled_returns: np.ndarray,
    initial_capital: float = 100_000.0,
) -> np.ndarray:
    """Calculate the equity curve from sampled returns."""
    cumulative_returns = np.cumprod(1 + sampled_returns)
    return cumulative_returns * initial_capital


def _max_drawdown(equity_curve: np.ndarray) -> float:
    """Return the maximum drawdown of an equity curve."""
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak
    return drawdown.min()


def _sharpe_ratio(
    sampled_returns: np.ndarray,
    risk_free_daily: float,
) -> float:
    """Calculate annualized Sharpe ratio for sampled returns."""
    excess = sampled_returns - risk_free_daily
    std = excess.std()
    if std == 0:
        return 0.0
    return excess.mean() / std * np.sqrt(252)


def _is_positive_return(equity_curve: np.ndarray, initial_capital: float = 100_000.0) -> bool:
    """Check if the final equity exceeds the initial capital."""
    return equity_curve[-1] > initial_capital


def monte_carlo_simulation(
    daily_returns: pd.Series,
    n_simulations: int = 1000,
    n_years: int = 3,
    risk_free_daily: float = 0.05 / 252,
) -> MonteCarloResult:
    """Bootstrap daily returns to simulate N years of paths."""
    n_days = n_years * 252
    returns_array = daily_returns.dropna().values

    sharpes: list[float] = []
    max_dds: list[float] = []
    positive_count = 0

    rng = np.random.default_rng(42)

    for _ in range(n_simulations):
        sampled = _sample_returns(rng, returns_array, n_days)
        equity = _compute_equity_path(sampled)

        max_dds.append(_max_drawdown(equity))
        sharpes.append(_sharpe_ratio(sampled, risk_free_daily))

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