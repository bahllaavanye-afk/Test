"""Monte Carlo simulation: bootstrap equity curve for robustness confidence intervals."""
from __future__ import annotations

import logging
import numbers
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Constants
TRADING_DAYS_PER_YEAR: int = 252
RANDOM_SEED: int = 42
INITIAL_EQUITY: float = 100_000.0
RISK_FREE_RATE_ANNUAL: float = 0.05
RISK_FREE_DAILY_DEFAULT: float = RISK_FREE_RATE_ANNUAL / TRADING_DAYS_PER_YEAR
PERCENTILE_LOW: int = 5
PERCENTILE_HIGH: int = 95
ROUND_DECIMALS: int = 4
SHARPE_SQRT_DAYS: float = np.sqrt(TRADING_DAYS_PER_YEAR)

logger = logging.getLogger(__name__)


class MonteCarloError(RuntimeError):
    """Raised when an unexpected error occurs during Monte Carlo simulation."""


@dataclass
class MonteCarloResult:
    median_sharpe: float
    p5_sharpe: float
    p95_sharpe: float
    median_max_dd: float
    p95_max_dd: float
    p5_max_dd: float
    prob_positive_return: float
    num_simulations: int = field(repr=False)

    def __post_init__(self) -> None:
        """Validate the dataclass fields after initialization."""
        numeric_fields = {
            "median_sharpe": self.median_sharpe,
            "p5_sharpe": self.p5_sharpe,
            "p95_sharpe": self.p95_sharpe,
            "median_max_dd": self.median_max_dd,
            "p95_max_dd": self.p95_max_dd,
            "p5_max_dd": self.p5_max_dd,
            "prob_positive_return": self.prob_positive_return,
        }
        for name, value in numeric_fields.items():
            if not isinstance(value, numbers.Real):
                raise ValueError(f"{name} must be a real number, got {type(value)}.")
        if not (0.0 <= self.prob_positive_return <= 1.0):
            raise ValueError(
                "prob_positive_return must be between 0 and 1 inclusive."
            )
        if not isinstance(self.num_simulations, int) or self.num_simulations <= 0:
            raise ValueError("num_simulations must be a positive integer.")


def monte_carlo_simulation(
    daily_returns: pd.Series,
    n_simulations: int = 1000,
    n_years: int = 3,
    risk_free_daily: float = RISK_FREE_DAILY_DEFAULT,
) -> MonteCarloResult:
    """Bootstrap daily returns to simulate N years of paths.

    Parameters
    ----------
    daily_returns : pd.Series
        Series of daily returns. Must be non‑empty, numeric, and contain finite values.
    n_simulations : int
        Number of Monte‑Carlo paths to generate. Must be a positive integer.
    n_years : int
        Number of years to simulate. Must be a positive integer.
    risk_free_daily : float
        Daily risk‑free rate. Must be a real number.

    Returns
    -------
    MonteCarloResult
        Aggregated statistics from the simulations.

    Raises
    ------
    ValueError
        If any input is invalid.
    MonteCarloError
        If an unexpected error occurs during the simulation.
    """
    # Input validation
    if not isinstance(daily_returns, pd.Series):
        raise ValueError("daily_returns must be a pandas Series.")
    if daily_returns.empty:
        raise ValueError("daily_returns series cannot be empty.")
    if not np.issubdtype(daily_returns.dtype, np.number):
        raise ValueError("daily_returns must contain numeric values.")
    if not np.isfinite(daily_returns.dropna()).all():
        raise ValueError("daily_returns contains non‑finite values (NaN or Inf).")
    if not isinstance(n_simulations, int) or n_simulations <= 0:
        raise ValueError("n_simulations must be a positive integer.")
    if not isinstance(n_years, (int, float)) or n_years <= 0:
        raise ValueError("n_years must be a positive number.")
    if not isinstance(risk_free_daily, numbers.Real):
        raise ValueError("risk_free_daily must be a real number.")

    n_days = int(n_years * TRADING_DAYS_PER_YEAR)
    returns_array = daily_returns.dropna().values.astype(float)

    rng = np.random.default_rng(RANDOM_SEED)

    try:
        # Vectorized sampling: shape (n_simulations, n_days)
        sampled = rng.choice(returns_array, size=(n_simulations, n_days), replace=True)

        # Equity curve per simulation
        equity = np.cumprod(1 + sampled, axis=1) * INITIAL_EQUITY

        # Maximum drawdown per simulation
        peak = np.maximum.accumulate(equity, axis=1)
        dd = (equity - peak) / peak
        max_dds = dd.min(axis=1)

        # Sharpe ratio per simulation
        excess = sampled - risk_free_daily
        mean_excess = excess.mean(axis=1)
        std_excess = excess.std(axis=1, ddof=0)
        sharpe = np.where(
            std_excess > 0,
            mean_excess / std_excess * SHARPE_SQRT_DAYS,
            0.0,
        )

        # Probability of positive return at horizon
        positive = np.sum(equity[:, -1] > INITIAL_EQUITY)
    except Exception as exc:  # pragma: no cover
        logger.exception(
            "Unexpected error during Monte Carlo simulation",
            extra={"n_simulations": n_simulations, "n_years": n_years},
        )
        raise MonteCarloError("Monte Carlo simulation failed") from exc

    return MonteCarloResult(
        median_sharpe=round(float(np.median(sharpe)), ROUND_DECIMALS),
        p5_sharpe=round(float(np.percentile(sharpe, PERCENTILE_LOW)), ROUND_DECIMALS),
        p95_sharpe=round(float(np.percentile(sharpe, PERCENTILE_HIGH)), ROUND_DECIMALS),
        median_max_dd=round(float(np.median(max_dds)), ROUND_DECIMALS),
        p95_max_dd=round(float(np.percentile(max_dds, PERCENTILE_HIGH)), ROUND_DECIMALS),
        p5_max_dd=round(float(np.percentile(max_dds, PERCENTILE_LOW)), ROUND_DECIMALS),
        prob_positive_return=round(positive / n_simulations, ROUND_DECIMALS),
        num_simulations=n_simulations,
    )