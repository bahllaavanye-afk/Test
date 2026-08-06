"""Monte Carlo simulation: bootstrap equity curve for robustness confidence intervals."""
from __future__ import annotations

import logging
import numbers
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Constants
START_EQUITY: int = 100_000
TRADING_DAYS_PER_YEAR: int = 252
DEFAULT_RNG_SEED: int = 42
RISK_FREE_RATE_ANNUAL: float = 0.05
SHARPE_ANNUALIZATION: float = np.sqrt(TRADING_DAYS_PER_YEAR)
PERCENTILE_LOW: int = 5
PERCENTILE_HIGH: int = 95
ROUND_PRECISION: int = 4

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


def _validate_inputs(
    daily_returns: pd.Series,
    n_simulations: int,
    n_years: int | float,
    risk_free_daily: float,
) -> None:
    """Validate inputs for the Monte Carlo simulation."""
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


def _sample_returns(
    rng: np.random.Generator,
    returns_array: np.ndarray,
    n_simulations: int,
    n_days: int,
) -> np.ndarray:
    """Draw bootstrap samples of daily returns."""
    return rng.choice(returns_array, size=(n_simulations, n_days), replace=True)


def _calculate_equity(sampled: np.ndarray) -> np.ndarray:
    """Convert sampled returns into equity curves, assuming a start equity."""
    return np.cumprod(1 + sampled, axis=1) * START_EQUITY


def _calculate_max_dd(equity: np.ndarray) -> np.ndarray:
    """Compute maximum drawdown for each simulated equity curve."""
    peak = np.maximum.accumulate(equity, axis=1)
    drawdown = (equity - peak) / peak
    return drawdown.min(axis=1)


def _calculate_sharpe(sampled: np.ndarray, risk_free_daily: float) -> np.ndarray:
    """Calculate annualized Sharpe ratio for each simulation."""
    excess = sampled - risk_free_daily
    mean_excess = excess.mean(axis=1)
    std_excess = excess.std(axis=1, ddof=0)
    # Avoid division by zero; assign zero Sharpe when std is zero.
    return np.where(
        std_excess > 0,
        mean_excess / std_excess * SHARPE_ANNUALIZATION,
        0.0,
    )


def _calculate_positive_return(equity: np.ndarray) -> int:
    """Count simulations that end with a positive return relative to the start."""
    return int(np.sum(equity[:, -1] > START_EQUITY))


def monte_carlo_simulation(
    daily_returns: pd.Series,
    n_simulations: int = 1000,
    n_years: int = 3,
    risk_free_daily: float = RISK_FREE_RATE_ANNUAL / TRADING_DAYS_PER_YEAR,
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
    _validate_inputs(daily_returns, n_simulations, n_years, risk_free_daily)

    n_days = int(n_years * TRADING_DAYS_PER_YEAR)
    returns_array = daily_returns.dropna().values.astype(float)
    rng = np.random.default_rng(DEFAULT_RNG_SEED)

    try:
        sampled = _sample_returns(rng, returns_array, n_simulations, n_days)
        equity = _calculate_equity(sampled)
        max_dds = _calculate_max_dd(equity)
        sharpe = _calculate_sharpe(sampled, risk_free_daily)
        positive = _calculate_positive_return(equity)
    except Exception as exc:  # pragma: no cover
        logger.exception(
            "Unexpected error during Monte Carlo simulation",
            extra={"n_simulations": n_simulations, "n_years": n_years},
        )
        raise MonteCarloError("Monte Carlo simulation failed") from exc

    return MonteCarloResult(
        median_sharpe=round(float(np.median(sharpe)), ROUND_PRECISION),
        p5_sharpe=round(float(np.percentile(sharpe, PERCENTILE_LOW)), ROUND_PRECISION),
        p95_sharpe=round(float(np.percentile(sharpe, PERCENTILE_HIGH)), ROUND_PRECISION),
        median_max_dd=round(float(np.median(max_dds)), ROUND_PRECISION),
        p95_max_dd=round(float(np.percentile(max_dds, PERCENTILE_HIGH)), ROUND_PRECISION),
        p5_max_dd=round(float(np.percentile(max_dds, PERCENTILE_LOW)), ROUND_PRECISION),
        prob_positive_return=round(positive / n_simulations, ROUND_PRECISION),
        num_simulations=n_simulations,
    )