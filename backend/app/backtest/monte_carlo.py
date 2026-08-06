"""Monte Carlo simulation: bootstrap equity curve for robustness confidence intervals."""
from __future__ import annotations

import logging
import numbers
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


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


def _compute_equity_curve(sampled: np.ndarray, initial_capital: float = 100_000) -> np.ndarray:
    """Calculate the equity curve for each simulated path."""
    return np.cumprod(1 + sampled, axis=1) * initial_capital


def _compute_metrics(
    equity: np.ndarray,
    sampled: np.ndarray,
    risk_free_daily: float,
    n_simulations: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Derive Sharpe ratios, maximum drawdowns, and positive‑return count."""
    # Maximum drawdown
    peak = np.maximum.accumulate(equity, axis=1)
    drawdown = (equity - peak) / peak
    max_dds = drawdown.min(axis=1)

    # Sharpe ratio
    excess = sampled - risk_free_daily
    mean_excess = excess.mean(axis=1)
    std_excess = excess.std(axis=1, ddof=0)
    sharpe = np.where(
        std_excess > 0,
        mean_excess / std_excess * np.sqrt(252),
        0.0,
    )

    # Positive return at horizon
    positive = np.sum(equity[:, -1] > equity[:, 0])

    return sharpe, max_dds, positive


def monte_carlo_simulation(
    daily_returns: pd.Series,
    n_simulations: int = 1000,
    n_years: int = 3,
    risk_free_daily: float = 0.05 / 252,
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

    n_days = int(n_years * 252)
    returns_array = daily_returns.dropna().values.astype(float)

    rng = np.random.default_rng(42)

    try:
        sampled = _sample_returns(rng, returns_array, n_simulations, n_days)
        equity = _compute_equity_curve(sampled)
        sharpe, max_dds, positive = _compute_metrics(equity, sampled, risk_free_daily, n_simulations)
    except Exception as exc:  # pragma: no cover
        logger.exception(
            "Unexpected error during Monte Carlo simulation",
            extra={"n_simulations": n_simulations, "n_years": n_years},
        )
        raise MonteCarloError("Monte Carlo simulation failed") from exc

    return MonteCarloResult(
        median_sharpe=round(float(np.median(sharpe)), 4),
        p5_sharpe=round(float(np.percentile(sharpe, 5)), 4),
        p95_sharpe=round(float(np.percentile(sharpe, 95)), 4),
        median_max_dd=round(float(np.median(max_dds)), 4),
        p95_max_dd=round(float(np.percentile(max_dds, 95)), 4),
        p5_max_dd=round(float(np.percentile(max_dds, 5)), 4),
        prob_positive_return=round(positive / n_simulations, 4),
        num_simulations=n_simulations,
    )