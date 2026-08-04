"""Monte Carlo simulation: bootstrap equity curve for robustness confidence intervals."""
from __future__ import annotations

import logging
import numbers
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Constants
TRADING_DAYS_PER_YEAR: int = 252
ANNUAL_RISK_FREE_RATE: float = 0.05
DEFAULT_RISK_FREE_DAILY: float = ANNUAL_RISK_FREE_RATE / TRADING_DAYS_PER_YEAR
RANDOM_SEED: int = 42
INITIAL_EQUITY: float = 100_000
PERCENTILE_LOW: int = 5
PERCENTILE_HIGH: int = 95
ROUND_PRECISION: int = 4

# Error message constants
ERR_DAILY_RETURNS_TYPE = "daily_returns must be a pandas Series."
ERR_DAILY_RETURNS_EMPTY = "daily_returns series cannot be empty."
ERR_DAILY_RETURNS_NUMERIC = "daily_returns must contain numeric values."
ERR_DAILY_RETURNS_FINITE = "daily_returns contains non‑finite values (NaN or Inf)."
ERR_N_SIMULATIONS = "n_simulations must be a positive integer."
ERR_N_YEARS = "n_years must be a positive number."
ERR_RISK_FREE_DAILY = "risk_free_daily must be a real number."

logger = logging.getLogger(__name__)


class MonteCarloError(RuntimeError):
    """Raised when an unexpected error occurs during Monte Carlo simulation."""


@dataclass
class MonteCarloResult:
    median_sharpe: float
    p5_sharpe: float
    p95_sharpe: float
    median_max_dd: float
    # NOTE ON SIGN: max_dd is `dd.min()`, so drawdowns are NEGATIVE. The 95th
    # percentile of a negative series is therefore the MILDEST drawdown, not the
    # worst — `p95_max_dd` is the lucky tail, despite reading like a risk
    # number. It is kept for compatibility and left as-is.
    #
    # `p5_max_dd` is the severe tail, and is the one to quote as risk. This
    # matches `p5_sharpe`, which is already the unlucky end.
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
    risk_free_daily: float = DEFAULT_RISK_FREE_DAILY,
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
        raise ValueError(ERR_DAILY_RETURNS_TYPE)
    if daily_returns.empty:
        raise ValueError(ERR_DAILY_RETURNS_EMPTY)
    if not np.issubdtype(daily_returns.dtype, np.number):
        raise ValueError(ERR_DAILY_RETURNS_NUMERIC)
    if not np.isfinite(daily_returns.dropna()).all():
        raise ValueError(ERR_DAILY_RETURNS_FINITE)
    if not isinstance(n_simulations, int) or n_simulations <= 0:
        raise ValueError(ERR_N_SIMULATIONS)
    if not isinstance(n_years, (int, float)) or n_years <= 0:
        raise ValueError(ERR_N_YEARS)
    if not isinstance(risk_free_daily, numbers.Real):
        raise ValueError(ERR_RISK_FREE_DAILY)

    n_days = int(n_years * TRADING_DAYS_PER_YEAR)
    returns_array = daily_returns.dropna().values
    sharpes: list[float] = []
    max_dds: list[float] = []
    positive = 0

    rng = np.random.default_rng(RANDOM_SEED)

    try:
        for _ in range(n_simulations):
            sampled = rng.choice(returns_array, size=n_days, replace=True)
            equity = np.cumprod(1 + sampled) * INITIAL_EQUITY
            peak = np.maximum.accumulate(equity)
            dd = (equity - peak) / peak
            max_dd = dd.min()

            excess = sampled - risk_free_daily
            sharpe = (
                (excess.mean() / excess.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
                if excess.std() > 0
                else 0.0
            )
            sharpes.append(sharpe)
            max_dds.append(max_dd)
            if equity[-1] > INITIAL_EQUITY:
                positive += 1
    except Exception as exc:  # pragma: no cover
        logger.exception(
            "Unexpected error during Monte Carlo simulation",
            extra={"n_simulations": n_simulations, "n_years": n_years},
        )
        raise MonteCarloError("Monte Carlo simulation failed") from exc

    return MonteCarloResult(
        median_sharpe=round(float(np.median(sharpes)), ROUND_PRECISION),
        p5_sharpe=round(float(np.percentile(sharpes, PERCENTILE_LOW)), ROUND_PRECISION),
        p95_sharpe=round(float(np.percentile(sharpes, PERCENTILE_HIGH)), ROUND_PRECISION),
        median_max_dd=round(float(np.median(max_dds)), ROUND_PRECISION),
        p95_max_dd=round(float(np.percentile(max_dds, PERCENTILE_HIGH)), ROUND_PRECISION),
        p5_max_dd=round(float(np.percentile(max_dds, PERCENTILE_LOW)), ROUND_PRECISION),
        prob_positive_return=round(positive / n_simulations, ROUND_PRECISION),
        num_simulations=n_simulations,
    )