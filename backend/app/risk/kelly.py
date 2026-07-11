"""Kelly criterion position sizing with fractional Kelly for safety."""
import numpy as np


def kelly_fraction(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    fraction: float = 0.25,
) -> float:
    """
    Full Kelly: f = (p*b - q) / b  where b = avg_win/avg_loss, q = 1-p
    Returns fractional Kelly (default 25%) to reduce variance.

    Parameters
    ----------
    win_rate : float
        Probability of a winning trade (0 ≤ win_rate ≤ 1).
    avg_win : float
        Average winning trade size (must be non‑negative).
    avg_loss : float
        Average losing trade size (must be non‑negative; zero is allowed, returns 0.0).
    fraction : float, optional
        Fraction of the full Kelly to use (0 ≤ fraction ≤ 1). Default is 0.25.

    Returns
    -------
    float
        Position size fraction, capped at 20 % of equity per position.

    Raises
    ------
    ValueError
        If any input is of an incorrect type or falls outside its allowed range.
    """
    # Input validation
    if not isinstance(win_rate, (int, float)):
        raise ValueError("win_rate must be a numeric type")
    if not 0 <= win_rate <= 1:
        raise ValueError("win_rate must be between 0 and 1 inclusive")
    if not isinstance(avg_win, (int, float)):
        raise ValueError("avg_win must be a numeric type")
    if avg_win < 0:
        raise ValueError("avg_win must be non‑negative")
    if not isinstance(avg_loss, (int, float)):
        raise ValueError("avg_loss must be a numeric type")
    if avg_loss < 0:
        raise ValueError("avg_loss must be non‑negative")
    if not isinstance(fraction, (int, float)):
        raise ValueError("fraction must be a numeric type")
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must be between 0 and 1 inclusive")

    # Guard against division by zero or non‑positive win probability
    if avg_loss == 0 or win_rate <= 0:
        return 0.0

    b = avg_win / avg_loss
    q = 1.0 - win_rate
    f_full = (win_rate * b - q) / b
    f_full = max(0.0, f_full)
    return min(f_full * fraction, 0.20)  # hard cap at 20% per position


def size_from_kelly(
    equity: float,
    win_rate: float,
    avg_win_pct: float,
    avg_loss_pct: float,
    price: float,
    max_pct: float = 0.05,
    kelly_fraction_pct: float = 0.25,
) -> int:
    """
    Return integer share count sized by Kelly criterion, capped at max_pct of equity.

    Parameters
    ----------
    equity : float
        Total account equity (must be non‑negative).
    win_rate : float
        Probability of a winning trade (0 ≤ win_rate ≤ 1).
    avg_win_pct : float
        Average winning trade return as a decimal (must be non‑negative).
    avg_loss_pct : float
        Average losing trade return as a decimal (must be non‑negative).
    price : float
        Current price of the instrument (must be positive).
    max_pct : float, optional
        Maximum proportion of equity to allocate to a single position (0 ≤ max_pct ≤ 1).
        Default is 0.05 (5 %).
    kelly_fraction_pct : float, optional
        Fraction of the full Kelly to use (0 ≤ kelly_fraction_pct ≤ 1). Default is 0.25.

    Returns
    -------
    int
        Number of shares to trade (minimum of 1).

    Raises
    ------
    ValueError
        If any input is of an incorrect type or falls outside its allowed range.
    """
    # Input validation
    if not isinstance(equity, (int, float)):
        raise ValueError("equity must be a numeric type")
    if equity < 0:
        raise ValueError("equity must be non‑negative")
    if not isinstance(win_rate, (int, float)):
        raise ValueError("win_rate must be a numeric type")
    if not 0 <= win_rate <= 1:
        raise ValueError("win_rate must be between 0 and 1 inclusive")
    if not isinstance(avg_win_pct, (int, float)):
        raise ValueError("avg_win_pct must be a numeric type")
    if avg_win_pct < 0:
        raise ValueError("avg_win_pct must be non‑negative")
    if not isinstance(avg_loss_pct, (int, float)):
        raise ValueError("avg_loss_pct must be a numeric type")
    if avg_loss_pct < 0:
        raise ValueError("avg_loss_pct must be non‑negative")
    if not isinstance(price, (int, float)):
        raise ValueError("price must be a numeric type")
    if price <= 0:
        raise ValueError("price must be positive")
    if not isinstance(max_pct, (int, float)):
        raise ValueError("max_pct must be a numeric type")
    if not 0 <= max_pct <= 1:
        raise ValueError("max_pct must be between 0 and 1 inclusive")
    if not isinstance(kelly_fraction_pct, (int, float)):
        raise ValueError("kelly_fraction_pct must be a numeric type")
    if not 0 <= kelly_fraction_pct <= 1:
        raise ValueError("kelly_fraction_pct must be between 0 and 1 inclusive")

    f = kelly_fraction(win_rate, avg_win_pct, avg_loss_pct, kelly_fraction_pct)
    f = min(f, max_pct)
    dollar_size = equity * f
    return max(1, int(dollar_size / price))