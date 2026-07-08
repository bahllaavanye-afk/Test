"""Kelly criterion position sizing with fractional Kelly for safety."""
import numbers
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

    Raises:
        ValueError: If any input is out of the expected range.
    """
    # Input validation
    if not isinstance(win_rate, numbers.Real) or isinstance(win_rate, bool):
        raise ValueError("win_rate must be a numeric value.")
    if not 0 < win_rate <= 1:
        raise ValueError("win_rate must be in the interval (0, 1].")
    if not isinstance(avg_win, numbers.Real) or isinstance(avg_win, bool):
        raise ValueError("avg_win must be a numeric value.")
    if avg_win <= 0:
        raise ValueError("avg_win must be greater than 0.")
    if not isinstance(avg_loss, numbers.Real) or isinstance(avg_loss, bool):
        raise ValueError("avg_loss must be a numeric value.")
    if avg_loss <= 0:
        raise ValueError("avg_loss must be greater than 0.")
    if not isinstance(fraction, numbers.Real) or isinstance(fraction, bool):
        raise ValueError("fraction must be a numeric value.")
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in the interval (0, 1].")

    b = avg_win / avg_loss
    q = 1.0 - win_rate
    f_full = (win_rate * b - q) / b
    f_full = max(0.0, f_full)
    # Hard cap at 20% per position
    return min(f_full * fraction, 0.20)


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

    Raises:
        ValueError: If any input is out of the expected range.
    """
    # Input validation
    if not isinstance(equity, numbers.Real) or isinstance(equity, bool):
        raise ValueError("equity must be a numeric value.")
    if equity <= 0:
        raise ValueError("equity must be greater than 0.")
    if not isinstance(price, numbers.Real) or isinstance(price, bool):
        raise ValueError("price must be a numeric value.")
    if price <= 0:
        raise ValueError("price must be greater than 0.")
    if not isinstance(win_rate, numbers.Real) or isinstance(win_rate, bool):
        raise ValueError("win_rate must be a numeric value.")
    if not 0 < win_rate <= 1:
        raise ValueError("win_rate must be in the interval (0, 1].")
    if not isinstance(avg_win_pct, numbers.Real) or isinstance(avg_win_pct, bool):
        raise ValueError("avg_win_pct must be a numeric value.")
    if avg_win_pct <= 0:
        raise ValueError("avg_win_pct must be greater than 0.")
    if not isinstance(avg_loss_pct, numbers.Real) or isinstance(avg_loss_pct, bool):
        raise ValueError("avg_loss_pct must be a numeric value.")
    if avg_loss_pct <= 0:
        raise ValueError("avg_loss_pct must be greater than 0.")
    if not isinstance(max_pct, numbers.Real) or isinstance(max_pct, bool):
        raise ValueError("max_pct must be a numeric value.")
    if not 0 < max_pct <= 1:
        raise ValueError("max_pct must be in the interval (0, 1].")
    if not isinstance(kelly_fraction_pct, numbers.Real) or isinstance(kelly_fraction_pct, bool):
        raise ValueError("kelly_fraction_pct must be a numeric value.")
    if not 0 < kelly_fraction_pct <= 1:
        raise ValueError("kelly_fraction_pct must be in the interval (0, 1].")

    f = kelly_fraction(win_rate, avg_win_pct, avg_loss_pct, kelly_fraction_pct)
    f = min(f, max_pct)
    dollar_size = equity * f
    return max(1, int(dollar_size / price))