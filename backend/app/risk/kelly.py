"""Kelly criterion position sizing with fractional Kelly for safety."""
import numpy as np


def _validate_probability(name: str, value: float) -> None:
    if not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a numeric type, got {type(value).__name__}")
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be between 0 and 1 inclusive, got {value}")


def _validate_positive_number(name: str, value: float, allow_zero: bool = False) -> None:
    if not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a numeric type, got {type(value).__name__}")
    if allow_zero:
        if value < 0:
            raise ValueError(f"{name} must be non‑negative, got {value}")
    else:
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")


def kelly_fraction(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    fraction: float = 0.25,
) -> float:
    """
    Full Kelly: f = (p*b - q) / b  where b = avg_win/avg_loss, q = 1-p
    Returns fractional Kelly (default 25%) to reduce variance.
    """
    # Input validation
    _validate_probability("win_rate", win_rate)
    _validate_positive_number("avg_win", avg_win, allow_zero=True)
    _validate_positive_number("avg_loss", avg_loss, allow_zero=True)
    _validate_probability("fraction", fraction)

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
    """Return integer share count sized by Kelly criterion, capped at max_pct of equity."""
    # Input validation
    _validate_positive_number("equity", equity)
    _validate_probability("win_rate", win_rate)
    _validate_positive_number("avg_win_pct", avg_win_pct, allow_zero=True)
    _validate_positive_number("avg_loss_pct", avg_loss_pct, allow_zero=True)
    _validate_positive_number("price", price)
    _validate_probability("max_pct", max_pct)
    _validate_probability("kelly_fraction_pct", kelly_fraction_pct)

    f = kelly_fraction(win_rate, avg_win_pct, avg_loss_pct, kelly_fraction_pct)
    f = min(f, max_pct)
    dollar_size = equity * f
    return max(1, int(dollar_size / price))