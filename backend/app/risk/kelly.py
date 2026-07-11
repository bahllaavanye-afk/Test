"""Kelly criterion position sizing with fractional Kelly for safety."""
import numpy as np


def kelly_fraction(win_rate: float, avg_win: float, avg_loss: float, fraction: float = 0.25) -> float:
    """
    Full Kelly: f = (p*b - q) / b  where b = avg_win/avg_loss, q = 1-p
    Returns fractional Kelly (default 25%) to reduce variance.
    """
    if avg_loss == 0 or win_rate <= 0:
        return 0.0
    b = avg_win / avg_loss
    q = 1.0 - win_rate
    f_full = (win_rate * b - q) / b
    f_full = max(0.0, f_full)
    return min(f_full * fraction, 0.20)  # hard cap at 20% per position


def _compute_fraction(
    win_rate: float,
    avg_win_pct: float,
    avg_loss_pct: float,
    fraction: float,
) -> float:
    """
    Compute the fractional Kelly value using the public kelly_fraction function.
    """
    return kelly_fraction(win_rate, avg_win_pct, avg_loss_pct, fraction)


def _apply_max_pct(fraction: float, max_pct: float) -> float:
    """
    Apply the maximum allowed percentage of equity to the Kelly fraction.
    """
    return min(fraction, max_pct)


def _calculate_dollar_size(equity: float, fraction: float) -> float:
    """
    Convert the fraction of equity into a dollar amount.
    """
    return equity * fraction


def _convert_to_shares(dollar_size: float, price: float) -> int:
    """
    Convert a dollar size to an integer share count, ensuring at least one share.
    """
    return max(1, int(dollar_size / price))


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
    """
    # Compute the base Kelly fraction
    fraction = _compute_fraction(win_rate, avg_win_pct, avg_loss_pct, kelly_fraction_pct)

    # Enforce the maximum percentage of equity allowed per position
    fraction = _apply_max_pct(fraction, max_pct)

    # Translate the fraction into a dollar size
    dollar_size = _calculate_dollar_size(equity, fraction)

    # Convert the dollar size to an integer number of shares
    return _convert_to_shares(dollar_size, price)