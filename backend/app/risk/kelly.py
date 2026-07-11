"""Kelly criterion position sizing with fractional Kelly for safety."""
import numpy as np


def _calculate_b(avg_win: float, avg_loss: float) -> float:
    """Calculate odds ratio b = avg_win / avg_loss."""
    return avg_win / avg_loss


def _calculate_full_kelly(win_rate: float, b: float) -> float:
    """Compute full Kelly fraction f = (p*b - q) / b where q = 1 - p."""
    q = 1.0 - win_rate
    return (win_rate * b - q) / b


def _apply_fraction_and_cap(full_kelly: float, fraction: float, cap: float = 0.20) -> float:
    """Apply fractional Kelly and enforce a hard cap."""
    fractional = max(0.0, full_kelly) * fraction
    return min(fractional, cap)


def kelly_fraction(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    fraction: float = 0.25,
) -> float:
    """
    Calculate fractional Kelly position size.

    Full Kelly: f = (p*b - q) / b where b = avg_win/avg_loss, q = 1-p.
    The result is scaled by `fraction` (default 25%) and capped at 20% of equity.
    """
    if avg_loss == 0 or win_rate <= 0:
        return 0.0

    b = _calculate_b(avg_win, avg_loss)
    full_kelly = _calculate_full_kelly(win_rate, b)
    return _apply_fraction_and_cap(full_kelly, fraction)


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
    Determine integer share count sized by Kelly criterion.

    The Kelly fraction is computed and then limited to `max_pct` of equity.
    The resulting dollar allocation is divided by the current price to obtain share count.
    """
    # Compute Kelly fraction and enforce max allocation percentage
    fraction = kelly_fraction(
        win_rate,
        avg_win_pct,
        avg_loss_pct,
        kelly_fraction_pct,
    )
    fraction = min(fraction, max_pct)

    # Convert equity allocation to share count, ensuring at least one share
    dollar_size = equity * fraction
    return max(1, int(dollar_size / price))