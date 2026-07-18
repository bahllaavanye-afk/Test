"""Kelly criterion position sizing with fractional Kelly for safety."""
import logging
import numpy as np

logger = logging.getLogger(__name__)


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
    try:
        # Validate input types
        if not all(isinstance(x, (int, float)) for x in (win_rate, avg_win, avg_loss, fraction)):
            raise TypeError("All inputs must be int or float")

        # Validate input values
        if any(x < 0 for x in (win_rate, avg_win, avg_loss, fraction)):
            raise ValueError("Inputs must be non‑negative")

        if avg_loss == 0:
            logger.warning(
                "avg_loss is zero; returning 0.0 to avoid division by zero",
                extra={"win_rate": win_rate, "avg_win": avg_win, "avg_loss": avg_loss, "fraction": fraction},
            )
            return 0.0

        if win_rate <= 0:
            logger.warning(
                "win_rate <= 0; returning 0.0 as Kelly fraction would be non‑positive",
                extra={"win_rate": win_rate},
            )
            return 0.0

        b = avg_win / avg_loss
        q = 1.0 - win_rate
        f_full = (win_rate * b - q) / b
        f_full = max(0.0, f_full)
        result = min(f_full * fraction, 0.20)  # hard cap at 20% per position
        return result

    except (TypeError, ValueError) as e:
        logger.error(
            "Invalid input for kelly_fraction",
            extra={
                "exception": str(e),
                "win_rate": win_rate,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "fraction": fraction,
            },
        )
        return 0.0
    except Exception as e:
        logger.error(
            "Unexpected error in kelly_fraction",
            extra={
                "exception": str(e),
                "win_rate": win_rate,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "fraction": fraction,
            },
        )
        return 0.0


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
    try:
        # Validate input types
        if not all(
            isinstance(x, (int, float))
            for x in (equity, win_rate, avg_win_pct, avg_loss_pct, price, max_pct, kelly_fraction_pct)
        ):
            raise TypeError("All inputs must be int or float")

        # Validate essential values
        if equity <= 0:
            raise ValueError("Equity must be positive")
        if price <= 0:
            raise ValueError("Price must be positive")
        if any(x < 0 for x in (win_rate, avg_win_pct, avg_loss_pct, max_pct, kelly_fraction_pct)):
            raise ValueError("Numeric inputs must be non‑negative")

        f = kelly_fraction(win_rate, avg_win_pct, avg_loss_pct, kelly_fraction_pct)
        f = min(f, max_pct)
        dollar_size = equity * f
        share_count = max(1, int(dollar_size / price))
        return share_count

    except (TypeError, ValueError) as e:
        logger.error(
            "Invalid input for size_from_kelly",
            extra={
                "exception": str(e),
                "equity": equity,
                "win_rate": win_rate,
                "avg_win_pct": avg_win_pct,
                "avg_loss_pct": avg_loss_pct,
                "price": price,
                "max_pct": max_pct,
                "kelly_fraction_pct": kelly_fraction_pct,
            },
        )
        # Fallback to minimal position size
        return 1
    except Exception as e:
        logger.error(
            "Unexpected error in size_from_kelly",
            extra={
                "exception": str(e),
                "equity": equity,
                "win_rate": win_rate,
                "price": price,
            },
        )
        return 1