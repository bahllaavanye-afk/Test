"""Kelly criterion position sizing with fractional Kelly for safety."""


def kelly_fraction(win_rate: float, avg_win: float, avg_loss: float, fraction: float = 0.25) -> float:
    """
    Full Kelly: f = (p*b - q) / b  where b = avg_win/avg_loss, q = 1-p
    Returns fractional Kelly (default 25%) to reduce variance.
    """
    if avg_loss < 1e-9 or win_rate <= 0:
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
    f = kelly_fraction(win_rate, avg_win_pct, avg_loss_pct, kelly_fraction_pct)
    f = min(f, max_pct)
    if price <= 0:
        return 0  # no valid price (halted / stale quote) — cannot size a position
    dollar_size = equity * f
    return max(1, int(dollar_size / price))
