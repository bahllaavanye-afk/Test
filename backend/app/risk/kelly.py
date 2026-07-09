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
    dollar_size = equity * f
    return max(1, int(dollar_size / price))


# ==============================
# Unit tests for edge conditions
# ==============================
def test_kelly_fraction_zero_win_rate():
    """When win_rate is zero, Kelly fraction should be zero regardless of other inputs."""
    assert kelly_fraction(0.0, avg_win=1.0, avg_loss=1.0) == 0.0


def test_kelly_fraction_zero_avg_loss():
    """Division by zero (avg_loss == 0) should safely return zero."""
    assert kelly_fraction(win_rate=0.6, avg_win=1.0, avg_loss=0.0) == 0.0


def test_kelly_fraction_hard_cap():
    """
    Verify that the hard 20% cap is applied.
    Choose parameters that would yield a full Kelly > 0.8 so that after applying the default
    fraction (0.25) the result would exceed the 0.20 cap.
    """
    # Construct a scenario with a very high win probability and favorable payoff ratio.
    win_rate = 0.99
    avg_win = 10.0
    avg_loss = 0.1
    # Full Kelly f_full ≈ (0.99*100 - 0.01)/100 ≈ 0.9899, fraction 0.25 => ≈0.2475 > 0.20
    result = kelly_fraction(win_rate, avg_win, avg_loss, fraction=0.25)
    assert result == 0.20


def test_size_from_kelly_min_one_share():
    """
    Ensure size_from_kelly never returns zero shares even when the calculated dollar size
    is less than the price of one share.
    """
    equity = 100.0          # small equity
    price = 200.0           # price higher than any possible allocation
    win_rate = 0.6
    avg_win_pct = 0.02
    avg_loss_pct = 0.01
    shares = size_from_kelly(equity, win_rate, avg_win_pct, avg_loss_pct, price)
    assert shares == 1


if __name__ == "__main__":
    # Simple ad‑hoc execution for quick sanity checks
    test_kelly_fraction_zero_win_rate()
    test_kelly_fraction_zero_avg_loss()
    test_kelly_fraction_hard_cap()
    test_size_from_kelly_min_one_share()
    print("All edge case tests passed.")