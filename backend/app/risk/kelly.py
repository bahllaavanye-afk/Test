"""Kelly criterion position sizing with fractional Kelly for safety."""
import numpy as np
import unittest


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


class TestKellyFunctions(unittest.TestCase):
    def test_zero_avg_loss_returns_zero_fraction(self):
        """When avg_loss is zero, Kelly fraction should be zero to avoid division errors."""
        self.assertEqual(kelly_fraction(win_rate=0.6, avg_win=1.0, avg_loss=0.0), 0.0)

    def test_full_win_rate_hits_hard_cap(self):
        """With win_rate=1, full Kelly is 1; fractional Kelly should be capped at 20%."""
        fraction = kelly_fraction(win_rate=1.0, avg_win=2.0, avg_loss=1.0)
        self.assertAlmostEqual(fraction, 0.20, places=7)

        # Verify size_from_kelly respects the capped fraction
        equity = 1000.0
        price = 10.0
        shares = size_from_kelly(equity, 1.0, 2.0, 1.0, price)
        expected_shares = int((equity * 0.20) / price)
        self.assertEqual(shares, expected_shares)

    def test_minimum_one_share_enforced(self):
        """Even when dollar size is less than price, function should return at least one share."""
        # Choose parameters that produce a very small dollar size
        equity = 0.5
        price = 1.0
        shares = size_from_kelly(equity, win_rate=0.5, avg_win_pct=1.0, avg_loss_pct=1.0, price=price)
        self.assertEqual(shares, 1)


if __name__ == "__main__":
    unittest.main()