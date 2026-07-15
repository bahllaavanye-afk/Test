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
    """Edge case unit tests for Kelly criterion utilities."""

    def test_zero_win_rate_returns_zero_fraction(self):
        """When win_rate is 0 the fraction should be 0."""
        self.assertEqual(kelly_fraction(0.0, 1.0, 1.0), 0.0)

    def test_zero_avg_loss_returns_zero_fraction(self):
        """Division by zero in avg_loss should be guarded."""
        self.assertEqual(kelly_fraction(0.5, 1.0, 0.0), 0.0)

    def test_fraction_capped_at_twenty_percent(self):
        """Full Kelly fraction multiplied by default fraction exceeds 0.20 and should be capped."""
        # Choose parameters that give a large full Kelly (e.g., win_rate=0.9, avg_win/avg_loss=10)
        result = kelly_fraction(0.9, 10.0, 1.0, fraction=0.5)  # 0.5 * full Kelly > 0.20
        self.assertAlmostEqual(result, 0.20, places=7)

    def test_size_from_kelly_minimum_one_share(self):
        """Even when calculated dollar size is less than price, at least one share is returned."""
        size = size_from_kelly(equity=1000, win_rate=0.6, avg_win_pct=0.02, avg_loss_pct=0.01,
                               price=5000, max_pct=0.05, kelly_fraction_pct=0.25)
        self.assertEqual(size, 1)

    def test_max_pct_cap_applied(self):
        """The max_pct parameter should limit the Kelly fraction."""
        # Create a scenario where Kelly fraction would be higher than max_pct
        size = size_from_kelly(equity=100000, win_rate=0.8, avg_win_pct=0.10,
                               avg_loss_pct=0.02, price=100, max_pct=0.03, kelly_fraction_pct=0.5)
        # Compute expected dollar size using the capped fraction
        expected_fraction = min(kelly_fraction(0.8, 0.10, 0.02, 0.5), 0.03)
        expected_dollar = 100000 * expected_fraction
        expected_shares = max(1, int(expected_dollar / 100))
        self.assertEqual(size, expected_shares)


if __name__ == "__main__":
    unittest.main()