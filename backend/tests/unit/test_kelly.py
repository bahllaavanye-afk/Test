"""Comprehensive Kelly criterion tests."""
import pytest
from app.risk.kelly import kelly_fraction, size_from_kelly


class TestKellyFraction:
    def test_basic_positive(self):
        f = kelly_fraction(win_rate=0.6, avg_win=1.0, avg_loss=1.0)
        assert 0 < f < 0.20

    def test_zero_when_unfavorable(self):
        f = kelly_fraction(win_rate=0.3, avg_win=1.0, avg_loss=1.0)
        assert f == 0.0

    def test_zero_loss_returns_zero(self):
        assert kelly_fraction(0.6, 1.0, 0.0) == 0.0

    def test_capped_at_20_percent(self):
        f = kelly_fraction(0.99, 100.0, 1.0, fraction=1.0)
        assert f <= 0.20

    def test_fractional_kelly(self):
        full = kelly_fraction(0.6, 2.0, 1.0, fraction=1.0)
        quarter = kelly_fraction(0.6, 2.0, 1.0, fraction=0.25)
        assert quarter < full
        assert abs(quarter - full * 0.25) < 0.01 or quarter <= 0.20

    @pytest.mark.parametrize("win_rate", [0.0, 0.5, 0.55, 0.65, 0.75])
    def test_monotonic_in_win_rate(self, win_rate):
        f = kelly_fraction(win_rate, 1.0, 1.0)
        assert 0 <= f <= 0.20


class TestKellyFractionEdgeCases:
    @pytest.mark.parametrize(
        "args",
        [
            (None, 1.0, 1.0),
            (0.5, None, 1.0),
            (0.5, 1.0, None),
        ],
    )
    def test_none_inputs(self, args):
        """Kelly fraction should raise for None inputs."""
        with pytest.raises((TypeError, ValueError)):
            kelly_fraction(*args)

    def test_win_rate_one(self):
        """Edge case where win_rate is 1.0 (off‑by‑one)."""
        f = kelly_fraction(1.0, 100.0, 1.0, fraction=1.0)
        assert f <= 0.20

    def test_empty_collection(self):
        """Passing an empty collection where a scalar is expected should error."""
        with pytest.raises((TypeError, ValueError)):
            kelly_fraction([], 1.0, 1.0)


class TestSizeFromKelly:
    def test_returns_positive_shares(self):
        shares = size_from_kelly(
            equity=100_000,
            win_rate=0.6,
            avg_win_pct=0.02,
            avg_loss_pct=0.01,
            price=100,
        )
        assert shares >= 1

    def test_respects_max_pct(self):
        shares = size_from_kelly(
            equity=100_000,
            win_rate=0.99,
            avg_win_pct=1.0,
            avg_loss_pct=0.01,
            price=100,
            max_pct=0.05,
        )
        max_shares = int(100_000 * 0.05 / 100)
        assert shares <= max_shares

    def test_zero_equity(self):
        shares = size_from_kelly(
            equity=0,
            win_rate=0.6,
            avg_win_pct=0.02,
            avg_loss_pct=0.01,
            price=100,
        )
        assert shares == 1  # falls to minimum

    def test_high_price_smaller_size(self):
        shares_low = size_from_kelly(100_000, 0.6, 0.02, 0.01, price=10)
        shares_high = size_from_kelly(100_000, 0.6, 0.02, 0.01, price=1000)
        assert shares_low >= shares_high


class TestSizeFromKellyEdgeCases:
    @pytest.mark.parametrize(
        "args",
        [
            (None, 0.6, 0.02, 0.01, 100),
            (100_000, None, 0.02, 0.01, 100),
            (100_000, 0.6, None, 0.01, 100),
            (100_000, 0.6, 0.02, None, 100),
            (100_000, 0.6, 0.02, 0.01, None),
        ],
    )
    def test_none_inputs(self, args):
        """size_from_kelly should raise for None inputs."""
        with pytest.raises((TypeError, ValueError)):
            size_from_kelly(*args)

    def test_zero_equity_and_price(self):
        """Both equity and price zero should still return the minimum share count."""
        shares = size_from_kelly(
            equity=0,
            win_rate=0.6,
            avg_win_pct=0.02,
            avg_loss_pct=0.01,
            price=0,
        )
        assert shares == 1

    def test_empty_collection(self):
        """Passing an empty collection where a scalar is expected should error."""
        with pytest.raises((TypeError, ValueError)):
            size_from_kelly([], 0.6, 0.02, 0.01, 100)