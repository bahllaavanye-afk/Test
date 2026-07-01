"""Unit tests for the CircuitBreaker risk management component."""

from app.risk.circuit_breaker import CircuitBreaker, BreakerState


class TestCircuitBreaker:
    def test_starts_normal(self):
        cb = CircuitBreaker(name="t", max_drawdown_pct=0.10)
        assert cb.state == BreakerState.NORMAL
        assert not cb.is_halted

    def test_tracks_peak(self):
        cb = CircuitBreaker(name="t", max_drawdown_pct=0.10)
        cb.update(100_000)
        cb.update(110_000)
        cb.update(105_000)
        assert cb.peak_equity == 110_000

    def test_trips_at_threshold(self):
        cb = CircuitBreaker(name="t", max_drawdown_pct=0.10)
        cb.update(100_000)
        cb.update(89_999)
        assert cb.is_halted

    def test_no_trip_below_threshold(self):
        cb = CircuitBreaker(name="t", max_drawdown_pct=0.10)
        cb.update(100_000)
        cb.update(91_000)  # -9%
        assert not cb.is_halted

    def test_reset_clears_halt(self):
        cb = CircuitBreaker(name="t", max_drawdown_pct=0.10)
        cb.update(100_000)
        cb.update(85_000)
        assert cb.is_halted
        cb.reset(85_000)
        assert not cb.is_halted

    def test_no_double_trip(self):
        cb = CircuitBreaker(name="t", max_drawdown_pct=0.10)
        cb.update(100_000)
        cb.update(85_000)
        cb.update(80_000)
        assert len(cb.halt_reasons) == 1  # only one reason recorded

    def test_drawdown_property(self):
        cb = CircuitBreaker(name="t", max_drawdown_pct=0.10)
        cb.update(100_000)
        cb.update(95_000)
        assert abs(cb.current_drawdown - 0.05) < 1e-6