"""Circuit breaker drawdown tests."""
import pytest
from app.risk.circuit_breaker import CircuitBreaker, BreakerState

# Constants
DEFAULT_MAX_DRAWDOWN_PCT = 0.10
TEST_NAME = "t"
EQ_INITIAL = 100_000
EQ_PEAK = 110_000
EQ_MID = 105_000
EQ_BELOW_THRESHOLD = 89_999
EQ_NEAR_THRESHOLD = 91_000
EQ_HALT = 85_000
EQ_LOWER = 80_000
EQ_DRAWDOWN = 95_000


class TestCircuitBreaker:
    def test_starts_normal(self):
        cb = CircuitBreaker(name=TEST_NAME, max_drawdown_pct=DEFAULT_MAX_DRAWDOWN_PCT)
        assert cb.state == BreakerState.NORMAL
        assert not cb.is_halted

    def test_tracks_peak(self):
        cb = CircuitBreaker(name=TEST_NAME, max_drawdown_pct=DEFAULT_MAX_DRAWDOWN_PCT)
        cb.update(EQ_INITIAL)
        cb.update(EQ_PEAK)
        cb.update(EQ_MID)
        assert cb.peak_equity == EQ_PEAK

    def test_trips_at_threshold(self):
        cb = CircuitBreaker(name=TEST_NAME, max_drawdown_pct=DEFAULT_MAX_DRAWDOWN_PCT)
        cb.update(EQ_INITIAL)
        cb.update(EQ_BELOW_THRESHOLD)
        assert cb.is_halted

    def test_no_trip_below_threshold(self):
        cb = CircuitBreaker(name=TEST_NAME, max_drawdown_pct=DEFAULT_MAX_DRAWDOWN_PCT)
        cb.update(EQ_INITIAL)
        cb.update(EQ_NEAR_THRESHOLD)  # -9%
        assert not cb.is_halted

    def test_reset_clears_halt(self):
        cb = CircuitBreaker(name=TEST_NAME, max_drawdown_pct=DEFAULT_MAX_DRAWDOWN_PCT)
        cb.update(EQ_INITIAL)
        cb.update(EQ_HALT)
        assert cb.is_halted
        cb.reset(EQ_HALT)
        assert not cb.is_halted

    def test_no_double_trip(self):
        cb = CircuitBreaker(name=TEST_NAME, max_drawdown_pct=DEFAULT_MAX_DRAWDOWN_PCT)
        cb.update(EQ_INITIAL)
        cb.update(EQ_HALT)
        cb.update(EQ_LOWER)
        assert len(cb.halt_reasons) == 1  # only one reason recorded

    def test_drawdown_property(self):
        cb = CircuitBreaker(name=TEST_NAME, max_drawdown_pct=DEFAULT_MAX_DRAWDOWN_PCT)
        cb.update(EQ_INITIAL)
        cb.update(EQ_DRAWDOWN)
        assert abs(cb.current_drawdown - 0.05) < 1e-6