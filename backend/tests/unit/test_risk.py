"""Unit tests for Kelly criterion and circuit breaker with enhanced coverage."""
import pytest
from app.risk.kelly import kelly_fraction, size_from_kelly
from app.risk.circuit_breaker import CircuitBreaker


def test_kelly_fraction_basic():
    f = kelly_fraction(win_rate=0.6, avg_win=1.0, avg_loss=1.0)
    assert 0 < f < 0.20


def test_kelly_fraction_zero_loss():
    assert kelly_fraction(0.5, 1.0, 0.0) == 0.0


def test_kelly_fraction_capped_at_20pct():
    f = kelly_fraction(0.99, 10.0, 1.0)
    assert f <= 0.20


def test_kelly_fraction_negative_win_rate():
    """Negative win rate should result in zero fraction."""
    assert kelly_fraction(-0.1, 1.0, 1.0) == 0.0


def test_kelly_fraction_win_rate_above_one():
    """Win rate greater than 1 should be treated as 1."""
    f = kelly_fraction(1.2, 2.0, 1.0)
    assert 0 < f <= 0.20


def test_kelly_fraction_loss_exceeds_win():
    """When avg_loss > avg_win, Kelly fraction should be zero."""
    assert kelly_fraction(0.6, 1.0, 2.0) == 0.0


def test_size_from_kelly():
    shares = size_from_kelly(equity=100_000, win_rate=0.6, avg_win_pct=0.02,
                             avg_loss_pct=0.01, price=100)
    assert shares >= 1


def test_size_from_kelly_zero_equity():
    """Zero equity should return zero shares."""
    shares = size_from_kelly(equity=0, win_rate=0.5, avg_win_pct=0.02,
                             avg_loss_pct=0.01, price=100)
    assert shares == 0


def test_circuit_breaker_normal():
    cb = CircuitBreaker(name="test", max_drawdown_pct=0.10)
    cb.update(100_000)
    assert not cb.is_halted


def test_circuit_breaker_triggers():
    cb = CircuitBreaker(name="test", max_drawdown_pct=0.10)
    cb.update(100_000)
    cb.update(89_000)
    assert cb.is_halted


def test_circuit_breaker_reset():
    cb = CircuitBreaker(name="test", max_drawdown_pct=0.10)
    cb.update(100_000)
    cb.update(89_000)
    assert cb.is_halted
    cb.reset(90_000)
    assert not cb.is_halted


def test_circuit_breaker_recovery_without_reset():
    """Circuit breaker should not clear halt state unless reset is called."""
    cb = CircuitBreaker(name="test", max_drawdown_pct=0.10)
    cb.update(100_000)
    cb.update(89_000)  # triggers halt
    cb.update(95_000)  # equity improves but still halted
    assert cb.is_halted
    cb.reset(95_000)
    assert not cb.is_halted


def test_circuit_breaker_multiple_updates():
    """Ensure consistent behavior over a series of updates."""
    cb = CircuitBreaker(name="test", max_drawdown_pct=0.15)
    equity_series = [200_000, 190_000, 170_000, 180_000, 160_000]
    for equity in equity_series:
        cb.update(equity)
    # 160k is a 20% drawdown from peak 200k, exceeding 15% threshold
    assert cb.is_halted
    cb.reset(170_000)
    assert not cb.is_halted