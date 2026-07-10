"""Unit tests for Kelly criterion and circuit breaker."""
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


def test_kelly_fraction_invalid_win_rate():
    with pytest.raises(ValueError, match="win_rate must be between 0 and 1"):
        kelly_fraction(win_rate=-0.1, avg_win=1.0, avg_loss=1.0)
    with pytest.raises(ValueError, match="win_rate must be between 0 and 1"):
        kelly_fraction(win_rate=1.5, avg_win=1.0, avg_loss=1.0)


def test_kelly_fraction_invalid_avg_win():
    with pytest.raises(ValueError, match="avg_win must be non‑negative"):
        kelly_fraction(win_rate=0.5, avg_win=-1.0, avg_loss=1.0)


def test_kelly_fraction_invalid_avg_loss():
    with pytest.raises(ValueError, match="avg_loss must be non‑negative"):
        kelly_fraction(win_rate=0.5, avg_win=1.0, avg_loss=-1.0)


def test_size_from_kelly():
    shares = size_from_kelly(equity=100_000, win_rate=0.6, avg_win_pct=0.02, avg_loss_pct=0.01, price=100)
    assert shares >= 1


def test_size_from_kelly_invalid_equity():
    with pytest.raises(ValueError, match="equity must be positive"):
        size_from_kelly(equity=0, win_rate=0.6, avg_win_pct=0.02, avg_loss_pct=0.01, price=100)
    with pytest.raises(ValueError, match="equity must be positive"):
        size_from_kelly(equity=-1000, win_rate=0.6, avg_win_pct=0.02, avg_loss_pct=0.01, price=100)


def test_size_from_kelly_invalid_win_rate():
    with pytest.raises(ValueError, match="win_rate must be between 0 and 1"):
        size_from_kelly(equity=100_000, win_rate=1.2, avg_win_pct=0.02, avg_loss_pct=0.01, price=100)


def test_size_from_kelly_invalid_avg_win_pct():
    with pytest.raises(ValueError, match="avg_win_pct must be non‑negative"):
        size_from_kelly(equity=100_000, win_rate=0.6, avg_win_pct=-0.01, avg_loss_pct=0.01, price=100)


def test_size_from_kelly_invalid_avg_loss_pct():
    with pytest.raises(ValueError, match="avg_loss_pct must be non‑negative"):
        size_from_kelly(equity=100_000, win_rate=0.6, avg_win_pct=0.02, avg_loss_pct=-0.01, price=100)


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


def test_circuit_breaker_invalid_max_drawdown():
    with pytest.raises(ValueError, match="max_drawdown_pct must be between 0 and 1"):
        CircuitBreaker(name="test", max_drawdown_pct=-0.1)
    with pytest.raises(ValueError, match="max_drawdown_pct must be between 0 and 1"):
        CircuitBreaker(name="test", max_drawdown_pct=1.5)


def test_circuit_breaker_invalid_name():
    with pytest.raises(ValueError, match="name must be a non‑empty string"):
        CircuitBreaker(name="", max_drawdown_pct=0.10)
    with pytest.raises(ValueError, match="name must be a non‑empty string"):
        CircuitBreaker(name=None, max_drawdown_pct=0.10)