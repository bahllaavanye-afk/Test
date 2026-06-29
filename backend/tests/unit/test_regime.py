"""Tests for regime detector and correlation monitor."""
import numpy as np
from app.ml.regime.detector import detect_regime, Regime, _hurst_exponent
from app.risk.correlation_monitor import CrossStrategyCorrelationMonitor


def _make_trending(n: int = 80) -> list[float]:
    """Generate a strongly trending price series."""
    prices = [100.0]
    for _ in range(n - 1):
        prices.append(prices[-1] * (1 + np.random.normal(0.002, 0.005)))
    return prices


def _make_mean_reverting(n: int = 80) -> list[float]:
    """Generate a mean‑reverting (OU process) price series."""
    prices = [100.0]
    mu, theta, sigma = 100.0, 0.3, 0.5
    for _ in range(n - 1):
        prev = prices[-1]
        prices.append(prev + theta * (mu - prev) + np.random.normal(0, sigma))
    return prices


def _make_high_vol(n: int = 60) -> list[float]:
    """Generate a high‑volatility price series."""
    prices = [100.0]
    for _ in range(n - 1):
        prices.append(max(1.0, prices[-1] * (1 + np.random.normal(0, 0.035))))
    return prices


class TestHurstExponent:
    def test_returns_float_in_range(self):
        prices = _make_trending(60)
        h = _hurst_exponent(np.array(prices))
        assert 0.1 <= h <= 0.9

    def test_short_series_returns_half(self):
        h = _hurst_exponent(np.array([100, 101, 102]))
        assert h == 0.5


class TestDetectRegime:
    def test_returns_regime_state(self):
        state = detect_regime(_make_trending(60))
        assert state.regime in list(Regime)
        assert 0.0 <= state.confidence <= 1.0
        assert state.sizing_multiplier > 0

    def test_high_vol_regime(self):
        np.random.seed(42)
        prices = _make_high_vol(60)
        state = detect_regime(prices, high_vol_threshold=0.20)
        assert state.regime == Regime.HIGH_VOL
        assert state.sizing_multiplier == 0.50

    def test_too_short_returns_unknown(self):
        state = detect_regime([100, 101, 102])
        assert state.regime == Regime.UNKNOWN
        assert state.sizing_multiplier == 0.75

    def test_to_dict_serializable(self):
        state = detect_regime(_make_trending(60))
        d = state.to_dict()
        assert "regime" in d
        assert "sizing_multiplier" in d
        assert "hurst_exponent" in d


class TestCorrelationMonitor:
    def test_no_correlation_below_threshold(self):
        mon = CrossStrategyCorrelationMonitor(window=5, kill_threshold=0.70)
        for i in range(5):
            mon.record_return("strat_a", float(i) * 0.01)
            mon.record_return("strat_b", float(4 - i) * 0.01)  # negatively correlated
        alerts = mon.scan()
        assert len(alerts) == 0
        assert not mon.is_reduced("strat_a")
        assert not mon.is_reduced("strat_b")

    def test_fires_alert_above_threshold(self):
        mon = CrossStrategyCorrelationMonitor(window=5, kill_threshold=0.70)
        for i in range(5):
            ret = float(i) * 0.02
            mon.record_return("strat_a", ret)
            mon.record_return("strat_b", ret + 0.001)  # almost perfectly correlated
        alerts = mon.scan()
        assert len(alerts) >= 1
        assert alerts[0].correlation > 0.70

    def test_sizing_multiplier_halved(self):
        mon = CrossStrategyCorrelationMonitor(window=5, kill_threshold=0.50)
        for i in range(5):
            ret = float(i) * 0.01
            mon.record_return("strat_x", ret)
            mon.record_return("strat_y", ret)
        mon.scan()
        # one of them should be reduced
        assert mon.sizing_multiplier("strat_x") == 0.5 or mon.sizing_multiplier("strat_y") == 0.5

    def test_matrix_as_list(self):
        mon = CrossStrategyCorrelationMonitor(window=5)
        for i in range(5):
            mon.record_return("a", float(i))
            mon.record_return("b", float(i))
        result = mon.matrix_as_list()
        assert isinstance(result, list)
        assert all("correlation" in r for r in result)