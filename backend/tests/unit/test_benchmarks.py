"""Benchmark stats tests."""
import pytest
from app.comparison.benchmarks import get_benchmark_stats, BENCHMARKS


@pytest.fixture(scope="module")
def benchmark_stats():
    """Cache benchmark stats for the entire test module."""
    return get_benchmark_stats()


def test_get_benchmark_stats_has_all_keys(benchmark_stats):
    stats = benchmark_stats
    assert "SPY" in stats
    assert "BRK-B" in stats
    assert "ALL_WEATHER" in stats
    required_keys = {"annual_return", "sharpe", "max_dd"}
    for v in stats.values():
        missing = required_keys - v.keys()
        assert not missing, f"Missing keys {missing} in benchmark stats"


def test_benchmarks_have_colors():
    for ticker, info in BENCHMARKS.items():
        assert "color" in info
        assert isinstance(info["color"], str)
        assert info["color"].startswith("#")