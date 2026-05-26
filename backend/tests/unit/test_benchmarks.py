"""Benchmark stats tests."""
from app.comparison.benchmarks import get_benchmark_stats, BENCHMARKS


def test_get_benchmark_stats_has_all_keys():
    stats = get_benchmark_stats()
    assert "SPY" in stats
    assert "BRK-B" in stats
    assert "ALL_WEATHER" in stats
    for v in stats.values():
        assert "annual_return" in v
        assert "sharpe" in v
        assert "max_dd" in v


def test_benchmarks_have_colors():
    for ticker, info in BENCHMARKS.items():
        assert "color" in info
        assert info["color"].startswith("#")
