"""Benchmark stats tests."""
from app.comparison.benchmarks import get_benchmark_stats, BENCHMARKS


def test_get_benchmark_stats_has_all_keys():
    stats = get_benchmark_stats()
    # Ensure all expected tickers are present
    for expected in ("SPY", "BRK-B", "ALL_WEATHER"):
        assert expected in stats, f"Missing {expected} in benchmark stats"
    # Verify each entry contains required fields
    required_fields = {"annual_return", "sharpe", "max_dd"}
    for ticker, data in stats.items():
        missing = required_fields - data.keys()
        assert not missing, f"{ticker} missing fields: {missing}"
        # Validate numeric types
        assert isinstance(data["annual_return"], (int, float)), f"{ticker} annual_return not numeric"
        assert isinstance(data["sharpe"], (int, float)), f"{ticker} sharpe not numeric"
        assert isinstance(data["max_dd"], (int, float)), f"{ticker} max_dd not numeric"


def test_benchmarks_have_colors():
    for ticker, info in BENCHMARKS.items():
        assert "color" in info, f"{ticker} missing color entry"
        color = info["color"]
        assert isinstance(color, str), f"{ticker} color is not a string"
        assert color.startswith("#"), f"{ticker} color does not start with '#'"
        hex_part = color[1:]
        assert len(hex_part) == 6, f"{ticker} color hex length is not 6"
        # Ensure all characters are valid hex digits
        try:
            int(hex_part, 16)
        except ValueError:
            assert False, f"{ticker} color contains non-hex characters"


def test_benchmark_stats_and_definitions_consistent():
    """Ensure that every benchmark defined in BENCHMARKS has a corresponding stats entry and vice‑versa."""
    stats = get_benchmark_stats()
    # Keys should match exactly
    stats_keys = set(stats.keys())
    benchmark_keys = set(BENCHMARKS.keys())
    missing_in_stats = benchmark_keys - stats_keys
    missing_in_benchmarks = stats_keys - benchmark_keys
    assert not missing_in_stats, f"Benchmarks missing in stats: {missing_in_stats}"
    assert not missing_in_benchmarks, f"Stats missing in benchmarks: {missing_in_benchmarks}"