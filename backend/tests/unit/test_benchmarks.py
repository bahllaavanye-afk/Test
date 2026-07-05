"""Benchmark stats tests."""
from app.comparison.benchmarks import get_benchmark_stats, BENCHMARKS


def test_get_benchmark_stats_has_all_keys():
    """Verify that the benchmark stats dictionary contains expected tickers and fields."""
    stats = get_benchmark_stats()
    assert "SPY" in stats
    assert "BRK-B" in stats
    assert "ALL_WEATHER" in stats
    for v in stats.values():
        assert "annual_return" in v
        assert "sharpe" in v
        assert "max_dd" in v


def test_benchmarks_have_colors():
    """Each benchmark entry must define a colour string starting with '#'. """
    for ticker, info in BENCHMARKS.items():
        assert "color" in info
        assert isinstance(info["color"], str)
        assert info["color"].startswith("#")


def test_get_benchmark_stats_len_matches_benchmarks():
    """The number of returned benchmark entries should match the number of defined benchmarks."""
    from app.comparison import benchmarks as bm_mod
    stats = bm_mod.get_benchmark_stats()
    assert isinstance(stats, dict)
    assert len(stats) == len(bm_mod.BENCHMARKS)


def test_get_benchmark_stats_none_input():
    """
    Ensure the function gracefully handles a None argument if it accepts one.
    If the implementation does not accept parameters, a TypeError is expected
    and the default call should still succeed.
    """
    try:
        stats = get_benchmark_stats(None)  # type: ignore[arg-type]
    except TypeError:
        # Function signature does not accept arguments; fallback to default call.
        stats = get_benchmark_stats()
    assert isinstance(stats, dict)


def test_benchmarks_empty(monkeypatch):
    """
    Verify that iterating over an empty BENCHMARKS mapping does not raise errors.
    The test temporarily replaces BENCHMARKS with an empty dict.
    """
    from app.comparison import benchmarks as bm_mod
    monkeypatch.setattr(bm_mod, "BENCHMARKS", {}, raising=False)

    # The loop should simply complete without executing any assertions.
    for ticker, info in bm_mod.BENCHMARKS.items():
        assert "color" in info
        assert isinstance(info["color"], str)
        assert info["color"].startswith("#")