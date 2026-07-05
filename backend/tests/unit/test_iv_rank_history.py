"""Unit tests for the IV-history rank functions in the options API.

_iv_rank_from_history is the honest replacement for the cross-sectional
"iv_rank" proxy: current IV ranked within its own trailing daily history.
Pure functions — no Redis, no network.
"""
from __future__ import annotations

from app.api.v1.options import (
    _IV_HIST_MIN_POINTS,
    _iv_rank_from_history,
    _median,
)


def _history(values: list[float]) -> dict[str, float]:
    return {f"2026-01-{i + 1:02d}": v for i, v in enumerate(values)}


class TestMedian:
    def test_odd(self):
        assert _median([3.0, 1.0, 2.0]) == 2.0

    def test_even(self):
        assert _median([4.0, 1.0, 3.0, 2.0]) == 2.5

    def test_empty(self):
        assert _median([]) is None


class TestIVRankFromHistory:
    def test_thin_history_returns_none(self):
        hist = _history([0.2] * (_IV_HIST_MIN_POINTS - 1))
        assert _iv_rank_from_history(0.25, hist) is None

    def test_none_iv_returns_none(self):
        hist = _history([0.2] * (_IV_HIST_MIN_POINTS + 5))
        assert _iv_rank_from_history(None, hist) is None

    def test_current_at_top_of_range_is_100(self):
        hist = _history([0.10 + i * 0.01 for i in range(30)])  # 0.10 … 0.39
        assert _iv_rank_from_history(0.50, hist) == 100.0

    def test_current_at_bottom_of_range_is_0(self):
        hist = _history([0.10 + i * 0.01 for i in range(30)])
        assert _iv_rank_from_history(0.05, hist) == 0.0

    def test_midrange_rank(self):
        hist = _history([0.10 + i * 0.01 for i in range(30)])
        rank = _iv_rank_from_history(0.25, hist)
        assert rank is not None and 45.0 <= rank <= 60.0
