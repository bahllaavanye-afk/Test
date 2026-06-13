"""
Unit tests for the execution learner (slippage feedback loop).

The decision logic and cache wiring are tested directly; the DB aggregation is
covered with a fake async session yielding rows so no real database is needed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import app.execution.execution_learner as el
from app.execution.execution_learner import (
    AlgoStats, _decide_best, get_best_algo, get_scorecard, refresh_scorecard,
)


@pytest.fixture(autouse=True)
def _clear_cache():
    el._BEST_ALGO_BY_SYMBOL.clear()
    el._SCORECARD.clear()
    yield
    el._BEST_ALGO_BY_SYMBOL.clear()
    el._SCORECARD.clear()


class TestDecideBest:
    def test_needs_two_eligible_algos(self):
        stats = {"market": AlgoStats("market", 100, 10.0, 10.0)}
        assert _decide_best(stats, min_samples=15, min_edge_bps=1.0) is None

    def test_insufficient_samples_excluded(self):
        stats = {
            "market": AlgoStats("market", 5, 1.0, 1.0),    # too few samples
            "twap": AlgoStats("twap", 100, 10.0, 10.0),
        }
        # Only one eligible → no decision.
        assert _decide_best(stats, min_samples=15, min_edge_bps=1.0) is None

    def test_picks_lowest_slippage_with_edge(self):
        stats = {
            "market": AlgoStats("market", 50, 11.0, 11.0),
            "limit_first": AlgoStats("limit_first", 40, 3.0, 3.0),
        }
        assert _decide_best(stats, min_samples=15, min_edge_bps=1.0) == "limit_first"

    def test_no_decision_when_edge_too_small(self):
        stats = {
            "market": AlgoStats("market", 50, 5.4, 5.4),
            "twap": AlgoStats("twap", 50, 5.0, 5.0),
        }
        # Edge is only 0.4 bps < 1.0 → not worth overriding.
        assert _decide_best(stats, min_samples=15, min_edge_bps=1.0) is None


class TestCache:
    def test_get_best_algo_empty_by_default(self):
        assert get_best_algo("AAPL") is None


# ── DB aggregation via a fake async session ────────────────────────────────────
class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows


class _FakeSession:
    def __init__(self, rows):
        self._rows = rows

    async def execute(self, _query):
        return _FakeResult(self._rows)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


def _factory(rows):
    def _make():
        return _FakeSession(rows)
    return _make


class TestRefreshScorecard:
    @pytest.mark.asyncio
    async def test_learns_best_algo_and_updates_cache(self):
        # (symbol, algo, slippage_bps, is_cost_bps)
        rows = []
        rows += [("AAPL", "market", 12.0, 12.0)] * 20
        rows += [("AAPL", "limit_first", 3.0, 3.0)] * 20
        summary = await refresh_scorecard(db_session_factory=_factory(rows),
                                          min_samples=15, min_edge_bps=1.0)
        assert summary["learned"] == 1
        assert get_best_algo("AAPL") == "limit_first"
        card = get_scorecard()["AAPL"]
        assert card["best_algo"] == "limit_first"
        assert card["algos"]["limit_first"]["n"] == 20

    @pytest.mark.asyncio
    async def test_no_learning_without_enough_samples(self):
        rows = [("MSFT", "market", 5.0, 5.0)] * 3 + [("MSFT", "twap", 2.0, 2.0)] * 3
        summary = await refresh_scorecard(db_session_factory=_factory(rows),
                                          min_samples=15, min_edge_bps=1.0)
        assert summary["learned"] == 0
        assert get_best_algo("MSFT") is None

    @pytest.mark.asyncio
    async def test_skips_rows_with_missing_fields(self):
        rows = [(None, "market", 5.0, 5.0), ("X", None, 5.0, 5.0), ("X", "market", None, None)]
        summary = await refresh_scorecard(db_session_factory=_factory(rows))
        # Nothing usable → no symbols with complete data.
        assert summary["learned"] == 0
