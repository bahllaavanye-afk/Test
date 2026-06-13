"""
Unit tests for profit forecasting math and the forecasting desk's report logic.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.analytics.forecasting import (
    summarize_daily_pnl, project, build_forecast, TRADING_DAYS,
)
from app.tasks.forecasting_desk import ForecastingDesk


class TestSummarize:
    def test_empty(self):
        s = summarize_daily_pnl([])
        assert s.n_days == 0 and s.total_pnl == 0.0 and s.sharpe_annual == 0.0

    def test_basic_stats(self):
        s = summarize_daily_pnl([10.0, 20.0, 30.0])
        assert s.n_days == 3
        assert s.total_pnl == 60.0
        assert s.mean_daily == 20.0
        assert s.positive_day_rate == 1.0

    def test_positive_day_rate(self):
        s = summarize_daily_pnl([5.0, -5.0, 5.0, -5.0])
        assert s.positive_day_rate == 0.5

    def test_zero_std_gives_zero_sharpe(self):
        s = summarize_daily_pnl([7.0, 7.0, 7.0])
        assert s.std_daily == 0.0
        assert s.sharpe_annual == 0.0


class TestProject:
    def test_expected_scales_linearly(self):
        pnls = [10.0] * 10  # mean 10, std 0
        p = project(pnls, horizon_days=5)
        assert p.expected_pnl == 50.0
        # Zero variance → tight band.
        assert p.low_pnl == 50.0 and p.high_pnl == 50.0

    def test_band_scales_with_sqrt_horizon(self):
        pnls = [10.0, -10.0, 10.0, -10.0, 10.0, -10.0]  # mean 0, nonzero std
        p = project(pnls, horizon_days=4, z=1.0)
        s = summarize_daily_pnl(pnls)
        expected_band = s.std_daily * math.sqrt(4)
        assert p.high_pnl == round(0.0 + expected_band, 2)


class TestBuildForecast:
    def test_insufficient_data_flag(self):
        f = build_forecast([1.0, 2.0])  # < 5 days
        assert f["sufficient_data"] is False

    def test_has_all_horizons(self):
        f = build_forecast([float(i % 7 - 3) for i in range(60)])
        assert set(f["projections"].keys()) == set(TRADING_DAYS.keys())
        assert f["sufficient_data"] is True

    def test_per_desk_breakdown(self):
        overall = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        by_desk = {"momentum": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                   "arb": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]}
        f = build_forecast(overall, by_desk=by_desk)
        assert "momentum" in f["by_desk"]
        assert "arb" in f["by_desk"]
        assert "yearly_expected_pnl" in f["by_desk"]["momentum"]


class TestDeskReport:
    def test_insufficient_report_text(self):
        desk = ForecastingDesk()
        text = desk._format_report({"sufficient_data": False, "stats": {"n_days": 2}})
        assert "insufficient history" in text

    def test_full_report_text(self):
        desk = ForecastingDesk()
        f = build_forecast([float(i % 5 - 2) for i in range(60)])
        text = desk._format_report(f)
        assert "Profit Forecast" in text
        assert "Weekly" in text and "Yearly" in text
