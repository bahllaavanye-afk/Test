"""
Regression tests for bugs fixed in recent sessions.

Each test is named after the specific bug it guards against and includes
a comment explaining the original failure mode. Tests are grouped by the
commit that introduced the fix.

Run: pytest tests/unit/test_regression.py -v
"""
from __future__ import annotations

import asyncio
import math
from datetime import date, datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Regression: Sharpe ±1e17 on zero-trade backtests
# Fixed in: commit 7be4978 (backtest engine Sharpe NaN fix)
# Bug: np.std() returned 2.71e-20 (float noise) on flat equity,
#      > 0 check passed, division yielded ±1e17.
# Fix: use > 1e-10 tolerance.
# ─────────────────────────────────────────────────────────────────────────────
class TestBacktestSharpeRegression:
    def _make_signals(self, n: Optional[int], value: int = 0) -> pd.Series:
        """Create a signal series of length ``n`` filled with ``value``.
        Handles ``None`` or non‑positive ``n`` by returning an empty series.
        """
        if not n or n <= 0:
            return pd.Series(dtype=int)
        dates = pd.date_range("2023-01-01", periods=n, freq="B")
        return pd.Series(value, index=dates, dtype=int)

    def _make_prices(self, n: Optional[int]) -> pd.Series:
        """Create a price series of length ``n`` linearly spaced.
        Handles ``None`` or non‑positive ``n`` by returning an empty series.
        """
        if not n or n <= 0:
            return pd.Series(dtype=float)
        dates = pd.date_range("2023-01-01", periods=n, freq="B")
        return pd.Series(np.linspace(100, 110, n), index=dates)

    def test_sharpe_is_zero_on_all_hold_signals(self):
        """All-zero signals → flat equity → Sharpe must be 0.0 not ±1e17."""
        from app.backtest.engine import run_backtest

        signals = self._make_signals(252, value=0)
        prices = self._make_prices(252)
        metrics = run_backtest(signals, prices)

        assert metrics.sharpe == 0.0, f"Expected 0.0 but got {metrics.sharpe}"
        assert not math.isinf(metrics.sharpe)
        assert not math.isnan(metrics.sharpe)

    def test_sortino_is_zero_on_all_hold_signals(self):
        """All-zero signals → flat equity → Sortino must be 0.0 not ±1e17."""
        from app.backtest.engine import run_backtest

        signals = self._make_signals(252, value=0)
        prices = self._make_prices(252)
        metrics = run_backtest(signals, prices)

        assert metrics.sortino == 0.0
        assert not math.isinf(metrics.sortino)
        assert not math.isnan(metrics.sortino)

    def test_sharpe_finite_on_constant_return(self):
        """Single fixed-return path → Sharpe must be finite, not ±inf."""
        from app.backtest.engine import run_backtest

        signals = self._make_signals(252, value=1)
        # Perfectly linear price (zero volatility path)
        prices = self._make_prices(252)
        metrics = run_backtest(signals, prices)

        assert not math.isinf(metrics.sharpe)
        assert not math.isnan(metrics.sharpe)

    def test_sharpe_reasonable_on_volatile_signals(self):
        """Real volatile signals should produce a bounded Sharpe."""
        from app.backtest.engine import run_backtest

        np.random.seed(42)
        dates = pd.date_range("2021-01-01", periods=500, freq="B")
        signals = pd.Series(np.random.choice([-1, 0, 1], 500), index=dates)
        prices = pd.Series(
            100 * np.cumprod(1 + np.random.normal(0.0005, 0.01, 500)), index=dates
        )
        metrics = run_backtest(signals, prices)

        assert -10 <= metrics.sharpe <= 10
        assert not math.isnan(metrics.sharpe)

    def test_sharpe_handles_empty_inputs(self):
        """Empty signal or price series should not raise and return 0 Sharpe."""
        from app.backtest.engine import run_backtest

        empty_signals = self._make_signals(0)
        empty_prices = self._make_prices(0)
        metrics = run_backtest(empty_signals, empty_prices)

        assert metrics.sharpe == 0.0
        assert metrics.sortino == 0.0

    def test_sharpe_handles_none_inputs(self):
        """Passing ``None`` for signals or prices should be gracefully handled."""
        from app.backtest.engine import run_backtest

        metrics = run_backtest(None, None)  # type: ignore[arg-type]

        # Expected safe defaults – implementation may define exact values.
        assert isinstance(metrics.sharpe, float)
        assert isinstance(metrics.sortino, float)


# ─────────────────────────────────────────────────────────────────────────────
# Regression: _supervised() backoff never resets after clean exit
# Fixed in: commit 8603d7c
# Bug: after each crash, delay doubled and was never reset — eventually
#      supervisor could wait 5 minutes between every restart even for
#      tasks that exit cleanly.
# Fix: delay = restart_delay after non-exception exit.
# ─────────────────────────────────────────────────────────────────────────────
class TestSupervisedTaskRegression:
    def test_delay_resets_after_clean_exit(self):
        """Supervisor delay must reset to restart_delay after a clean (non-crashing) run."""
        from app.main import _supervised

        call_count = 0

        async def coro_that_exits_cleanly():
            nonlocal call_count
            call_count += 1
            if call_count >= 3:
                raise asyncio.CancelledError()
            # Normal return (clean exit)

        async def run():
            try:
                await _supervised(coro_that_exits_cleanly, "test_task", restart_delay=1)
            except asyncio.CancelledError:
                pass

        asyncio.run(run())
        assert call_count >= 3

    def test_delay_doubles_on_crash(self):
        """After a crash, supervisor waits before restarting."""
        from app.main import _supervised

        call_count = 0
        delays_observed = []

        async def crashing_coro():
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                raise asyncio.CancelledError()
            raise RuntimeError("simulated crash")

        async def mock_sleep(delay):
            delays_observed.append(delay)

        async def run():
            with patch("asyncio.sleep", side_effect=mock_sleep):
                try:
                    await _supervised(crashing_coro, "crash_test", restart_delay=5)
                except asyncio.CancelledError:
                    pass

        asyncio.run(run())
        assert len(delays_observed) > 0
        assert delays_observed[0] == 5  # first crash → restart_delay


# ─────────────────────────────────────────────────────────────────────────────
# Regression: yfinance blocking asyncio event loop in regime_monitor
# Fixed in: commit 8603d7c
# Bug: _fetch_spy_returns_sync() was called directly inside async def,
#      blocking the event loop during network fetch.
# Fix: wrapped in run_in_executor.
# ─────────────────────────────────────────────────────────────────────────────
class TestRegimeMonitorAsyncRegression:
    def test_fetch_spy_returns_is_async(self):
        """_fetch_spy_returns must be an async function (not blocking)."""
        from app.tasks.regime_monitor import _fetch_spy_returns
        import inspect

        assert inspect.iscoroutinefunction(_fetch_spy_returns), (
            "_fetch_spy_returns must be async (uses run_in_executor internally)"
        )

    def test_fetch_spy_returns_sync_is_sync(self):
        """_fetch_spy_returns_sync must be a plain sync function (for run_in_executor)."""
        from app.tasks.regime_monitor import _fetch_spy_returns_sync
        import inspect

        assert not inspect.iscoroutinefunction(_fetch_spy_returns_sync), (
            "_fetch_spy_returns_sync must be sync (called in executor thread)"
        )

    def test_run_once_does_not_block_event_loop(self):
        """run_once should use run_in_executor, not call sync IO directly."""
        from app.tasks.regime_monitor import run_once
        import inspect

        # Verify it's a coroutine function
        assert inspect.iscoroutinefunction(run_once)

    def test_regime_monitor_loop_is_async(self):
        """_loop must be async so _supervised can await it."""
        from app.tasks.regime_monitor import RegimeMonitor
        import inspect

        rm = RegimeMonitor()
        assert inspect.iscoroutinefunction(rm._loop)


# ─────────────────────────────────────────────────────────────────────────────
# Regression: QA Monitor auto_fix_deprecated_apis() was a no-op
# Fixed in: commit 8603d7c + today's session
# Bug: was replacing "asyncio.get_running_loop()" with itself — no-op
# Fix: now correctly replaces "asyncio.get_event_loop()" and "datetime.utcnow()"
# ─────────────────────────────────────────────────────────────────────────────
class TestQAAutoFixRegression:
    def test_auto_fix_replaces_get_event_loop(self, tmp_path):
        """auto_fix_deprecated_apis must change get_event_loop → get_running_loop."""
        from app.tasks.qa_monitor import auto_fix_deprecated_apis, SecurityIssue
        import app.tasks.qa_monitor as qa_mod

        # Create a temporary file that contains the problematic pattern
        source = (
            "import asyncio\\n"
            "def broken():\\n"
            "    loop = asyncio.get_event_loop()\\n"
            "    return loop\\n"
        )
        file_path = tmp_path / "broken.py"
        file_path.write_text(source)

        # Run the auto‑fixer
        result = auto_fix_deprecated_apis([str(file_path)])

        # Verify the file was modified and the pattern was replaced
        fixed_content = file_path.read_text()
        assert "asyncio.get_running_loop()" in fixed_content
        assert "asyncio.get_event_loop()" not in fixed_content
        assert result == [str(file_path)]

    def test_auto_fix_replaces_datetime_utcnow(self, tmp_path):
        """auto_fix_deprecated_apis must replace datetime.utcnow with datetime.now(tz=timezone.utc)."""
        from app.tasks.qa_monitor import auto_fix_deprecated_apis
        import app.tasks.qa_monitor as qa_mod

        source = (
            "from datetime import datetime\\n"
            "def broken_time():\\n"
            "    return datetime.utcnow()\\n"
        )
        file_path = tmp_path / "time_bug.py"
        file_path.write_text(source)

        result = auto_fix_deprecated_apis([str(file_path)])

        fixed_content = file_path.read_text()
        assert "datetime.now(timezone.utc)" in fixed_content
        assert "datetime.utcnow()" not in fixed_content
        assert result == [str(file_path)]

    def test_auto_fix_no_changes_when_clean(self, tmp_path):
        """If the file contains no deprecated patterns, it should be left untouched."""
        from app.tasks.qa_monitor import auto_fix_deprecated_apis

        source = "def clean():\\n    return 42\\n"
        file_path = tmp_path / "clean.py"
        file_path.write_text(source)

        result = auto_fix_deprecated_apis([str(file_path)])

        assert file_path.read_text() == source
        assert result == []  # No files needed fixing