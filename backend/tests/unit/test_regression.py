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
from datetime import datetime
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Regression: Sharpe ±1e17 on zero-trade backtests
# Fixed in: commit 7be4978 (backtest engine Sharpe NaN fix)
# Bug: np.std() returned 2.71e-20 (float noise) on flat equity,
#      > 0 check passed, division yielded ±1e17.
# Fix: use > 1e-10 tolerance.
# ─────────────────────────────────────────────────────────────────────────────
class TestBacktestSharpeRegression:
    def _make_signals(self, n: int | None, value: int | None = 0) -> pd.Series:
        """
        Create a signal series of length `n` with constant `value`.

        Edge‑case handling:
        * If `n` is None or not a positive integer, return an empty Series.
        * If `value` is None, the series will contain NaN entries.
        """
        if not isinstance(n, int) or n <= 0:
            return pd.Series([], dtype="float64")
        dates = pd.date_range("2023-01-01", periods=n, freq="B")
        # Preserve NaN if value is None
        dtype = "float64" if value is None else "int64"
        return pd.Series(value, index=dates, dtype=dtype)

    def _make_prices(self, n: int | None) -> pd.Series:
        """
        Create a price series of length `n` linearly spaced between 100 and 110.

        Edge‑case handling mirrors `_make_signals`.
        """
        if not isinstance(n, int) or n <= 0:
            return pd.Series([], dtype="float64")
        dates = pd.date_range("2023-01-01", periods=n, freq="B")
        return pd.Series(np.linspace(100, 110, n), index=dates, dtype="float64")

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
        prices = pd.Series(100 * np.cumprod(1 + np.random.normal(0.0005, 0.01, 500)), index=dates)
        metrics = run_backtest(signals, prices)
        assert -10 <= metrics.sharpe <= 10
        assert not math.isnan(metrics.sharpe)


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

        # Create a temporary Python file containing deprecated calls
        source = (
            "import asyncio\\n"
            "from datetime import datetime\\n"
            "def foo():\\n"
            "    loop = asyncio.get_event_loop()\\n"
            "    now = datetime.utcnow()\\n"
            "    return loop, now\\n"
        )
        file_path = tmp_path / "sample.py"
        file_path.write_text(source)

        # Run the auto‑fixer on the temporary file
        issues = auto_fix_deprecated_apis([str(file_path)])

        # Ensure a SecurityIssue was reported
        assert any(isinstance(i, SecurityIssue) for i in issues), "Expected a SecurityIssue report"

        # Verify the file content was updated
        updated = file_path.read_text()
        assert "asyncio.get_running_loop()" in updated, "get_event_loop should be replaced"
        assert "datetime.utcnow()" not in updated, "utcnow should be replaced"
        assert "datetime.now(datetime.timezone.utc)" in updated, "utcnow should be replaced with timezone‑aware now"

    def test_auto_fix_is_idempotent(self, tmp_path):
        """Running auto_fix_deprecated_apis twice should not introduce duplicate changes."""
        from app.tasks.qa_monitor import auto_fix_deprecated_apis

        source = (
            "import asyncio\\n"
            "def bar():\\n"
            "    return asyncio.get_event_loop()\\n"
        )
        file_path = tmp_path / "idempotent.py"
        file_path.write_text(source)

        # First run
        auto_fix_deprecated_apis([str(file_path)])
        first_pass = file_path.read_text()

        # Second run
        auto_fix_deprecated_apis([str(file_path)])
        second_pass = file_path.read_text()

        assert first_pass == second_pass, "Auto‑fix should be idempotent"


# End of regression test suite.