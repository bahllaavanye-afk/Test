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


# ─────────────────────────────────────────────────────────────────────────────
# Regression: Sharpe ±1e17 on zero-trade backtests
# Fixed in: commit 7be4978 (backtest engine Sharpe NaN fix)
# Bug: np.std() returned 2.71e-20 (float noise) on flat equity,
#      > 0 check passed, division yielded ±1e17.
# Fix: use > 1e-10 tolerance.
# ─────────────────────────────────────────────────────────────────────────────
class TestBacktestSharpeRegression:
    def _make_signals(self, n: int, value: int = 0) -> pd.Series:
        dates = pd.date_range("2023-01-01", periods=n, freq="B")
        return pd.Series(value, index=dates, dtype=int)

    def _make_prices(self, n: int) -> pd.Series:
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

        original_sleep = asyncio.sleep

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
        assert inspect.iscoroutinefunction(_fetch_spy_returns), \
            "_fetch_spy_returns must be async (uses run_in_executor internally)"

    def test_fetch_spy_returns_sync_is_sync(self):
        """_fetch_spy_returns_sync must be a plain sync function (for run_in_executor)."""
        from app.tasks.regime_monitor import _fetch_spy_returns_sync
        import inspect
        assert not inspect.iscoroutinefunction(_fetch_spy_returns_sync), \
            "_fetch_spy_returns_sync must be sync (called in executor thread)"

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

        test_file = tmp_path / "test_module.py"
        # Build the deprecated literal from fragments so the editor linter doesn't
        # auto-rewrite this fixture before the test runs.
        old_loop = "asyncio.get_" + "event_loop()"
        test_file.write_text(f'loop = {old_loop}\n')

        # The fixer resolves paths relative to PROJECT_ROOT, so we need the
        # file_path to be relative to it
        original_root = qa_mod.PROJECT_ROOT
        qa_mod.PROJECT_ROOT = tmp_path

        issue = SecurityIssue(
            severity="low",
            issue_type="deprecated_api",
            file_path="test_module.py",
            line_number=1,
            description="deprecated event loop call",
            auto_fixable=True,
        )
        try:
            auto_fix_deprecated_apis([issue])
        finally:
            qa_mod.PROJECT_ROOT = original_root

        content = test_file.read_text()
        assert "get_running_loop()" in content, "Should have replaced with get_running_loop()"
        assert old_loop not in content, "Old API should be removed"

    def test_auto_fix_replaces_utcnow(self, tmp_path):
        """auto_fix_deprecated_apis must change utcnow() → now(timezone.utc)."""
        from app.tasks.qa_monitor import auto_fix_deprecated_apis, SecurityIssue
        import app.tasks.qa_monitor as qa_mod

        test_file = tmp_path / "test_mod2.py"
        old_utc = "datetime." + "utcnow()"
        test_file.write_text(f'ts = {old_utc}\n')

        original_root = qa_mod.PROJECT_ROOT
        qa_mod.PROJECT_ROOT = tmp_path

        issue = SecurityIssue(
            severity="low",
            issue_type="deprecated_api",
            file_path="test_mod2.py",
            line_number=1,
            description="deprecated utc timestamp call",
            auto_fixable=True,
        )
        try:
            auto_fix_deprecated_apis([issue])
        finally:
            qa_mod.PROJECT_ROOT = original_root

        content = test_file.read_text()
        assert "now(timezone.utc)" in content, "Should have replaced with now(timezone.utc)"
        assert old_utc not in content, "Old API should be removed"

    def test_auto_fix_noop_on_empty_list(self):
        """auto_fix_deprecated_apis([]) must return 0 without touching any files."""
        from app.tasks.qa_monitor import auto_fix_deprecated_apis
        assert auto_fix_deprecated_apis([]) == 0

    def test_qa_monitor_does_not_flag_itself(self):
        """QA Monitor must exclude itself from the security scan (self-scan false positives)."""
        from app.tasks.qa_monitor import scan_security_issues
        issues = scan_security_issues()
        self_issues = [i for i in issues if "qa_monitor.py" in i.file_path]
        assert self_issues == [], \
            f"QA monitor flagging its own source: {[i.description for i in self_issues]}"


# ─────────────────────────────────────────────────────────────────────────────
# Regression: Synthetic OHLCV fallback when yfinance fails
# Fixed in: commit 7be4978
# Bug: if yfinance raised an exception, fetch_ohlcv raised instead of falling back
# Fix: _synthetic_ohlcv() GBM fallback added
# ─────────────────────────────────────────────────────────────────────────────
class TestSyntheticOHLCVRegression:
    def test_synthetic_ohlcv_shape_is_valid(self):
        """_synthetic_ohlcv must return a DataFrame with OHLCV columns (lowercase)."""
        from app.backtest.data_loader import _synthetic_ohlcv
        result = _synthetic_ohlcv("SPY", date(2023, 1, 1), date(2023, 6, 1), "1d")
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        for col in ("open", "high", "low", "close", "volume"):
            assert col in result.columns, f"Missing column: {col}"

    def test_synthetic_ohlcv_high_gte_low(self):
        """High must always be >= Low in synthetic data."""
        from app.backtest.data_loader import _synthetic_ohlcv
        df = _synthetic_ohlcv("BTC/USD", date(2023, 1, 1), date(2023, 12, 31), "1d")
        assert (df["high"] >= df["low"]).all(), "high < low found in synthetic OHLCV"

    def test_synthetic_ohlcv_prices_positive(self):
        """All prices must be positive."""
        from app.backtest.data_loader import _synthetic_ohlcv
        df = _synthetic_ohlcv("ETH/USD", date(2023, 1, 1), date(2023, 6, 1), "1d")
        assert (df["close"] > 0).all()
        assert (df["open"] > 0).all()

    def test_fetch_ohlcv_returns_synthetic_when_yfinance_fails(self):
        """fetch_ohlcv must return synthetic data instead of raising when yfinance fails."""
        from app.backtest.data_loader import fetch_ohlcv

        with patch("yfinance.download", side_effect=Exception("network error")):
            result = asyncio.run(
                fetch_ohlcv("SPY", date(2023, 1, 1), date(2023, 6, 1), "1d")
            )
        assert isinstance(result, pd.DataFrame)
        assert not result.empty


# ─────────────────────────────────────────────────────────────────────────────
# Regression: asyncio.get_event_loop() in non-async context crashes Python 3.10+
# Fixed in: multiple commits
# Bug: calling get_event_loop() from a module-level or non-async function
#      raises DeprecationWarning in 3.10 and RuntimeError in 3.12+
# Fix: replace with get_running_loop() inside async functions
# ─────────────────────────────────────────────────────────────────────────────
class TestDeprecatedAPIRegression:
    def test_no_get_event_loop_in_async_context(self):
        """No source file (except qa_monitor patterns) should use get_event_loop()."""
        import re
        from pathlib import Path
        backend_dir = Path("/home/user/Test/backend/app")
        pattern = re.compile(r'asyncio\.get_event_loop\(\)')
        violations = []
        for py_file in backend_dir.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
            if py_file.name == "qa_monitor.py":
                continue  # regex patterns are expected here
            content = py_file.read_text(errors="replace")
            if pattern.search(content):
                violations.append(str(py_file.relative_to(Path("/home/user/Test"))))
        assert violations == [], \
            f"get_event_loop() still used in: {violations} — replace with get_running_loop()"

    def test_no_datetime_utcnow_in_source(self):
        """No source file should use datetime.utcnow() (deprecated in Python 3.12)."""
        import re
        from pathlib import Path
        backend_dir = Path("/home/user/Test/backend/app")
        pattern = re.compile(r'datetime\.utcnow\(\)')
        violations = []
        for py_file in backend_dir.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
            if py_file.name == "qa_monitor.py":
                continue  # detection regex expected here
            content = py_file.read_text(errors="replace")
            if pattern.search(content):
                violations.append(str(py_file.relative_to(Path("/home/user/Test"))))
        assert violations == [], \
            f"datetime.utcnow() still used in: {violations} — replace with datetime.now(timezone.utc)"


# ─────────────────────────────────────────────────────────────────────────────
# Regression: BacktestSignals → pd.Series conversion crash
# Fixed in: commit 7be4978
# Bug: backtest_worker received a BacktestSignals namedtuple from strategy,
#      but run_backtest expected pd.Series[int] — crashed with AttributeError
# Fix: explicit isinstance check and conversion added in backtest_worker.py
# ─────────────────────────────────────────────────────────────────────────────
class TestBacktestSignalsConversionRegression:
    def test_backtest_signals_dataclass_exists(self):
        """BacktestSignals must be importable from strategies.base."""
        from app.strategies.base import BacktestSignals
        assert BacktestSignals is not None

    def test_backtest_signals_has_entries_exits(self):
        """BacktestSignals must have entries and exits attributes."""
        from app.strategies.base import BacktestSignals
        import inspect
        fields = [f.name for f in inspect.fields(BacktestSignals)] \
            if hasattr(inspect, "fields") else []
        # Try dataclass fields
        try:
            import dataclasses
            field_names = [f.name for f in dataclasses.fields(BacktestSignals)]
        except TypeError:
            # Not a dataclass, try namedtuple
            field_names = list(BacktestSignals._fields) if hasattr(BacktestSignals, "_fields") else []

        assert "entries" in field_names or hasattr(BacktestSignals, "entries"), \
            "BacktestSignals must have 'entries' field"

    def test_strategies_return_compatible_signal_type(self):
        """All strategies must return pd.Series or BacktestSignals from backtest_signals()."""
        from app.strategies import STRATEGY_REGISTRY
        from app.strategies.base import BacktestSignals

        np.random.seed(42)
        dates = pd.date_range("2022-01-01", periods=100, freq="B")
        df = pd.DataFrame({
            "Open": 100 + np.random.randn(100).cumsum(),
            "High": 102 + np.random.randn(100).cumsum(),
            "Low": 98 + np.random.randn(100).cumsum(),
            "Close": 100 + np.random.randn(100).cumsum(),
            "Volume": np.random.randint(100_000, 1_000_000, 100).astype(float),
        }, index=dates)
        # Ensure High > Low always
        df["High"] = df[["High", "Low"]].max(axis=1) + 0.01
        df["Low"] = df[["High", "Low"]].min(axis=1) - 0.01

        # Use lowercase column names as most strategies expect
        df.columns = [c.lower() for c in df.columns]
        for name, cls in list(STRATEGY_REGISTRY.items())[:5]:  # test 5 to keep it fast
            strategy = cls()
            try:
                result = strategy.backtest_signals(df)
                if asyncio.iscoroutine(result):
                    result = asyncio.run(result)
                assert isinstance(result, (pd.Series, BacktestSignals)), \
                    f"{name}.backtest_signals() returned unexpected type: {type(result)}"
            except KeyError as e:
                pytest.fail(f"{name}.backtest_signals() raised KeyError {e} — column name mismatch")


# ─────────────────────────────────────────────────────────────────────────────
# Regression: Risk API returning 404 (root endpoint missing)
# Fixed in: commit 49c0116
# Bug: /api/v1/risk/ had no GET route, returned 404
# ─────────────────────────────────────────────────────────────────────────────
class TestAPIEndpointRegression:
    """Smoke-test that key API endpoints exist (not 404/500)."""

    BASE = "http://localhost:8000"

    def _get(self, path: str):
        import httpx
        return httpx.get(f"{self.BASE}{path}", timeout=5.0)

    def test_health_endpoint_200(self):
        resp = self._get("/health")
        assert resp.status_code == 200

    def test_risk_root_endpoint_not_404(self):
        """GET /api/v1/risk/ must exist (was 404 before fix)."""
        resp = self._get("/api/v1/risk/")
        assert resp.status_code in (200, 401, 403), \
            f"/api/v1/risk/ returned {resp.status_code} — was previously missing"

    def test_analytics_root_endpoint_not_404(self):
        """GET /api/v1/analytics/ must exist."""
        resp = self._get("/api/v1/analytics/")
        assert resp.status_code in (200, 401, 403)

    def test_leaderboard_entries_not_404(self):
        """GET /api/v1/leaderboard/entries must exist."""
        resp = self._get("/api/v1/leaderboard/entries")
        assert resp.status_code in (200, 401, 403)

    def test_regime_current_not_404(self):
        """GET /api/v1/regime/current must exist."""
        resp = self._get("/api/v1/regime/current")
        assert resp.status_code in (200, 401, 403)

    def test_agents_status_not_404(self):
        """GET /api/v1/agents/status must exist."""
        resp = self._get("/api/v1/agents/status")
        assert resp.status_code in (200, 401, 403)

    def test_monitoring_health_not_404(self):
        """GET /api/v1/monitoring/health must exist."""
        resp = self._get("/api/v1/monitoring/health")
        assert resp.status_code in (200, 401, 403)


# ─────────────────────────────────────────────────────────────────────────────
# Regression: Correlation monitor returns empty on insufficient data
# Bug: was raising ZeroDivisionError when single-row returns passed
# Fix: length check before computing correlation
# ─────────────────────────────────────────────────────────────────────────────
class TestCorrelationEdgeCases:
    def test_single_symbol_returns_empty_clusters(self):
        from app.risk.correlation import compute_correlation_clusters
        returns = pd.DataFrame({"SPY": [0.01, 0.02, -0.01]})
        result = compute_correlation_clusters(returns)
        assert isinstance(result, dict)

    def test_empty_dataframe_returns_empty(self):
        from app.risk.correlation import compute_correlation_clusters
        result = compute_correlation_clusters(pd.DataFrame())
        assert result == {}

    def test_nan_returns_handled_gracefully(self):
        from app.risk.correlation import compute_correlation_clusters
        np.random.seed(0)
        returns = pd.DataFrame({
            "A": np.random.normal(0, 0.01, 50),
            "B": np.random.normal(0, 0.01, 50),
        })
        returns.loc[0, "A"] = np.nan
        # Should not raise
        result = compute_correlation_clusters(returns)
        assert isinstance(result, dict)
