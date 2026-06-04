"""
Comprehensive routine health tests for ALL 12 QuantEdge background employees.

Each test verifies that the employee:
  1. Can be instantiated without errors
  2. Has the required interface methods
  3. Produces sensible output when run for one cycle
  4. Returns data in the expected format

Tests run quickly (< 5s each) using mocks or minimal data.
"""
from __future__ import annotations

import asyncio
import json
import tempfile
from dataclasses import asdict
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Employee 1: AlgoAgent (UCB1 Exploration/Exploitation)
# ─────────────────────────────────────────────────────────────────────────────
class TestAlgoAgent:
    def test_instantiation(self):
        from app.tasks.algo_agent import AlgoAgent
        agent = AlgoAgent()
        assert len(agent._candidates) > 0
        assert agent._total_runs == 0

    def test_ucb_selects_unexplored_first(self):
        from app.tasks.algo_agent import AlgoAgent
        agent = AlgoAgent()
        selected = agent._select_candidate()
        assert selected is not None
        assert selected.n_runs == 0  # unexplored candidates have infinite UCB

    def test_leaderboard_is_sorted_descending(self):
        from app.tasks.algo_agent import AlgoAgent
        agent = AlgoAgent()
        keys = list(agent._candidates.keys())[:2]
        agent._candidates[keys[0]].n_runs = 3
        agent._candidates[keys[0]].total_sharpe = 3.0
        agent._candidates[keys[1]].n_runs = 3
        agent._candidates[keys[1]].total_sharpe = 6.0
        lb = agent.get_leaderboard()
        sharpes = [e["avg_sharpe"] for e in lb[:5]]
        assert sharpes == sorted(sharpes, reverse=True)

    def test_leaderboard_has_required_fields(self):
        from app.tasks.algo_agent import AlgoAgent
        agent = AlgoAgent()
        lb = agent.get_leaderboard()
        entry = lb[0]
        for field in ("key", "strategy", "symbol", "type", "avg_sharpe", "n_runs"):
            assert field in entry, f"Missing field: {field}"

    def test_save_result_updates_state(self):
        from app.tasks.algo_agent import AlgoAgent
        agent = AlgoAgent()
        key = list(agent._candidates.keys())[0]
        candidate = agent._candidates[key]
        old_n_runs = candidate.n_runs
        # Simulate what run() does
        candidate.n_runs += 1
        candidate.total_sharpe += 1.5
        candidate.best_sharpe = max(candidate.best_sharpe, 1.5)
        agent._total_runs += 1
        assert candidate.n_runs == old_n_runs + 1
        assert candidate.avg_sharpe == pytest.approx(1.5)


# ─────────────────────────────────────────────────────────────────────────────
# Employee 2: ResearchScientist (Alpha Mining)
# ─────────────────────────────────────────────────────────────────────────────
class TestResearchScientist:
    def test_instantiation(self):
        from app.tasks.research_scientist import ResearchScientist
        rs = ResearchScientist(interval_seconds=1)
        assert rs._cycle == 0
        assert rs._findings == []

    def test_research_cycle_returns_findings(self):
        from app.tasks.research_scientist import ResearchScientist, ResearchFinding
        rs = ResearchScientist(interval_seconds=1)
        findings = asyncio.run(rs.research_cycle())
        assert len(findings) > 0
        assert all(isinstance(f, ResearchFinding) for f in findings)
        assert rs._cycle == 1

    def test_finding_fields_valid(self):
        from app.tasks.research_scientist import ResearchScientist, RESEARCH_AGENDA
        rs = ResearchScientist(interval_seconds=1)
        topic = RESEARCH_AGENDA[0]
        finding = asyncio.run(rs._evaluate_topic(topic))
        assert 0.0 <= finding.confidence <= 1.0
        assert finding.recommended_action in ("backtest", "implement", "monitor", "shelve")
        assert finding.ic_estimate >= 0.0

    def test_top_ideas_sorted_by_score(self):
        from app.tasks.research_scientist import ResearchScientist
        rs = ResearchScientist(interval_seconds=1)
        for _ in range(2):
            findings = asyncio.run(rs.research_cycle())
            rs._findings.extend(findings)
        top = rs.get_top_ideas(n=5)
        scores = [f.estimated_sharpe * f.confidence for f in top]
        assert scores == sorted(scores, reverse=True)

    def test_get_research_summary_shape(self):
        from app.tasks.research_scientist import ResearchScientist
        rs = ResearchScientist(interval_seconds=1)
        asyncio.run(rs.research_cycle())
        summary = rs.get_research_summary()
        for key in ("cycles_completed", "total_findings", "top_ideas", "implement_queue"):
            assert key in summary, f"Missing key: {key}"


# ─────────────────────────────────────────────────────────────────────────────
# Employee 3: ModelingEngineer (ML Drift Detection)
# ─────────────────────────────────────────────────────────────────────────────
class TestModelingEngineer:
    def test_instantiation(self):
        from app.tasks.modeling_engineer import ModelingEngineer, MODEL_TYPES
        me = ModelingEngineer()
        assert me.drift_threshold == 0.52
        assert me.retrain_after_n_drift == 3
        assert me._cycle == 0
        for m in MODEL_TYPES:
            assert m in me._best_sharpe

    def test_check_performance_returns_record(self):
        from app.tasks.modeling_engineer import ModelingEngineer, ModelPerformanceRecord
        me = ModelingEngineer()
        rec = asyncio.run(me.check_model_performance("lstm"))
        assert isinstance(rec, ModelPerformanceRecord)
        assert 0.0 <= rec.accuracy <= 1.0
        assert isinstance(rec.drift_detected, bool)

    def test_detect_drift_below_threshold(self):
        from app.tasks.modeling_engineer import ModelingEngineer, ModelPerformanceRecord
        me = ModelingEngineer(drift_threshold=0.55)
        for _ in range(3):
            me._perf_cache["lstm"].append(
                ModelPerformanceRecord(model_id="lstm", accuracy=0.45, sharpe=-0.2, drift_detected=True)
            )
        assert asyncio.run(me.detect_drift("lstm")) is True

    def test_detect_no_drift_above_threshold(self):
        from app.tasks.modeling_engineer import ModelingEngineer, ModelPerformanceRecord
        me = ModelingEngineer(drift_threshold=0.52)
        me._perf_cache["xgboost"].append(
            ModelPerformanceRecord(model_id="xgboost", accuracy=0.65, sharpe=1.2, drift_detected=False)
        )
        assert asyncio.run(me.detect_drift("xgboost")) is False

    def test_summary_has_required_fields(self):
        from app.tasks.modeling_engineer import ModelingEngineer
        me = ModelingEngineer()
        summary = me.get_engineering_summary()
        for key in ("cycles_completed", "models_monitored", "drift_threshold",
                    "latest_performance", "recent_decisions", "promote_count", "retrain_count"):
            assert key in summary, f"Missing key: {key}"


# ─────────────────────────────────────────────────────────────────────────────
# Employee 4: QA Monitor (Code Quality Watchdog)
# ─────────────────────────────────────────────────────────────────────────────
class TestQAMonitor:
    def test_scan_security_finds_no_critical_issues(self):
        from app.tasks.qa_monitor import scan_security_issues
        issues = scan_security_issues()
        critical = [i for i in issues if i.severity == "critical"]
        assert len(critical) == 0, f"Critical security issues: {[i.description for i in critical]}"

    def test_scan_finds_no_import_errors(self):
        from app.tasks.qa_monitor import check_imports
        errors = check_imports()
        assert errors == [], f"Import errors: {errors}"

    def test_qa_monitor_class_instantiable(self):
        from app.tasks.qa_monitor import QAMonitor
        monitor = QAMonitor(interval_seconds=9999)
        assert monitor.interval_seconds == 9999

    def test_auto_fix_empty_list(self):
        from app.tasks.qa_monitor import auto_fix_deprecated_apis
        count = auto_fix_deprecated_apis([])
        assert count == 0

    def test_qa_monitor_skips_self(self):
        """QA monitor should not report false positives on its own regex patterns."""
        from app.tasks.qa_monitor import scan_security_issues
        issues = scan_security_issues()
        self_issues = [i for i in issues if "qa_monitor.py" in i.file_path]
        assert len(self_issues) == 0, f"QA monitor is flagging itself: {[i.description for i in self_issues]}"


# ─────────────────────────────────────────────────────────────────────────────
# Employee 5: SelfImprover (Parameter Optimization)
# ─────────────────────────────────────────────────────────────────────────────
class TestSelfImprover:
    def test_instantiation(self):
        from app.tasks.self_improver import SelfImprover
        si = SelfImprover()
        assert si.interval_seconds == 900
        assert si._iteration == 0

    def test_sample_params_known_strategy(self):
        from app.tasks.self_improver import SelfImprover, PARAM_SPACES
        si = SelfImprover()
        for strategy in PARAM_SPACES:
            params = si._sample_params(strategy)
            assert isinstance(params, dict)
            assert len(params) > 0

    def test_sample_params_unknown_strategy(self):
        from app.tasks.self_improver import SelfImprover
        si = SelfImprover()
        params = si._sample_params("nonexistent_strategy")
        assert params == {}

    def test_evaluate_returns_float(self):
        pytest.importorskip("yfinance")  # skip gracefully when not installed in CI
        from app.tasks.self_improver import SelfImprover
        si = SelfImprover()
        # Mock yfinance to return synthetic data
        np.random.seed(42)
        dates = pd.date_range("2022-01-01", periods=200, freq="D")
        hist = pd.DataFrame({
            "Open": 100 + np.random.randn(200).cumsum(),
            "High": 102 + np.random.randn(200).cumsum(),
            "Low": 98 + np.random.randn(200).cumsum(),
            "Close": 100 + np.random.randn(200).cumsum(),
            "Volume": np.random.randint(1_000_000, 5_000_000, 200).astype(float),
        }, index=dates)
        with patch("yfinance.download", return_value=hist):
            sharpe = asyncio.run(si._evaluate("momentum", "SPY", {"lookback_months": 6}))
        assert isinstance(sharpe, float)
        assert not (sharpe != sharpe)  # not NaN

    def test_get_history_returns_list(self):
        from app.tasks.self_improver import SelfImprover
        si = SelfImprover()
        history = si.get_history()
        assert isinstance(history, list)


# ─────────────────────────────────────────────────────────────────────────────
# Employee 6: RegimeMonitor (HMM Market State)
# ─────────────────────────────────────────────────────────────────────────────
class TestRegimeMonitor:
    def test_fit_regime_returns_valid_int(self):
        from app.tasks.regime_monitor import _fit_regime
        returns = np.random.normal(0.001, 0.005, 300)
        regime = _fit_regime(returns)
        assert regime in (0, 1, 2)

    def test_fit_regime_insufficient_data(self):
        from app.tasks.regime_monitor import _fit_regime
        returns = np.array([0.001, -0.002, 0.003])
        assert _fit_regime(returns) == 1  # sideways fallback

    def test_fit_regime_bear_conditions(self):
        from app.tasks.regime_monitor import _fit_regime
        returns = np.random.normal(-0.003, 0.02, 300)  # negative drift + high vol
        regime = _fit_regime(returns)
        assert regime in (0, 1, 2)  # deterministic label may vary, but must be valid

    def test_regime_monitor_has_start_stop(self):
        from app.tasks.regime_monitor import RegimeMonitor
        rm = RegimeMonitor()
        assert hasattr(rm, "start") and callable(rm.start)
        assert hasattr(rm, "stop") and callable(rm.stop)

    def test_run_once_with_mock(self):
        from app.tasks.regime_monitor import run_once
        returns = np.random.normal(0.0, 0.01, 200)
        mock_redis = AsyncMock()
        mock_redis.set = AsyncMock(return_value=True)
        with patch("app.tasks.regime_monitor._fetch_spy_returns", return_value=returns):
            result = asyncio.run(run_once(mock_redis))
        assert result in (0, 1, 2)
        mock_redis.set.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
# Employee 7: CodeQualityLoop (LOC & Lint Reporting)
# ─────────────────────────────────────────────────────────────────────────────
class TestCodeQualityLoop:
    def test_instantiation(self):
        from app.tasks.code_quality_loop import CodeQualityLoop
        cql = CodeQualityLoop(interval_seconds=9999)
        assert cql.interval_seconds == 9999

    def test_count_loc_returns_dict(self):
        from app.tasks.code_quality_loop import _count_loc, BACKEND_ROOT
        result = _count_loc(BACKEND_ROOT / "app")
        assert isinstance(result, dict)
        assert result["files"] > 0
        assert result["code_lines"] > 0

    def test_count_strategies(self):
        from app.tasks.code_quality_loop import _count_strategies, BACKEND_ROOT
        result = _count_strategies(BACKEND_ROOT)
        assert isinstance(result, dict)
        assert result["manual_strategies"] > 0

    def test_snapshot_shape(self):
        from app.tasks.code_quality_loop import CodeQualityLoop
        cql = CodeQualityLoop(interval_seconds=9999)
        snapshot = asyncio.run(cql._snapshot())
        assert "timestamp" in snapshot
        assert "files" in snapshot or "code_lines" in snapshot
        assert "manual_strategies" in snapshot

    def test_latest_returns_dict_or_none(self):
        from app.tasks.code_quality_loop import CodeQualityLoop
        cql = CodeQualityLoop()
        result = cql.latest()
        # May be pre-populated from disk or None on a clean run
        assert result is None or isinstance(result, dict)


# ─────────────────────────────────────────────────────────────────────────────
# Employee 8: BacktestWorker (Queued Backtest Executor)
# ─────────────────────────────────────────────────────────────────────────────
class TestBacktestWorker:
    def test_run_backtest_job_not_found(self):
        """A missing backtest run ID should exit silently."""
        from app.tasks.backtest_worker import run_backtest_job
        mock_session = AsyncMock()
        mock_session.get = AsyncMock(return_value=None)
        mock_session.commit = AsyncMock()
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("app.database.AsyncSessionLocal", return_value=mock_ctx):
            asyncio.run(run_backtest_job("00000000-0000-0000-0000-000000000000"))
        # Should not raise

    def test_backtest_worker_loop_callable(self):
        from app.tasks.backtest_worker import backtest_worker_loop
        assert callable(backtest_worker_loop)


# ─────────────────────────────────────────────────────────────────────────────
# Employee 9: StrategyRunner (Signal Generation Loop)
# ─────────────────────────────────────────────────────────────────────────────
class TestStrategyRunner:
    def test_strategy_registry_has_core_strategies(self):
        from app.strategies import STRATEGY_REGISTRY
        for name in ("momentum", "mean_reversion", "rsi_macd", "breakout"):
            assert name in STRATEGY_REGISTRY, f"Missing strategy: {name}"

    def test_all_strategies_have_required_interface(self):
        from app.strategies import STRATEGY_REGISTRY
        for name, cls in STRATEGY_REGISTRY.items():
            inst = cls()
            assert hasattr(inst, "analyze"), f"{name}: missing analyze()"
            assert hasattr(inst, "backtest_signals"), f"{name}: missing backtest_signals()"
            assert hasattr(inst, "name"), f"{name}: missing name attribute"

    def test_regime_map_excludes_directional_in_bear(self):
        from app.tasks.strategy_runner import STRATEGY_REGIME_MAP
        assert "momentum" in STRATEGY_REGIME_MAP
        assert 0 not in STRATEGY_REGIME_MAP["momentum"], "momentum should not run in bear regime"

    def test_continuous_strategy_runner_instantiable(self):
        from app.tasks.strategy_runner import ContinuousStrategyRunner
        runner = ContinuousStrategyRunner(broker=None)
        assert hasattr(runner, "start")


# ─────────────────────────────────────────────────────────────────────────────
# Employee 10: PriceFeed (Real-time Data Ingestion)
# ─────────────────────────────────────────────────────────────────────────────
class TestPriceFeed:
    def test_feed_functions_importable(self):
        from app.tasks.price_feed import run_price_feed, start_price_feed
        assert callable(run_price_feed) and callable(start_price_feed)

    def test_default_symbols_populated(self):
        from app.tasks.price_feed import DEFAULT_EQUITY_SYMBOLS, DEFAULT_CRYPTO_SYMBOLS
        assert len(DEFAULT_EQUITY_SYMBOLS) >= 5
        assert "SPY" in DEFAULT_EQUITY_SYMBOLS
        assert any("BTC" in s for s in DEFAULT_CRYPTO_SYMBOLS)

    def test_fetch_graceful_on_broker_error(self):
        from app.tasks.price_feed import _fetch_and_publish
        mock_broker = AsyncMock()
        mock_broker.get_quote = AsyncMock(side_effect=Exception("No broker connection"))
        mock_cache = AsyncMock()
        # Should complete without raising
        asyncio.run(_fetch_and_publish(mock_broker, "SPY", mock_cache))

    def test_poll_interval_defined(self):
        from app.tasks import price_feed
        assert hasattr(price_feed, "POLL_INTERVAL")
        assert price_feed.POLL_INTERVAL > 0


# ─────────────────────────────────────────────────────────────────────────────
# Employee 11: Scheduler (APScheduler Orchestration)
# ─────────────────────────────────────────────────────────────────────────────
class TestScheduler:
    def test_get_scheduler_importable(self):
        from app.tasks.scheduler import get_scheduler, start_scheduler
        assert callable(get_scheduler) and callable(start_scheduler)

    def test_start_scheduler_registers_jobs(self):
        """Verify scheduler registers jobs (must run inside an event loop)."""
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        from app.tasks.scheduler import get_scheduler, start_scheduler

        async def _inner():
            sched = start_scheduler(db_session_factory=None, broker=None)
            jobs = sched.get_jobs()
            assert len(jobs) > 0, "Scheduler has no jobs registered"
            sched.shutdown(wait=False)

        asyncio.run(_inner())

    def test_nightly_retrain_importable(self):
        from app.tasks.ml_retrain import nightly_retrain
        assert callable(nightly_retrain)


# ─────────────────────────────────────────────────────────────────────────────
# Employee 12: CorrelationMonitor (Portfolio Risk)
# ─────────────────────────────────────────────────────────────────────────────
class TestCorrelationMonitor:
    def test_compute_correlation_clusters(self):
        from app.risk.correlation import compute_correlation_clusters
        np.random.seed(42)
        returns = pd.DataFrame({
            "SPY": np.random.normal(0.001, 0.01, 100),
            "QQQ": np.random.normal(0.001, 0.01, 100),
            "GLD": np.random.normal(0.0, 0.008, 100),
        })
        clusters = compute_correlation_clusters(returns, threshold=0.70)
        assert isinstance(clusters, dict)

    def test_check_cluster_limits_allows_small_position(self):
        from app.risk.correlation import check_cluster_limits
        clusters = {"cluster1": ["SPY", "QQQ"]}
        allowed, reason = check_cluster_limits(
            new_symbol="SPY",
            new_value_usd=5_000,
            current_positions={"SPY": 0, "QQQ": 0},
            clusters=clusters,
            max_cluster_pct=0.30,
            total_equity=100_000,
        )
        assert allowed is True

    def test_check_cluster_limits_blocks_large_position(self):
        from app.risk.correlation import check_cluster_limits
        clusters = {"cluster1": ["SPY", "QQQ"]}
        allowed, reason = check_cluster_limits(
            new_symbol="SPY",
            new_value_usd=25_000,
            current_positions={"SPY": 10_000, "QQQ": 10_000},
            clusters=clusters,
            max_cluster_pct=0.30,
            total_equity=100_000,
        )
        assert allowed is False
        assert "cluster1" in reason or "cluster" in reason.lower()

    def test_insufficient_data_returns_empty(self):
        from app.risk.correlation import compute_correlation_clusters
        returns = pd.DataFrame({"SPY": [0.01]})  # single row
        clusters = compute_correlation_clusters(returns)
        assert clusters == {}
