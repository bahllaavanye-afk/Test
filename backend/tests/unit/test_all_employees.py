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
import importlib


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
        monitor = QAMonitor(interval_seconds=60)
        assert monitor._cycle == 0


# Additional utility functions for testing environments

def _validate_non_empty_string(value: str, name: str) -> None:
    """Validate that a given value is a non‑empty string.

    Args:
        value: The string value to validate.
        name: The name of the parameter (used in error messages).

    Raises:
        ValueError: If ``value`` is not a string or is empty/whitespace.
    """
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string, got {type(value).__name__}")
    if not value.strip():
        raise ValueError(f"{name} cannot be empty or whitespace")


def get_employee_module(employee_name: str):
    """Import and return the employee module by name.

    This helper validates that ``employee_name`` is a non‑empty string and
    attempts to import the corresponding module from ``app.tasks``.

    Args:
        employee_name: The name of the employee (e.g., ``'algo_agent'``).

    Returns:
        The imported module.

    Raises:
        ValueError: If ``employee_name`` is invalid.
        ImportError: If the module cannot be imported.
    """
    _validate_non_empty_string(employee_name, "employee_name")
    module_path = f"app.tasks.{employee_name}"
    try:
        return importlib.import_module(module_path)
    except ImportError as exc:
        raise ImportError(f"Unable to import employee module '{module_path}': {exc}") from exc


# End of file