"""
Unit tests for the two autonomous "employee" background tasks:
  - ResearchScientist
  - ModelingEngineer
"""
from __future__ import annotations

import asyncio
import json
import tempfile
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# ResearchScientist tests
# ---------------------------------------------------------------------------

class TestResearchScientist:
    """Tests for ResearchScientist."""

    def _make_scientist(self, tmp_log: Path | None = None):
        """Helper: create a ResearchScientist with an optional patched log path."""
        from app.tasks.research_scientist import ResearchScientist, RESEARCH_AGENDA  # noqa: F401
        rs = ResearchScientist(interval_seconds=1)
        return rs

    def test_research_scientist_cycle(self):
        """run one cycle, check findings list is populated."""
        from app.tasks.research_scientist import ResearchScientist

        rs = ResearchScientist(interval_seconds=1)
        assert rs._cycle == 0
        assert rs._findings == []

        findings = asyncio.run(rs.research_cycle())

        assert len(findings) > 0, "research_cycle should return at least one finding"
        assert rs._cycle == 1
        # All findings should be ResearchFinding instances
        from app.tasks.research_scientist import ResearchFinding
        for f in findings:
            assert isinstance(f, ResearchFinding)

    def test_research_scientist_evaluate_topic(self):
        """Evaluate a specific topic and check all expected fields are present."""
        from app.tasks.research_scientist import ResearchScientist, RESEARCH_AGENDA, ResearchFinding

        rs = ResearchScientist(interval_seconds=1)
        topic = RESEARCH_AGENDA[0]  # yield_curve_momentum

        finding = asyncio.run(rs._evaluate_topic(topic))

        assert isinstance(finding, ResearchFinding)
        # All required fields must be non-empty / in range
        assert finding.topic == topic["topic"]
        assert finding.description == topic["description"]
        assert finding.estimated_sharpe == topic["expected_sharpe"]
        assert 0.0 <= finding.novelty_score <= 1.0
        assert finding.complexity in ("low", "medium", "high")
        assert finding.data_source == topic["data_source"]
        assert finding.ic_estimate >= 0.0
        assert len(finding.sample_signal) > 0
        assert finding.recommended_action in ("backtest", "implement", "monitor", "shelve")
        assert 0.0 <= finding.confidence <= 1.0
        assert finding.researched_at  # non-empty ISO timestamp

    def test_research_scientist_top_ideas(self):
        """get_top_ideas returns results sorted by estimated_sharpe * confidence desc."""
        from app.tasks.research_scientist import ResearchScientist

        rs = ResearchScientist(interval_seconds=1)

        # Run 3 cycles to accumulate findings
        for _ in range(3):
            findings = asyncio.run(rs.research_cycle())
            rs._findings.extend(findings)

        top = rs.get_top_ideas(n=5)

        assert len(top) > 0
        # Verify descending sort
        scores = [f.estimated_sharpe * f.confidence for f in top]
        assert scores == sorted(scores, reverse=True), "top ideas should be sorted desc by score"

    def test_research_scientist_log_finding(self, tmp_path):
        """Log a finding, check the file was written with valid JSON."""
        from app.tasks.research_scientist import ResearchScientist, ResearchFinding

        rs = ResearchScientist(interval_seconds=1)
        log_file = tmp_path / "research_log.jsonl"

        # Patch the RESEARCH_LOG constant used inside the module
        import app.tasks.research_scientist as rs_module
        original_log = rs_module.RESEARCH_LOG
        rs_module.RESEARCH_LOG = log_file

        try:
            finding = ResearchFinding(
                topic="test_topic",
                description="A test finding",
                estimated_sharpe=1.5,
                novelty_score=0.7,
                complexity="low",
                data_source="test_public",
                ic_estimate=0.075,
                sample_signal="Signal from test_public: IC=0.0750",
                recommended_action="backtest",
                confidence=0.85,
            )
            rs._log_finding(finding)

            assert log_file.exists(), "Research log file should be created"
            lines = log_file.read_text().strip().splitlines()
            assert len(lines) == 1, "Should write exactly one line"
            parsed = json.loads(lines[0])
            assert parsed["topic"] == "test_topic"
            assert parsed["estimated_sharpe"] == 1.5
            assert parsed["recommended_action"] == "backtest"
        finally:
            rs_module.RESEARCH_LOG = original_log


# ---------------------------------------------------------------------------
# ModelingEngineer tests
# ---------------------------------------------------------------------------

class TestModelingEngineer:
    """Tests for ModelingEngineer."""

    def test_modeling_engineer_init(self):
        """Verify correct defaults after construction."""
        from app.tasks.modeling_engineer import ModelingEngineer, MODEL_TYPES

        me = ModelingEngineer()

        assert me.interval_seconds == 1800
        assert me.drift_threshold == 0.52
        assert me.retrain_after_n_drift == 3
        assert me._cycle == 0
        assert me._decisions == []
        # Perf cache starts empty
        assert len(me._perf_cache) == 0
        # Best sharpe should contain all model types from INCUMBENT_SHARPE
        for model in MODEL_TYPES:
            assert model in me._best_sharpe

    def test_modeling_engineer_summary_empty(self):
        """get_engineering_summary with no cycles run returns a valid dict."""
        from app.tasks.modeling_engineer import ModelingEngineer, MODEL_TYPES

        me = ModelingEngineer()
        summary = me.get_engineering_summary()

        assert isinstance(summary, dict)
        assert summary["cycles_completed"] == 0
        assert summary["models_monitored"] == MODEL_TYPES
        assert summary["drift_threshold"] == 0.52
        assert isinstance(summary["latest_performance"], dict)
        assert isinstance(summary["recent_decisions"], list)
        assert summary["promote_count"] == 0
        assert summary["retrain_count"] == 0

    def test_modeling_engineer_performance_record(self):
        """ModelPerformanceRecord dataclass serializes correctly via asdict."""
        from app.tasks.modeling_engineer import ModelPerformanceRecord

        record = ModelPerformanceRecord(
            model_id="lstm",
            accuracy=0.57,
            sharpe=1.1,
            n_predictions=100,
            drift_detected=False,
        )
        d = asdict(record)

        assert d["model_id"] == "lstm"
        assert d["accuracy"] == 0.57
        assert d["sharpe"] == 1.1
        assert d["n_predictions"] == 100
        assert d["drift_detected"] is False
        assert "checked_at" in d
        assert d["checked_at"]  # non-empty

    def test_modeling_engineer_check_performance(self):
        """check_model_performance returns a valid record with values in range."""
        from app.tasks.modeling_engineer import ModelingEngineer, ModelPerformanceRecord

        me = ModelingEngineer()
        record = asyncio.run(me.check_model_performance("lstm"))

        assert isinstance(record, ModelPerformanceRecord)
        assert record.model_id == "lstm"
        assert 0.0 <= record.accuracy <= 1.0
        assert isinstance(record.sharpe, float)
        assert record.n_predictions == 100
        assert isinstance(record.drift_detected, bool)

    def test_modeling_engineer_detect_drift_below_threshold(self):
        """detect_drift returns True when accuracy is consistently below threshold."""
        from app.tasks.modeling_engineer import ModelingEngineer, ModelPerformanceRecord

        me = ModelingEngineer(drift_threshold=0.55)

        # Inject records all below threshold
        for i in range(3):
            me._perf_cache["lstm"].append(
                ModelPerformanceRecord(
                    model_id="lstm",
                    accuracy=0.48,
                    sharpe=-0.1,
                    drift_detected=True,
                )
            )

        drifted = asyncio.run(me.detect_drift("lstm"))
        assert drifted is True

    def test_modeling_engineer_detect_no_drift_above_threshold(self):
        """detect_drift returns False when accuracy is above threshold."""
        from app.tasks.modeling_engineer import ModelingEngineer, ModelPerformanceRecord

        me = ModelingEngineer(drift_threshold=0.52)

        me._perf_cache["xgboost"].append(
            ModelPerformanceRecord(
                model_id="xgboost",
                accuracy=0.60,
                sharpe=1.0,
                drift_detected=False,
            )
        )

        drifted = asyncio.run(me.detect_drift("xgboost"))
        assert drifted is False

    def test_modeling_decision_dataclass(self):
        """ModelingDecision dataclass has all expected fields."""
        from app.tasks.modeling_engineer import ModelingDecision

        decision = ModelingDecision(
            decision_type="promote",
            model_id="ensemble",
            reason="Sharpe improved by 0.25",
            confidence=0.9,
        )
        d = asdict(decision)

        assert d["decision_type"] == "promote"
        assert d["model_id"] == "ensemble"
        assert d["confidence"] == 0.9
        assert "decided_at" in d
