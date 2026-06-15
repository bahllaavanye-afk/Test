"""
Unit tests for the GitHub issue escalation bridge.

Pure logic only — no network. We test that QA findings map to the right
escalations, fingerprints are stable, and disabled escalators no-op cleanly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.tasks.qa_monitor import QAReport, SecurityIssue, FailureRecord
from app.tasks.issue_escalation import IssueEscalator, _fingerprint, _role_for_path


def _empty_report(**overrides) -> QAReport:
    base = dict(
        timestamp="2026-06-13T00:00:00+00:00",
        overall_status="degraded",
        tests_total=10,
        tests_passed=9,
        tests_failed=1,
        test_failures=[],
        security_issues=[],
        import_errors=[],
        auto_fixes_applied=0,
        auto_fixes_failed=0,
        duration_seconds=1.0,
        next_check_in_seconds=300,
    )
    base.update(overrides)
    return QAReport(**base)


class TestRoleRouting:
    def test_strategy_path(self):
        assert _role_for_path("app/strategies/manual/momentum.py") == "strategy"

    def test_ml_path(self):
        assert _role_for_path("app/ml/models/lstm.py") == "ml"

    def test_risk_path(self):
        assert _role_for_path("app/risk/manager.py") == "risk"

    def test_unknown_defaults_backend(self):
        assert _role_for_path("some/random/path.py") == "backend"


class TestFingerprintStability:
    def test_same_input_same_fingerprint(self):
        assert _fingerprint("test_failure", "t::a", "X") == _fingerprint("test_failure", "t::a", "X")

    def test_different_input_different_fingerprint(self):
        assert _fingerprint("test_failure", "t::a") != _fingerprint("test_failure", "t::b")


class TestBuildEscalations:
    def test_import_error_is_p0(self):
        report = _empty_report(import_errors=["app.main: boom"])
        escs = IssueEscalator.build_escalations(report)
        assert len(escs) == 1
        assert escs[0]["priority"] == "P0"
        assert "Import error" in escs[0]["title"]

    def test_critical_security_is_p0(self):
        issue = SecurityIssue(
            severity="critical", issue_type="hardcoded_secret",
            file_path="app/config.py", line_number=5,
            description="secret", auto_fixable=False,
        )
        report = _empty_report(security_issues=[issue])
        escs = IssueEscalator.build_escalations(report)
        assert len(escs) == 1
        assert escs[0]["priority"] == "P0"
        assert escs[0]["role"] == "backend"

    def test_auto_fixable_security_not_escalated(self):
        issue = SecurityIssue(
            severity="low", issue_type="deprecated_api",
            file_path="app/x.py", line_number=1,
            description="deprecated", auto_fixable=True,
        )
        report = _empty_report(security_issues=[issue])
        assert IssueEscalator.build_escalations(report) == []

    def test_fixable_test_failure_not_escalated(self):
        fail = FailureRecord(
            test_id="tests/unit/test_x.py::test_y",
            error_type="DeprecationWarning", error_msg="old api",
            file_path="tests/unit/test_x.py", line_number=10,
            fixable=True, fix_strategy="upgrade",
        )
        report = _empty_report(test_failures=[fail])
        assert IssueEscalator.build_escalations(report) == []

    def test_unfixable_test_failure_is_p1(self):
        fail = FailureRecord(
            test_id="tests/unit/test_x.py::test_y",
            error_type="AssertionError", error_msg="expected 1 got 2",
            file_path="tests/unit/test_x.py", line_number=10,
            fixable=False, fix_strategy=None,
        )
        report = _empty_report(test_failures=[fail])
        escs = IssueEscalator.build_escalations(report)
        assert len(escs) == 1
        assert escs[0]["priority"] == "P1"
        assert "fingerprint" in escs[0]


class TestDisabledEscalator:
    @pytest.mark.asyncio
    async def test_disabled_escalate_noops(self):
        esc = IssueEscalator(github_token="", github_repo="")
        assert esc.enabled is False
        report = _empty_report(import_errors=["app.main: boom"])
        summary = await esc.escalate(report)
        assert summary["opened"] == 0
        assert "escalation_disabled" in summary["errors"]
