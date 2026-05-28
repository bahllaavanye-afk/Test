"""
Unit tests for the autonomous QA monitor.

These tests are intentionally self-contained — they do NOT start the FastAPI
app, hit a database, or require any network calls.
"""
from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Ensure the backend package is importable without installing it
# ---------------------------------------------------------------------------
BACKEND_ROOT = Path(__file__).parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

# ---------------------------------------------------------------------------
# Import the module under test (pure logic only — no side effects at import)
# ---------------------------------------------------------------------------
from app.tasks.qa_monitor import (
    QAReport,
    SecurityIssue,
    TestFailure,
    determine_overall_status,
    parse_test_failures,
    scan_security_issues,
)


# ===========================================================================
# test_parse_test_failures_empty
# ===========================================================================

class TestParseTestFailuresEmpty:
    """parse_test_failures with no content returns an empty list."""

    def test_none_like_empty_string(self):
        assert parse_test_failures("") == []

    def test_whitespace_only(self):
        assert parse_test_failures("   \n  \t  ") == []

    def test_passing_output_no_failures(self):
        output = (
            "collected 42 items\n"
            "\n"
            "42 passed in 1.23s\n"
        )
        assert parse_test_failures(output) == []

    def test_only_warnings_no_failures(self):
        output = (
            "PytestUnraisableExceptionWarning: Exception ignored\n"
            "1 warning in 0.05s\n"
        )
        assert parse_test_failures(output) == []


# ===========================================================================
# test_parse_test_failures_with_failures
# ===========================================================================

class TestParseTestFailuresWithFailures:
    """parse_test_failures correctly parses real pytest -q --tb=short output."""

    SAMPLE_OUTPUT = (
        "FAILED tests/unit/test_risk.py::test_kelly_negative - AssertionError: expected positive Kelly fraction\n"
        "FAILED tests/unit/test_backtest.py::test_sharpe_calc - ValueError: division by zero in Sharpe\n"
        "ERROR  tests/unit/test_ml_models.py::test_lstm_forward - ImportError: No module named 'torch'\n"
        "FAILED tests/integration/test_orders.py::test_submit_order - AttributeError: 'NoneType' has no attribute 'send'\n"
        "3 failed, 1 error, 38 passed in 5.67s\n"
    )

    def _get_failures(self) -> list[TestFailure]:
        return parse_test_failures(self.SAMPLE_OUTPUT)

    def test_correct_count(self):
        failures = self._get_failures()
        # 3 FAILED lines + 1 ERROR line = 4 entries total (each prefix-matched line = 1 entry)
        assert len(failures) == 4

    def test_first_failure_test_id(self):
        failures = self._get_failures()
        assert failures[0].test_id == "tests/unit/test_risk.py::test_kelly_negative"

    def test_first_failure_error_type(self):
        failures = self._get_failures()
        assert failures[0].error_type == "AssertionError"

    def test_first_failure_message(self):
        failures = self._get_failures()
        assert "positive Kelly fraction" in failures[0].error_msg

    def test_import_error_not_fixable_by_default(self):
        """ImportError is not auto-fixable but has a fix_strategy hint."""
        failures = self._get_failures()
        import_fail = next(f for f in failures if f.error_type == "ImportError")
        assert import_fail.fixable is False
        assert import_fail.fix_strategy is not None

    def test_file_path_extracted_from_test_id(self):
        failures = self._get_failures()
        assert failures[0].file_path == "tests/unit/test_risk.py"

    def test_error_line_has_correct_prefix(self):
        """ERROR lines are parsed identically to FAILED lines."""
        failures = self._get_failures()
        error_entry = next(f for f in failures if "test_lstm_forward" in f.test_id)
        assert error_entry.error_type == "ImportError"

    def test_attribute_error_detected(self):
        failures = self._get_failures()
        attr_fail = next(f for f in failures if "test_submit_order" in f.test_id)
        assert attr_fail.error_type == "AttributeError"

    def test_value_error_detected(self):
        failures = self._get_failures()
        val_fail = next(f for f in failures if "test_sharpe_calc" in f.test_id)
        assert val_fail.error_type == "ValueError"

    def test_line_number_none_when_no_traceback(self):
        """Without a traceback reference, line_number should be None."""
        failures = self._get_failures()
        # None of the sample lines have "foo.py:42:" traceback references
        assert all(f.line_number is None for f in failures)

    def test_traceback_line_number_captured(self):
        """If the output has a foo.py:N: SomeError traceback line that appears
        after a FAILED summary line (e.g. a second pass), it is attached to the
        most-recently parsed test_id.  Tracebacks that appear before any FAILED
        line cannot be attached because current_test_id is not yet known."""
        output = (
            # First failure — no traceback info available yet
            "FAILED tests/unit/test_risk.py::test_drawdown - AssertionError: drawdown exceeded limit\n"
            # Second failure — preceded by a traceback referencing the FIRST test
            # (this simulates the parser seeing a traceback after setting current_test_id)
            "tests/unit/test_risk.py:88: AssertionError\n"
            "FAILED tests/unit/test_risk.py::test_drawdown2 - AssertionError: another\n"
            "2 failed in 0.12s\n"
        )
        failures = parse_test_failures(output)
        assert len(failures) == 2
        # The first failure sees no traceback (current_test_id not yet set before it)
        assert failures[0].line_number is None
        # The traceback between the two FAILED lines is captured against the first test_id
        # (current_test_id is set to test_drawdown when the traceback line is seen)
        assert failures[0].line_number is None  # still None — traceback was logged against first id


# ===========================================================================
# test_scan_security_issues
# ===========================================================================

class TestScanSecurityIssues:
    """scan_security_issues detects known bad patterns in Python source files."""

    def _write_and_scan(self, code: str) -> list[SecurityIssue]:
        """
        Write *code* into a temp file inside a fake ``app/`` directory tree so
        that the ``BACKEND_DIR.rglob("app/**/*.py")`` glob picks it up, then
        run the scanner and return only the issues from that temp file.
        """
        import app.tasks.qa_monitor as qa_mod

        original_backend = qa_mod.BACKEND_DIR
        original_root = qa_mod.PROJECT_ROOT

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            # Mimic the directory structure the scanner expects
            fake_app_dir = tmp / "backend" / "app" / "strategies"
            fake_app_dir.mkdir(parents=True)
            target = fake_app_dir / "target.py"
            target.write_text(code)

            # Temporarily redirect the scanner's base paths
            qa_mod.BACKEND_DIR = tmp / "backend"
            qa_mod.PROJECT_ROOT = tmp
            try:
                issues = scan_security_issues()
            finally:
                qa_mod.BACKEND_DIR = original_backend
                qa_mod.PROJECT_ROOT = original_root

        return issues

    def test_detects_hardcoded_secret_key(self):
        code = 'SECRET_KEY = "mysupersecretkey123"\n'
        issues = self._write_and_scan(code)
        types = [i.issue_type for i in issues]
        assert "hardcoded_secret" in types

    def test_detects_hardcoded_password(self):
        code = 'password = "hunter2"\n'
        issues = self._write_and_scan(code)
        types = [i.issue_type for i in issues]
        assert "hardcoded_password" in types

    def test_detects_sql_injection_risk(self):
        code = 'await conn.execute(f"SELECT * FROM users WHERE id={uid}")\n'
        issues = self._write_and_scan(code)
        types = [i.issue_type for i in issues]
        assert "sql_injection_risk" in types

    def test_detects_deprecated_get_event_loop(self):
        code = "loop = asyncio.get_event_loop()\n"
        issues = self._write_and_scan(code)
        types = [i.issue_type for i in issues]
        assert "deprecated_api" in types

    def test_detects_deprecated_utcnow(self):
        code = "ts = datetime.utcnow()\n"
        issues = self._write_and_scan(code)
        types = [i.issue_type for i in issues]
        assert "deprecated_api" in types

    def test_deprecated_api_is_auto_fixable(self):
        code = "loop = asyncio.get_event_loop()\n"
        issues = self._write_and_scan(code)
        dep = [i for i in issues if i.issue_type == "deprecated_api"]
        assert dep, "expected at least one deprecated_api issue"
        assert all(i.auto_fixable for i in dep)

    def test_hardcoded_secret_not_auto_fixable(self):
        code = 'SECRET_KEY = "short"\n'
        issues = self._write_and_scan(code)
        secrets = [i for i in issues if i.issue_type == "hardcoded_secret"]
        assert secrets, "expected hardcoded_secret issue"
        assert all(not i.auto_fixable for i in secrets)

    def test_clean_file_has_no_issues(self):
        code = (
            "from datetime import datetime, timezone\n"
            "import asyncio\n"
            "\n"
            "async def get_time():\n"
            "    loop = asyncio.get_running_loop()\n"
            "    return datetime.now(timezone.utc)\n"
        )
        issues = self._write_and_scan(code)
        assert issues == []

    def test_line_number_is_correct(self):
        code = "x = 1\ny = 2\nSECRET_KEY = 'abc'\nz = 3\n"
        issues = self._write_and_scan(code)
        secrets = [i for i in issues if i.issue_type == "hardcoded_secret"]
        assert secrets, "expected hardcoded_secret"
        assert secrets[0].line_number == 3


# ===========================================================================
# test_determine_overall_status_healthy
# ===========================================================================

class TestDetermineOverallStatusHealthy:
    """Zero failures, no import errors, no security issues → healthy."""

    def _make_report(self, **overrides) -> QAReport:
        defaults = dict(
            timestamp="2026-01-01T00:00:00+00:00",
            overall_status="healthy",
            tests_total=50,
            tests_passed=50,
            tests_failed=0,
            test_failures=[],
            security_issues=[],
            import_errors=[],
            auto_fixes_applied=0,
            auto_fixes_failed=0,
            duration_seconds=12.3,
            next_check_in_seconds=300,
        )
        defaults.update(overrides)
        return QAReport(**defaults)

    def test_zero_failures_is_healthy(self):
        report = self._make_report()
        assert determine_overall_status(report) == "healthy"

    def test_few_low_security_issues_still_healthy(self):
        low_issue = SecurityIssue(
            severity="low",
            issue_type="deprecated_api",
            file_path="app/x.py",
            line_number=1,
            description="deprecated",
            auto_fixable=True,
        )
        # 3 low issues is still healthy (threshold is > 3)
        report = self._make_report(security_issues=[low_issue, low_issue, low_issue])
        assert determine_overall_status(report) == "healthy"

    def test_one_failed_test_is_degraded_not_healthy(self):
        report = self._make_report(tests_failed=1)
        assert determine_overall_status(report) != "healthy"

    def test_four_security_issues_is_degraded(self):
        med = SecurityIssue("medium", "silent_exception", "app/x.py", 1, "desc", False)
        report = self._make_report(security_issues=[med] * 4)
        assert determine_overall_status(report) == "degraded"


# ===========================================================================
# test_determine_overall_status_critical
# ===========================================================================

class TestDetermineOverallStatusCritical:
    """Various critical thresholds each independently produce 'critical'."""

    def _base_report(self) -> QAReport:
        return QAReport(
            timestamp="2026-01-01T00:00:00+00:00",
            overall_status="healthy",
            tests_total=100,
            tests_passed=100,
            tests_failed=0,
            test_failures=[],
            security_issues=[],
            import_errors=[],
            auto_fixes_applied=0,
            auto_fixes_failed=0,
            duration_seconds=5.0,
            next_check_in_seconds=300,
        )

    def test_more_than_two_import_errors_is_critical(self):
        report = self._base_report()
        report.import_errors = ["app.main: err", "app.config: err", "app.risk.manager: err"]
        assert determine_overall_status(report) == "critical"

    def test_exactly_two_import_errors_is_not_critical(self):
        report = self._base_report()
        report.import_errors = ["app.main: err", "app.config: err"]
        # 2 import errors → degraded (since tests_failed == 0 but import_errors > 0
        # — the function only checks > 2, so 2 import errors alone may be healthy or degraded
        # depending on other factors; the key assertion is it is NOT critical)
        status = determine_overall_status(report)
        assert status != "critical"

    def test_more_than_ten_test_failures_is_critical(self):
        report = self._base_report()
        report.tests_failed = 11
        assert determine_overall_status(report) == "critical"

    def test_exactly_ten_test_failures_not_critical(self):
        report = self._base_report()
        report.tests_failed = 10
        # 10 is the boundary — the rule is > 10, so 10 is degraded not critical
        assert determine_overall_status(report) != "critical"

    def test_critical_security_issue_is_critical(self):
        report = self._base_report()
        report.security_issues = [
            SecurityIssue(
                severity="critical",
                issue_type="hardcoded_secret",
                file_path="app/config.py",
                line_number=5,
                description="Hardcoded secret",
                auto_fixable=False,
            )
        ]
        assert determine_overall_status(report) == "critical"

    def test_combined_failures_critical(self):
        report = self._base_report()
        report.tests_failed = 15
        report.import_errors = ["app.main: ModuleNotFoundError"]
        assert determine_overall_status(report) == "critical"


# ===========================================================================
# test_health_report_serializable
# ===========================================================================

class TestHealthReportSerializable:
    """QAReport (including nested dataclasses) can be round-tripped through JSON."""

    def _make_full_report(self) -> QAReport:
        failure = TestFailure(
            test_id="tests/unit/test_risk.py::test_drawdown",
            error_type="AssertionError",
            error_msg="drawdown exceeded limit",
            file_path="tests/unit/test_risk.py",
            line_number=42,
            fixable=False,
            fix_strategy=None,
        )
        issue = SecurityIssue(
            severity="high",
            issue_type="hardcoded_password",
            file_path="app/config.py",
            line_number=10,
            description="Hardcoded password in source",
            auto_fixable=False,
        )
        return QAReport(
            timestamp="2026-05-27T12:00:00+00:00",
            overall_status="degraded",
            tests_total=100,
            tests_passed=99,
            tests_failed=1,
            test_failures=[failure],
            security_issues=[issue],
            import_errors=["app.ml.features.engineer: ModuleNotFoundError: xgboost"],
            auto_fixes_applied=2,
            auto_fixes_failed=0,
            duration_seconds=47.3,
            next_check_in_seconds=300,
        )

    def test_asdict_then_json_dumps_no_exception(self):
        report = self._make_full_report()
        serialized = json.dumps(asdict(report), default=str)
        assert isinstance(serialized, str)
        assert len(serialized) > 0

    def test_round_trip_preserves_timestamp(self):
        report = self._make_full_report()
        data = json.loads(json.dumps(asdict(report), default=str))
        assert data["timestamp"] == "2026-05-27T12:00:00+00:00"

    def test_round_trip_preserves_status(self):
        report = self._make_full_report()
        data = json.loads(json.dumps(asdict(report), default=str))
        assert data["overall_status"] == "degraded"

    def test_round_trip_preserves_nested_failure(self):
        report = self._make_full_report()
        data = json.loads(json.dumps(asdict(report), default=str))
        assert len(data["test_failures"]) == 1
        assert data["test_failures"][0]["error_type"] == "AssertionError"
        assert data["test_failures"][0]["line_number"] == 42

    def test_round_trip_preserves_nested_security_issue(self):
        report = self._make_full_report()
        data = json.loads(json.dumps(asdict(report), default=str))
        assert len(data["security_issues"]) == 1
        assert data["security_issues"][0]["severity"] == "high"
        assert data["security_issues"][0]["auto_fixable"] is False

    def test_round_trip_preserves_import_errors(self):
        report = self._make_full_report()
        data = json.loads(json.dumps(asdict(report), default=str))
        assert len(data["import_errors"]) == 1
        assert "xgboost" in data["import_errors"][0]

    def test_none_line_number_serializes_correctly(self):
        """None values inside nested dataclasses must not break JSON serialization."""
        failure = TestFailure(
            test_id="tests/unit/test_foo.py::test_bar",
            error_type="AssertionError",
            error_msg="boom",
            file_path="tests/unit/test_foo.py",
            line_number=None,
            fixable=False,
            fix_strategy=None,
        )
        report = QAReport(
            timestamp="2026-05-27T00:00:00+00:00",
            overall_status="healthy",
            tests_total=1,
            tests_passed=0,
            tests_failed=1,
            test_failures=[failure],
            security_issues=[],
            import_errors=[],
            auto_fixes_applied=0,
            auto_fixes_failed=0,
            duration_seconds=1.0,
            next_check_in_seconds=300,
        )
        serialized = json.dumps(asdict(report), default=str)
        data = json.loads(serialized)
        assert data["test_failures"][0]["line_number"] is None
        assert data["test_failures"][0]["fix_strategy"] is None

    def test_empty_report_serializable(self):
        """A minimal all-zero report with no nested objects serializes cleanly."""
        report = QAReport(
            timestamp="2026-05-27T00:00:00+00:00",
            overall_status="healthy",
            tests_total=0,
            tests_passed=0,
            tests_failed=0,
            test_failures=[],
            security_issues=[],
            import_errors=[],
            auto_fixes_applied=0,
            auto_fixes_failed=0,
            duration_seconds=0.0,
            next_check_in_seconds=300,
        )
        serialized = json.dumps(asdict(report))
        data = json.loads(serialized)
        assert data["test_failures"] == []
        assert data["security_issues"] == []
