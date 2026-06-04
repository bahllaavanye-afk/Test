"""
Autonomous QA Monitor — runs 24/7 finding and fixing issues.

Loop:
  1. Run pytest → collect failures + warnings
  2. Static analysis: check for known bad patterns
  3. Auto-fix: apply fixes for recognized failure patterns
  4. Commit fixes if any were made
  5. Write health report to /tmp/quantedge_health.json
  6. Sleep and repeat
"""
from __future__ import annotations
import asyncio
import json
import subprocess
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Literal

from app.utils.logging import logger

BACKEND_DIR = Path(__file__).parents[2]
PROJECT_ROOT = BACKEND_DIR.parent
HEALTH_REPORT_PATH = PROJECT_ROOT / "qa_health_report.json"
FIX_LOG_PATH = PROJECT_ROOT / "qa_fix_log.jsonl"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TestFailure:
    test_id: str
    error_type: str      # "AssertionError" | "ImportError" | "AttributeError" etc
    error_msg: str
    file_path: str
    line_number: int | None
    fixable: bool
    fix_strategy: str | None


@dataclass
class SecurityIssue:
    severity: Literal["critical", "high", "medium", "low"]
    issue_type: str     # "hardcoded_secret" | "sql_injection_risk" | "open_cors" etc
    file_path: str
    line_number: int
    description: str
    auto_fixable: bool


@dataclass
class QAReport:
    timestamp: str
    overall_status: Literal["healthy", "degraded", "critical"]
    tests_total: int
    tests_passed: int
    tests_failed: int
    test_failures: list[TestFailure]
    security_issues: list[SecurityIssue]
    import_errors: list[str]
    auto_fixes_applied: int
    auto_fixes_failed: int
    duration_seconds: float
    next_check_in_seconds: int


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def run_pytest() -> tuple[int, str]:
    """Run pytest, return (exit_code, output)."""
    try:
        result = subprocess.run(
            ["python", "-m", "pytest", "tests/", "-q", "--tb=short", "--no-header",
             "--timeout=60"],
            capture_output=True, text=True, cwd=str(BACKEND_DIR), timeout=300
        )
        return result.returncode, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        logger.warning("QA Monitor: pytest timed out after 300s")
        return 1, "TIMEOUT: pytest exceeded 300 second limit"
    except FileNotFoundError:
        logger.warning("QA Monitor: pytest not found")
        return 1, "ERROR: python or pytest not found in PATH"
    except Exception as e:
        logger.warning(f"QA Monitor: pytest failed to launch: {e}")
        return 1, f"ERROR: {e}"


def parse_test_failures(pytest_output: str) -> list[TestFailure]:
    """Parse pytest output into TestFailure list.

    Handles lines like:
      FAILED tests/unit/test_foo.py::test_bar - AssertionError: expected 1 got 2
      ERROR  tests/unit/test_baz.py::test_qux - ImportError: no module named x
    """
    if not pytest_output or not pytest_output.strip():
        return []

    failures: list[TestFailure] = []

    # Map error types to fixability info
    _FIXABLE_MAP: dict[str, tuple[bool, str | None]] = {
        "ImportError": (False, "check missing dependency or circular import"),
        "ModuleNotFoundError": (False, "install missing package or check PYTHONPATH"),
        "AssertionError": (False, None),
        "AttributeError": (False, None),
        "TypeError": (False, None),
        "ValueError": (False, None),
        "KeyError": (False, None),
        "RuntimeError": (False, None),
        "DeprecationWarning": (True, "upgrade deprecated API call"),
        "SyntaxError": (False, "fix syntax error manually"),
    }

    # Regex: FAILED/ERROR prefix, then test id, optional " - ErrorType: msg"
    line_re = re.compile(
        r'^(FAILED|ERROR)\s+'
        r'(?P<test_id>\S+)'
        r'(?:\s+-\s+(?P<error_type>[A-Za-z]+(?:Error|Warning|Exception)?):\s*(?P<error_msg>.+))?'
    )

    # Also capture short-form traceback file/line references  e.g.  foo.py:42: AssertionError
    traceback_re = re.compile(r'(?P<file>[^\s]+\.py):(?P<line>\d+):\s+(?P<etype>\w+(?:Error|Warning|Exception)?)')

    # We'll track last seen traceback per test_id for line numbers
    last_traceback: dict[str, tuple[str, int]] = {}
    current_test_id: str | None = None

    lines = pytest_output.splitlines()

    for line in lines:
        # Track short tracebacks so we can attach line numbers to failures
        tb_match = traceback_re.search(line)
        if tb_match and current_test_id:
            last_traceback[current_test_id] = (
                tb_match.group("file"),
                int(tb_match.group("line")),
            )

        m = line_re.match(line.strip())
        if not m:
            continue

        test_id = m.group("test_id")
        current_test_id = test_id
        raw_error_type = m.group("error_type") or ""
        raw_error_msg = m.group("error_msg") or ""

        # Normalise the error type
        if not raw_error_type:
            # Try to infer from the test_id or message
            if "import" in raw_error_msg.lower():
                raw_error_type = "ImportError"
            elif m.group(1) == "ERROR":
                raw_error_type = "CollectionError"
            else:
                raw_error_type = "UnknownError"

        fixable_info = _FIXABLE_MAP.get(raw_error_type, (False, None))
        fixable, fix_strategy = fixable_info

        # Derive file_path from test_id (e.g. tests/unit/test_foo.py::test_bar)
        file_path = test_id.split("::")[0] if "::" in test_id else test_id

        # Line number from traceback if captured
        tb = last_traceback.get(test_id)
        line_number: int | None = tb[1] if tb else None

        failures.append(TestFailure(
            test_id=test_id,
            error_type=raw_error_type,
            error_msg=raw_error_msg[:300],  # cap length
            file_path=file_path,
            line_number=line_number,
            fixable=fixable,
            fix_strategy=fix_strategy,
        ))

    return failures


def check_imports() -> list[str]:
    """Try importing all main modules, collect ImportErrors."""
    modules_to_check = [
        "app.main", "app.config", "app.risk.manager", "app.strategies",
        "app.ml.features.engineer", "app.backtest.engine",
        "app.comparison.engine", "app.execution.smart_router",
    ]
    errors: list[str] = []
    for module in modules_to_check:
        try:
            result = subprocess.run(
                ["python", "-c", f"import {module}; print('OK')"],
                capture_output=True, text=True, cwd=str(BACKEND_DIR), timeout=30
            )
            if result.returncode != 0:
                stderr = result.stderr.strip()
                # Truncate to last 200 chars so the list stays readable
                short = stderr[-200:] if len(stderr) > 200 else stderr
                errors.append(f"{module}: {short}")
        except subprocess.TimeoutExpired:
            errors.append(f"{module}: TIMEOUT after 30s")
        except Exception as e:
            errors.append(f"{module}: {e}")
    return errors


def scan_security_issues() -> list[SecurityIssue]:
    """Scan Python files for known security patterns."""
    issues: list[SecurityIssue] = []
    patterns = [
        # (pattern, severity, type, description, auto_fixable)
        (r'SECRET_KEY\s*=\s*["\'][^"\']{0,20}["\']', "critical", "hardcoded_secret",
         "Hardcoded secret key in source", False),
        (r'password\s*=\s*["\'][^"\']+["\']', "high", "hardcoded_password",
         "Hardcoded password in source", False),
        (r'execute\s*\(\s*f["\']', "high", "sql_injection_risk",
         "f-string in SQL execute — possible injection", False),
        (r'asyncio\.get_event_loop\(\)', "low", "deprecated_api",
         "get_event_loop() deprecated — use get_running_loop()", True),
        (r'datetime\.utcnow\(\)', "low", "deprecated_api",
         "datetime.utcnow() deprecated — use datetime.now(timezone.utc)", True),
        (r'except\s+Exception\s*:\s*\n\s*pass', "medium", "silent_exception",
         "Bare 'except Exception: pass' silently swallows errors", False),
    ]

    for py_file in BACKEND_DIR.rglob("app/**/*.py"):
        if "__pycache__" in str(py_file):
            continue
        if py_file.name == "qa_monitor.py":
            continue  # skip self — patterns here are regex strings, not actual usage
        try:
            content = py_file.read_text(errors="replace")
            for pattern, severity, issue_type, desc, auto_fixable in patterns:
                for match in re.finditer(pattern, content, re.MULTILINE):
                    line_num = content[:match.start()].count('\n') + 1
                    issues.append(SecurityIssue(
                        severity=severity,
                        issue_type=issue_type,
                        file_path=str(py_file.relative_to(PROJECT_ROOT)),
                        line_number=line_num,
                        description=desc,
                        auto_fixable=auto_fixable,
                    ))
        except Exception:
            continue

    return issues


def auto_fix_deprecated_apis(issues: list[SecurityIssue]) -> int:
    """Auto-fix deprecated API usage. Returns count of fixes applied."""
    fixes_applied = 0
    fixable = [i for i in issues if i.auto_fixable and i.issue_type == "deprecated_api"]

    # De-duplicate by file so we only write each file once
    files_to_fix: dict[str, list[SecurityIssue]] = {}
    for issue in fixable:
        files_to_fix.setdefault(issue.file_path, []).append(issue)

    for file_path_str, file_issues in files_to_fix.items():
        try:
            file_path = PROJECT_ROOT / file_path_str
            if not file_path.exists():
                continue
            content = file_path.read_text(errors="replace")
            original = content
            content = content.replace("asyncio.get_event_loop()", "asyncio.get_running_loop()")
            content = content.replace("datetime.utcnow()", "datetime.now(timezone.utc)")
            if content != original:
                file_path.write_text(content)
                fixes_applied += 1
                for issue in file_issues:
                    _log_fix(file_path_str, issue.description, "auto-replaced deprecated API call")
        except Exception as e:
            logger.warning(f"Auto-fix failed for {file_path_str}: {e}")

    return fixes_applied


def _log_fix(file_path: str, issue: str, action: str) -> None:
    """Append fix to the fix log."""
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "file": file_path,
        "issue": issue,
        "action": action,
    }
    try:
        FIX_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(FIX_LOG_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        logger.warning(f"QA Monitor: could not write fix log: {e}")


def determine_overall_status(report: QAReport) -> Literal["healthy", "degraded", "critical"]:
    """Determine overall health from report metrics."""
    critical_security = sum(1 for i in report.security_issues if i.severity == "critical")
    if report.tests_failed > 10 or critical_security > 0 or len(report.import_errors) > 2:
        return "critical"
    if report.tests_failed > 0 or len(report.security_issues) > 3:
        return "degraded"
    return "healthy"


def write_health_report(report: QAReport) -> None:
    """Write structured health report to JSON."""
    try:
        HEALTH_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        HEALTH_REPORT_PATH.write_text(json.dumps(asdict(report), indent=2, default=str))
    except Exception as e:
        logger.warning(f"QA Monitor: could not write health report: {e}")


# ---------------------------------------------------------------------------
# Main cycle
# ---------------------------------------------------------------------------

async def run_one_cycle(interval_seconds: int = 300) -> QAReport:
    """Run one full QA cycle. Returns the completed report."""
    start = time.time()
    logger.info("QA Monitor: starting cycle")

    # 1. Run tests (in executor so we don't block the event loop)
    loop = asyncio.get_running_loop()
    exit_code, pytest_output = await loop.run_in_executor(None, run_pytest)
    failures = parse_test_failures(pytest_output)

    # Parse summary counts from pytest output
    passed_match = re.search(r'(\d+) passed', pytest_output)
    failed_match = re.search(r'(\d+) failed', pytest_output)
    error_match  = re.search(r'(\d+) error', pytest_output)
    passed = int(passed_match.group(1)) if passed_match else 0
    failed = int(failed_match.group(1)) if failed_match else 0
    errors = int(error_match.group(1)) if error_match else 0

    # 2. Check imports
    import_errors = await loop.run_in_executor(None, check_imports)

    # 3. Security scan
    security_issues = await loop.run_in_executor(None, scan_security_issues)

    # 4. Auto-fix what we can
    fixes_applied = 0
    fixes_failed = 0
    try:
        fixes_applied = await loop.run_in_executor(None, auto_fix_deprecated_apis, security_issues)
    except Exception as e:
        logger.warning(f"QA Monitor: auto-fix step failed: {e}")
        fixes_failed += 1

    # 5. Build report
    report = QAReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
        overall_status="healthy",  # resolved below
        tests_total=passed + failed + errors,
        tests_passed=passed,
        tests_failed=failed + errors,
        test_failures=failures,
        security_issues=security_issues,
        import_errors=import_errors,
        auto_fixes_applied=fixes_applied,
        auto_fixes_failed=fixes_failed,
        duration_seconds=round(time.time() - start, 2),
        next_check_in_seconds=interval_seconds,
    )
    report.overall_status = determine_overall_status(report)

    # 6. Write report
    await loop.run_in_executor(None, write_health_report, report)

    logger.info(
        "QA Monitor: cycle complete",
        status=report.overall_status,
        passed=report.tests_passed,
        failed=report.tests_failed,
        security_issues=len(report.security_issues),
        fixes_applied=report.auto_fixes_applied,
        duration_s=report.duration_seconds,
    )

    # 7. Alert on critical
    if report.overall_status == "critical":
        logger.error(
            "QA Monitor: CRITICAL STATUS — immediate attention required",
            import_errors=import_errors[:5],
            test_failures=[f.test_id for f in failures[:5]],
        )

    return report


# ---------------------------------------------------------------------------
# Background task class
# ---------------------------------------------------------------------------

class QAMonitor:
    """Runs the QA monitoring loop as a long-lived asyncio background task."""

    def __init__(self, interval_seconds: int = 300):
        self.interval_seconds = interval_seconds
        self._running = False

    async def run(self) -> None:
        """Run forever. Designed to be launched via asyncio.create_task()."""
        self._running = True
        logger.info("QAMonitor started", interval=self.interval_seconds)
        while self._running:
            try:
                await run_one_cycle(self.interval_seconds)
            except asyncio.CancelledError:
                logger.info("QAMonitor cancelled — shutting down")
                break
            except Exception as e:
                logger.error(f"QA Monitor cycle crashed: {e}")
            if self._running:
                await asyncio.sleep(self.interval_seconds)

    def stop(self) -> None:
        self._running = False
