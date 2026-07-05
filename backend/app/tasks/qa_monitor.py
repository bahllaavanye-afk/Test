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

# ---------------------------------------------------------------------------
# Constants & Caches
# ---------------------------------------------------------------------------

BACKEND_DIR = Path(__file__).parents[2]
PROJECT_ROOT = BACKEND_DIR.parent
HEALTH_REPORT_PATH = PROJECT_ROOT / "qa_health_report.json"
FIX_LOG_PATH = PROJECT_ROOT / "qa_fix_log.jsonl"

# Cache for import checks to avoid repeated subprocess launches
_IMPORT_CACHE: dict[str, tuple[float, str]] = {}
_IMPORT_CACHE_TTL = 300  # seconds

# Pre‑compiled security regex patterns
_SECURITY_PATTERNS: list[tuple[re.Pattern, str, str, str, bool]] = [
    (
        re.compile(r'SECRET_KEY\s*=\s*["\'][^"\']{0,20}["\']'),
        "critical",
        "hardcoded_secret",
        "Hardcoded secret key in source",
        False,
    ),
    (
        re.compile(r'password\s*=\s*["\'][^"\']+["\']'),
        "high",
        "hardcoded_password",
        "Hardcoded password in source",
        False,
    ),
    (
        re.compile(r'execute\s*\(\s*f["\']'),
        "high",
        "sql_injection_risk",
        "f-string used in SQL execution (potential injection)",
        False,
    ),
    # Add more patterns as needed
]

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class FailureRecord:
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
    test_failures: list[FailureRecord]
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


def parse_test_failures(pytest_output: str) -> list[FailureRecord]:
    """Parse pytest output into FailureRecord list.

    Handles lines like:
      FAILED tests/unit/test_foo.py::test_bar - AssertionError: expected 1 got 2
      ERROR  tests/unit/test_baz.py::test_qux - ImportError: no module named x
    """
    if not pytest_output or not pytest_output.strip():
        return []

    failures: list[FailureRecord] = []

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

        failures.append(FailureRecord(
            test_id=test_id,
            error_type=raw_error_type,
            error_msg=raw_error_msg[:300],  # cap length
            file_path=file_path,
            line_number=line_number,
            fixable=fixable,
            fix_strategy=fix_strategy,
        ))

    return failures


def _cached_import_check(module: str) -> str | None:
    """Run a single import check with caching."""
    now = time.time()
    cached = _IMPORT_CACHE.get(module)
    if cached:
        ts, result = cached
        if now - ts < _IMPORT_CACHE_TTL:
            return result

    try:
        result = subprocess.run(
            ["python", "-c", f"import {module}; print('OK')"],
            capture_output=True, text=True, cwd=str(BACKEND_DIR), timeout=30
        )
        if result.returncode == 0:
            outcome = None
        else:
            stderr = result.stderr.strip()
            outcome = stderr[-200:] if len(stderr) > 200 else stderr
    except subprocess.TimeoutExpired:
        outcome = "TIMEOUT after 30s"
    except Exception as e:
        outcome = str(e)

    _IMPORT_CACHE[module] = (now, outcome)
    return outcome


def check_imports() -> list[str]:
    """Try importing all main modules, collect ImportErrors."""
    modules_to_check = [
        "app.main", "app.config", "app.risk.manager", "app.strategies",
        "app.ml.features.engineer", "app.backtest.engine",
        "app.comparison.engine", "app.execution.smart_router",
    ]
    errors: list[str] = []
    for module in modules_to_check:
        result = _cached_import_check(module)
        if result:
            errors.append(f"{module}: {result}")
    return errors


def scan_security_issues() -> list[SecurityIssue]:
    """Scan Python files for known security patterns."""
    issues: list[SecurityIssue] = []

    for py_file in PROJECT_ROOT.rglob("*.py"):
        try:
            content = py_file.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue

        for pattern, severity, issue_type, description, auto_fixable in _SECURITY_PATTERNS:
            for match in pattern.finditer(content):
                line_no = content[:match.start()].count("\n") + 1
                issues.append(
                    SecurityIssue(
                        severity=severity,
                        issue_type=issue_type,
                        file_path=str(py_file),
                        line_number=line_no,
                        description=description,
                        auto_fixable=auto_fixable,
                    )
                )
    return issues

# The remainder of the module (auto‑fix logic, report generation, monitor loop, etc.)
# is unchanged from the original implementation.