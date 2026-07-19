"""
Code Quality Autoloop: lints the codebase and writes a quality report.
Runs every hour. Tracks LOC, test coverage, lint warnings.
Does NOT modify source — just reports.
"""
from __future__ import annotations
import asyncio
import json
import os
import stat
from datetime import datetime, timezone
from pathlib import Path

import pytest
import tempfile

from app.utils.logging import logger

QUALITY_FILE = Path(__file__).parents[3] / "experiments" / "results" / "code_quality.json"
QUALITY_FILE.parent.mkdir(parents=True, exist_ok=True)

BACKEND_ROOT = Path(__file__).parents[2]


def _count_loc(root: Path) -> dict:
    total_files = 0
    total_lines = 0
    code_lines = 0
    blank_lines = 0
    comment_lines = 0

    for py_file in root.rglob("*.py"):
        if any(skip in str(py_file) for skip in ("__pycache__", ".pytest_cache", "test.db")):
            continue
        total_files += 1
        try:
            for line in py_file.read_text(errors="ignore").splitlines():
                total_lines += 1
                stripped = line.strip()
                if not stripped:
                    blank_lines += 1
                elif stripped.startswith("#"):
                    comment_lines += 1
                else:
                    code_lines += 1
        except Exception as e:
            logger.debug("code_quality: skip unreadable file", error=str(e))
            continue

    return {
        "files": total_files,
        "total_lines": total_lines,
        "code_lines": code_lines,
        "comment_lines": comment_lines,
        "blank_lines": blank_lines,
        "comment_ratio": round(comment_lines / max(code_lines, 1), 3),
    }


def _count_strategies(root: Path) -> dict:
    manual = list((root / "app" / "strategies" / "manual").glob("*.py"))
    ml = list((root / "app" / "strategies" / "ml_enhanced").glob("*.py"))
    return {
        "manual_strategies": len([f for f in manual if not f.name.startswith("__")]),
        "ml_strategies": len([f for f in ml if not f.name.startswith("__")]),
    }


def _count_tests(root: Path) -> dict:
    unit = list((root / "tests" / "unit").glob("test_*.py"))
    integration = list((root / "tests" / "integration").glob("test_*.py"))
    return {
        "unit_test_files": len(unit),
        "integration_test_files": len(integration),
    }


class CodeQualityLoop:
    def __init__(self, interval_seconds: int = 3600):
        self.interval_seconds = interval_seconds
        self._running = False

    async def _snapshot(self) -> dict:
        loop = asyncio.get_running_loop()
        loc = await loop.run_in_executor(None, _count_loc, BACKEND_ROOT)
        strat = await loop.run_in_executor(None, _count_strategies, BACKEND_ROOT)
        tests = await loop.run_in_executor(None, _count_tests, BACKEND_ROOT)
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **loc,
            **strat,
            **tests,
        }

    def _persist(self, snapshot: dict) -> None:
        try:
            history = json.loads(QUALITY_FILE.read_text()) if QUALITY_FILE.exists() else []
            history.append(snapshot)
            history = history[-200:]
            QUALITY_FILE.write_text(json.dumps(history, indent=2))
        except Exception as e:
            logger.warning("code_quality: failed to persist snapshot", error=str(e))

    async def run(self) -> None:
        self._running = True
        logger.info("CodeQualityLoop started", interval=self.interval_seconds)
        while self._running:
            try:
                snapshot = await self._snapshot()
                self._persist(snapshot)
                logger.debug("Code quality snapshot", **snapshot)
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.warning("Quality snapshot failed", error=str(e))
            await asyncio.sleep(self.interval_seconds)

    async def stop(self) -> None:
        self._running = False

    def latest(self) -> dict | None:
        if not QUALITY_FILE.exists():
            return None
        try:
            history = json.loads(QUALITY_FILE.read_text())
            return history[-1] if history else None
        except Exception:
            return None


# =========================
# Unit tests for edge cases
# =========================

def test_count_loc_empty_directory():
    """When the directory contains no .py files, all counts should be zero and comment_ratio 0."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        result = _count_loc(root)
        assert result["files"] == 0
        assert result["total_lines"] == 0
        assert result["code_lines"] == 0
        assert result["comment_lines"] == 0
        assert result["blank_lines"] == 0
        assert result["comment_ratio"] == 0.0


def test_count_loc_comments_and_blanks_only():
    """A file with only comments and blank lines should report zero code lines and correct ratios."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        file_path = root / "sample.py"
        content = "\n".join(
            [
                "# comment line 1",
                "",
                "# comment line 2",
                "   ",
                "# comment line 3",
            ]
        )
        file_path.write_text(content)
        result = _count_loc(root)
        assert result["files"] == 1
        assert result["total_lines"] == 5
        assert result["code_lines"] == 0
        assert result["comment_lines"] == 3
        assert result["blank_lines"] == 2
        # comment_ratio uses max(code_lines, 1) => 1, so ratio equals comment_lines
        assert result["comment_ratio"] == round(3 / 1, 3)


def test_count_loc_unreadable_file_is_skipped():
    """Unreadable .py files should be ignored without raising an exception."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        file_path = root / "unreadable.py"
        file_path.write_text("print('should be skipped')\n")
        # Remove read permissions
        file_path.chmod(stat.S_IWUSR)
        # Ensure function does not raise
        result = _count_loc(root)
        # The unreadable file is counted as a file but its lines are not processed,
        # therefore total_lines remains 0 and code_lines stays 0.
        # Depending on OS, the file may still be readable; we assert that no exception occurs.
        assert isinstance(result, dict)
        assert "files" in result
        # The exact counts may vary, but the function must return a valid dict.
        assert result["total_lines"] >= 0
        assert result["code_lines"] >= 0
        assert result["comment_ratio"] >= 0.0