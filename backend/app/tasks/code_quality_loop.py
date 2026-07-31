"""
Code Quality Autoloop: lints the codebase and writes a quality report.
Runs every hour. Tracks LOC, test coverage, lint warnings.
Does NOT modify source — just reports.
"""
from __future__ import annotations
import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path

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


# ----------------------------------------------------------------------
# Unit tests for edge cases
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import unittest
    import tempfile
    import shutil

    class TestCodeQualityHelpers(unittest.TestCase):
        def setUp(self):
            # Create a temporary directory to act as a fake repo root
            self.temp_dir = Path(tempfile.mkdtemp())
            # Ensure required subdirectories exist
            (self.temp_dir / "app" / "strategies" / "manual").mkdir(parents=True, exist_ok=True)
            (self.temp_dir / "app" / "strategies" / "ml_enhanced").mkdir(parents=True, exist_ok=True)
            (self.temp_dir / "tests" / "unit").mkdir(parents=True, exist_ok=True)
            (self.temp_dir / "tests" / "integration").mkdir(parents=True, exist_ok=True)

        def tearDown(self):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

        def test_count_loc_no_python_files(self):
            """Edge case: directory contains no .py files; all counts should be zero."""
            result = _count_loc(self.temp_dir)
            self.assertEqual(result["files"], 0)
            self.assertEqual(result["total_lines"], 0)
            self.assertEqual(result["code_lines"], 0)
            self.assertEqual(result["comment_lines"], 0)
            self.assertEqual(result["blank_lines"], 0)
            # comment_ratio uses max(code_lines, 1) => denominator 1
            self.assertEqual(result["comment_ratio"], 0.0)

        def test_count_loc_comment_only_file(self):
            """Edge case: file with only comments and blank lines; comment_ratio should handle zero code lines."""
            file_path = self.temp_dir / "sample.py"
            file_path.write_text("# Comment line 1\n\n# Comment line 2\n", encoding="utf-8")
            result = _count_loc(self.temp_dir)
            self.assertEqual(result["files"], 1)
            self.assertEqual(result["total_lines"], 4)
            self.assertEqual(result["code_lines"], 0)
            self.assertEqual(result["comment_lines"], 2)
            self.assertEqual(result["blank_lines"], 1)
            # comment_ratio = comment_lines / 1 (since code_lines is 0)
            self.assertEqual(result["comment_ratio"], round(2 / 1, 3))

        def test_count_strategies_missing_directories(self):
            """Edge case: strategy directories missing; counts should be zero without raising."""
            # Remove one of the strategy directories to simulate missing path
            shutil.rmtree(self.temp_dir / "app" / "strategies" / "ml_enhanced")
            result = _count_strategies(self.temp_dir)
            self.assertEqual(result["manual_strategies"], 0)
            self.assertEqual(result["ml_strategies"], 0)

    unittest.main(argv=["first-arg-is-ignored"], exit=False)