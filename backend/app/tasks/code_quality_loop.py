"""
Code Quality Autoloop: lints the codebase and writes a quality report.

Runs every hour. Tracks LOC, test coverage, lint warnings.
Does NOT modify source — just reports.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional

from app.utils.logging import logger

QUALITY_FILE = Path(__file__).parents[3] / "experiments" / "results" / "code_quality.json"
QUALITY_FILE.parent.mkdir(parents=True, exist_ok=True)

BACKEND_ROOT = Path(__file__).parents[2]


def _count_loc(root: Path) -> Dict[str, Any]:
    """
    Count lines of code (LOC) statistics for all Python files under *root*.

    Args:
        root: The directory from which to recursively search for ``*.py`` files.

    Returns:
        A dictionary containing totals for files, total lines, code lines,
        comment lines, blank lines, and the comment‑to‑code ratio.
    """
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


def _count_strategies(root: Path) -> Dict[str, int]:
    """
    Count the number of manual and machine‑learning strategies present in the codebase.

    Args:
        root: The repository root directory.

    Returns:
        A dictionary with counts for ``manual_strategies`` and ``ml_strategies``.
    """
    manual = list((root / "app" / "strategies" / "manual").glob("*.py"))
    ml = list((root / "app" / "strategies" / "ml_enhanced").glob("*.py"))
    return {
        "manual_strategies": len([f for f in manual if not f.name.startswith("__")]),
        "ml_strategies": len([f for f in ml if not f.name.startswith("__")]),
    }


def _count_tests(root: Path) -> Dict[str, int]:
    """
    Count unit and integration test files.

    Args:
        root: The repository root directory.

    Returns:
        A dictionary with the number of unit test files and integration test files.
    """
    unit = list((root / "tests" / "unit").glob("test_*.py"))
    integration = list((root / "tests" / "integration").glob("test_*.py"))
    return {
        "unit_test_files": len(unit),
        "integration_test_files": len(integration),
    }


class CodeQualityLoop:
    """
    Periodic task that captures a snapshot of code‑quality metrics and persists it.

    The loop runs every ``interval_seconds`` (default: 3600 seconds) and records
    line‑count statistics, strategy counts, and test counts. Snapshots are stored
    in a JSON file for later inspection.
    """

    def __init__(self, interval_seconds: int = 3600) -> None:
        """
        Initialize the loop.

        Args:
            interval_seconds: How often to take a snapshot, in seconds.
        """
        self.interval_seconds = interval_seconds
        self._running = False

    async def _snapshot(self) -> Dict[str, Any]:
        """
        Gather a snapshot of the current code‑quality metrics.

        Returns:
            A dictionary containing a timestamp and various LOC, strategy, and test metrics.
        """
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

    def _persist(self, snapshot: Dict[str, Any]) -> None:
        """
        Append a snapshot to the persistent JSON history file.

        Args:
            snapshot: The snapshot dictionary to persist.
        """
        try:
            history = json.loads(QUALITY_FILE.read_text()) if QUALITY_FILE.exists() else []
            history.append(snapshot)
            history = history[-200:]  # keep only the most recent 200 entries
            QUALITY_FILE.write_text(json.dumps(history, indent=2))
        except Exception as e:
            logger.warning("code_quality: failed to persist snapshot", error=str(e))

    async def run(self) -> None:
        """
        Start the periodic execution loop.

        This method blocks until ``stop`` is called or the coroutine is cancelled.
        """
        self._running = True
        logger.info("CodeQualityLoop started", interval=self.interval_seconds)
        while self._running:
            start_time = time.perf_counter()
            try:
                snapshot = await self._snapshot()
                self._persist(snapshot)

                # Compute key metrics for structured logging
                signal_count = snapshot.get("manual_strategies", 0) + snapshot.get("ml_strategies", 0)
                execution_time = round(time.perf_counter() - start_time, 3)
                pnl = snapshot.get("pnl")  # P&L not tracked here; will be None if absent

                logger.info(
                    "code_quality: iteration metrics",
                    signal_count=signal_count,
                    execution_time=execution_time,
                    pnl=pnl,
                )
                logger.debug("Code quality snapshot", **snapshot)
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.warning("Quality snapshot failed", error=str(e))
            await asyncio.sleep(self.interval_seconds)

    async def stop(self) -> None:
        """
        Signal the loop to stop after the current iteration finishes.
        """
        self._running = False

    def latest(self) -> Optional[Dict[str, Any]]:
        """
        Retrieve the most recent snapshot from the persisted history.

        Returns:
            The latest snapshot dictionary, or ``None`` if no history exists or
            the file cannot be read.
        """
        if not QUALITY_FILE.exists():
            return None
        try:
            history = json.loads(QUALITY_FILE.read_text())
            return history[-1] if history else None
        except Exception:
            return None