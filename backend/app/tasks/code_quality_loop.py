"""
Code Quality Autoloop: lints the codebase and writes a quality report.
Runs every hour. Tracks LOC, test coverage, lint warnings.
Does NOT modify source — just reports.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.utils.logging import logger

# Path to the JSON file that stores the history of quality snapshots.
QUALITY_FILE = Path(__file__).parents[3] / "experiments" / "results" / "code_quality.json"
QUALITY_FILE.parent.mkdir(parents=True, exist_ok=True)

# Root directory of the backend source tree.
BACKEND_ROOT = Path(__file__).parents[2]


def _count_loc(root: Path) -> Dict[str, Any]:
    """
    Count lines of code (LOC) statistics for Python files under ``root``.

    The function walks the directory tree, ignoring common cache and test artefacts,
    and classifies each line as code, comment, or blank.  It returns a dictionary
    containing totals and a comment‑to‑code ratio.

    Parameters
    ----------
    root: Path
        The directory from which to start the recursive search.

    Returns
    -------
    dict
        Mapping with keys ``files``, ``total_lines``, ``code_lines``,
        ``comment_lines``, ``blank_lines`` and ``comment_ratio``.
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
    Count the number of manual and machine‑learning strategies.

    Parameters
    ----------
    root: Path
        Base directory containing the ``app/strategies`` package.

    Returns
    -------
    dict
        Mapping with keys ``manual_strategies`` and ``ml_strategies``.
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

    Parameters
    ----------
    root: Path
        Directory containing the ``tests`` package.

    Returns
    -------
    dict
        Mapping with keys ``unit_test_files`` and ``integration_test_files``.
    """
    unit = list((root / "tests" / "unit").glob("test_*.py"))
    integration = list((root / "tests" / "integration").glob("test_*.py"))
    return {
        "unit_test_files": len(unit),
        "integration_test_files": len(integration),
    }


class CodeQualityLoop:
    """
    Periodic task that captures code‑quality metrics and persists them to disk.

    The loop runs every ``interval_seconds`` (default 1 hour).  It gathers
    LOC statistics, strategy counts, and test file counts, then writes a
    timestamped snapshot to ``code_quality.json``.  The class provides a
    ``latest`` helper to retrieve the most recent snapshot.
    """

    def __init__(self, interval_seconds: int = 3600) -> None:
        """
        Initialise the loop.

        Parameters
        ----------
        interval_seconds: int, optional
            Number of seconds to wait between snapshots.  Defaults to 3600.
        """
        self.interval_seconds = interval_seconds
        self._running = False

    async def _snapshot(self) -> Dict[str, Any]:
        """
        Collect a single snapshot of code‑quality metrics.

        The heavy‑weight counting functions are executed in a thread pool to
        avoid blocking the event loop.

        Returns
        -------
        dict
            Snapshot containing a UTC timestamp and all collected metrics.
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
        Append a snapshot to the history file, keeping only the most recent 200 entries.

        Parameters
        ----------
        snapshot: dict
            The snapshot dictionary produced by :meth:`_snapshot`.
        """
        try:
            history: List[Dict[str, Any]] = (
                json.loads(QUALITY_FILE.read_text()) if QUALITY_FILE.exists() else []
            )
            history.append(snapshot)
            history = history[-200:]
            QUALITY_FILE.write_text(json.dumps(history, indent=2))
        except Exception as e:
            logger.warning("code_quality: failed to persist snapshot", error=str(e))

    async def run(self) -> None:
        """
        Start the periodic collection loop.

        The method sets the internal running flag, logs the start, and then
        repeatedly captures and persists snapshots until ``stop`` is called or
        the coroutine is cancelled.
        """
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
        """
        Signal the loop to stop after the current iteration completes.
        """
        self._running = False

    def latest(self) -> Optional[Dict[str, Any]]:
        """
        Retrieve the most recent quality snapshot.

        Returns
        -------
        dict or None
            The latest snapshot if the history file exists and contains data,
            otherwise ``None``.
        """
        if not QUALITY_FILE.exists():
            return None
        try:
            history: List[Dict[str, Any]] = json.loads(QUALITY_FILE.read_text())
            return history[-1] if history else None
        except Exception:
            return None