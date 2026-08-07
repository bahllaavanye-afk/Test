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
from typing import Any, Dict, List, Tuple

from app.utils.logging import logger

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
DEFAULT_INTERVAL_SECONDS = 3600
HISTORY_LIMIT = 200
SKIP_PATTERNS = ("__pycache__", ".pytest_cache", "test.db")

MANUAL_STRATEGY_REL = "app/strategies/manual"
ML_STRATEGY_REL = "app/strategies/ml_enhanced"
UNIT_TEST_REL = "tests/unit"
INTEGRATION_TEST_REL = "tests/integration"

QUALITY_FILE = Path(__file__).parents[3] / "experiments" / "results" / "code_quality.json"
QUALITY_FILE.parent.mkdir(parents=True, exist_ok=True)

BACKEND_ROOT = Path(__file__).parents[2]

# ----------------------------------------------------------------------
# Internal cache for LOC counting to avoid re‑reading unchanged files.
# The cache maps a file path to a tuple of (modification timestamp, stats dict).
# ----------------------------------------------------------------------
_loc_cache: Dict[Path, Tuple[float, Dict[str, int]]] = {}


def _compute_file_stats(py_file: Path) -> Dict[str, int]:
    """
    Compute line‑of‑code statistics for a single Python file.

    Parameters
    ----------
    py_file: Path
        Path to the Python source file.

    Returns
    -------
    Dict[str, int]
        Mapping with keys ``total_lines``, ``code_lines``, ``blank_lines``,
        and ``comment_lines``.
    """
    stats = {
        "total_lines": 0,
        "code_lines": 0,
        "blank_lines": 0,
        "comment_lines": 0,
    }
    try:
        for line in py_file.read_text(errors="ignore").splitlines():
            stats["total_lines"] += 1
            stripped = line.strip()
            if not stripped:
                stats["blank_lines"] += 1
            elif stripped.startswith("#"):
                stats["comment_lines"] += 1
            else:
                stats["code_lines"] += 1
    except Exception as e:
        logger.debug("code_quality: skip unreadable file", error=str(e))
    return stats


def _count_loc(root: Path) -> Dict[str, Any]:
    """
    Count lines of code under ``root`` with caching.

    Files whose modification time has not changed are read from the internal
    cache to avoid unnecessary I/O.

    Parameters
    ----------
    root: Path
        Directory from which to start the recursive search for ``*.py`` files.

    Returns
    -------
    Dict[str, Any]
        Aggregated statistics including file count, total lines, code lines,
        comment lines, blank lines, and a comment‑to‑code ratio.
    """
    global _loc_cache

    total_files = 0
    total_lines = 0
    code_lines = 0
    blank_lines = 0
    comment_lines = 0

    # Gather current python files, respecting skip patterns
    current_files: List[Path] = [
        py_file
        for py_file in root.rglob("*.py")
        if not any(skip in str(py_file) for skip in SKIP_PATTERNS)
    ]

    # Remove cache entries for files that disappeared
    vanished = set(_loc_cache) - set(current_files)
    for dead in vanished:
        del _loc_cache[dead]

    for py_file in current_files:
        total_files += 1
        mtime = py_file.stat().st_mtime
        cached = _loc_cache.get(py_file)

        if cached and cached[0] == mtime:
            stats = cached[1]
        else:
            stats = _compute_file_stats(py_file)
            _loc_cache[py_file] = (mtime, stats)

        total_lines += stats["total_lines"]
        code_lines += stats["code_lines"]
        blank_lines += stats["blank_lines"]
        comment_lines += stats["comment_lines"]

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
    Count user‑defined strategy files.

    Parameters
    ----------
    root: Path
        Project root containing the strategy directories.

    Returns
    -------
    Dict[str, int]
        Number of manual and ML‑enhanced strategy files (excluding ``__init__``‑style
        files).
    """
    manual = list((root / MANUAL_STRATEGY_REL).glob("*.py"))
    ml = list((root / ML_STRATEGY_REL).glob("*.py"))
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
        Project root containing the test directories.

    Returns
    -------
    Dict[str, int]
        Number of unit test files and integration test files.
    """
    unit = list((root / UNIT_TEST_REL).glob("test_*.py"))
    integration = list((root / INTEGRATION_TEST_REL).glob("test_*.py"))
    return {
        "unit_test_files": len(unit),
        "integration_test_files": len(integration),
    }


class CodeQualityLoop:
    """
    Periodic task that snapshots code‑base metrics and persists them.

    The loop runs every ``interval_seconds`` (default 1 hour) and writes a JSON
    history file that can be inspected by downstream monitoring tools.
    """

    def __init__(self, interval_seconds: int = DEFAULT_INTERVAL_SECONDS) -> None:
        """
        Initialise the loop.

        Parameters
        ----------
        interval_seconds: int, optional
            Number of seconds to wait between successive snapshots.
        """
        self.interval_seconds = interval_seconds
        self._running = False

    async def _snapshot(self) -> Dict[str, Any]:
        """
        Gather a snapshot of code metrics.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing timestamp, LOC statistics, strategy counts,
            and test counts.
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

        Parameters
        ----------
        snapshot: Dict[str, Any]
            The snapshot data to be stored.
        """
        try:
            history = json.loads(QUALITY_FILE.read_text()) if QUALITY_FILE.exists() else []
            history.append(snapshot)
            history = history[-HISTORY_LIMIT:]
            QUALITY_FILE.write_text(json.dumps(history, indent=2))
        except Exception as e:
            logger.warning("code_quality: failed to persist snapshot", error=str(e))

    async def run(self) -> None:
        """
        Start the periodic execution loop.

        The method runs until ``stop`` is called or the coroutine is cancelled.
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
        Signal the loop to stop after the current iteration completes.
        """
        self._running = False

    def latest(self) -> Dict[str, Any] | None:
        """
        Retrieve the most recent snapshot from the history file.

        Returns
        -------
        Dict[str, Any] | None
            The latest snapshot if the file exists and is readable; otherwise ``None``.
        """
        if not QUALITY_FILE.exists():
            return None
        try:
            history = json.loads(QUALITY_FILE.read_text())
            return history[-1] if history else None
        except Exception:
            return None