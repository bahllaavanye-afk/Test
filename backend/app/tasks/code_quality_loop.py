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
from typing import Dict, Tuple, List

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
    """Compute LOC statistics for a single python file."""
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


def _gather_python_files(root: Path) -> List[Path]:
    """Return a list of python files under ``root`` excluding skip patterns."""
    return [
        py_file
        for py_file in root.rglob("*.py")
        if not any(skip in str(py_file) for skip in SKIP_PATTERNS)
    ]


def _purge_vanished_cache(current_files: List[Path]) -> None:
    """Remove cache entries for files that no longer exist."""
    vanished = set(_loc_cache) - set(current_files)
    for dead in vanished:
        del _loc_cache[dead]


def _get_file_stats(py_file: Path) -> Dict[str, int]:
    """
    Retrieve LOC statistics for ``py_file`` using the cache when possible.
    Updates the cache if the file has changed.
    """
    mtime = py_file.stat().st_mtime
    cached = _loc_cache.get(py_file)

    if cached and cached[0] == mtime:
        return cached[1]

    stats = _compute_file_stats(py_file)
    _loc_cache[py_file] = (mtime, stats)
    return stats


def _aggregate_loc_stats(files: List[Path]) -> Dict[str, int]:
    """Aggregate LOC statistics across a list of python files."""
    totals = {
        "files": 0,
        "total_lines": 0,
        "code_lines": 0,
        "blank_lines": 0,
        "comment_lines": 0,
    }

    for py_file in files:
        totals["files"] += 1
        stats = _get_file_stats(py_file)
        totals["total_lines"] += stats["total_lines"]
        totals["code_lines"] += stats["code_lines"]
        totals["blank_lines"] += stats["blank_lines"]
        totals["comment_lines"] += stats["comment_lines"]

    totals["comment_ratio"] = round(
        totals["comment_lines"] / max(totals["code_lines"], 1), 3
    )
    return totals


def _count_loc(root: Path) -> dict:
    """
    Count lines of code under ``root`` with caching.
    Only files whose modification time has changed are re‑processed.
    """
    python_files = _gather_python_files(root)
    _purge_vanished_cache(python_files)
    return _aggregate_loc_stats(python_files)


def _count_strategies(root: Path) -> dict:
    manual = list((root / MANUAL_STRATEGY_REL).glob("*.py"))
    ml = list((root / ML_STRATEGY_REL).glob("*.py"))
    return {
        "manual_strategies": len([f for f in manual if not f.name.startswith("__")]),
        "ml_strategies": len([f for f in ml if not f.name.startswith("__")]),
    }


def _count_tests(root: Path) -> dict:
    unit = list((root / UNIT_TEST_REL).glob("test_*.py"))
    integration = list((root / INTEGRATION_TEST_REL).glob("test_*.py"))
    return {
        "unit_test_files": len(unit),
        "integration_test_files": len(integration),
    }


class CodeQualityLoop:
    def __init__(self, interval_seconds: int = DEFAULT_INTERVAL_SECONDS):
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
            history = history[-HISTORY_LIMIT:]
            QUALITY_FILE.write_text(json.dumps(history, indent=2))
        except Exception as e:
            logger.warning("code_quality: failed to persist snapshot", error=str(e))

    async def run(self) -> None:
        self._running = True
        logger.info("CodeQualityLoop started", interval=self.interval_seconds)
        while self._running:
            start_time = time.perf_counter()
            try:
                snapshot = await self._snapshot()
                self._persist(snapshot)

                # Compute key metrics for structured logging
                signal_count = snapshot.get("manual_strategies", 0) + snapshot.get(
                    "ml_strategies", 0
                )
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
        self._running = False

    def latest(self) -> dict | None:
        if not QUALITY_FILE.exists():
            return None
        try:
            history = json.loads(QUALITY_FILE.read_text())
            return history[-1] if history else None
        except Exception:
            return None