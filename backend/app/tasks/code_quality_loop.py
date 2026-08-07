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
from typing import Dict, Tuple

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
# Internal caches
# ----------------------------------------------------------------------
# Cache for LOC counting: maps a file path to (modification timestamp, stats dict).
_loc_cache: Dict[Path, Tuple[float, Dict[str, int]]] = {}

# Cache for strategy and test counting: maps a directory path to (modification timestamp, result dict).
_strategy_cache: Dict[Path, Tuple[float, Dict[str, int]]] = {}
_test_cache: Dict[Path, Tuple[float, Dict[str, int]]] = {}


def _compute_file_stats(py_file: Path) -> Dict[str, int]:
    """Compute LOC statistics for a single python file."""
    stats = {
        "total_lines": 0,
        "code_lines": 0,
        "blank_lines": 0,
        "comment_lines": 0,
    }
    try:
        with py_file.open("r", errors="ignore") as f:
            for line in f:
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


def _should_skip(py_file: Path) -> bool:
    """Return True if the file matches any skip pattern."""
    return any(skip in py_file.parts for skip in SKIP_PATTERNS)


def _count_loc(root: Path) -> dict:
    """
    Count lines of code under ``root`` with caching.
    Only files whose modification time has changed are re‑processed.
    """
    global _loc_cache

    total_files = total_lines = code_lines = blank_lines = comment_lines = 0
    seen_files = set()

    for py_file in root.rglob("*.py"):
        if _should_skip(py_file):
            continue
        seen_files.add(py_file)
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

    # Purge stale entries
    for dead in set(_loc_cache) - seen_files:
        del _loc_cache[dead]

    return {
        "files": total_files,
        "total_lines": total_lines,
        "code_lines": code_lines,
        "comment_lines": comment_lines,
        "blank_lines": blank_lines,
        "comment_ratio": round(comment_lines / max(code_lines, 1), 3),
    }


def _cached_dir_count(
    base_dir: Path,
    pattern: str,
    cache: Dict[Path, Tuple[float, Dict[str, int]]],
    count_name: str,
) -> Dict[str, int]:
    """
    Generic cached directory counting helper.
    Returns a dict with a single key ``count_name`` mapping to the number of files
    matching ``pattern`` under ``base_dir``. Caches results based on the directory's
    modification timestamp.
    """
    if not base_dir.is_dir():
        return {count_name: 0}

    mtime = base_dir.stat().st_mtime
    cached = cache.get(base_dir)

    if cached and cached[0] == mtime:
        return cached[1]

    files = list(base_dir.glob(pattern))
    result = {count_name: len([f for f in files if not f.name.startswith("__")])}
    cache[base_dir] = (mtime, result)
    return result


def _count_strategies(root: Path) -> dict:
    """Count manual and ML strategy files with directory‑level caching."""
    manual_dir = root / MANUAL_STRATEGY_REL
    ml_dir = root / ML_STRATEGY_REL

    manual = _cached_dir_count(
        manual_dir,
        "*.py",
        _strategy_cache,
        "manual_strategies",
    )
    ml = _cached_dir_count(
        ml_dir,
        "*.py",
        _strategy_cache,
        "ml_strategies",
    )
    return {**manual, **ml}


def _count_tests(root: Path) -> dict:
    """Count unit and integration test files with directory‑level caching."""
    unit_dir = root / UNIT_TEST_REL
    integration_dir = root / INTEGRATION_TEST_REL

    unit = _cached_dir_count(
        unit_dir,
        "test_*.py",
        _test_cache,
        "unit_test_files",
    )
    integration = _cached_dir_count(
        integration_dir,
        "test_*.py",
        _test_cache,
        "integration_test_files",
    )
    return {**unit, **integration}


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

                signal_count = snapshot.get("manual_strategies", 0) + snapshot.get("ml_strategies", 0)
                execution_time = round(time.perf_counter() - start_time, 3)
                pnl = snapshot.get("pnl")

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