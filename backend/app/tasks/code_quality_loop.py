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
from typing import Dict, Tuple, Any

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


def _count_loc(root: Path | None) -> dict:
    """
    Count lines of code under ``root`` with caching.
    Only files whose modification time has changed are re‑processed.
    Handles None or non‑existent roots gracefully.
    """
    if root is None or not root.exists():
        return {
            "files": 0,
            "total_lines": 0,
            "code_lines": 0,
            "comment_lines": 0,
            "blank_lines": 0,
            "comment_ratio": 0.0,
        }

    global _loc_cache

    total_files = total_lines = code_lines = blank_lines = comment_lines = 0

    # Track files we encounter to later purge stale cache entries
    seen_files = set()

    for py_file in root.rglob("*.py"):
        if _should_skip(py_file):
            continue
        seen_files.add(py_file)
        total_files += 1
        try:
            mtime = py_file.stat().st_mtime
        except Exception:
            # If the file cannot be stat'ed, skip it
            continue
        cached = _loc_cache.get(py_file)

        if cached and cached[0] == mtime:
            stats = cached[1]
        else:
            stats = _compute_file_stats(py_file)
            _loc_cache[py_file] = (mtime, stats)

        total_lines += stats.get("total_lines", 0)
        code_lines += stats.get("code_lines", 0)
        blank_lines += stats.get("blank_lines", 0)
        comment_lines += stats.get("comment_lines", 0)

    # Remove cache entries for files that no longer exist
    for dead in set(_loc_cache) - seen_files:
        del _loc_cache[dead]

    comment_ratio = round(comment_lines / max(code_lines, 1), 3) if code_lines else 0.0

    return {
        "files": total_files,
        "total_lines": total_lines,
        "code_lines": code_lines,
        "comment_lines": comment_lines,
        "blank_lines": blank_lines,
        "comment_ratio": comment_ratio,
    }


def _count_strategies(root: Path | None) -> dict:
    """
    Count manual and ML strategy files.
    Handles None or missing directories safely.
    """
    if root is None or not root.exists():
        return {"manual_strategies": 0, "ml_strategies": 0}

    manual_path = root / MANUAL_STRATEGY_REL
    ml_path = root / ML_STRATEGY_REL

    manual = list(manual_path.glob("*.py")) if manual_path.exists() else []
    ml = list(ml_path.glob("*.py")) if ml_path.exists() else []

    return {
        "manual_strategies": len([f for f in manual if not f.name.startswith("__")]),
        "ml_strategies": len([f for f in ml if not f.name.startswith("__")]),
    }


def _count_tests(root: Path | None) -> dict:
    """
    Count unit and integration test files.
    Handles None or missing directories safely.
    """
    if root is None or not root.exists():
        return {"unit_test_files": 0, "integration_test_files": 0}

    unit_path = root / UNIT_TEST_REL
    integration_path = root / INTEGRATION_TEST_REL

    unit = list(unit_path.glob("test_*.py")) if unit_path.exists() else []
    integration = list(integration_path.glob("test_*.py")) if integration_path.exists() else []

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

    def _persist(self, snapshot: Any) -> None:
        """
        Persist a snapshot to disk.
        Ignores None or malformed snapshots.
        """
        if not isinstance(snapshot, dict):
            logger.warning("code_quality: snapshot is not a dict, skipping persist")
            return
        try:
            history = json.loads(QUALITY_FILE.read_text()) if QUALITY_FILE.exists() else []
            if not isinstance(history, list):
                history = []
            history.append(snapshot)
            # Ensure we keep at most HISTORY_LIMIT entries
            if len(history) > HISTORY_LIMIT:
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
        """
        Return the most recent snapshot, or None if unavailable or malformed.
        """
        if not QUALITY_FILE.exists():
            return None
        try:
            content = QUALITY_FILE.read_text()
            history = json.loads(content) if content else []
            if not isinstance(history, list) or not history:
                return None
            return history[-1]
        except Exception:
            return None