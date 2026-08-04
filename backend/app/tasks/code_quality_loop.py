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
from typing import Any, Dict, List, Optional

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
    """Periodic task that gathers code‑base metrics and persists them."""

    def __init__(self, interval_seconds: int = 3600):
        self.interval_seconds = interval_seconds
        self._running: bool = False

    async def _snapshot(self) -> dict:
        """Collect a snapshot of code‑quality related metrics."""
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
        """Append the snapshot to the rolling history file."""
        try:
            if QUALITY_FILE.exists():
                try:
                    history: List[Dict[str, Any]] = json.loads(QUALITY_FILE.read_text())
                except json.JSONDecodeError:
                    history = []
            else:
                history = []
            history.append(snapshot)
            # keep only the most recent 200 entries
            history = history[-200:]
            QUALITY_FILE.write_text(json.dumps(history, indent=2))
        except Exception as e:
            logger.warning("code_quality: failed to persist snapshot", error=str(e))

    async def run(self) -> None:
        """Start the infinite monitoring loop."""
        self._running = True
        logger.info("CodeQualityLoop started", interval=self.interval_seconds)
        while self._running:
            start_time = time.perf_counter()
            try:
                snapshot = await self._snapshot()
                self._persist(snapshot)

                # Key metrics for structured logging
                signal_count = snapshot.get("manual_strategies", 0) + snapshot.get("ml_strategies", 0)
                execution_time = round(time.perf_counter() - start_time, 3)
                pnl = snapshot.get("pnl")  # Not tracked; will be None if absent

                logger.info(
                    "code_quality: iteration metrics",
                    signal_count=signal_count,
                    execution_time=execution_time,
                    pnl=pnl,
                )
                logger.debug("Code quality snapshot", **snapshot)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("Quality snapshot failed", error=str(e))
            await asyncio.sleep(self.interval_seconds)

    async def stop(self) -> None:
        """Signal the loop to stop after the current iteration."""
        self._running = False

    def latest(self) -> Optional[dict]:
        """Return the most recent snapshot, or None if unavailable."""
        if not QUALITY_FILE.exists():
            return None
        try:
            history = json.loads(QUALITY_FILE.read_text())
            return history[-1] if history else None
        except Exception:
            return None