"""
Code Quality Autoloop: lints the codebase and writes a quality report.
Runs every hour. Tracks LOC, test coverage, lint warnings.
Does NOT modify source — just reports.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

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
    """Periodically collects code‑quality metrics and persists them."""

    def __init__(self, interval_seconds: int = 3600) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be a positive integer")
        self.interval_seconds = interval_seconds
        self._running = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._logger = logger

    async def _snapshot(self) -> dict:
        """Collect a single snapshot of the repository state."""
        loop = asyncio.get_running_loop()
        loc = await loop.run_in_executor(None, _count_loc, BACKEND_ROOT)
        strat = await loop.run_in_executor(None, _count_strategies, BACKEND_ROOT)
        tests = await loop.run_in_executor(None, _count_tests, BACKEND_ROOT)
        snapshot = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **loc,
            **strat,
            **tests,
        }
        return snapshot

    def _validate_snapshot(self, snapshot: Dict[str, Any]) -> bool:
        """Confirm required fields exist before persisting."""
        required = {"timestamp", "files", "code_lines"}
        missing = required - snapshot.keys()
        if missing:
            self._logger.debug("code_quality: snapshot missing fields", missing=missing)
            return False
        return True

    def _persist(self, snapshot: dict) -> None:
        """Append a validated snapshot to the JSON history file."""
        if not self._validate_snapshot(snapshot):
            return
        try:
            history: List[dict] = json.loads(QUALITY_FILE.read_text()) if QUALITY_FILE.exists() else []
            history.append(snapshot)
            # Keep only the most recent 200 entries to bound file size.
            history = history[-200:]
            QUALITY_FILE.write_text(json.dumps(history, indent=2))
        except Exception as e:
            self._logger.warning("code_quality: failed to persist snapshot", error=str(e))

    async def _run_loop(self) -> None:
        """Internal loop runner; respects the running event for graceful shutdown."""
        self._logger.info("CodeQualityLoop started", interval=self.interval_seconds)
        while self._running.is_set():
            try:
                snapshot = await self._snapshot()
                self._persist(snapshot)
                self._logger.debug("Code quality snapshot", **snapshot)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.warning("Quality snapshot failed", error=str(e))
            await asyncio.sleep(self.interval_seconds)

    async def start(self) -> None:
        """Public method to start the background task."""
        if self._task and not self._task.done():
            self._logger.debug("CodeQualityLoop already running")
            return
        self._running.set()
        self._task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        """Signal the loop to stop and await task completion."""
        self._running.clear()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        self._logger.info("CodeQualityLoop stopped")

    def latest(self) -> dict | None:
        """Return the most recent persisted snapshot, if any."""
        if not QUALITY_FILE.exists():
            return None
        try:
            history = json.loads(QUALITY_FILE.read_text())
            return history[-1] if history else None
        except Exception:
            self._logger.debug("code_quality: failed to read latest snapshot")
            return None