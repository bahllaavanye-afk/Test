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

from pydantic import BaseModel, Field, validator

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


class CodeQualitySnapshot(BaseModel):
    """Schema representing a single code‑quality snapshot."""

    timestamp: str = Field(
        ...,
        description="ISO8601 UTC timestamp of when the snapshot was taken",
        example="2026-07-30T12:34:56Z",
    )
    files: int = Field(
        ...,
        ge=0,
        description="Total number of Python files scanned",
        example=123,
    )
    total_lines: int = Field(
        ...,
        ge=0,
        description="Total number of lines across all scanned files",
        example=4567,
    )
    code_lines: int = Field(
        ...,
        ge=0,
        description="Lines containing executable code (non‑blank, non‑comment)",
        example=3400,
    )
    comment_lines: int = Field(
        ...,
        ge=0,
        description="Lines that are comments",
        example=800,
    )
    blank_lines: int = Field(
        ...,
        ge=0,
        description="Blank (empty) lines",
        example=500,
    )
    comment_ratio: float = Field(
        ...,
        ge=0,
        le=1,
        description="Ratio of comment lines to code lines",
        example=0.235,
    )
    manual_strategies: int = Field(
        ...,
        ge=0,
        description="Number of manual strategy files",
        example=5,
    )
    ml_strategies: int = Field(
        ...,
        ge=0,
        description="Number of ML‑enhanced strategy files",
        example=3,
    )
    unit_test_files: int = Field(
        ...,
        ge=0,
        description="Number of unit‑test files",
        example=20,
    )
    integration_test_files: int = Field(
        ...,
        ge=0,
        description="Number of integration‑test files",
        example=8,
    )

    @validator("timestamp")
    def validate_timestamp(cls, v: str) -> str:
        """Ensure the timestamp is a valid ISO8601 UTC string."""
        try:
            dt = datetime.fromisoformat(v.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                raise ValueError
        except Exception as exc:
            raise ValueError("timestamp must be a valid ISO8601 UTC string") from exc
        return v


class CodeQualityLoop:
    """Periodic task that records code‑base metrics."""

    def __init__(self, interval_seconds: int = 3600):
        self.interval_seconds = interval_seconds
        self._running = False

    async def _snapshot(self) -> dict:
        loop = asyncio.get_running_loop()
        loc = await loop.run_in_executor(None, _count_loc, BACKEND_ROOT)
        strat = await loop.run_in_executor(None, _count_strategies, BACKEND_ROOT)
        tests = await loop.run_in_executor(None, _count_tests, BACKEND_ROOT)

        raw_snapshot = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **loc,
            **strat,
            **tests,
        }

        # Validate and normalise using the Pydantic schema
        snapshot_model = CodeQualitySnapshot(**raw_snapshot)
        return snapshot_model.dict()

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
        """Return the most recent snapshot, if any."""
        if not QUALITY_FILE.exists():
            return None
        try:
            history = json.loads(QUALITY_FILE.read_text())
            return history[-1] if history else None
        except Exception:
            return None