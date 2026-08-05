"""
Trade Archiver: writes every order, fill, and signal to JSON-lines files
for long-term audit and replay. Files rotate daily.
"""
from __future__ import annotations

import json
import asyncio
import time
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator

from app.utils.logging import logger

ARCHIVE_DIR = Path(__file__).parents[3] / "archive"
ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
_lock = asyncio.Lock()

# In‑memory counters for monitoring
_stats: dict[str, dict[str, float]] = defaultdict(lambda: {"count": 0, "pnl": 0.0})

# Signal quality thresholds
_MIN_CONFIDENCE = 0.6  # Minimum confidence required for entry signals
_REQUIRED_SIGNAL_FIELDS = {"signal_id", "action", "confidence"}
_REQUIRED_EXIT_FIELDS = {"reason"}


# -------------------------------------------------------------------------
# Pydantic Schemas
# -------------------------------------------------------------------------
class Order(BaseModel):
    """Schema for order events."""

    ts: datetime = Field(
        ...,
        description="Timestamp of the order event in UTC.",
        example="2023-01-01T12:00:00Z",
    )
    order_id: str = Field(
        ...,
        description="Unique identifier for the order.",
        example="ord_12345",
    )
    symbol: str = Field(
        ...,
        description="Ticker symbol of the instrument.",
        example="AAPL",
    )
    side: str = Field(
        ...,
        description="Side of the order, either 'buy' or 'sell'.",
        example="buy",
    )
    quantity: float = Field(
        ...,
        description="Quantity of the order.",
        example=100.0,
    )
    price: Optional[float] = Field(
        None,
        description="Limit price if the order is not market.",
        example=150.25,
    )
    status: str = Field(
        ...,
        description="Current status of the order.",
        example="filled",
    )
    pnl: Optional[float] = Field(
        None,
        description="Realized PnL associated with the order, if any.",
        example=12.5,
    )

    @validator("pnl")
    def pnl_must_be_number(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and not isinstance(v, (int, float)):
            raise ValueError("pnl must be a numeric type")
        return v


class Fill(BaseModel):
    """Schema for fill events."""

    ts: datetime = Field(
        ...,
        description="Timestamp of the fill event in UTC.",
        example="2023-01-01T12:01:00Z",
    )
    order_id: str = Field(
        ...,
        description="Identifier of the order this fill belongs to.",
        example="ord_12345",
    )
    fill_id: str = Field(
        ...,
        description="Unique identifier for the fill.",
        example="fill_98765",
    )
    quantity: float = Field(
        ...,
        description="Quantity filled.",
        example=50.0,
    )
    price: float = Field(
        ...,
        description="Price at which the fill occurred.",
        example=150.30,
    )
    pnl: Optional[float] = Field(
        None,
        description="PnL realized from this fill.",
        example=6.2,
    )

    @validator("pnl")
    def pnl_must_be_number(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and not isinstance(v, (int, float)):
            raise ValueError("pnl must be a numeric type")
        return v


class Signal(BaseModel):
    """Schema for trading signals."""

    ts: datetime = Field(
        ...,
        description="Timestamp when the signal was generated.",
        example="2023-01-01T11:55:00Z",
    )
    signal_id: str = Field(
        ...,
        description="Unique identifier for the signal.",
        example="sig_abc123",
    )
    action: str = Field(
        ...,
        description="Action to be taken: 'enter' or 'exit'.",
        example="enter",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence level of the signal (0‑1).",
        example=0.85,
    )
    reason: Optional[str] = Field(
        None,
        description="Reason for an exit signal; required when action is 'exit'.",
        example="Target reached",
    )
    extra: Optional[Dict[str, Any]] = Field(
        None,
        description="Any additional free‑form data attached to the signal.",
    )

    @validator("confidence")
    def confidence_above_min(cls, v: float) -> float:
        if v < _MIN_CONFIDENCE:
            raise ValueError(f"confidence {v} below minimum {_MIN_CONFIDENCE}")
        return v

    @validator("reason", always=True)
    def reason_required_for_exit(cls, v: Optional[str], values: dict) -> Optional[str]:
        if values.get("action") == "exit" and not v:
            raise ValueError("reason is required for exit signals")
        return v


class Decision(BaseModel):
    """Schema for decision events derived from signals."""

    ts: datetime = Field(
        ...,
        description="Timestamp of the decision.",
        example="2023-01-01T12:05:00Z",
    )
    decision_id: str = Field(
        ...,
        description="Unique identifier for the decision.",
        example="dec_45678",
    )
    signal_id: str = Field(
        ...,
        description="Identifier of the originating signal.",
        example="sig_abc123",
    )
    order_id: Optional[str] = Field(
        None,
        description="Associated order identifier if a trade was placed.",
        example="ord_12345",
    )
    outcome: Optional[str] = Field(
        None,
        description="Result of the decision, e.g., 'filled', 'rejected'.",
        example="filled",
    )
    pnl: Optional[float] = Field(
        None,
        description="Profit or loss resulting from the decision.",
        example=-3.4,
    )

    @validator("pnl")
    def pnl_must_be_number(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and not isinstance(v, (int, float)):
            raise ValueError("pnl must be a numeric type")
        return v


class Risk(BaseModel):
    """Schema for risk‑related events."""

    ts: datetime = Field(
        ...,
        description="Timestamp of the risk event.",
        example="2023-01-01T12:10:00Z",
    )
    metric: str = Field(
        ...,
        description="Risk metric name, e.g., 'var', 'drawdown'.",
        example="var",
    )
    value: float = Field(
        ...,
        description="Numeric value of the risk metric.",
        example=0.02,
    )
    details: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional context for the risk event.",
    )


# -------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------
def _today_file(category: str) -> Path:
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return ARCHIVE_DIR / f"{category}_{date_str}.jsonl"


def _sync_append(path: str, line: str) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)


def _validate_signal(data: dict) -> bool:
    """
    Apply tighter entry conditions and exit confirmation filters.
    Returns True if the signal passes validation, otherwise False.
    """
    # Basic required fields
    if not _REQUIRED_SIGNAL_FIELDS.issubset(data):
        logger.debug("Signal missing required fields", data=data)
        return False

    confidence = data.get("confidence")
    if not isinstance(confidence, (int, float)) or confidence < _MIN_CONFIDENCE:
        logger.debug("Signal confidence below threshold", confidence=confidence)
        return False

    action = data.get("action")
    if action == "enter":
        # Entry signal: ensure confidence is high and required fields exist
        return True
    elif action == "exit":
        # Exit signal: require additional confirmation fields
        if not _REQUIRED_EXIT_FIELDS.issubset(data):
            logger.debug("Exit signal missing confirmation fields", data=data)
            return False
        return True
    # Other actions are allowed without extra checks
    return True


# -------------------------------------------------------------------------
# Core Archiving Functions
# -------------------------------------------------------------------------
async def archive_event(category: str, data: dict) -> None:
    """
    category: 'orders' | 'fills' | 'signals' | 'decisions' | 'risk'
    Appends a single JSON line to today's file. Atomic (lock‑guarded).
    """
    # Apply signal quality filters only for signal‑related categories
    if category in {"signals", "decisions"}:
        if not _validate_signal(data):
            logger.info("Signal filtered out by validation", category=category, data=data)
            return

    start = time.monotonic()
    record = {"ts": datetime.now(timezone.utc).isoformat(), **data}
    line = json.dumps(record, default=str) + "\n"
    file = _today_file(category)
    try:
        loop = asyncio.get_running_loop()
        async with _lock:
            await loop.run_in_executor(None, _sync_append, str(file), line)

            # Update monitoring metrics
            stats = _stats[category]
            stats["count"] += 1
            pnl = data.get("pnl")
            if isinstance(pnl, (int, float)):
                stats["pnl"] += float(pnl)

        duration = time.monotonic() - start
        logger.info(
            "Archived event",
            category=category,
            count=_stats[category]["count"],
            duration=duration,
            pnl=_stats[category]["pnl"],
        )
    except Exception as e:
        logger.warning("Archive failed", category=category, error=str(e))


def replay(category: str, date_str: str | None = None, limit: int = 1000) -> list[dict]:
    """Read back archived events for a category and date (YYYY-MM-DD)."""
    if date_str is None:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    file = ARCHIVE_DIR / f"{category}_{date_str}.jsonl"
    if not file.exists():
        return []
    out = []
    with open(file, encoding="utf-8") as f:
        for line in f:
            try:
                out.append(json.loads(line))
                if len(out) >= limit:
                    break
            except Exception:
                continue
    return out


def list_archives() -> dict[str, list[str]]:
    """Return {category: [date1, date2, ...]} listing."""
    result: dict[str, list[str]] = {}
    for f in sorted(ARCHIVE_DIR.glob("*.jsonl")):
        category, date_str = f.stem.rsplit("_", 1)
        result.setdefault(category, []).append(date_str)
    return result