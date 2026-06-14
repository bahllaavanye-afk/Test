"""
Centralized agent activity logger.
Call log_action() from any background task, strategy runner, ML trainer, etc.
Writes to DB and broadcasts to WebSocket subscribers.
"""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

from app.utils.logging import logger as _logger


class AgentLogger:
    """Singleton logger — access via `from app.services.agent_logger import agent_logger`"""

    async def log_action(
        self,
        action: str,
        employee_id: str = "system",
        agent_type: str = "system",
        tool_used: str | None = None,
        input_summary: str | None = None,
        output_summary: str | None = None,
        duration_ms: int | None = None,
        status: str = "ok",
        error_message: str | None = None,
        strategy_name: str | None = None,
        symbol: str | None = None,
        account_id: str | None = None,
        extra: dict | None = None,
    ) -> None:
        """Write a log entry to DB and broadcast to WebSocket subscribers."""
        # Compute heuristic anomaly score
        score = 0.0
        if status == "error":
            score += 0.8
        elif status == "warning":
            score += 0.3
        if duration_ms and duration_ms > 30_000:
            score += 0.4
        score = min(score, 1.0)
        is_anomaly = score >= 0.6

        try:
            from app.database import AsyncSessionLocal
            from app.models.agent_log import AgentActivityLog
            from app.ws.manager import manager

            entry = AgentActivityLog(
                employee_id=employee_id,
                agent_type=agent_type,
                action=action,
                tool_used=tool_used,
                input_summary=input_summary[:200] if input_summary else None,
                output_summary=output_summary[:200] if output_summary else None,
                duration_ms=duration_ms,
                status=status,
                error_message=error_message,
                anomaly_score=score,
                is_anomaly=is_anomaly,
                strategy_name=strategy_name,
                symbol=symbol,
                account_id=account_id,
                extra=extra,
            )

            async with AsyncSessionLocal() as db:
                db.add(entry)
                await db.commit()
                await db.refresh(entry)

            # Broadcast to all WebSocket subscribers on the agent-logs channel
            payload = {
                "type": "agent_log",
                "id": entry.id,
                "employee_id": entry.employee_id,
                "agent_type": entry.agent_type,
                "action": entry.action,
                "tool_used": entry.tool_used,
                "input_summary": entry.input_summary,
                "output_summary": entry.output_summary,
                "duration_ms": entry.duration_ms,
                "status": entry.status,
                "error_message": entry.error_message,
                "anomaly_score": entry.anomaly_score,
                "is_anomaly": entry.is_anomaly,
                "strategy_name": entry.strategy_name,
                "symbol": entry.symbol,
                "account_id": entry.account_id,
                "created_at": entry.created_at.isoformat() if entry.created_at else None,
            }
            await manager.broadcast("agent_logs", payload)

        except Exception as exc:
            _logger.warning("AgentLogger: failed to write log entry", error=str(exc))

    def log_action_fire_and_forget(
        self,
        action: str,
        employee_id: str = "system",
        agent_type: str = "system",
        **kwargs: Any,
    ) -> None:
        """
        Schedule log_action as a background task in the current event loop.
        Safe to call from sync or async context; silently drops if no loop is running.
        """
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(
                self.log_action(
                    action=action,
                    employee_id=employee_id,
                    agent_type=agent_type,
                    **kwargs,
                )
            )
        except RuntimeError:
            # No running event loop — silently skip (e.g. during tests)
            pass


agent_logger = AgentLogger()
