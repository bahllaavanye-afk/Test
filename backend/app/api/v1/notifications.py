"""Notifications, activity tracker, and Slack CTO-agent endpoints."""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import os
import time
from datetime import datetime, timezone
from typing import Any, List

import structlog

logger = structlog.get_logger()

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel

from app.api.deps import get_current_user
from app.models.user import User
from app.notifications.tracker import tracker
from app.notifications.slack import slack


def _verify_slack_signature(
    raw_body: bytes, timestamp: str, signature: str, secret: str
) -> bool:
    """Verify a Slack request signature (X-Slack-Signature) with replay protection.

    Restores a Codex P1 security finding that regressed: without this, anyone can POST
    forged events to /slack/events. https://api.slack.com/authentication/verifying-requests
    """
    if not (secret and signature and timestamp):
        return False
    try:
        if abs(time.time() - int(timestamp)) > 60 * 5:  # reject stale (replay) requests
            return False
    except ValueError:
        return False
    basestring = b"v0:" + timestamp.encode() + b":" + raw_body
    expected = "v0=" + hmac.new(secret.encode(), basestring, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


router = APIRouter(prefix="/notifications", tags=["notifications"])


# ── Models ────────────────────────────────────────────────────────────────────

class SlackEventPayload(BaseModel):
    type: str
    event: dict | None = None
    challenge: str | None = None


class SlackReviewRequest(BaseModel):
    channel: str
    message: str
    context: str | None = None


# ── Standard endpoints ────────────────────────────────────────────────────────

@router.get("/activity")
async def get_activity(
    limit: int = Query(100, le=500),
    category: str | None = None,
    current_user: User = Depends(get_current_user),
):
    return tracker.recent(limit=limit, category=category)


@router.get("/stats")
async def get_stats(current_user: User = Depends(get_current_user)):
    return tracker.stats()


@router.post("/slack/test")
async def slack_test(current_user: User = Depends(get_current_user)):
    """Send a test message to confirm Slack webhook is configured."""
    ok = await slack.notify_system(
        "QuantEdge Slack notifications are working ✓", level="info"
    )
    return {"sent": ok, "enabled": slack._enabled}


# ── CTO Agent: Employee Status Broadcast ──────────────────────────────────────

@router.post("/slack/employee-report")
async def post_employee_report(current_user: User = Depends(get_current_user)):
    """
    Post full employee status report to Slack #engineering channel.
    Called by the scheduler every hour.
    """
    try:
        from app.main import app as _app
        algo = getattr(_app.state, "algo_agent", None)
        research = getattr(_app.state, "research_scientist", None)
        modeling = getattr(_app.state, "modeling_engineer", None)
        regime = getattr(_app.state, "regime_monitor", None)
    except Exception:
        algo = research = modeling = regime = None

    lines: List[str] = [
        f"*QuantEdge Employee Status Report* — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
    ]

    # AlgoAgent
    if algo:
        try:
            lb = algo.get_leaderboard() or []
            if not isinstance(lb, list):
                lb = list(lb)
        except Exception:
            lb = []
        best = lb[0] if lb else {}
        lines.append(
            f"*👤 AlgoAgent*: {getattr(algo, '_total_runs', '?')} runs | top: "
            f"{best.get('strategy', '?')}:{best.get('symbol', '?')} sharpe={best.get('avg_sharpe', 0):.3f}"
        )
    else:
        lines.append("*👤 AlgoAgent*: status unavailable")

    # ResearchScientist
    if research:
        try:
            summary = research.get_research_summary() or {}
            if not isinstance(summary, dict):
                summary = {}
        except Exception:
            summary = {}
        top = summary.get("top_ideas", [{}])
        if not isinstance(top, list):
            top = [top] if top else []
        t = top[0] if top else {}
        lines.append(
            f"*👤 ResearchScientist*: {summary.get('cycles_completed', 0)} cycles | top: "
            f"{t.get('topic', '?')} (sharpe≈{t.get('estimated_sharpe', '?')})"
        )
    else:
        lines.append("*👤 ResearchScientist*: status unavailable")

    # ModelingEngineer
    if modeling:
        try:
            eng = modeling.get_engineering_summary() or {}
            if not isinstance(eng, dict):
                eng = {}
        except Exception:
            eng = {}
        promoted = eng.get("promote_count", 0)
        latest_perf = eng.get("latest_performance", {})
        if not isinstance(latest_perf, dict):
            latest_perf = {}
        drifted = sum(
            1
            for v in latest_perf.values()
            if isinstance(v, dict) and v.get("drift_detected")
        )
        lines.append(
            f"*👤 ModelingEngineer*: {eng.get('cycles_completed', 0)} cycles | "
            f"{promoted} promotions | {drifted} models drifting"
        )
    else:
        lines.append("*👤 ModelingEngineer*: status unavailable")

    # QAMonitor (read from health report file)
    try:
        import json
        from pathlib import Path

        report_path = Path(
            os.getenv("QA_HEALTH_REPORT_PATH", "qa_health_report.json")
        )
        if report_path.exists():
            try:
                rpt = json.loads(report_path.read_text())
                if not isinstance(rpt, dict):
                    raise ValueError("Report JSON is not an object")
            except Exception:
                rpt = {}
            passed = rpt.get("tests_passed", 0)
            total = rpt.get("tests_total", 0)
            fixes = rpt.get("auto_fixes_applied", 0)
            status = rpt.get("overall_status", "unknown")
            lines.append(
                f"*👤 QAMonitor*: {passed}/{total} tests ✅ | {fixes} auto-fixes | status={status}"
            )
        else:
            lines.append("*👤 QAMonitor*: no report yet")
    except Exception:
        lines.append("*👤 QAMonitor*: report unavailable")

    lines.extend(
        [
            "*👤 RegimeMonitor*: running (5min HMM cycle)",
            "*👤 SelfImprover*: parameter sweep active",
            "*👤 BacktestWorker*: polling queue every 30s",
            "*👤 StrategyRunner*: regime-gated 24/7",
            "*👤 PriceFeed*: 2s poll cycle (stub mode — no broker keys)",
            "*👤 Scheduler*: hourly snapshots + nightly retrain",
            "*👤 CorrelationMonitor*: 6-symbol cluster watch",
            "",
            "_All employees supervised by `_supervised()` with exponential backoff restart._",
        ]
    )

    report_text = "\n".join(lines)

    ok = await slack.send(
        "system", "system", "📋 Employee Status Report", text=report_text
    )
    return {"sent": ok, "enabled": slack._enabled, "report": report_text}


# ── Slack helper: resolve username from user_id ────────────────────────────────

async def _resolve_slack_username(token: str, user_id: str) -> str:
    """Return display name for a Slack user ID, or the raw ID if lookup fails."""
    if not user_id or not token:
        return user_id or "unknown"
    try:
        import httpx
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                "https://slack.com/api/users.info",
                headers={"Authorization": f"Bearer {token}"},
                params={"user": user_id},
            )
            data = resp.json()
            if data.get("ok"):
                profile = data.get("user", {}).get("profile", {})
                return (
                    profile.get("display_name")
                    or profile.get("real_name")
                    or user_id
                )
    except Exception as _e:
        logger.debug(
            "slack username resolve failed", user_id=user_id, error=str(_e)
        )
    return user_id


# ── Slack helper: fetch thread context ────────────────────────────────────────

async def _fetch_slack_thread_context(
    token: str, channel_id: str, thread_ts: str, limit: int = 4
) -> str:
    """Fetch the last `limit` messages from an existing thread for LLM context."""
    if not token or not channel_id or not thread_ts:
        return ""
    if limit <= 0:
        return ""
    try:
        import httpx
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(
                "https://slack.com/api/conversations.replies",
                headers={"Authorization": f"Bearer {token}"},
                params={"channel": channel_id, "ts": thread_ts, "limit": limit + 1},
            )
            data = resp.json()
            if not data.get("ok"):
                return ""
            msgs = data.get("messages", [])
            if not isinstance(msgs, list):
                return ""
            # Exclude the original thread message (identified by matching ts)
            filtered = [
                m
                for m in msgs
                if isinstance(m, dict) and m.get("ts") != thread_ts
            ]
            # Keep only the most recent `limit` messages
            recent = filtered[-limit:] if limit < len(filtered) else filtered
            # Concatenate the text of each message, safely handling missing fields
            parts = [
                (m.get("text") or "").strip()
                for m in recent
                if isinstance(m, dict)
            ]
            return "\n".join(part for part in parts if part)
    except Exception as _e:
        logger.debug(
            "fetch slack thread context failed",
            channel_id=channel_id,
            thread_ts=thread_ts,
            error=str(_e),
        )
    return ""