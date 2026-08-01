"""Notifications, activity tracker, and the CTO review endpoint.

Discord-only. The Slack Events receiver, thread helpers, follow‑up tracker and
conversations.history backfill were removed 2026-07-25 together with the rest of
the Slack integration: they were bound to Slack's Events API / thread model and
had been dead since the free‑plan quota expired. The equivalent Discord flows
live in .github/scripts (channel_monitor, multi_agent_discussion).
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger()

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from app.api.deps import get_current_user
from app.models.user import User
from app.notifications.tracker import tracker
from app.notifications.discord import discord

router = APIRouter(prefix="/notifications", tags=["notifications"])

# ── Models ───────────────────────────────────────────────────────────────

class ReviewRequest(BaseModel):
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


@router.post("/discord/test")
async def discord_test(current_user: User = Depends(get_current_user)):
    """Send a test message to confirm Discord delivery is configured."""
    ok = await discord.notify_system(
        "QuantEdge Discord notifications are working ✓", level="info"
    )
    return {"sent": ok, "enabled": discord._enabled}


# ── CTO Agent: Employee Status Broadcast ──────────────────────────────────────

@router.post("/discord/employee-report")
async def post_employee_report(current_user: User = Depends(get_current_user)):
    """
    Post the full employee status report to the Discord #engineering channel.
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

    lines: list[str] = [
        f"*QuantEdge Employee Status Report* — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
    ]

    # AlgoAgent
    if algo:
        try:
            lb = algo.get_leaderboard() or []
            best = lb[0] if isinstance(lb, (list, tuple)) and len(lb) > 0 else {}
        except Exception:
            best = {}
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
            top = summary.get("top_ideas", [{}])
            t = top[0] if isinstance(top, (list, tuple)) and len(top) > 0 else {}
        except Exception:
            summary = {}
            t = {}
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
            promoted = eng.get("promote_count", 0)
            latest_perf = eng.get("latest_performance", {})
            drifted = sum(
                1
                for v in latest_perf.values()
                if isinstance(v, dict) and v.get("drift_detected")
            )
        except Exception:
            eng = {}
            promoted = drifted = 0
        lines.append(
            f"*👤 ModelingEngineer*: {eng.get('cycles_completed', 0)} cycles | "
            f"{promoted} promotions | {drifted} models drifting"
        )
    else:
        lines.append("*👤 ModelingEngineer*: status unavailable")

    # QAMonitor (read from health report file)
    try:
        report_path = Path(os.getenv("QA_HEALTH_REPORT_PATH", "qa_health_report.json"))
        if report_path.exists():
            try:
                rpt = json.loads(report_path.read_text())
            except json.JSONDecodeError:
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

    ok = await discord.send(
        "system", "system", "📋 Employee Status Report", text=report_text
    )
    return {"sent": ok, "enabled": discord._enabled, "report": report_text}


# ── CTO Agent: Manual Review Trigger ─────────────────────────────────────────

@router.post("/discord/cto-review")
async def cto_manual_review(
    payload: ReviewRequest,
    current_user: User = Depends(get_current_user),
):
    """
    Manually trigger a CTO review of any message.
    Reviews the given message with Claude and posts the result to Discord.
    """
    try:
        import anthropic
        from app.config import settings

        api_key = getattr(settings, "anthropic_api_key", "") or ""
        if not api_key:
            return {"error": "ANTHROPIC_API_KEY not configured", "review": None}

        client = anthropic.Anthropic(api_key=api_key)

        system_prompt = (
            """You are the AI CTO of QuantEdge, an institutional quantitative trading platform.
Your role: review employee messages, give concise technical guidance, assign follow-up tasks.
Keep replies under 4 sentences. Be direct, technical, and action-oriented."""
        )

        # Guard against missing message content
        if not payload.message:
            return {"error": "Message payload is empty", "review": None}

        content = payload.message
        if payload.context:
            content = f"Context: {payload.context}\n\nMessage: {payload.message}"

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=400,
            system=system_prompt,
            messages=[{"role": "user", "content": content}],
        )
        review = response.content[0].text if getattr(response, "content", None) else ""

        # Post the review to Discord
        sent = False
        if review:
            target_channel = payload.channel or "general"
            sent = await discord.send(
                target_channel,
                "system",
                f"🤖 CTO Review: {target_channel}",
                text=review,
            )

        return {"review": review, "sent_to_discord": sent}

    except Exception as e:
        return {"error": str(e), "review": None}