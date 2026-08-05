"""Notifications, activity tracker, and the CTO review endpoint.

Discord-only. The Slack Events receiver, thread helpers, follow-up tracker and
conversations.history backfill were removed 2026-07-25 together with the rest of
the Slack integration: they were bound to Slack's Events API / thread model and
had been dead since the free-plan quota expired. The equivalent Discord flows
live in .github/scripts (channel_monitor, multi_agent_discussion).
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import structlog
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from app.api.deps import get_current_user
from app.models.user import User
from app.notifications.tracker import tracker
from app.notifications.discord import discord

# ── Constants ─────────────────────────────────────────────────────────────
DEFAULT_ACTIVITY_LIMIT = 100
MAX_ACTIVITY_LIMIT = 500

DISCORD_TEST_MESSAGE = "QuantEdge Discord notifications are working ✓"

DISCORD_SYSTEM_CHANNEL = "system"
DISCORD_SYSTEM_USERNAME = "system"
DISCORD_EMPLOYEE_REPORT_TITLE = "📋 Employee Status Report"

EMPLOYEE_STATUS_STATIC_LINES = [
    "*👤 RegimeMonitor*: running (5min HMM cycle)",
    "*👤 SelfImprover*: parameter sweep active",
    "*👤 BacktestWorker*: polling queue every 30s",
    "*👤 StrategyRunner*: regime-gated 24/7",
    "*👤 PriceFeed*: 2s poll cycle (stub mode — no broker keys)",
    "*👤 Scheduler*: hourly snapshots + nightly retrain",
    "*👤 CorrelationMonitor*: 6-symbol cluster watch",
]

EMPLOYEE_STATUS_FOOTER = "_All employees supervised by `_supervised()` with exponential backoff restart._"

CLAUDE_MODEL = "claude-haiku-4-5-20251001"
CLAUDE_MAX_TOKENS = 400
CTA_SYSTEM_PROMPT = """You are the AI CTO of QuantEdge, an institutional quantitative trading platform.
Your role: review employee messages, give concise technical guidance, assign follow-up tasks.
Keep replies under 4 sentences. Be direct, technical, and action-oriented."""

ANTHROPIC_API_KEY_ERROR = "ANTHROPIC_API_KEY not configured"

QA_HEALTH_REPORT_ENV = "QA_HEALTH_REPORT_PATH"
DEFAULT_QA_REPORT_FILE = "qa_health_report.json"

# ── Router ────────────────────────────────────────────────────────────────
router = APIRouter(prefix="/notifications", tags=["notifications"])

# ── Models ───────────────────────────────────────────────────────────────

class ReviewRequest(BaseModel):
    channel: str
    message: str
    context: str | None = None

# ── Standard endpoints ────────────────────────────────────────────────────────

@router.get("/activity")
async def get_activity(
    limit: int = Query(DEFAULT_ACTIVITY_LIMIT, le=MAX_ACTIVITY_LIMIT),
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
    ok = await discord.notify_system(DISCORD_TEST_MESSAGE, level="info")
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
        lb = algo.get_leaderboard()
        best = lb[0] if lb else {}
        lines.append(f"*👤 AlgoAgent*: {algo._total_runs} runs | top: {best.get('strategy','?')}:{best.get('symbol','?')} sharpe={best.get('avg_sharpe',0):.3f}")
    else:
        lines.append("*👤 AlgoAgent*: status unavailable")

    # ResearchScientist
    if research:
        summary = research.get_research_summary()
        top = summary.get("top_ideas", [{}])
        t = top[0] if top else {}
        lines.append(f"*👤 ResearchScientist*: {summary.get('cycles_completed',0)} cycles | top: {t.get('topic','?')} (sharpe≈{t.get('estimated_sharpe','?')})")
    else:
        lines.append("*👤 ResearchScientist*: status unavailable")

    # ModelingEngineer
    if modeling:
        eng = modeling.get_engineering_summary()
        promoted = eng.get("promote_count", 0)
        drifted = sum(1 for v in eng.get("latest_performance", {}).values() if isinstance(v, dict) and v.get("drift_detected"))
        lines.append(f"*👤 ModelingEngineer*: {eng.get('cycles_completed',0)} cycles | {promoted} promotions | {drifted} models drifting")
    else:
        lines.append("*👤 ModelingEngineer*: status unavailable")

    # QAMonitor (read from health report file)
    try:
        import json
        from pathlib import Path
        report_path = Path(os.getenv(QA_HEALTH_REPORT_ENV, DEFAULT_QA_REPORT_FILE))
        if report_path.exists():
            rpt = json.loads(report_path.read_text())
            passed = rpt.get("tests_passed", 0)
            total = rpt.get("tests_total", 0)
            fixes = rpt.get("auto_fixes_applied", 0)
            status = rpt.get("overall_status", "unknown")
            lines.append(f"*👤 QAMonitor*: {passed}/{total} tests ✅ | {fixes} auto-fixes | status={status}")
        else:
            lines.append("*👤 QAMonitor*: no report yet")
    except Exception:
        lines.append("*👤 QAMonitor*: report unavailable")

    lines.extend(EMPLOYEE_STATUS_STATIC_LINES + ["", EMPLOYEE_STATUS_FOOTER])

    report_text = "\n".join(lines)

    ok = await discord.send(
        DISCORD_SYSTEM_CHANNEL,
        DISCORD_SYSTEM_USERNAME,
        DISCORD_EMPLOYEE_REPORT_TITLE,
        text=report_text,
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
            return {"error": ANTHROPIC_API_KEY_ERROR, "review": None}

        client = anthropic.Anthropic(api_key=api_key)

        content = payload.message
        if payload.context:
            content = f"Context: {payload.context}\n\nMessage: {payload.message}"

        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=CLAUDE_MAX_TOKENS,
            system=CTA_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": content}],
        )
        review = response.content[0].text if response.content else ""

        # Post the review to Discord
        sent = False
        if review:
            sent = await discord.send(
                payload.channel,
                DISCORD_SYSTEM_USERNAME,
                f"🤖 CTO Review: {payload.channel}",
                text=review,
            )

        return {"review": review, "sent_to_discord": sent}

    except Exception as e:
        return {"error": str(e), "review": None}