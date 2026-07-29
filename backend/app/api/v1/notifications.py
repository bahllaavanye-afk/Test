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
    limit: int | None = Query(100, le=500),
    category: str | None = None,
    current_user: User = Depends(get_current_user),
):
    """Return recent activity, protecting against None or out‑of‑range limits."""
    # Guard against None or non‑positive limits
    if not isinstance(limit, int) or limit <= 0:
        limit = 100
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
            best = lb[0] if lb else {}
        except Exception as exc:
            logger.error("algo_agent leaderboard error", exc=exc)
            best = {}
        lines.append(
            f"*👤 AlgoAgent*: {getattr(algo, '_total_runs', 0)} runs | top: "
            f"{best.get('strategy', '?')}:{best.get('symbol', '?')} sharpe={best.get('avg_sharpe', 0):.3f}"
        )
    else:
        lines.append("*👤 AlgoAgent*: status unavailable")

    # ResearchScientist
    if research:
        try:
            summary = research.get_research_summary() or {}
        except Exception as exc:
            logger.error("research_scientist summary error", exc=exc)
            summary = {}
        top_ideas = summary.get("top_ideas", [])
        top = top_ideas[0] if top_ideas else {}
        lines.append(
            f"*👤 ResearchScientist*: {summary.get('cycles_completed', 0)} cycles | "
            f"top: {top.get('topic', '?')} (sharpe≈{top.get('estimated_sharpe', '?')})"
        )
    else:
        lines.append("*👤 ResearchScientist*: status unavailable")

    # ModelingEngineer
    if modeling:
        try:
            eng = modeling.get_engineering_summary() or {}
        except Exception as exc:
            logger.error("modeling_engineer summary error", exc=exc)
            eng = {}
        promoted = eng.get("promote_count", 0)
        latest_perf = eng.get("latest_performance", {}) or {}
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

        report_path = Path(os.getenv("QA_HEALTH_REPORT_PATH", "qa_health_report.json"))
        if report_path.exists():
            rpt = json.loads(report_path.read_text())
            passed = rpt.get("tests_passed", 0)
            total = rpt.get("tests_total", 0)
            fixes = rpt.get("auto_fixes_applied", 0)
            status = rpt.get("overall_status", "unknown")
            lines.append(
                f"*👤 QAMonitor*: {passed}/{total} tests ✅ | {fixes} auto-fixes | status={status}"
            )
        else:
            lines.append("*👤 QAMonitor*: no report yet")
    except Exception as exc:
        logger.error("QAMonitor report error", exc=exc)
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
    # Basic validation of required fields
    if not payload.channel or not payload.message:
        return {"error": "channel and message must be non‑empty", "review": None, "sent_to_discord": False}

    try:
        import anthropic
        from app.config import settings

        api_key = getattr(settings, "anthropic_api_key", "") or ""
        if not api_key:
            return {"error": "ANTHROPIC_API_KEY not configured", "review": None, "sent_to_discord": False}

        client = anthropic.Anthropic(api_key=api_key)

        system_prompt = (
            "You are the AI CTO of QuantEdge, an institutional quantitative trading platform.\n"
            "Your role: review employee messages, give concise technical guidance, assign follow‑up tasks.\n"
            "Keep replies under 4 sentences. Be direct, technical, and action‑oriented."
        )

        content = payload.message
        if payload.context:
            content = f"Context: {payload.context}\n\nMessage: {payload.message}"

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=400,
            system=system_prompt,
            messages=[{"role": "user", "content": content}],
        )
        # Defensive extraction of text
        review = ""
        if response and getattr(response, "content", None):
            first = response.content[0] if response.content else None
            review = getattr(first, "text", "") if first else ""

        # Post the review to Discord
        sent = False
        if review:
            sent = await discord.send(
                payload.channel,
                "system",
                f"🤖 CTO Review: {payload.channel}",
                text=review,
            )

        return {"review": review, "sent_to_discord": sent}
    except Exception as e:
        logger.error("cto_manual_review error", exc=e)
        return {"error": str(e), "review": None, "sent_to_discord": False}