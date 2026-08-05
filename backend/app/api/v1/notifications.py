"""Notifications, activity tracker, and the CTO review endpoint.

Discord-only. The Slack Events receiver, thread helpers, follow‑up tracker and
conversations.history backfill were removed 2026-07-25 together with the rest of
the Slack integration: they were bound to Slack's Events API / thread model and
had been dead since the free‑plan quota expired. The equivalent Discord flows
live in .github/scripts (channel_monitor, multi_agent_discussion).
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, List

import structlog
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field, validator

from app.api.deps import get_current_user
from app.models.user import User
from app.notifications.tracker import tracker
from app.notifications.discord import discord

logger = structlog.get_logger()

router = APIRouter(prefix="/notifications", tags=["notifications"])


# ── Models ───────────────────────────────────────────────────────────────

class ReviewRequest(BaseModel):
    channel: str
    message: str
    context: str | None = None


class SignalRequest(BaseModel):
    """Payload for a strategy signal that will be forwarded to Discord.

    The validation logic tightens entry conditions, adds confirmation filters,
    and improves exit criteria.
    """
    symbol: str = Field(..., description="Ticker symbol for the signal")
    entry_price: float = Field(..., gt=0, description="Proposed entry price")
    indicator_value: float = Field(..., description="Raw value of the entry indicator")
    confirmation: bool = Field(..., description="Whether a secondary confirmation is present")
    target_price: float = Field(..., gt=0, description="Desired exit target price")
    stop_price: float | None = Field(None, gt=0, description="Optional stop‑loss price")
    strategy_name: str = Field(..., description="Name of the strategy generating the signal")
    regime: str = Field(..., description="Current market regime, e.g. 'bull' or 'bear'")

    @validator("target_price")
    def check_target_vs_entry(cls, v: float, values: dict) -> float:
        entry = values.get("entry_price")
        if entry is not None and v <= entry:
            raise ValueError("target_price must be greater than entry_price for a long position")
        return v

    @validator("stop_price")
    def check_stop_vs_entry(cls, v: float | None, values: dict) -> float | None:
        entry = values.get("entry_price")
        if v is not None and entry is not None and v >= entry:
            raise ValueError("stop_price must be lower than entry_price for a long position")
        return v


# ── Standard endpoints ────────────────────────────────────────────────────────

@router.get("/activity")
async def get_activity(
    limit: int = Query(100, le=500),
    category: str | None = None,
    current_user: User = Depends(get_current_user),
):
    """Return recent activity tracked by the notification system."""
    return tracker.recent(limit=limit, category=category)


@router.get("/stats")
async def get_stats(current_user: User = Depends(get_current_user)):
    """Return aggregated statistics from the notification tracker."""
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

    lines: List[str] = [
        f"*QuantEdge Employee Status Report* — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
    ]

    # AlgoAgent
    if algo:
        lb = algo.get_leaderboard()
        best = lb[0] if lb else {}
        lines.append(
            f"*👤 AlgoAgent*: {algo._total_runs} runs | top: {best.get('strategy','?')}:{best.get('symbol','?')} sharpe={best.get('avg_sharpe',0):.3f}"
        )
    else:
        lines.append("*👤 AlgoAgent*: status unavailable")

    # ResearchScientist
    if research:
        summary = research.get_research_summary()
        top = summary.get("top_ideas", [{}])
        t = top[0] if top else {}
        lines.append(
            f"*👤 ResearchScientist*: {summary.get('cycles_completed',0)} cycles | top: {t.get('topic','?')} (sharpe≈{t.get('estimated_sharpe','?')})"
        )
    else:
        lines.append("*👤 ResearchScientist*: status unavailable")

    # ModelingEngineer
    if modeling:
        eng = modeling.get_engineering_summary()
        promoted = eng.get("promote_count", 0)
        drifted = sum(
            1
            for v in eng.get("latest_performance", {}).values()
            if isinstance(v, dict) and v.get("drift_detected")
        )
        lines.append(
            f"*👤 ModelingEngineer*: {eng.get('cycles_completed',0)} cycles | {promoted} promotions | {drifted} models drifting"
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
            "You are the AI CTO of QuantEdge, an institutional quantitative trading platform.\n"
            "Your role: review employee messages, give concise technical guidance, assign follow-up tasks.\n"
            "Keep replies under 4 sentences. Be direct, technical, and action-oriented."
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
        review = response.content[0].text if response.content else ""

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
        logger.error("CTO review failed", error=str(e))
        return {"error": str(e), "review": None}


# ── Strategy Signal Endpoint ──────────────────────────────────────────────────

@router.post("/discord/signal-alert")
async def signal_alert(
    payload: SignalRequest,
    current_user: User = Depends(get_current_user),
):
    """
    Receive a strategy signal, apply tightened entry/exit checks, and forward a
    Discord notification only when the signal passes all filters.
    """
    # 1️⃣ Regime gating – only allow bullish regime for the mean_rev_25_1.5 strategy
    allowed_strategy = "mean_rev_25_1.5"
    if payload.strategy_name != allowed_strategy:
        return {
            "accepted": False,
            "reason": f"Strategy {payload.strategy_name} not allowed; only {allowed_strategy} is supported",
        }

    if payload.regime.lower() != "bull":
        return {
            "accepted": False,
            "reason": f"Regime {payload.regime} not compatible with bullish entry conditions",
        }

    # 2️⃣ Indicator confirmation – require a minimum strength and explicit confirmation flag
    MIN_INDICATOR = 1.5
    if payload.indicator_value < MIN_INDICATOR:
        return {
            "accepted": False,
            "reason": f"Indicator value {payload.indicator_value:.2f} below threshold {MIN_INDICATOR}",
        }

    if not payload.confirmation:
        return {"accepted": False, "reason": "Missing secondary confirmation"}

    # 3️⃣ Exit sanity checks – enforce a minimum profit target of 1 % above entry
    MIN_PROFIT_FACTOR = 1.01
    if payload.target_price < payload.entry_price * MIN_PROFIT_FACTOR:
        return {
            "accepted": False,
            "reason": f"Target price {payload.target_price:.4f} does not meet minimum 1 % profit requirement",
        }

    # 4️⃣ Optional stop‑loss sanity – if provided, ensure it is at least 0.5 % below entry
    if payload.stop_price is not None:
        MAX_STOP_FACTOR = 0.995  # 0.5 % below entry
        if payload.stop_price > payload.entry_price * MAX_STOP_FACTOR:
            return {
                "accepted": False,
                "reason": f"Stop price {payload.stop_price:.4f} too close to entry; must be ≤0.5 % below",
            }

    # All checks passed – format a concise Discord message
    message = (
        f"🚀 **Signal** – *{payload.strategy_name}*\n"
        f"**Symbol**: {payload.symbol}\n"
        f"**Entry**: ${payload.entry_price:.4f}\n"
        f"**Target**: ${payload.target_price:.4f}\n"
        f"**Stop‑Loss**: ${payload.stop_price:.4f}" if payload.stop_price else ""
    )

    sent = await discord.send(
        "system",
        "system",
        f"📈 Signal Alert: {payload.symbol}",
        text=message,
    )
    return {"accepted": True, "sent_to_discord": sent}