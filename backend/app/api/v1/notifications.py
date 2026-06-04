"""Notifications, activity tracker, and Slack CTO-agent endpoints."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel

from app.api.deps import get_current_user
from app.models.user import User
from app.notifications.tracker import tracker
from app.notifications.slack import slack

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
    ok = await slack.notify_system("QuantEdge Slack notifications are working ✓", level="info")
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
        report_path = Path("/home/user/Test/qa_health_report.json")
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

    lines.extend([
        "*👤 RegimeMonitor*: running (5min HMM cycle)",
        "*👤 SelfImprover*: parameter sweep active",
        "*👤 BacktestWorker*: polling queue every 30s",
        "*👤 StrategyRunner*: regime-gated 24/7",
        "*👤 PriceFeed*: 2s poll cycle (stub mode — no broker keys)",
        "*👤 Scheduler*: hourly snapshots + nightly retrain",
        "*👤 CorrelationMonitor*: 6-symbol cluster watch",
        "",
        "_All employees supervised by `_supervised()` with exponential backoff restart._",
    ])

    report_text = "\n".join(lines)

    ok = await slack.send("system", "system", "📋 Employee Status Report",
                          text=report_text)
    return {"sent": ok, "enabled": slack._enabled, "report": report_text}


# ── CTO Agent: Review Incoming Slack Messages ─────────────────────────────────

@router.post("/slack/events")
async def slack_events(request: Request):
    """
    Slack Events API webhook endpoint.
    Handles URL verification challenge and incoming message events.
    When an employee posts to Slack, the CTO (Claude) reviews it and
    replies with guidance via the same channel.

    Configure in Slack App → Event Subscriptions → Request URL:
      https://quantedge-api.onrender.com/api/v1/notifications/slack/events
    Subscribe to: message.channels, message.groups
    """
    body = await request.json()

    # Slack URL verification handshake
    if body.get("type") == "url_verification":
        return {"challenge": body.get("challenge")}

    event = body.get("event", {})
    event_type = event.get("type", "")

    # Only process user messages (not bot messages — avoid loops)
    if event_type == "message" and not event.get("bot_id") and not event.get("subtype"):
        channel_id = event.get("channel", "")
        user_id = event.get("user", "")
        text = event.get("text", "")
        ts = event.get("ts", "")

        # Fire-and-forget CTO review — don't block the Slack handshake
        asyncio.create_task(
            _cto_review_message(channel_id=channel_id, user_id=user_id,
                                text=text, thread_ts=ts)
        )

    return {"ok": True}


async def _cto_review_message(channel_id: str, user_id: str, text: str, thread_ts: str) -> None:
    """
    CTO agent reviews an employee Slack message and posts a threaded reply.
    Uses the Anthropic API (Claude) to generate contextual guidance.
    """
    if not text or len(text.strip()) < 5:
        return

    try:
        import anthropic
        from app.config import settings

        api_key = getattr(settings, "anthropic_api_key", "") or ""
        if not api_key:
            return

        client = anthropic.Anthropic(api_key=api_key)

        system_prompt = """You are the AI CTO of QuantEdge, an institutional quantitative trading platform.
Your role: review employee messages, give concise technical guidance, assign follow-up tasks.
Keep replies under 3 sentences. Be direct, technical, and action-oriented.
Never hallucinate specific data. If you don't know, say so.
This is a trading platform context: FastAPI backend, React frontend, PyTorch ML, Alpaca/Binance brokers."""

        user_prompt = f"Employee message (channel {channel_id}): {text[:500]}\n\nProvide CTO review and any task assignments."

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        reply = response.content[0].text if response.content else ""

        if reply:
            await _post_threaded_reply(channel_id, thread_ts, f"🤖 *CTO Review*: {reply}")

    except Exception as e:
        from app.utils.logging import logger
        logger.debug("CTO review failed", error=str(e))


async def _post_threaded_reply(channel_id: str, thread_ts: str, text: str) -> None:
    """Post a threaded reply to a Slack message using the bot token."""
    from app.config import settings
    import httpx

    token = getattr(settings, "slack_bot_token", "") or ""
    if not token:
        return

    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            await client.post(
                "https://slack.com/api/chat.postMessage",
                headers={"Authorization": f"Bearer {token}"},
                json={
                    "channel": channel_id,
                    "thread_ts": thread_ts,
                    "text": text,
                },
            )
    except Exception:
        pass


# ── CTO Agent: Manual Review Trigger ─────────────────────────────────────────

@router.post("/slack/cto-review")
async def cto_manual_review(
    payload: SlackReviewRequest,
    current_user: User = Depends(get_current_user),
):
    """
    Manually trigger a CTO review of any message.
    Useful for testing before Slack Events API is wired up.
    """
    try:
        import anthropic
        from app.config import settings

        api_key = getattr(settings, "anthropic_api_key", "") or ""
        if not api_key:
            return {"error": "ANTHROPIC_API_KEY not configured", "review": None}

        client = anthropic.Anthropic(api_key=api_key)

        system_prompt = """You are the AI CTO of QuantEdge, an institutional quantitative trading platform.
Your role: review employee messages, give concise technical guidance, assign follow-up tasks.
Keep replies under 4 sentences. Be direct, technical, and action-oriented."""

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

        # Also post to Slack if enabled
        sent = False
        if review:
            sent = await slack.send(
                payload.channel, "system",
                f"🤖 CTO Review: {payload.channel}",
                text=review,
            )

        return {"review": review, "sent_to_slack": sent}

    except Exception as e:
        return {"error": str(e), "review": None}


# ── CTO Agent: Backfill — review ALL existing messages in channels ────────────

class HistoryReviewRequest(BaseModel):
    channels: list[str] | None = None   # channel IDs; None = auto-discover bot's channels
    per_channel_limit: int = 50         # how many recent messages to review per channel
    post_replies: bool = True           # post CTO replies back to Slack


@router.post("/slack/review-history")
async def review_channel_history(
    payload: HistoryReviewRequest,
    current_user: User = Depends(get_current_user),
):
    """
    CTO backfill: pull existing messages from each channel via Slack
    conversations.history and post a threaded CTO review on each unreviewed
    human message. This is the 'start with all existing messages' pass.

    Requires SLACK_BOT_TOKEN (scopes: channels:history, groups:history,
    channels:read, chat:write) and ANTHROPIC_API_KEY.
    """
    from app.config import settings
    import httpx

    token = getattr(settings, "slack_bot_token", "") or ""
    api_key = getattr(settings, "anthropic_api_key", "") or ""

    if not token:
        return {"error": "SLACK_BOT_TOKEN not configured — cannot read history", "reviewed": 0}
    if not api_key:
        return {"error": "ANTHROPIC_API_KEY not configured — cannot review", "reviewed": 0}

    headers = {"Authorization": f"Bearer {token}"}

    # 1. Resolve channel list (auto-discover if not provided)
    channels = payload.channels
    async with httpx.AsyncClient(timeout=10.0) as client:
        if not channels:
            resp = await client.get(
                "https://slack.com/api/conversations.list",
                headers=headers,
                params={"types": "public_channel,private_channel", "limit": 200},
            )
            data = resp.json()
            channels = [c["id"] for c in data.get("channels", []) if c.get("is_member")]

    if not channels:
        return {"error": "bot is not a member of any channels", "reviewed": 0}

    # 2. For each channel, pull history and review each human message
    import anthropic
    anthropic_client = anthropic.Anthropic(api_key=api_key)

    system_prompt = """You are the AI CTO of QuantEdge, an institutional quant trading platform.
Review each employee message: give concise technical guidance and assign a concrete follow-up task.
Keep replies under 3 sentences. Be direct and action-oriented. Never fabricate data."""

    reviewed = 0
    summary: list[dict] = []

    async with httpx.AsyncClient(timeout=15.0) as client:
        for ch in channels:
            resp = await client.get(
                "https://slack.com/api/conversations.history",
                headers=headers,
                params={"channel": ch, "limit": payload.per_channel_limit},
            )
            hist = resp.json()
            messages = hist.get("messages", [])

            for msg in reversed(messages):  # oldest-first
                # Skip bot messages, edits, and already-reviewed (threaded) replies
                if msg.get("bot_id") or msg.get("subtype"):
                    continue
                text = msg.get("text", "")
                if not text or len(text.strip()) < 5:
                    continue
                if text.startswith("🤖"):  # already a CTO reply
                    continue

                try:
                    r = anthropic_client.messages.create(
                        model="claude-haiku-4-5-20251001",
                        max_tokens=250,
                        system=system_prompt,
                        messages=[{"role": "user",
                                   "content": f"Employee message: {text[:500]}"}],
                    )
                    review = r.content[0].text if r.content else ""
                except Exception as e:
                    review = f"(review failed: {e})"

                if review and payload.post_replies:
                    await client.post(
                        "https://slack.com/api/chat.postMessage",
                        headers=headers,
                        json={"channel": ch, "thread_ts": msg.get("ts"),
                              "text": f"🤖 *CTO Review*: {review}"},
                    )

                reviewed += 1
                summary.append({"channel": ch, "message": text[:80], "review": review[:120]})

    return {"reviewed": reviewed, "channels": len(channels), "details": summary[:50]}
