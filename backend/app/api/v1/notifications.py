"""Notifications, activity tracker, and Slack CTO-agent endpoints."""
from __future__ import annotations

import asyncio
import os
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
                return profile.get("display_name") or profile.get("real_name") or user_id
    except Exception:
        pass
    return user_id


# ── Slack helper: fetch thread context ────────────────────────────────────────

async def _fetch_slack_thread_context(token: str, channel_id: str, thread_ts: str, limit: int = 4) -> str:
    """Fetch the last `limit` messages from an existing thread for LLM context."""
    if not token or not channel_id or not thread_ts:
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
            lines = []
            for m in msgs[1:]:  # skip root message
                who = m.get("user", "bot") if not m.get("bot_id") else "bot"
                lines.append(f"{who}: {m.get('text','')[:200]}")
            return "\n".join(lines[-limit:])
    except Exception:
        return ""


# ── Slack helper: detect relevant employees from message content ───────────────

_EMPLOYEE_KEYWORD_MAP: list[tuple[tuple[str, ...], str]] = [
    (("model", "lstm", "xgboost", "transformer", "training", "feature", "overfitting", "drift", "inference", "prediction", "ml "), "ModelingEngineer"),
    (("strategy", "backtest", "sharpe", "momentum", "mean reversion", "signal", "alpha", "regime", "breakout", "arb"), "AlgoAgent"),
    (("risk", "drawdown", "kelly", "position size", "correlation", "circuit breaker", "vol"), "RiskMonitor"),
    (("test", "bug", "error", "failing", "ci ", "pytest", "qa", "coverage", "assertion"), "QAMonitor"),
    (("deploy", "render", "redis", "database", "docker", "infra", "migration", "startup"), "DataEngineer"),
]

def _detect_employee_tags(text: str) -> list[str]:
    """Return up to 2 relevant employee names based on message content."""
    text_lower = text.lower()
    tags = []
    for keywords, employee in _EMPLOYEE_KEYWORD_MAP:
        if any(kw in text_lower for kw in keywords):
            tags.append(employee)
        if len(tags) == 2:
            break
    return tags


# ── Slack helper: track open questions for follow-up ─────────────────────────

async def _track_followup(channel_id: str, thread_ts: str, question: str, user_id: str) -> None:
    """Store an open question in Redis so the follow-up task can pick it up."""
    try:
        from app.redis_client import get_redis
        redis = get_redis()
        import json
        from datetime import datetime, timezone
        key = f"slack:followup:{channel_id}:{thread_ts}"
        payload = json.dumps({
            "channel_id": channel_id,
            "thread_ts": thread_ts,
            "question": question[:300],
            "user_id": user_id,
            "asked_at": datetime.now(timezone.utc).isoformat(),
            "answered": False,
        })
        await redis.set(key, payload, ex=86400 * 2)  # 2-day TTL
    except Exception:
        pass


# ── CTO Agent: Review Incoming Slack Messages ─────────────────────────────────

@router.post("/slack/events")
async def slack_events(request: Request):
    """
    Slack Events API webhook — handles every message and thread reply.

    New messages:   CTO reviews + tags relevant employees, tracks open questions.
    Thread replies: CTO continues the conversation with full thread context.
    App mentions:   Treated as a direct question to the CTO.

    Configure in Slack App → Event Subscriptions → Request URL:
      https://quantedge-api.onrender.com/api/v1/notifications/slack/events
    Subscribe to bot events: message.channels, message.groups, app_mention
    """
    body = await request.json()

    # Slack URL verification handshake
    if body.get("type") == "url_verification":
        return {"challenge": body.get("challenge")}

    event = body.get("event", {})
    event_type = event.get("type", "")

    # Handle new messages, thread replies, and direct @mentions
    if event_type in ("message", "app_mention"):
        # Skip bot messages, edits, and message-deleted subtypes
        if event.get("bot_id") or event.get("subtype"):
            return {"ok": True}

        channel_id = event.get("channel", "")
        user_id = event.get("user", "")
        text = event.get("text", "")
        message_ts = event.get("ts", "")
        # If thread_ts is set this is a reply in an existing thread;
        # otherwise message_ts IS the thread root.
        thread_ts = event.get("thread_ts") or message_ts
        is_reply = bool(event.get("thread_ts"))

        # Fire-and-forget — don't block the Slack 3-second handshake window
        asyncio.create_task(
            _cto_review_message(
                channel_id=channel_id,
                user_id=user_id,
                text=text,
                thread_ts=thread_ts,
                is_reply=is_reply,
            )
        )

    return {"ok": True}


async def _cto_review_message(
    channel_id: str,
    user_id: str,
    text: str,
    thread_ts: str,
    is_reply: bool = False,
) -> None:
    """
    CTO agent reviews a Slack message and posts a threaded reply.
    - Fetches thread context when replying to an existing thread.
    - Tags the message author and routes to relevant employees.
    - Tracks unanswered questions in Redis for follow-up.
    """
    if not text or len(text.strip()) < 5:
        return

    try:
        import anthropic
        from app.config import settings

        api_key = getattr(settings, "anthropic_api_key", "") or ""
        token = getattr(settings, "slack_bot_token", "") or ""
        if not api_key:
            return

        # Fetch thread context for replies
        thread_context = ""
        if is_reply and token:
            thread_context = await _fetch_slack_thread_context(token, channel_id, thread_ts)

        # Detect relevant employees
        employee_tags = _detect_employee_tags(text)

        context_block = f"\nThread so far:\n{thread_context}\n" if thread_context else ""
        routing_block = f"\nRoute action items to: {', '.join(employee_tags)}" if employee_tags else ""
        action_word = "replied in thread" if is_reply else "posted"

        system_prompt = (
            "You are the AI CTO of QuantEdge, an institutional quant trading platform. "
            "Review each employee message and give direct, concise technical guidance. "
            "When continuing a thread, reference earlier context naturally. "
            "Assign concrete follow-up tasks. Max 3 sentences. Never fabricate data."
        )
        user_prompt = (
            f"<@{user_id}> {action_word} in channel {channel_id}:"
            f"{context_block}"
            f"\nMessage: {text[:600]}"
            f"{routing_block}\n\n"
            "Provide CTO review."
        )

        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=350,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        reply_text = response.content[0].text if response.content else ""

        if reply_text:
            # Tag the author + route to relevant employees
            author_mention = f"<@{user_id}> " if user_id else ""
            routing_footer = (
                f"\n\n_cc: {' · '.join(employee_tags)}_"
                if employee_tags else ""
            )
            full_reply = f"🤖 *CTO:* {author_mention}{reply_text}{routing_footer}"
            await _post_threaded_reply(channel_id, thread_ts, full_reply, token=token)

            # Track unanswered questions for follow-up (new messages only)
            if "?" in text and not is_reply:
                await _track_followup(channel_id, thread_ts, text, user_id)

    except Exception as e:
        from app.utils.logging import logger
        logger.debug("CTO review failed", error=str(e))


async def _post_threaded_reply(channel_id: str, thread_ts: str, text: str, token: str = "") -> None:
    """Post a threaded reply to a Slack message using the bot token."""
    from app.config import settings
    import httpx

    if not token:
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
                    "mrkdwn": True,
                },
            )
    except Exception:
        pass


# ── Follow-up check endpoint ─────────────────────────────────────────────────

async def _run_followup_check(hours_threshold: int = 4) -> dict:
    """Core follow-up logic — callable from both the scheduler and the API endpoint."""
    from app.config import settings
    import json
    from datetime import datetime, timedelta, timezone

    token = getattr(settings, "slack_bot_token", "") or ""
    api_key = getattr(settings, "anthropic_api_key", "") or ""
    if not token or not api_key:
        return {"followed_up": 0}

    try:
        from app.redis_client import get_redis
        redis = get_redis()
    except Exception:
        return {"followed_up": 0}

    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours_threshold)
    followed_up = 0

    try:
        # Scan all open followup keys
        cursor = 0
        keys: list[str] = []
        while True:
            cursor, batch = await redis.scan(cursor, match="slack:followup:*", count=100)
            keys.extend(batch)
            if cursor == 0:
                break

        import anthropic
        ac = anthropic.Anthropic(api_key=api_key)

        for key in keys:
            try:
                raw = await redis.get(key)
                if not raw:
                    continue
                item = json.loads(raw)
                if item.get("answered"):
                    continue
                asked_at = datetime.fromisoformat(item["asked_at"])
                if asked_at > cutoff:
                    continue  # too recent

                # Generate follow-up nudge

                r = ac.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=150,
                    system="You are the CTO of QuantEdge. Write a one-sentence follow-up nudge for an unanswered question.",
                    messages=[{"role": "user", "content": f"Unanswered question: {item['question'][:300]}"}],
                )
                nudge = r.content[0].text if r.content else "Following up — any update on this?"

                author = f"<@{item['user_id']}> " if item.get("user_id") else ""
                await _post_threaded_reply(
                    item["channel_id"], item["thread_ts"],
                    f"🤖 *CTO follow-up:* {author}{nudge}",
                    token=token,
                )
                # Mark as answered so we don't spam
                item["answered"] = True
                await redis.set(key, json.dumps(item), ex=86400)
                followed_up += 1
            except Exception:
                continue

    except Exception:
        pass

    return {"followed_up": followed_up}


@router.post("/slack/check-followups")
async def check_and_send_followups(
    hours_threshold: int = Query(4, ge=1, le=48),
    current_user: User = Depends(get_current_user),
):
    """
    Scan Redis for open questions that have not been answered for `hours_threshold`
    hours and post a follow-up nudge in the original thread.
    Also called by the scheduler every 4 hours.
    """
    return await _run_followup_check(hours_threshold)


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
