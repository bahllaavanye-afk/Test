"""
Slack slash-command handler for /qe commands.
POST /slack/commands — validated with SLACK_SIGNING_SECRET.
"""
import hashlib
import hmac
import json
import os
import time
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/slack", tags=["slack"])

SLACK_SIGNING_SECRET = os.environ.get("SLACK_SIGNING_SECRET", "")


def _verify_slack_signature(request_body: bytes, timestamp: str, signature: str) -> bool:
    if not SLACK_SIGNING_SECRET:
        return True  # Skip verification in dev (no secret configured)
    if abs(time.time() - int(timestamp)) > 300:
        return False
    sig_base = f"v0:{timestamp}:{request_body.decode()}"
    expected = "v0=" + hmac.new(
        SLACK_SIGNING_SECRET.encode(), sig_base.encode(), hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature)


def _block_text(text: str) -> dict:
    return {"type": "section", "text": {"type": "mrkdwn", "text": text}}


@router.post("/commands")
async def slack_command(request: Request):
    body = await request.body()
    timestamp = request.headers.get("X-Slack-Request-Timestamp", "0")
    signature = request.headers.get("X-Slack-Signature", "")
    if not _verify_slack_signature(body, timestamp, signature):
        raise HTTPException(status_code=403, detail="Invalid Slack signature")

    form = await request.form()
    command_text = str(form.get("text", "")).strip()
    parts = command_text.split()
    sub = parts[0].lower() if parts else "status"

    blocks = []
    if sub == "status":
        blocks = await _status_blocks()
    elif sub == "risk":
        blocks = await _risk_blocks()
    elif sub == "signal" and len(parts) > 1:
        blocks = await _signal_blocks(parts[1].upper())
    elif sub == "audit":
        blocks = await _audit_blocks()
    elif sub == "help":
        blocks = [_block_text(
            "*QuantEdge Slack Commands*\n"
            "• `/qe status` — live trading mode + system health\n"
            "• `/qe risk` — real circuit-breaker + drawdown state\n"
            "• `/qe signal <SYMBOL>` — live ML ensemble signal\n"
            "• `/qe audit` — channel health audit (what's not working)\n"
            "• `/qe help` — this message"
        )]
    else:
        blocks = [_block_text(f"Unknown command: `{command_text}`. Try `/qe help`")]

    return JSONResponse({"response_type": "in_channel", "blocks": blocks})


async def _status_blocks() -> list[dict]:
    """Real trading mode + background-task health from app state."""
    from app.config import settings
    mode = (getattr(settings, "trading_mode", None) or "paper").upper()
    try:
        from app.main import app as _app
        bg = getattr(_app.state, "bg_tasks", []) or []
        alive = sum(1 for t in bg if not t.done())
        health = f"{alive}/{len(bg)} background workers alive" if bg else "workers not started"
    except Exception:
        health = "status unavailable"
    return [
        _block_text("*QuantEdge Status* :chart_with_upwards_trend:"),
        _block_text(f"• Trading mode: `{mode}`\n• {health}\n• Use `/qe risk` for risk details"),
    ]


async def _risk_blocks() -> list[dict]:
    """
    Real risk metrics from the live risk manager on app.state. There is no
    global singleton, and a fresh RiskManager() has no equity history (its
    drawdown would be a meaningless zero) — so if the live instance isn't
    attached yet we say so rather than report fabricated numbers.
    """
    try:
        from app.main import app as _app
        rm = getattr(_app.state, "risk_manager", None)
        if rm is None:
            return [_block_text("*Risk Status*\nRisk manager not yet attached "
                                "(strategy runner may still be starting).")]
        breaker = getattr(rm, "global_breaker", None)
        lines = []
        if breaker is not None:
            lines.append(f"• Circuit breaker: `{breaker.state.value}`")
            # current_drawdown is a @property — access it, don't call it.
            lines.append(f"• Current drawdown: `{breaker.current_drawdown * 100:.1f}%`")
            lines.append(f"• Max drawdown limit: `{rm.max_drawdown_pct * 100:.0f}%`")
        else:
            lines.append("• Circuit breaker: `unavailable`")
        return [_block_text("*Risk Status*\n" + "\n".join(lines))]
    except Exception as e:
        return [_block_text(f"*Risk Status*\nUnavailable: `{e}`")]


async def _signal_blocks(symbol: str) -> list[dict]:
    """Real ML ensemble signal for a symbol, or 503-style message if no models."""
    try:
        from app.ml.inference import get_inference_service
        inference = get_inference_service()
        if not inference.has_any_model():
            return [_block_text(f"*Signal for {symbol}*\nNo trained models available yet.")]
        import yfinance as yf
        df = yf.Ticker(symbol).history(period="6mo", interval="1d")
        if df.empty or len(df) < 60:
            return [_block_text(f"*Signal for {symbol}*\nNot enough market data.")]
        df.columns = [c.lower() for c in df.columns]
        result = await inference.predict(df, symbol)
        if not result:
            return [_block_text(f"*Signal for {symbol}*\nCould not generate a prediction.")]
        return [_block_text(
            f"*Signal for {symbol}*\n"
            f"• Direction: `{result.get('prediction', '?')}`\n"
            f"• Confidence: `{result.get('confidence', 0):.2f}`"
        )]
    except Exception as e:
        return [_block_text(f"*Signal for {symbol}*\nError: `{e}`")]


async def _audit_blocks() -> list[dict]:
    """Run a Slack channel health audit and summarise what's not working."""
    try:
        from app.integrations.slack_bot import from_env
        bot = from_env()
        if bot is None:
            return [_block_text("*Channel Audit*\nSLACK_BOT_TOKEN not configured.")]
        import asyncio
        report = await asyncio.to_thread(bot.audit_channels)
        s = report["summary"]
        problem = [c for c in report["channels"]
                   if c["status"] in ("flagged", "stale", "silent", "unreadable", "missing")]
        lines = [
            f"*Channel Audit* — {s['total']} channels",
            f"• ✅ healthy: {s['healthy']}  |  🚩 flagged: {s['flagged']}  |  "
            f"💤 stale: {s['stale']}  |  🔇 silent: {s['silent']}  |  "
            f"🚫 unreadable: {s['unreadable']}  |  ❓ missing: {s['missing']}",
        ]
        for c in problem[:15]:
            detail = c.get("detail") or f"{c.get('flagged_messages', 0)} flagged, last {c.get('last_activity_hours')}h ago"
            lines.append(f"• `#{c['channel']}` → *{c['status']}* — {detail}")
        return [_block_text("\n".join(lines))]
    except Exception as e:
        return [_block_text(f"*Channel Audit*\nError: `{e}`")]
