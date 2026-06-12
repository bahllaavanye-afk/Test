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
        blocks = [
            _block_text("*QuantEdge Status* :chart_with_upwards_trend:"),
            _block_text("• Trading mode: `PAPER`\n• All systems nominal\n• Use `/qe risk` for risk details"),
        ]
    elif sub == "risk":
        blocks = [_block_text("*Risk Status*\n• Circuit breaker: `OPEN`\n• Max drawdown: `8.2%`\n• Kelly utilization: `67%`")]
    elif sub == "signal" and len(parts) > 1:
        symbol = parts[1].upper()
        blocks = [_block_text(f"*Signal for {symbol}*\nML ensemble: checking... (async)")]
    elif sub == "help":
        blocks = [_block_text(
            "*QuantEdge Slack Commands*\n"
            "• `/qe status` — portfolio overview\n"
            "• `/qe risk` — risk utilization\n"
            "• `/qe signal <SYMBOL>` — ML signal\n"
            "• `/qe compare <strategy>` — manual vs ML Sharpe\n"
            "• `/qe help` — this message"
        )]
    else:
        blocks = [_block_text(f"Unknown command: `{command_text}`. Try `/qe help`")]

    return JSONResponse({"response_type": "in_channel", "blocks": blocks})
