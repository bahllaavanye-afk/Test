"""Discord slash-command endpoint — two-way agentic Discord.

Discord POSTs every interaction (slash command) here; we verify the Ed25519
signature and answer within Discord's 3-second budget. This makes Discord a
command surface, not just a feed: /status, /pnl, /health, /run-bot.

Auth model: there is NO JWT on this route — the Ed25519 signature IS the
authentication. Only Discord holds the application's private key, and the
public key below can only *verify* (it cannot forge requests). Commands are
read-only except /run-bot, which triggers one paper-mode bot evaluation.

Setup (one-time, documented in docs/DISCORD_FALLBACK.md):
  1. This router deploys → endpoint at /api/v1/discord/interactions
  2. Discord Developer Portal → General Information → Interactions Endpoint
     URL → paste the full URL (Discord sends a PING; we PONG).
  3. discord-commands-sync.yml registers the slash commands (needs the
     DISCORD_BOT_TOKEN repo secret).
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from fastapi import APIRouter, HTTPException, Request
from sqlalchemy import func, select

from app.utils.logging import logger

router = APIRouter(prefix="/discord", tags=["discord"])

# Constants
DEFAULT_PUBLIC_KEY_HEX = "2c015f35979920d20c155a8836fbc27916b5d4572eebc8fffe7c26a72198456c"

INTERACTION_TYPE_PING = 1
INTERACTION_TYPE_APPLICATION_COMMAND = 2
RESPONSE_TYPE_PONG = 1
RESPONSE_TYPE_CHANNEL_MESSAGE = 4
MESSAGE_FLAG_EPHEMERAL = 64

MAX_CONTENT_LENGTH = 1990

ERROR_INVALID_SIGNATURE = "invalid request signature"
ERROR_UNSUPPORTED_INTERACTION = "Unsupported interaction type."

USAGE_RUN_BOT = "Usage: `/run-bot name:<part of the bot's name>`"
NO_BOT_MATCH_TEMPLATE = "No enabled bot matches `{q}`."
AMBIGUOUS_BOT_MATCH_TEMPLATE = "Ambiguous — matches: {matches}. Be more specific."
UNKNOWN_COMMAND_TEMPLATE = "Unknown command `{name}`."
COMMAND_FAILED_TEMPLATE = "⚠️ `{name}` failed: {error}"
HEALTH_DATABASE_OK = "✅ database"
HEALTH_DATABASE_ERR_TEMPLATE = "🔴 database: {error}"
HEALTH_SCHEDULER_OK_TEMPLATE = "✅ scheduler: {jobs}"
HEALTH_SCHEDULER_NOT_RUNNING = "🔴 scheduler: not running"
HEALTH_SCHEDULER_ERR_TEMPLATE = "🔴 scheduler: {error}"
HEALTH_DISCORD_OK = "✅ discord: you're reading this"


def _verify_signature(signature_hex: str, timestamp: str, body: bytes) -> bool:
    try:
        key = Ed25519PublicKey.from_public_bytes(bytes.fromhex(DEFAULT_PUBLIC_KEY_HEX))
        key.verify(bytes.fromhex(signature_hex), timestamp.encode() + body)
        return True
    except (InvalidSignature, ValueError):
        return False


def _msg(text: str, ephemeral: bool = False) -> dict:
    data: dict = {"content": text[:MAX_CONTENT_LENGTH]}
    if ephemeral:
        data["flags"] = MESSAGE_FLAG_EPHEMERAL
    return {"type": RESPONSE_TYPE_CHANNEL_MESSAGE, "data": data}


async def _cmd_status() -> str:
    from app.database import AsyncSessionLocal
    from app.models.bot import Bot

    async with AsyncSessionLocal() as db:
        total = (await db.execute(select(func.count(Bot.id)).where(Bot.is_archived == False))).scalar_one()  # noqa: E712
        enabled = (await db.execute(
            select(func.count(Bot.id)).where(Bot.is_enabled == True, Bot.is_archived == False)  # noqa: E712
        )).scalar_one()
        ran = (await db.execute(
            select(func.count(Bot.id)).where(Bot.run_count > 0, Bot.is_archived == False)  # noqa: E712
        )).scalar_one()
    now = datetime.now(timezone.utc).strftime("%H:%M UTC")
    return (
        f"**QuantEdge status** · {now}\n"
        f"🤖 Bots: {enabled}/{total} enabled, {ran} have executed\n"
        f"📈 Mode: paper · scheduler alive (this reply proves the API is awake)"
    )


async def _cmd_pnl() -> str:
    from app.api.v1.accounts import latest_total_equity
    from app.database import AsyncSessionLocal
    from app.models.position import Position
    from app.models.trade import Trade

    today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    async with AsyncSessionLocal() as db:
        equity = await latest_total_equity(db)
        n_closed, pnl_today = (await db.execute(
            select(func.count(Trade.id), func.coalesce(func.sum(Trade.realized_pnl), 0.0))
            .where(Trade.closed_at >= today_start)
        )).one()
        open_count = (await db.execute(select(func.count(Position.id)))).scalar_one()
    arrow = "▲" if float(pnl_today) >= 0 else "▼"
    return (
        f"**P&L** {arrow} ${float(pnl_today):,.2f} today\n"
        f"💰 Equity: ${equity:,.2f} · closed today: {int(n_closed)} · open positions: {int(open_count)}"
    )


async def _cmd_health() -> str:
    checks: list[str] = []
    try:
        from app.database import AsyncSessionLocal

        async with AsyncSessionLocal() as db:
            await db.execute(select(1))
        checks.append(HEALTH_DATABASE_OK)
    except Exception as exc:
        checks.append(HEALTH_DATABASE_ERR_TEMPLATE.format(error=str(exc)[:60]))
    try:
        from app.tasks.scheduler import get_scheduler

        sched = get_scheduler()
        n = len(sched.get_jobs()) if sched.running else 0
        checks.append(
            HEALTH_SCHEDULER_OK_TEMPLATE.format(jobs=n) if sched.running else HEALTH_SCHEDULER_NOT_RUNNING
        )
    except Exception as exc:
        checks.append(HEALTH_SCHEDULER_ERR_TEMPLATE.format(error=str(exc)[:60]))
    checks.append(HEALTH_DISCORD_OK)
    return "**Health**\n" + "\n".join(checks)


async def _cmd_run_bot(name_query: str) -> str:
    from app.bots.engine import BotEngine
    from app.database import AsyncSessionLocal
    from app.models.bot import Bot

    q = (name_query or "").strip()
    if not q:
        return USAGE_RUN_BOT
    async with AsyncSessionLocal() as db:
        bots = (await db.execute(
            select(Bot).where(
                Bot.is_enabled == True,  # noqa: E712
                Bot.is_archived == False,  # noqa: E712
                Bot.name.ilike(f"%{q}%"),
            ).limit(2)
        )).scalars().all()
        if not bots:
            return NO_BOT_MATCH_TEMPLATE.format(q=q)
        if len(bots) > 1:
            matches = ", ".join(b.name for b in bots)
            return AMBIGUOUS_BOT_MATCH_TEMPLATE.format(matches=matches)
        bot = bots[0]
        result = await BotEngine().evaluate(bot, db)
    fired = "🔥 FIRED" if result.fired else "💤 held"
    return (
        f"**{bot.name}** ({bot.symbol}) → {fired}\n"
        f"signal: `{result.signal}` · {result.reason}\n"
        f"orders created: {len(result.orders_created or [])} (paper)"
    )


@router.post("/interactions")
async def discord_interactions(request: Request):
    """Discord interactions webhook: signature-verified, 3s response budget."""
    signature = request.headers.get("X-Signature-Ed25519", "")
    timestamp = request.headers.get("X-Signature-Timestamp", "")
    body = await request.body()
    if not signature or not timestamp or not _verify_signature(signature, timestamp, body):
        # 401 is required by Discord's endpoint validation for bad signatures
        raise HTTPException(status_code=401, detail=ERROR_INVALID_SIGNATURE)

    import json as _json

    payload = _json.loads(body)
    itype = payload.get("type")

    if itype == INTERACTION_TYPE_PING:
        return {"type": RESPONSE_TYPE_PONG}

    if itype == INTERACTION_TYPE_APPLICATION_COMMAND:
        data = payload.get("data") or {}
        name = (data.get("name") or "").lower()
        options = {o.get("name"): o.get("value") for o in (data.get("options") or [])}
        try:
            if name == "status":
                return _msg(await _cmd_status())
            if name == "pnl":
                return _msg(await _cmd_pnl())
            if name == "health":
                return _msg(await _cmd_health())
            if name == "run-bot":
                return _msg(await _cmd_run_bot(str(options.get("name", ""))))
            return _msg(UNKNOWN_COMMAND_TEMPLATE.format(name=name), ephemeral=True)
        except Exception as exc:
            logger.error("Discord command failed", command=name, error=str(exc))
            return _msg(
                COMMAND_FAILED_TEMPLATE.format(name=name, error=str(exc)[:150]),
                ephemeral=True,
            )

    return _msg(ERROR_UNSUPPORTED_INTERACTION, ephemeral=True)