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
import json as _json
from datetime import datetime, timezone

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from fastapi import APIRouter, HTTPException, Request
from sqlalchemy import func, select

from app.utils.logging import logger

router = APIRouter(prefix="/discord", tags=["discord"])

# The application PUBLIC key — verification-only, not a secret (it cannot sign
# anything). Env-overridable for rotation or a different Discord application.
_PUBLIC_KEY_HEX = os.environ.get(
    "DISCORD_PUBLIC_KEY",
    "2c015f35979920d20c155a8836fbc27916b5d4572eebc8fffe7c26a72198456c",
)

# Interaction types / response types (Discord API v10)
_PING, _APPLICATION_COMMAND = 1, 2
_PONG, _CHANNEL_MESSAGE = 1, 4
_EPHEMERAL = 64


def _verify_signature(signature_hex: str, timestamp: str, body: bytes) -> bool:
    try:
        key = Ed25519PublicKey.from_public_bytes(bytes.fromhex(_PUBLIC_KEY_HEX))
        key.verify(bytes.fromhex(signature_hex), timestamp.encode() + body)
        return True
    except (InvalidSignature, ValueError):
        return False


def _msg(text: str | None, ephemeral: bool = False) -> dict:
    """Create a Discord message payload, safely handling None text."""
    if text is None:
        text = ""
    data: dict = {"content": text[:1990]}
    if ephemeral:
        data["flags"] = _EPHEMERAL
    return {"type": _CHANNEL_MESSAGE, "data": data}


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
        checks.append("✅ database")
    except Exception as exc:
        checks.append(f"🔴 database: {str(exc)[:60]}")
    try:
        from app.tasks.scheduler import get_scheduler

        sched = get_scheduler()
        n = len(sched.get_jobs()) if sched and getattr(sched, "running", False) else 0
        checks.append(f"✅ scheduler: {n} jobs" if getattr(sched, "running", False) else "🔴 scheduler: not running")
    except Exception as exc:
        checks.append(f"🔴 scheduler: {str(exc)[:60]}")
    checks.append("✅ discord: you're reading this")
    return "**Health**\n" + "\n".join(checks)


async def _cmd_run_bot(name_query: str) -> str:
    from app.bots.engine import BotEngine
    from app.database import AsyncSessionLocal
    from app.models.bot import Bot

    q = (name_query or "").strip()
    if not q:
        return "Usage: `/run-bot name:<part of the bot's name>`"
    async with AsyncSessionLocal() as db:
        bots = (await db.execute(
            select(Bot).where(
                Bot.is_enabled == True,  # noqa: E712
                Bot.is_archived == False,  # noqa: E712
                Bot.name.ilike(f"%{q}%"),
            ).limit(2)
        )).scalars().all()
        if not bots:
            return f"No enabled bot matches `{q}`."
        if len(bots) > 1:
            return f"Ambiguous — matches: {', '.join(b.name for b in bots)}. Be more specific."
        bot = bots[0]
        result = await BotEngine().evaluate(bot, db)
    if result is None:
        return f"⚠️ Bot evaluation failed for `{bot.name}`."
    fired = "🔥 FIRED" if getattr(result, "fired", False) else "💤 held"
    orders_created = getattr(result, "orders_created", []) or []
    return (
        f"**{bot.name}** ({bot.symbol}) → {fired}\n"
        f"signal: `{getattr(result, 'signal', 'N/A')}` · {getattr(result, 'reason', '')}\n"
        f"orders created: {len(orders_created)} (paper)"
    )


@router.post("/interactions")
async def discord_interactions(request: Request):
    """Discord interactions webhook: signature-verified, 3s response budget."""
    signature = request.headers.get("X-Signature-Ed25519", "")
    timestamp = request.headers.get("X-Signature-Timestamp", "")
    body = await request.body()
    if not signature or not timestamp or not _verify_signature(signature, timestamp, body):
        # 401 is required by Discord's endpoint validation for bad signatures
        raise HTTPException(status_code=401, detail="invalid request signature")

    try:
        payload = _json.loads(body)
    except _json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="invalid JSON payload")

    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="payload must be a JSON object")

    itype = payload.get("type")
    if itype == _PING:
        return {"type": _PONG}

    if itype == _APPLICATION_COMMAND:
        data = payload.get("data") or {}
        name = (data.get("name") or "").lower()
        raw_options = data.get("options") or []
        # Safely build options dict, ignoring malformed entries
        options = {}
        for opt in raw_options:
            if isinstance(opt, dict):
                opt_name = opt.get("name")
                opt_value = opt.get("value")
                if opt_name is not None:
                    options[opt_name] = opt_value
        try:
            if name == "status":
                return _msg(await _cmd_status())
            if name == "pnl":
                return _msg(await _cmd_pnl())
            if name == "health":
                return _msg(await _cmd_health())
            if name == "run-bot":
                return _msg(await _cmd_run_bot(str(options.get("name", ""))))
            return _msg(f"Unknown command `{name}`.", ephemeral=True)
        except Exception as exc:
            logger.error("Discord command failed", command=name, error=str(exc))
            return _msg(f"⚠️ `{name}` failed: {str(exc)[:150]}", ephemeral=True)

    return _msg("Unsupported interaction type.", ephemeral=True)