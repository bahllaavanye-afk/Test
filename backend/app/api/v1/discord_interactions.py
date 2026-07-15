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
import time
from datetime import datetime, timezone

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from fastapi import APIRouter, HTTPException, Request
from sqlalchemy import case, func, select

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

# Simple in‑memory caches for expensive commands
_STATUS_CACHE_TTL = 30  # seconds
_PNL_CACHE_TTL = 60  # seconds
_status_cache = {"value": None, "ts": 0}
_pnl_cache = {"value": None, "ts": 0}


def _verify_signature(signature_hex: str, timestamp: str, body: bytes) -> bool:
    try:
        key = Ed25519PublicKey.from_public_bytes(bytes.fromhex(_PUBLIC_KEY_HEX))
        key.verify(bytes.fromhex(signature_hex), timestamp.encode() + body)
        return True
    except (InvalidSignature, ValueError):
        return False


def _msg(text: str, ephemeral: bool = False) -> dict:
    data: dict = {"content": text[:1990]}
    if ephemeral:
        data["flags"] = _EPHEMERAL
    return {"type": _CHANNEL_MESSAGE, "data": data}


async def _cmd_status() -> str:
    """Return a short status string, using a cached aggregated DB query."""
    now_ts = time.time()
    if _status_cache["value"] is not None and now_ts - _status_cache["ts"] < _STATUS_CACHE_TTL:
        return _status_cache["value"]

    from app.database import AsyncSessionLocal
    from app.models.bot import Bot

    async with AsyncSessionLocal() as db:
        stmt = (
            select(
                func.count(Bot.id).label("total"),
                func.sum(case((Bot.is_enabled == True, 1), else_=0)).label("enabled"),
                func.sum(case((Bot.run_count > 0, 1), else_=0)).label("ran"),
            )
            .where(Bot.is_archived == False)  # noqa: E712
        )
        result = await db.execute(stmt)
        total, enabled, ran = result.one()
    now = datetime.now(timezone.utc).strftime("%H:%M UTC")
    status_msg = (
        f"**QuantEdge status** · {now}\n"
        f"🤖 Bots: {enabled}/{total} enabled, {ran} have executed\n"
        f"📈 Mode: paper · scheduler alive (this reply proves the API is awake)"
    )
    _status_cache.update({"value": status_msg, "ts": now_ts})
    return status_msg


async def _cmd_pnl() -> str:
    """Return P&L summary, cached for a short interval."""
    now_ts = time.time()
    if _pnl_cache["value"] is not None and now_ts - _pnl_cache["ts"] < _PNL_CACHE_TTL:
        return _pnl_cache["value"]

    from app.api.v1.accounts import latest_total_equity
    from app.database import AsyncSessionLocal
    from app.models.position import Position
    from app.models.trade import Trade

    today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    async with AsyncSessionLocal() as db:
        equity = await latest_total_equity(db)
        n_closed, pnl_today = (
            await db.execute(
                select(func.count(Trade.id), func.coalesce(func.sum(Trade.realized_pnl), 0.0))
                .where(Trade.closed_at >= today_start)
            )
        ).one()
        open_count = (await db.execute(select(func.count(Position.id)))).scalar_one()
    arrow = "▲" if float(pnl_today) >= 0 else "▼"
    pnl_msg = (
        f"**P&L** {arrow} ${float(pnl_today):,.2f} today\n"
        f"💰 Equity: ${equity:,.2f} · closed today: {int(n_closed)} · open positions: {int(open_count)}"
    )
    _pnl_cache.update({"value": pnl_msg, "ts": now_ts})
    return pnl_msg


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
        n = len(sched.get_jobs()) if sched.running else 0
        checks.append(f"✅ scheduler: {n} jobs" if sched.running else "🔴 scheduler: not running")
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
        bots = (
            await db.execute(
                select(Bot)
                .where(
                    Bot.is_enabled == True,  # noqa: E712
                    Bot.is_archived == False,  # noqa: E712
                    Bot.name.ilike(f"%{q}%"),
                )
                .limit(2)
            )
        ).scalars().all()
        if not bots:
            return f"No enabled bot matches `{q}`."
        if len(bots) > 1:
            return f"Ambiguous — matches: {', '.join(b.name for b in bots)}. Be more specific."
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
        raise HTTPException(status_code=401, detail="invalid request signature")

    import json as _json

    payload = _json.loads(body)
    itype = payload.get("type")

    if itype == _PING:
        return {"type": _PONG}

    if itype == _APPLICATION_COMMAND:
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
            return _msg(f"Unknown command `{name}`.", ephemeral=True)
        except Exception as exc:
            logger.error("Discord command failed", command=name, error=str(exc))
            return _msg(f"⚠️ `{name}` failed: {str(exc)[:150]}", ephemeral=True)

    return _msg("Unsupported interaction type.", ephemeral=True)