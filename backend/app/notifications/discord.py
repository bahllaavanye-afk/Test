"""
Discord notification client — the single notification path for the backend.

Replaces the old Slack-first client (`app/notifications/slack.py`, removed
2026-07-25). Slack was retired: its free-plan message quota exhausted on
2026-06-29 and never recovered, so every alert had been travelling the Discord
failover path for weeks while still paying the cost of a Slack attempt first.

Two delivery routes, tried in order:
  1. **Bot token** (`DISCORD_BOT_TOKEN`) → resolves `#channel-name` to a real
     channel id and posts there. This is what puts each alert in its OWN
     channel; without it everything piles into one webhook target.
  2. **Webhook failover** — `DISCORD_WEBHOOK_URL_<SLUG>` for a specific channel,
     else the catch-all `DISCORD_WEBHOOK_URL` (which gets a `[#channel]` prefix
     so the intended destination is still readable).

Messages are sent as native Discord **embeds**, not flattened text. The old
code built a rich coloured attachment for Slack and then, on failover, threw
that structure away and posted `**title**\\n• k: v` to Discord — so in practice
(Slack being dead) every alert rendered as the degraded plain-text version.
Embeds restore the colour coding, field layout and timestamp.
"""
from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from typing import Any

import httpx

from app.config import settings
from app.utils.logging import logger

DISCORD_API = "https://discord.com/api/v10"
# Discord's REST API returns 403 for a browser User-Agent on bot endpoints; it
# requires this exact "DiscordBot (url, version)" form. Webhooks are lenient,
# but sending it everywhere keeps one rule instead of two.
BOT_UA = "DiscordBot (https://github.com/quantedge/quantedge, 1.0)"

# Logical event stream → the Discord channel that owns it.
CHANNEL_MAP = {
    "orders":      "pnl-daily",
    "signals":     "alpha-research",
    "alerts":      "risk-alerts",
    "experiments": "ml-experiments",
    "system":      "engineering",
}

# Embed colours are ints in Discord (Slack used "#rrggbb" strings).
COLORS = {
    "order_filled":    0x00C853,
    "order_cancelled": 0x888888,
    "signal_fired":    0xF5A623,
    "order_rejected":  0xFF1744,
    "risk_event":      0xFF1744,
    "circuit_breaker": 0x9C27B0,
    "experiment_done": 0x2979FF,
    "system":          0x888888,
}
DEFAULT_COLOR = 0x888888

MAX_EMBED_FIELDS = 25      # Discord hard limit
MAX_DESCRIPTION = 4096     # Discord hard limit
MAX_CONTENT = 2000         # plain-content fallback limit


class DiscordClient:
    """Multi-channel Discord notifier. Bot token preferred, webhook failover."""

    def __init__(self) -> None:
        self._bot_token: str = (
            getattr(settings, "discord_bot_token", "") or os.environ.get("DISCORD_BOT_TOKEN", "")
        ).strip()
        self._guild_id: str = (
            getattr(settings, "discord_guild_id", "") or os.environ.get("DISCORD_GUILD_ID", "")
        ).strip()
        self._default_webhook: str = (
            getattr(settings, "discord_webhook_url", "") or os.environ.get("DISCORD_WEBHOOK_URL", "")
        )
        self._channel_ids: dict[str, str] | None = None
        self._lock = asyncio.Lock()

    @property
    def _enabled(self) -> bool:
        """True when at least one delivery route is configured.

        A property (not a cached attribute) so tests and runtime env changes are
        picked up — the old client froze this at import time, which made it
        impossible to enable notifications without a process restart.
        """
        return bool(self._bot_token or self._default_webhook or self._any_channel_webhook())

    @staticmethod
    def _any_channel_webhook() -> bool:
        return any(k.startswith("DISCORD_WEBHOOK_URL_") and v for k, v in os.environ.items())

    # ── channel resolution ───────────────────────────────────────────────────

    async def _channel_id(self, channel: str) -> str | None:
        """Resolve `#channel-name` → channel id via the bot token, cached.

        A pinned DISCORD_GUILD_ID skips `/users/@me/guilds`, which some bot
        installs are not permitted to call.
        """
        if not self._bot_token:
            return None
        if self._channel_ids is None:
            async with self._lock:
                if self._channel_ids is None:  # re-check inside the lock
                    self._channel_ids = await self._load_channel_map()
        return self._channel_ids.get(self._slug(channel))

    @staticmethod
    def _slug(channel: str) -> str:
        return str(channel).lower().lstrip("#")

    async def _load_channel_map(self) -> dict[str, str]:
        headers = {"Authorization": f"Bot {self._bot_token}", "User-Agent": BOT_UA}
        mapping: dict[str, str] = {}
        try:
            async with httpx.AsyncClient(timeout=10.0, headers=headers) as client:
                if self._guild_id:
                    guild_ids = [self._guild_id]
                else:
                    resp = await client.get(f"{DISCORD_API}/users/@me/guilds")
                    guild_ids = [g["id"] for g in resp.json()]
                for gid in guild_ids:
                    resp = await client.get(f"{DISCORD_API}/guilds/{gid}/channels")
                    for ch in resp.json():
                        if ch.get("type") == 0:  # GUILD_TEXT
                            mapping.setdefault(ch["name"].lower().lstrip("#"), ch["id"])
        except Exception as e:  # noqa: BLE001 — notification path must never raise
            logger.warning("Discord channel map failed", error=str(e))
        return mapping

    # ── delivery ─────────────────────────────────────────────────────────────

    @staticmethod
    def _build_embed(event_type: str, title: str, text: str | None,
                     fields: dict[str, Any] | None) -> dict[str, Any]:
        embed: dict[str, Any] = {
            "title": title[:256],
            "color": COLORS.get(event_type, DEFAULT_COLOR),
            "footer": {"text": "QuantEdge"},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if text:
            embed["description"] = text[:MAX_DESCRIPTION]
        if fields:
            embed["fields"] = [
                {"name": str(k)[:256], "value": str(v)[:1024], "inline": True}
                for k, v in list(fields.items())[:MAX_EMBED_FIELDS]
            ]
        return embed

    async def _post_bot(self, channel_id: str, embed: dict[str, Any]) -> bool:
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                resp = await client.post(
                    f"{DISCORD_API}/channels/{channel_id}/messages",
                    headers={"Authorization": f"Bot {self._bot_token}", "User-Agent": BOT_UA},
                    json={"embeds": [embed]},
                )
                return resp.status_code in (200, 201)
        except Exception as e:  # noqa: BLE001
            logger.warning("Discord bot post failed", error=str(e))
            return False

    def _webhook_for(self, channel: str) -> tuple[str, bool]:
        """(webhook_url, is_default). Per-channel override wins over the catch-all."""
        slug = self._slug(channel).upper().replace("-", "_")
        specific = os.environ.get(f"DISCORD_WEBHOOK_URL_{slug}", "")
        if specific:
            return specific, False
        return self._default_webhook, True

    async def _post_webhook(self, channel: str, embed: dict[str, Any]) -> bool:
        webhook, is_default = self._webhook_for(channel)
        if not webhook:
            return False
        payload: dict[str, Any] = {"embeds": [embed], "username": "QuantEdge"}
        if is_default:
            # One shared webhook posts everything to the same place; name the
            # intended channel so the message is still traceable.
            payload["content"] = f"**[#{self._slug(channel)}]**"[:MAX_CONTENT]
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                resp = await client.post(webhook, json=payload)
                return resp.status_code in (200, 204)
        except Exception as e:  # noqa: BLE001
            logger.warning("Discord webhook post failed", error=str(e))
            return False

    async def send(self, channel: str, event_type: str, title: str,
                   fields: dict[str, Any] | None = None, text: str | None = None) -> bool:
        """Deliver one notification. Never raises; returns True if it landed."""
        if not self._enabled:
            return False
        target = CHANNEL_MAP.get(channel, channel or "engineering")
        embed = self._build_embed(event_type, title, text, fields)

        cid = await self._channel_id(target)
        if cid and await self._post_bot(cid, embed):
            return True
        return await self._post_webhook(target, embed)

    # ── typed helpers (same interface the old Slack client exposed) ──────────

    async def notify_order_filled(self, symbol: str, side: str, quantity: float,
                                  fill_price: float, slippage_bps: float | None = None,
                                  algo: str | None = None) -> bool:
        fields: dict[str, Any] = {
            "Symbol": symbol, "Side": side.upper(),
            "Qty": f"{quantity:.4f}", "Fill": f"${fill_price:.4f}",
        }
        if slippage_bps is not None:
            fields["Slippage"] = f"{slippage_bps:.2f} bps"
        if algo:
            fields["Algo"] = algo
        return await self.send("orders", "order_filled", f"✅ Filled: {symbol}", fields)

    async def notify_signal(self, strategy: str, symbol: str, side: str,
                            confidence: float, target_price: float | None = None) -> bool:
        fields: dict[str, Any] = {
            "Strategy": strategy, "Symbol": symbol,
            "Side": side.upper(), "Confidence": f"{confidence:.1%}",
        }
        if target_price:
            fields["Target"] = f"${target_price:.4f}"
        return await self.send("signals", "signal_fired", f"📡 {strategy} → {symbol}", fields)

    async def notify_risk_event(self, event_type: str, description: str,
                                value: float | None = None) -> bool:
        fields: dict[str, Any] = {"Event": event_type}
        if value is not None:
            fields["Value"] = f"{value:.4f}"
        return await self.send("alerts", "risk_event", f"⚠️ Risk: {event_type}",
                               fields, text=description)

    async def notify_circuit_breaker(self, name: str, drawdown: float, threshold: float) -> bool:
        fields = {"Breaker": name, "Drawdown": f"{drawdown:.2%}", "Threshold": f"{threshold:.2%}"}
        return await self.send("alerts", "circuit_breaker", f"🛑 CIRCUIT BREAKER: {name}",
                               fields, text="Trading halted. Manual review required.")

    async def notify_experiment_done(self, name: str, val_sharpe: float | None,
                                     test_sharpe: float | None,
                                     val_accuracy: float | None = None) -> bool:
        fields: dict[str, Any] = {"Name": name}
        if val_sharpe is not None:
            fields["Val Sharpe"] = f"{val_sharpe:.3f}"
        if test_sharpe is not None:
            fields["Test Sharpe"] = f"{test_sharpe:.3f}"
        if val_accuracy is not None:
            fields["Val Acc"] = f"{val_accuracy:.1%}"
        return await self.send("experiments", "experiment_done", f"🧪 Experiment: {name}", fields)

    async def notify_daily_summary(self, total_pnl: float, total_trades: int,
                                   win_rate: float, best_strategy: str | None = None) -> bool:
        fields: dict[str, Any] = {
            "P&L": f"${total_pnl:.2f}", "Trades": str(total_trades),
            "Win Rate": f"{win_rate:.1%}",
        }
        if best_strategy:
            fields["Best"] = best_strategy
        return await self.send("system", "system", "📊 Daily Summary", fields)

    async def notify_system(self, message: str, level: str = "info") -> bool:
        emoji = {"info": "ℹ️", "warning": "⚠️", "error": "🔴"}.get(level, "ℹ️")
        return await self.send("system", "system", f"{emoji} {message}")


discord = DiscordClient()
