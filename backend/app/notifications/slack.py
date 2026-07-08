"""
Slack notification client (with Discord failover).

Supports two modes (auto-detected, bot token takes priority):
  1. Bot token (SLACK_BOT_TOKEN=xoxb-...): uses chat.postMessage API.
     Posts to named channels (#pnl-daily, #risk-alerts, etc.).
  2. Incoming webhooks (SLACK_WEBHOOK_*): legacy, one URL per channel.

Required bot token scopes: chat:write, chat:write.public

Discord failover: Slack's free-plan message quota can exhaust entirely
(message_limit_exceeded — it did on 2026-06-29 and stayed dead for days).
When DISCORD_WEBHOOK_URL is set, any notification Slack can't deliver is
posted to Discord instead, so alerts are never silently lost.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import httpx

from app.config import settings
from app.utils.logging import logger

SLACK_API = "https://slack.com/api/chat.postMessage"

# Maps logical channel → Slack channel name
CHANNEL_MAP = {
    "orders":      "pnl-daily",
    "signals":     "alpha-research",
    "alerts":      "risk-alerts",
    "experiments": "ml-experiments",
    "system":      "engineering",
}

COLORS = {
    "order_filled":    "#00c853",
    "order_cancelled": "#888888",
    "order_rejected":  "#ff1744",
    "signal_fired":    "#f5a623",
    "risk_event":      "#ff1744",
    "circuit_breaker": "#9c27b0",
    "experiment_done": "#2979ff",
    "system":          "#888888",
}


class SlackClient:
    """Multi-channel Slack notifier. Bot token takes priority over webhooks."""

    def __init__(self) -> None:
        self._token: str = getattr(settings, "slack_bot_token", "") or ""
        self._use_bot = bool(self._token and self._token.startswith("xoxb-"))
        self._webhooks = {
            "orders":      getattr(settings, "slack_webhook_orders", ""),
            "signals":     getattr(settings, "slack_webhook_signals", ""),
            "alerts":      getattr(settings, "slack_webhook_alerts", ""),
            "experiments": getattr(settings, "slack_webhook_experiments", ""),
            "system":      getattr(settings, "slack_webhook_system", ""),
        }
        self._default_webhook = getattr(settings, "slack_webhook_default", "")
        self._discord_webhook = (
            getattr(settings, "discord_webhook_url", "") or os.environ.get("DISCORD_WEBHOOK_URL", "")
        )
        self._enabled = (
            self._use_bot
            or bool(self._default_webhook or any(self._webhooks.values()))
            or bool(self._discord_webhook)
        )

    async def _discord_channel_id(self, channel: str) -> str | None:
        """Resolve #channel-name → id via the bot token (cached). Needs the bot
        in the server. This is what routes to the REAL channel instead of the
        single default webhook (the reason everything landed in #general)."""
        token = os.environ.get("DISCORD_BOT_TOKEN", "").strip()
        if not token:
            return None
        if getattr(self, "_discord_channel_ids", None) is None:
            self._discord_channel_ids = {}
            api = "https://discord.com/api/v10"
            hdr = {"Authorization": f"Bot {token}",
                   "User-Agent": "Mozilla/5.0 QuantEdge-Notify/1.0"}
            try:
                async with httpx.AsyncClient(timeout=10.0, headers=hdr) as client:
                    guilds = (await client.get(f"{api}/users/@me/guilds")).json()
                    for g in guilds:
                        chans = (await client.get(f"{api}/guilds/{g['id']}/channels")).json()
                        for c in chans:
                            if c.get("type") == 0:
                                self._discord_channel_ids.setdefault(
                                    c["name"].lower().lstrip("#"), c["id"])
            except Exception as e:  # noqa: BLE001
                logger.warning("Discord channel map failed", error=str(e))
        return self._discord_channel_ids.get(str(channel).lower().lstrip("#"))

    async def _post_discord(self, channel: str, title: str, text: str,
                            fields: dict[str, Any] | None = None) -> bool:
        """Deliver to Discord — real channel via bot token, else webhook failover."""
        lines = [f"**{title}**"]
        if text:
            lines.append(text)
        for k, v in (fields or {}).items():
            lines.append(f"• {k}: {v}")
        body = "\n".join(lines)

        # 1) bot-token routing to the actual channel
        cid = await self._discord_channel_id(channel)
        if cid:
            token = os.environ.get("DISCORD_BOT_TOKEN", "").strip()
            try:
                async with httpx.AsyncClient(timeout=8.0) as client:
                    resp = await client.post(
                        f"https://discord.com/api/v10/channels/{cid}/messages",
                        headers={"Authorization": f"Bot {token}",
                                 "User-Agent": "Mozilla/5.0 QuantEdge-Notify/1.0"},
                        json={"content": body[:2000]},
                    )
                    if resp.status_code in (200, 201):
                        return True
            except Exception as e:  # noqa: BLE001
                logger.warning("Discord bot post failed", error=str(e))

        # 2) webhook failover (per-channel override, else default → #general w/ prefix)
        slug = str(channel).lstrip("#").upper().replace("-", "_")
        webhook = os.environ.get(f"DISCORD_WEBHOOK_URL_{slug}", "") or self._discord_webhook
        if not webhook:
            return False
        using_default = webhook == self._discord_webhook
        content = (f"**[#{channel}]** {body}" if using_default else body)[:2000]
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                resp = await client.post(webhook, json={"content": content, "username": "QuantEdge"})
                return resp.status_code in (200, 204)
        except Exception as e:
            logger.warning("Discord post failed", error=str(e))
            return False

    async def _post_bot(self, channel: str, payload: dict) -> bool:
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                resp = await client.post(
                    SLACK_API,
                    headers={"Authorization": f"Bearer {self._token}"},
                    json={"channel": f"#{channel}", **payload},
                )
                data = resp.json()
                if not data.get("ok"):
                    logger.warning("Slack bot error", error=data.get("error"), channel=channel)
                return data.get("ok", False)
        except Exception as e:
            logger.warning("Slack bot post failed", error=str(e))
            return False

    async def _post_webhook(self, webhook: str, payload: dict) -> bool:
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                resp = await client.post(webhook, json=payload)
                return resp.status_code in (200, 204)
        except Exception as e:
            logger.warning("Slack webhook post failed", error=str(e))
            return False

    async def send(self, channel: str, event_type: str, title: str,
                   fields: dict[str, Any] | None = None, text: str | None = None) -> bool:
        if not self._enabled:
            return False

        color = COLORS.get(event_type, "#888888")
        attachment_fields = [
            {"title": k, "value": str(v), "short": True}
            for k, v in (fields or {}).items()
        ]
        attachment = {
            "color": color,
            "title": title,
            "text": text or "",
            "fields": attachment_fields,
            "footer": "QuantEdge",
            "ts": int(datetime.now(timezone.utc).timestamp()),
        }

        slack_channel = CHANNEL_MAP.get(channel, "engineering")
        if self._use_bot:
            ok = await self._post_bot(slack_channel, {"attachments": [attachment]})
            if ok:
                return True
            # Slack rejected it (quota, revoked token, ...) — don't lose the alert
            return await self._post_discord(slack_channel, title, text or "", fields)

        webhook = self._webhooks.get(channel, "") or self._default_webhook
        if webhook:
            ok = await self._post_webhook(webhook, {"attachments": [attachment]})
            if ok:
                return True
            return await self._post_discord(slack_channel, title, text or "", fields)
        # No Slack path configured at all — Discord is the primary channel
        return await self._post_discord(slack_channel, title, text or "", fields)

    # ── Typed helpers ────────────────────────────────────────────────────────

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
        return await self.send("alerts", "risk_event", f"⚠️ Risk: {event_type}", fields, text=description)

    async def notify_circuit_breaker(self, name: str, drawdown: float, threshold: float) -> bool:
        fields = {"Breaker": name, "Drawdown": f"{drawdown:.2%}", "Threshold": f"{threshold:.2%}"}
        return await self.send("alerts", "circuit_breaker",
                               f"🛑 CIRCUIT BREAKER: {name}", fields,
                               text="Trading halted. Manual review required.")

    async def notify_experiment_done(self, name: str, val_sharpe: float | None,
                                     test_sharpe: float | None,
                                     val_accuracy: float | None = None) -> bool:
        fields: dict[str, Any] = {"Name": name}
        if val_sharpe is not None:   fields["Val Sharpe"] = f"{val_sharpe:.3f}"
        if test_sharpe is not None:  fields["Test Sharpe"] = f"{test_sharpe:.3f}"
        if val_accuracy is not None: fields["Val Acc"] = f"{val_accuracy:.1%}"
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


slack = SlackClient()
