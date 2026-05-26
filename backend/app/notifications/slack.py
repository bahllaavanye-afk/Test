"""
Slack notification client. Uses incoming webhooks (free, no OAuth needed).
Configure SLACK_WEBHOOK_* env vars to enable different channels per event type.
"""
from __future__ import annotations
import asyncio
import json
from datetime import datetime, timezone
from typing import Any
import httpx
from app.config import settings
from app.utils.logging import logger


class SlackClient:
    """
    Multi-channel Slack notifier.

    Channels (each optional, configured via separate webhook URLs):
      - orders:    fills, cancels, rejections
      - signals:   strategy signals fired
      - alerts:    risk events, circuit breakers
      - experiments: ML training completed
      - system:    startup/shutdown, errors, daily summary
    """

    COLORS = {
        "order_filled":   "#00c853",   # green
        "order_cancelled": "#888888",  # gray
        "order_rejected": "#ff1744",   # red
        "signal_fired":   "#f5a623",   # amber
        "risk_event":     "#ff1744",   # red
        "circuit_breaker": "#9c27b0",  # purple
        "experiment_done": "#2979ff",  # blue
        "system":         "#888888",
    }

    def __init__(self):
        self._enabled = bool(getattr(settings, "slack_webhook_orders", "") or
                             getattr(settings, "slack_webhook_signals", "") or
                             getattr(settings, "slack_webhook_alerts", "") or
                             getattr(settings, "slack_webhook_default", ""))

    def _channel_webhook(self, channel: str) -> str | None:
        mapping = {
            "orders": getattr(settings, "slack_webhook_orders", ""),
            "signals": getattr(settings, "slack_webhook_signals", ""),
            "alerts": getattr(settings, "slack_webhook_alerts", ""),
            "experiments": getattr(settings, "slack_webhook_experiments", ""),
            "system": getattr(settings, "slack_webhook_system", ""),
        }
        webhook = mapping.get(channel, "") or getattr(settings, "slack_webhook_default", "")
        return webhook or None

    async def _post(self, webhook: str, payload: dict) -> bool:
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                resp = await client.post(webhook, json=payload)
                return resp.status_code in (200, 204)
        except Exception as e:
            logger.warning("Slack post failed", error=str(e))
            return False

    async def send(self, channel: str, event_type: str, title: str, fields: dict[str, Any] | None = None,
                    text: str | None = None) -> bool:
        """Send a formatted Slack message. Returns True if delivered."""
        if not self._enabled:
            return False
        webhook = self._channel_webhook(channel)
        if not webhook:
            return False

        color = self.COLORS.get(event_type, "#888888")
        attachment_fields = []
        if fields:
            for k, v in fields.items():
                attachment_fields.append({"title": str(k), "value": str(v), "short": True})

        payload = {
            "attachments": [{
                "color": color,
                "title": title,
                "text": text or "",
                "fields": attachment_fields,
                "footer": "QuantEdge",
                "ts": int(datetime.now(timezone.utc).timestamp()),
            }]
        }
        return await self._post(webhook, payload)

    async def notify_order_filled(self, symbol: str, side: str, quantity: float, fill_price: float,
                                    slippage_bps: float | None = None, algo: str | None = None) -> bool:
        fields = {
            "Symbol": symbol,
            "Side": side.upper(),
            "Quantity": f"{quantity:.4f}",
            "Fill Price": f"${fill_price:.4f}",
        }
        if slippage_bps is not None:
            fields["Slippage"] = f"{slippage_bps:.2f} bps"
        if algo:
            fields["Algorithm"] = algo
        return await self.send("orders", "order_filled", f"✅ Order Filled: {symbol}", fields)

    async def notify_signal(self, strategy: str, symbol: str, side: str, confidence: float,
                             target_price: float | None = None) -> bool:
        fields = {
            "Strategy": strategy,
            "Symbol": symbol,
            "Side": side.upper(),
            "Confidence": f"{confidence:.1%}",
        }
        if target_price:
            fields["Target"] = f"${target_price:.4f}"
        return await self.send("signals", "signal_fired", f"📡 Signal: {strategy} → {symbol}", fields)

    async def notify_risk_event(self, event_type: str, description: str, value: float | None = None) -> bool:
        fields = {"Event": event_type}
        if value is not None:
            fields["Value"] = f"{value:.4f}"
        return await self.send("alerts", "risk_event", f"⚠️ Risk Event: {event_type}", fields, text=description)

    async def notify_circuit_breaker(self, name: str, drawdown: float, threshold: float) -> bool:
        fields = {
            "Breaker": name,
            "Drawdown": f"{drawdown:.2%}",
            "Threshold": f"{threshold:.2%}",
        }
        return await self.send("alerts", "circuit_breaker",
                                f"🛑 CIRCUIT BREAKER TRIPPED: {name}", fields,
                                text="Trading halted. Manual review required.")

    async def notify_experiment_done(self, name: str, val_sharpe: float | None, test_sharpe: float | None,
                                       val_accuracy: float | None = None) -> bool:
        fields = {"Name": name}
        if val_sharpe is not None: fields["Val Sharpe"] = f"{val_sharpe:.3f}"
        if test_sharpe is not None: fields["Test Sharpe"] = f"{test_sharpe:.3f}"
        if val_accuracy is not None: fields["Val Acc"] = f"{val_accuracy:.1%}"
        return await self.send("experiments", "experiment_done",
                                f"🧪 Experiment complete: {name}", fields)

    async def notify_daily_summary(self, total_pnl: float, total_trades: int, win_rate: float,
                                     best_strategy: str | None = None) -> bool:
        fields = {
            "Total P&L": f"${total_pnl:.2f}",
            "Trades": str(total_trades),
            "Win Rate": f"{win_rate:.1%}",
        }
        if best_strategy:
            fields["Best Strategy"] = best_strategy
        return await self.send("system", "system", "📊 Daily Trading Summary", fields)

    async def notify_system(self, message: str, level: str = "info") -> bool:
        emoji = {"info": "ℹ️", "warning": "⚠️", "error": "🔴"}.get(level, "ℹ️")
        return await self.send("system", "system", f"{emoji} {message}")


slack = SlackClient()
