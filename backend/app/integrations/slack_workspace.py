"""
Full Slack workspace integration for the engineering team.

Posts standups, alpha reviews, daily P&L, deploys, incidents, and CI failures
to dedicated channels. Each channel has its own webhook URL so cost is zero
on Slack's free tier (10k messages, 90-day retention).

Channels (create these in your Slack workspace):
  #engineering-standup    — daily standups (one post per squad)
  #alpha-research         — new strategy proposals, paper reviews
  #pnl-daily              — EOD P&L attribution
  #risk-alerts            — VaR breaches, circuit breaker fires
  #incidents              — P0/P1 incidents + postmortems
  #deploys                — deploy notifications
  #ci-failures            — CI / test failures
  #ml-experiments         — training run completions, model leaderboard

Config: set the channel-specific webhook URLs as env vars
  SLACK_WEBHOOK_STANDUP, SLACK_WEBHOOK_ALPHA, SLACK_WEBHOOK_PNL,
  SLACK_WEBHOOK_RISK, SLACK_WEBHOOK_INCIDENTS, SLACK_WEBHOOK_DEPLOYS,
  SLACK_WEBHOOK_CI, SLACK_WEBHOOK_ML
Or use the legacy SLACK_WEBHOOK_DEFAULT fallback.

Get webhook URLs at: https://api.slack.com/apps → Create New App → From Scratch
→ Incoming Webhooks → Add per-channel webhooks → copy each URL.
"""
from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Any

import httpx

from app.utils.logging import logger


class SlackChannel:
    STANDUP = "standup"
    ALPHA = "alpha"
    PNL = "pnl"
    RISK = "risk"
    INCIDENTS = "incidents"
    DEPLOYS = "deploys"
    CI = "ci"
    ML = "ml"


_ENV_BY_CHANNEL = {
    SlackChannel.STANDUP:   "SLACK_WEBHOOK_STANDUP",
    SlackChannel.ALPHA:     "SLACK_WEBHOOK_ALPHA",
    SlackChannel.PNL:       "SLACK_WEBHOOK_PNL",
    SlackChannel.RISK:      "SLACK_WEBHOOK_RISK",
    SlackChannel.INCIDENTS: "SLACK_WEBHOOK_INCIDENTS",
    SlackChannel.DEPLOYS:   "SLACK_WEBHOOK_DEPLOYS",
    SlackChannel.CI:        "SLACK_WEBHOOK_CI",
    SlackChannel.ML:        "SLACK_WEBHOOK_ML",
}


def _resolve_webhook(channel: str) -> str | None:
    """Pick the most specific webhook env var, fall back to the default."""
    env_name = _ENV_BY_CHANNEL.get(channel)
    if env_name:
        url = os.getenv(env_name, "").strip()
        if url:
            return url
    return os.getenv("SLACK_WEBHOOK_DEFAULT", "").strip() or None


async def post(
    channel: str,
    text: str,
    *,
    blocks: list[dict] | None = None,
    color: str | None = None,
) -> bool:
    """Post a message to the given channel. Returns True on success."""
    webhook = _resolve_webhook(channel)
    if not webhook:
        logger.debug("slack: no webhook for channel", channel=channel)
        return False

    payload: dict[str, Any] = {"text": text}
    if blocks:
        payload["blocks"] = blocks
    if color:
        # Slack attachment color shows on the left side
        payload["attachments"] = [{"color": color, "text": text}]

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(webhook, json=payload)
            r.raise_for_status()
            return True
    except Exception as e:
        logger.warning("slack post failed", channel=channel, error=str(e))
        return False


# ── High-level helpers used across the codebase ───────────────────────────

async def post_standup(squad: str, shipped: list[str], planned: list[str], blockers: list[str]) -> bool:
    blocks = [
        {"type": "header", "text": {"type": "plain_text", "text": f"🌅 {squad} standup — {datetime.now(UTC):%Y-%m-%d}"}},
        {"type": "section", "fields": [
            {"type": "mrkdwn", "text": "*Shipped*\n" + ("\n".join(f"• {x}" for x in shipped) or "_nothing yet_")},
            {"type": "mrkdwn", "text": "*Planned*\n" + ("\n".join(f"• {x}" for x in planned) or "_to be set_")},
        ]},
    ]
    if blockers:
        blocks.append({"type": "section", "text": {"type": "mrkdwn",
            "text": "*🚧 Blockers*\n" + "\n".join(f"• {x}" for x in blockers)}})
    return await post(SlackChannel.STANDUP, f"{squad} standup", blocks=blocks)


async def post_alpha_review(strategy: str, sharpe: float, maxdd: float, decision: str) -> bool:
    color = "good" if decision == "promoted" else "warning" if decision == "iterate" else "danger"
    text = f"📈 *{strategy}* — Sharpe {sharpe:.2f}, MaxDD {maxdd:.1%}, decision: {decision}"
    return await post(SlackChannel.ALPHA, text, color=color)


async def post_eod_pnl(date: str, total_pnl: float, top: list[tuple[str, float]], bottom: list[tuple[str, float]]) -> bool:
    sign = "+" if total_pnl >= 0 else ""
    top_str = "\n".join(f"  🟢 {s}: {sign}${p:,.0f}" for s, p in top[:5])
    bot_str = "\n".join(f"  🔴 {s}: ${p:,.0f}" for s, p in bottom[:5])
    text = f"💰 *EOD P&L {date}* — Total: {sign}${total_pnl:,.0f}\n\n*Top 5*\n{top_str}\n\n*Bottom 5*\n{bot_str}"
    color = "good" if total_pnl >= 0 else "danger"
    return await post(SlackChannel.PNL, text, color=color)


async def post_risk_alert(severity: str, message: str, metric: str | None = None, value: float | None = None) -> bool:
    color = {"P0": "danger", "P1": "warning", "P2": "good"}.get(severity, "warning")
    detail = f" ({metric}={value:.4f})" if metric and value is not None else ""
    return await post(SlackChannel.RISK, f"⚠️ *[{severity}] Risk*: {message}{detail}", color=color)


async def post_deploy(service: str, version: str, status: str, url: str | None = None) -> bool:
    color = "good" if status == "success" else "danger"
    link = f"\n<{url}|View deploy>" if url else ""
    return await post(SlackChannel.DEPLOYS, f"🚀 *{service}* deploy `{version}` → {status}{link}", color=color)


async def post_ci_failure(branch: str, run_url: str, failing_step: str) -> bool:
    text = f"❌ CI failed on `{branch}` — step *{failing_step}*\n<{run_url}|View logs>"
    return await post(SlackChannel.CI, text, color="danger")


async def post_incident(severity: str, component: str, description: str, oncall: str) -> bool:
    color = {"P0": "danger", "P1": "warning"}.get(severity, "good")
    text = f"🚨 *[{severity}] Incident* — {component}\n{description}\n_On-call: {oncall}_"
    return await post(SlackChannel.INCIDENTS, text, color=color)


async def post_ml_run_complete(model: str, symbol: str, val_sharpe: float, run_id: str) -> bool:
    color = "good" if val_sharpe > 1.0 else "warning"
    text = f"🧠 ML training complete — *{model}* on *{symbol}* — val Sharpe {val_sharpe:.2f} (run `{run_id}`)"
    return await post(SlackChannel.ML, text, color=color)
